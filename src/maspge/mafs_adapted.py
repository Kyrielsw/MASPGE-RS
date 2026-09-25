"""Task-adapted MAFS for 60-to-15 SMARTEOLE power forecasting.

This module preserves the official method's multi-scale iTransformer agents,
inter-agent graph communication, two-stage adjacency learning, and
sample-dependent voting.  History/future scales are proportionally adapted
because the official implementation hard-codes 96-step and longer tasks.
"""

from __future__ import annotations

import torch
from torch import nn


def normalized_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("agent adjacency must be square")
    with_self = adjacency + torch.eye(
        adjacency.shape[0], device=adjacency.device, dtype=adjacency.dtype
    )
    degree = with_self.sum(dim=1).clamp_min(1.0)
    inverse_sqrt = degree.rsqrt()
    return inverse_sqrt[:, None] * with_self * inverse_sqrt[None, :]


def topology_mask(agent_count: int, mode: str) -> torch.Tensor:
    mask = torch.zeros(agent_count, agent_count)
    if mode == "fully":
        mask.fill_(1)
        mask.fill_diagonal_(0)
    elif mode == "ring":
        for index in range(agent_count):
            mask[index, (index + 1) % agent_count] = 1
            mask[index, (index - 1) % agent_count] = 1
    elif mode == "chain":
        for index in range(agent_count - 1):
            mask[index, index + 1] = 1
            mask[index + 1, index] = 1
    elif mode == "star":
        for index in range(1, agent_count):
            mask[0, index] = 1
            mask[index, 0] = 1
    else:
        raise ValueError(f"unknown MAFS topology: {mode}")
    return mask


class LearnableAgentAdjacency(nn.Module):
    def __init__(self, agent_count: int, topology: str) -> None:
        super().__init__()
        self.edge_logits = nn.Parameter(torch.rand(agent_count, agent_count))
        mask = topology_mask(agent_count, topology)
        self.register_buffer("off_diagonal_mask", mask)

    def forward(self) -> torch.Tensor:
        weighted = torch.sigmoid(self.edge_logits) * self.off_diagonal_mask
        symmetric = (weighted + weighted.transpose(0, 1)) / 2
        return normalized_adjacency(symmetric)


class AgentGraphCommunication(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, d_model)

    def forward(
        self, embeddings: list[torch.Tensor], adjacency: torch.Tensor
    ) -> torch.Tensor:
        stacked = torch.stack(embeddings, dim=0)
        communicated = torch.einsum("ij,jbud->ibud", adjacency, stacked)
        return self.projection(communicated)


class MultiScaleITransformerAgent(nn.Module):
    def __init__(
        self,
        *,
        history_steps: int,
        horizon_steps: int,
        d_model: int,
        n_heads: int,
        layers: int,
        feedforward_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.history_steps = int(history_steps)
        self.horizon_steps = int(horizon_steps)
        self.embedding = nn.Linear(self.history_steps, d_model)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=feedforward_size,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, self.horizon_steps)

    def encode(
        self, full_history: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history = full_history[:, -self.history_steps :]
        mean = history.mean(dim=1, keepdim=True).detach()
        scale = (
            history.var(dim=1, keepdim=True, unbiased=False) + 1e-5
        ).sqrt().detach()
        normalized = (history - mean) / scale
        tokens = self.embedding(normalized.transpose(1, 2))
        return tokens, mean, scale

    def project(
        self,
        tokens: torch.Tensor,
        mean: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        sequence = self.projection(self.norm(tokens)).permute(0, 2, 1)
        return sequence * scale + mean


class PerUnitRoleStateAdapter(nn.Module):
    """Encode A1--A4 histories per unit and condition each MAFS specialist.

    The final residual projections are zero initialized.  Consequently a new
    adapter is exactly equivalent to the frozen MAFS base before training.
    """

    def __init__(
        self,
        *,
        history_scales: tuple[int, ...],
        role_feature_counts: tuple[int, ...],
        role_hidden_size: int,
        d_model: int,
    ) -> None:
        super().__init__()
        self.history_scales = tuple(int(value) for value in history_scales)
        self.role_feature_counts = tuple(
            int(value) for value in role_feature_counts
        )
        self.active_role_indices = tuple(
            index for index, count in enumerate(self.role_feature_counts)
            if count > 0
        )
        if not self.active_role_indices:
            raise ValueError("role-state adapter requires at least one role")
        self.encoders = nn.ModuleDict()
        self.fusions = nn.ModuleList()
        self.residual_projections = nn.ModuleList()
        for agent_index, _ in enumerate(self.history_scales):
            for role_index in self.active_role_indices:
                self.encoders[f"{agent_index}:{role_index}"] = nn.GRU(
                    input_size=self.role_feature_counts[role_index],
                    hidden_size=role_hidden_size,
                    batch_first=True,
                )
            self.fusions.append(
                nn.Sequential(
                    nn.Linear(
                        len(self.active_role_indices) * role_hidden_size,
                        d_model,
                    ),
                    nn.GELU(),
                    nn.LayerNorm(d_model),
                )
            )
            residual = nn.Linear(d_model, d_model)
            nn.init.zeros_(residual.weight)
            nn.init.zeros_(residual.bias)
            self.residual_projections.append(residual)

    def forward(
        self,
        x_role: torch.Tensor,
        *,
        ablation_mode: str = "full",
    ) -> list[torch.Tensor]:
        if x_role.ndim != 5:
            raise ValueError(
                "per-unit role state requires x_role shaped [B,H,U,R,F]"
            )
        if ablation_mode not in {
            "full", "shared_units", "unscaled_history",
            "without_a1", "without_a2", "without_a3", "without_a4",
        }:
            raise ValueError(
                f"unknown role-state ablation mode: {ablation_mode}"
            )
        batch, _, units, _, _ = x_role.shape
        adapter_input = x_role
        if ablation_mode == "shared_units":
            # Remove unit-specific conditioning while preserving the adapter
            # architecture and parameter count.  Every unit receives the same
            # role summary formed from the unit-average history.
            adapter_input = x_role.mean(dim=2, keepdim=True).expand(
                -1, -1, units, -1, -1
            )
        outputs = []
        for agent_index, history_steps in enumerate(self.history_scales):
            role_states = []
            selected_steps = (
                self.history_scales[-1]
                if ablation_mode == "unscaled_history"
                else history_steps
            )
            history = adapter_input[:, -selected_steps:]
            for role_index in self.active_role_indices:
                feature_count = self.role_feature_counts[role_index]
                masked_role = (
                    int(ablation_mode[-1]) - 1
                    if ablation_mode.startswith("without_a") else None
                )
                if role_index == masked_role:
                    role_states.append(
                        torch.zeros(
                            batch, units,
                            self.encoders[f"{agent_index}:{role_index}"].hidden_size,
                            device=x_role.device,
                            dtype=x_role.dtype,
                        )
                    )
                else:
                    sequence = history[:, :, :, role_index, :feature_count]
                    sequence = torch.nan_to_num(
                        sequence, nan=0.0, posinf=0.0, neginf=0.0
                    )
                    sequence = sequence.permute(0, 2, 1, 3).reshape(
                        batch * units, selected_steps, feature_count
                    )
                    _, hidden = self.encoders[
                        f"{agent_index}:{role_index}"
                    ](sequence)
                    role_states.append(hidden[-1].reshape(batch, units, -1))
            fused = self.fusions[agent_index](torch.cat(role_states, dim=-1))
            outputs.append(self.residual_projections[agent_index](fused))
        return outputs


class PerUnitGenericStateAdapter(nn.Module):
    """Encode all valid sensors of each unit without A1--A4 factorization.

    This control preserves the unit axis, multi-scale histories, zero-initialized
    pre-communication residual injection, sensor values, and frozen MAFS
    backbone used by :class:`PerUnitRoleStateAdapter`.  Its sole encoder at each
    scale receives the concatenation of every valid role slot, so the network
    does not contain role-specific encoders or role-wise fusion.
    """

    def __init__(
        self,
        *,
        history_scales: tuple[int, ...],
        role_feature_counts: tuple[int, ...],
        state_hidden_size: int,
        d_model: int,
        residual_initialization: str = "zero",
        residual_initialization_std: float = 1e-3,
    ) -> None:
        super().__init__()
        self.history_scales = tuple(int(value) for value in history_scales)
        self.role_feature_counts = tuple(
            int(value) for value in role_feature_counts
        )
        self.total_feature_count = sum(self.role_feature_counts)
        self.residual_initialization = str(residual_initialization)
        self.residual_initialization_std = float(residual_initialization_std)
        if self.residual_initialization not in {"zero", "small_random", "default"}:
            raise ValueError(
                "residual_initialization must be zero, small_random, or default"
            )
        if self.total_feature_count <= 0:
            raise ValueError("generic-state adapter requires valid sensors")
        if state_hidden_size <= 0:
            raise ValueError("state_hidden_size must be positive")
        self.encoders = nn.ModuleList()
        self.fusions = nn.ModuleList()
        self.residual_projections = nn.ModuleList()
        for _ in self.history_scales:
            self.encoders.append(
                nn.GRU(
                    input_size=self.total_feature_count,
                    hidden_size=int(state_hidden_size),
                    batch_first=True,
                )
            )
            self.fusions.append(
                nn.Sequential(
                    nn.Linear(int(state_hidden_size), d_model),
                    nn.GELU(),
                    nn.LayerNorm(d_model),
                )
            )
            residual = nn.Linear(d_model, d_model)
            self.residual_projections.append(residual)
        # Apply the controlled projection initialization only after every GRU
        # and fusion module has been constructed. This keeps all upstream
        # parameters bitwise matched between zero and small-random variants.
        if self.residual_initialization != "default":
            for residual in self.residual_projections:
                if self.residual_initialization == "zero":
                    nn.init.zeros_(residual.weight)
                else:
                    nn.init.normal_(
                        residual.weight,
                        mean=0.0,
                        std=self.residual_initialization_std,
                    )
                nn.init.zeros_(residual.bias)

    def flatten_valid_sensors(self, x_role: torch.Tensor) -> torch.Tensor:
        if x_role.ndim != 5:
            raise ValueError(
                "per-unit generic state requires x_role shaped [B,H,U,R,F]"
            )
        if x_role.shape[3] != len(self.role_feature_counts):
            raise ValueError("role axis does not match feature-count contract")
        pieces = [
            x_role[:, :, :, role_index, :feature_count]
            for role_index, feature_count in enumerate(self.role_feature_counts)
            if feature_count > 0
        ]
        return torch.nan_to_num(
            torch.cat(pieces, dim=-1), nan=0.0, posinf=0.0, neginf=0.0
        )

    def forward(
        self,
        x_role: torch.Tensor,
        *,
        ablation_mode: str = "full",
        unit_state_mode: str = "unitwise",
    ) -> list[torch.Tensor]:
        if ablation_mode != "full":
            raise ValueError(
                "generic-state control supports only ablation_mode='full'"
            )
        flat = self.flatten_valid_sensors(x_role)
        if unit_state_mode == "pooled_broadcast":
            flat = flat.mean(dim=2, keepdim=True).expand_as(flat)
        elif unit_state_mode != "unitwise":
            raise ValueError(
                "unit_state_mode must be unitwise or pooled_broadcast"
            )
        batch, _, units, _ = flat.shape
        outputs = []
        for index, history_steps in enumerate(self.history_scales):
            sequence = flat[:, -history_steps:].permute(0, 2, 1, 3).reshape(
                batch * units, history_steps, self.total_feature_count
            )
            _, hidden = self.encoders[index](sequence)
            fused = self.fusions[index](hidden[-1].reshape(batch, units, -1))
            outputs.append(self.residual_projections[index](fused))
        return outputs


class MAFSAdaptedPower(nn.Module):
    """Four multi-scale agents with MAFS communication and voting."""

    def __init__(
        self,
        *,
        history_scales: tuple[int, ...],
        horizon_scales: tuple[int, ...],
        unit_count: int,
        full_horizon_steps: int,
        d_model: int,
        n_heads: int,
        layers: int,
        feedforward_size: int,
        dropout: float,
        topology: str = "fully",
        role_feature_counts: tuple[int, ...] | None = None,
        context_size: int | None = None,
        router_hidden_size: int = 0,
        role_state_hidden_size: int = 0,
    ) -> None:
        super().__init__()
        if len(history_scales) != len(horizon_scales):
            raise ValueError("history and horizon scales must align")
        if len(history_scales) < 2:
            raise ValueError("MAFS requires multiple specialized agents")
        if max(horizon_scales) != full_horizon_steps:
            raise ValueError("one agent must cover the full future horizon")
        self.history_scales = history_scales
        self.horizon_scales = horizon_scales
        self.full_history_steps = max(history_scales)
        self.full_horizon_steps = int(full_horizon_steps)
        self.agent_count = len(history_scales)
        self.agents = nn.ModuleList(
            [
                MultiScaleITransformerAgent(
                    history_steps=history,
                    horizon_steps=horizon,
                    d_model=d_model,
                    n_heads=n_heads,
                    layers=layers,
                    feedforward_size=feedforward_size,
                    dropout=dropout,
                )
                for history, horizon in zip(history_scales, horizon_scales)
            ]
        )
        fixed = topology_mask(self.agent_count, topology)
        self.register_buffer("fixed_adjacency", normalized_adjacency(fixed))
        self.learnable_adjacency = LearnableAgentAdjacency(
            self.agent_count, topology
        )
        self.communication = AgentGraphCommunication(d_model)
        self.context_embedding = nn.Linear(self.full_history_steps, d_model)
        self.context_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.Sigmoid()
        )
        self.vote_over_variables = nn.Linear(unit_count, 1)
        self.vote_over_time = nn.Linear(
            self.full_history_steps, self.agent_count
        )
        self.final_projection = nn.Linear(d_model, self.full_horizon_steps)
        self.role_router: nn.Module | None = None
        self.role_temporal_embeddings = nn.ModuleDict()
        self.router_context_embedding: nn.Module | None = None
        self.role_state_adapter: PerUnitRoleStateAdapter | None = None
        if role_feature_counts is not None:
            self.active_role_indices = tuple(
                index
                for index, count in enumerate(role_feature_counts)
                if int(count) > 0
            )
            if not self.active_role_indices or context_size is None:
                raise ValueError("role routing requires active roles and context")
            if router_hidden_size <= 0:
                raise ValueError("router_hidden_size must be positive")
            self.role_feature_counts = tuple(int(x) for x in role_feature_counts)
            self.role_temporal_embeddings = nn.ModuleDict(
                {
                    str(index): nn.Linear(
                        self.full_history_steps, router_hidden_size
                    )
                    for index in self.active_role_indices
                }
            )
            self.router_context_embedding = nn.Linear(
                self.full_history_steps, router_hidden_size
            )
            self.role_router = nn.Sequential(
                nn.Linear(
                    (len(self.active_role_indices) + 1) * router_hidden_size,
                    router_hidden_size,
                ),
                nn.GELU(),
                nn.Linear(router_hidden_size, self.agent_count),
            )
            final_router = self.role_router[-1]
            assert isinstance(final_router, nn.Linear)
            nn.init.zeros_(final_router.weight)
            nn.init.zeros_(final_router.bias)
        if role_state_hidden_size > 0:
            if role_feature_counts is None:
                raise ValueError(
                    "role-state conditioning requires role feature counts"
                )
            self.role_state_adapter = PerUnitRoleStateAdapter(
                history_scales=self.history_scales,
                role_feature_counts=tuple(int(x) for x in role_feature_counts),
                role_hidden_size=int(role_state_hidden_size),
                d_model=d_model,
            )

    def freeze_specialist_agents(self) -> None:
        for agent in self.agents:
            for parameter in agent.parameters():
                parameter.requires_grad = False

    def freeze_for_role_routing(self) -> None:
        """Freeze the complete MAFS base and train only the role router."""

        if self.role_router is None:
            raise RuntimeError("this MAFS model has no role router")
        for parameter in self.parameters():
            parameter.requires_grad = False
        for module in (
            self.role_temporal_embeddings,
            self.router_context_embedding,
            self.role_router,
        ):
            assert module is not None
            for parameter in module.parameters():
                parameter.requires_grad = True

    def freeze_for_role_state(self) -> None:
        """Freeze the MAFS base and train only per-unit role conditioning."""

        if self.role_state_adapter is None:
            raise RuntimeError("this MAFS model has no role-state adapter")
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.role_state_adapter.parameters():
            parameter.requires_grad = True

    def role_route_logits(
        self,
        x_role: torch.Tensor,
        context: torch.Tensor,
        *,
        router_input_mode: str = "full",
    ) -> torch.Tensor:
        if self.role_router is None or self.router_context_embedding is None:
            raise RuntimeError("role router is not configured")
        if router_input_mode not in {
            "full", "roles_only", "context_only", "constant"
        }:
            raise ValueError(f"unknown router input mode: {router_input_mode}")
        summaries = []
        for role_index in self.active_role_indices:
            feature_count = self.role_feature_counts[role_index]
            if x_role.ndim == 3:
                temporal = x_role[:, :, role_index]
            elif x_role.ndim == 5:
                temporal = x_role[:, :, :, role_index, :feature_count].mean(
                    dim=(2, 3)
                )
            else:
                raise ValueError(
                    "x_role must be compact [B,H,R] or dense [B,H,U,R,F]"
                )
            embedded = self.role_temporal_embeddings[str(role_index)](temporal)
            if router_input_mode in {"context_only", "constant"}:
                embedded = torch.zeros_like(embedded)
            summaries.append(embedded)
        context_temporal = context.mean(dim=-1)
        embedded_context = self.router_context_embedding(context_temporal)
        if router_input_mode in {"roles_only", "constant"}:
            embedded_context = torch.zeros_like(embedded_context)
        summaries.append(embedded_context)
        return self.role_router(torch.cat(summaries, dim=-1))

    def forward(
        self,
        power_history: torch.Tensor,
        *,
        finetune: bool,
        x_role: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        use_role_router: bool = False,
        router_input_mode: str = "full",
        use_role_state: bool = False,
        role_state_ablation_mode: str = "full",
        generic_state_mode: str = "unitwise",
        role_state_injection: str = "pre_communication",
        role_state_unit_shift: int = 0,
        return_communication_trace: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if power_history.shape[1] != self.full_history_steps:
            raise ValueError("unexpected MAFS history length")
        encoded = [agent.encode(power_history) for agent in self.agents]
        states = [item[0] for item in encoded]
        role_state_residuals = [torch.zeros_like(state) for state in states]
        if use_role_state:
            if self.role_state_adapter is None or x_role is None:
                raise ValueError("role-state conditioning requires dense x_role")
            if isinstance(self.role_state_adapter, PerUnitGenericStateAdapter):
                role_state_residuals = self.role_state_adapter(
                    x_role,
                    ablation_mode=role_state_ablation_mode,
                    unit_state_mode=generic_state_mode,
                )
            else:
                if generic_state_mode != "unitwise":
                    raise ValueError(
                        "generic_state_mode applies only to the generic adapter"
                    )
                role_state_residuals = self.role_state_adapter(
                    x_role, ablation_mode=role_state_ablation_mode
                )
            if role_state_unit_shift:
                role_state_residuals = [
                    torch.roll(
                        residual,
                        shifts=int(role_state_unit_shift),
                        dims=1,
                    )
                    for residual in role_state_residuals
                ]
            if role_state_injection == "pre_communication":
                states = [
                    state + residual
                    for state, residual in zip(states, role_state_residuals)
                ]
            elif role_state_injection != "post_communication":
                raise ValueError(
                    "role_state_injection must be pre_communication or "
                    "post_communication"
                )
        adjacency = (
            self.learnable_adjacency()
            if finetune
            else self.fixed_adjacency
        )
        messages: torch.Tensor | None = None
        communication_trace: list[torch.Tensor] = []
        for layer_index in range(len(self.agents[0].layers)):
            outputs = []
            for agent_index, agent in enumerate(self.agents):
                layer_input = states[agent_index]
                if messages is not None:
                    layer_input = layer_input + messages[agent_index]
                outputs.append(agent.layers[layer_index](layer_input))
            messages = self.communication(outputs, adjacency)
            if return_communication_trace:
                communication_trace.append(messages)
            states = outputs

        if use_role_state and role_state_injection == "post_communication":
            states = [
                state + residual
                for state, residual in zip(states, role_state_residuals)
            ]

        agent_sequences = [
            agent.project(state, encoded[index][1], encoded[index][2])
            for index, (agent, state) in enumerate(zip(self.agents, states))
        ]

        full_mean = power_history.mean(dim=1, keepdim=True).detach()
        full_scale = (
            power_history.var(dim=1, keepdim=True, unbiased=False) + 1e-5
        ).sqrt().detach()
        normalized = (power_history - full_mean) / full_scale
        power_context = self.context_embedding(normalized.transpose(1, 2))
        gated_states = []
        for state in states:
            gate = self.context_gate(torch.cat([power_context, state], dim=-1))
            gated_states.append(gate * state + (1 - gate) * power_context)
        vote_features = self.vote_over_variables(normalized).squeeze(-1)
        vote_logits = self.vote_over_time(vote_features)
        route_logits = torch.zeros_like(vote_logits)
        if use_role_router:
            if x_role is None or context is None:
                raise ValueError("role routing requires x_role and context")
            route_logits = self.role_route_logits(
                x_role, context, router_input_mode=router_input_mode
            )
            vote_logits = vote_logits + route_logits
        vote_weights = torch.softmax(vote_logits, dim=-1)
        stacked_states = torch.stack(gated_states, dim=1)
        combined = (
            stacked_states * vote_weights[:, :, None, None]
        ).sum(dim=1)
        final_sequence = self.final_projection(combined).permute(0, 2, 1)
        final_sequence = final_sequence * full_scale + full_mean
        role_state_mean_square = torch.stack(
            [residual.square().mean() for residual in role_state_residuals]
        ).mean()
        return {
            "final_sequence": final_sequence,
            "agent_sequences": agent_sequences,
            "adjacency": adjacency,
            "vote_weights": vote_weights,
            "role_route_logits": route_logits,
            # RMS is diagnostic only. Detaching avoids the undefined gradient
            # of sqrt(x) at the deliberately zero-initialized residual x=0.
            "role_state_rms": role_state_mean_square.detach().sqrt(),
            "role_state_mean_square": role_state_mean_square,
            "role_state_residuals": role_state_residuals,
            "communication_trace": communication_trace,
        }
