"""Role-aware local, graph, and sparse dynamic communication models."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .graphs import normalize_by_incoming_degree


class InvertedTemporalEncoder(nn.Module):
    """Encode a multivariate history as variable tokens over the time axis."""

    def __init__(
        self,
        *,
        history_steps: int,
        d_model: int,
        n_heads: int,
        layers: int,
        feedforward_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.history_steps = int(history_steps)
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

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3 or sequence.shape[1] != self.history_steps:
            raise ValueError("unexpected inverted-encoder input shape")
        mean = sequence.mean(dim=1, keepdim=True).detach()
        scale = (
            sequence.var(dim=1, keepdim=True, unbiased=False) + 1e-5
        ).sqrt().detach()
        tokens = self.embedding(((sequence - mean) / scale).transpose(1, 2))
        for layer in self.layers:
            tokens = layer(tokens)
        return self.norm(tokens).mean(dim=1)


class RoleGraphITransformer(nn.Module):
    """Drop-in iTransformer replacement for the role-aware GRU backbone."""

    def __init__(
        self,
        *,
        history_steps: int,
        role_feature_counts: tuple[int, ...],
        active_roles: tuple[bool, ...],
        context_size: int,
        d_model: int,
        n_heads: int,
        layers: int,
        feedforward_size: int,
        dropout: float,
        fusion_hidden_size: int,
    ) -> None:
        super().__init__()
        if len(role_feature_counts) != len(active_roles):
            raise ValueError("role feature counts and masks must align")
        self.active_role_indices = tuple(
            index for index, active in enumerate(active_roles) if active
        )
        self.role_feature_counts = role_feature_counts
        encoder_args = {
            "history_steps": history_steps,
            "d_model": d_model,
            "n_heads": n_heads,
            "layers": layers,
            "feedforward_size": feedforward_size,
            "dropout": dropout,
        }
        self.role_encoders = nn.ModuleDict(
            {
                str(index): InvertedTemporalEncoder(**encoder_args)
                for index in self.active_role_indices
            }
        )
        self.context_encoder = InvertedTemporalEncoder(**encoder_args)
        self.state_size = (len(self.active_role_indices) + 1) * d_model
        self.message_projection = nn.Linear(self.state_size, self.state_size)
        self.fusion = nn.Sequential(
            nn.Linear(self.state_size, fusion_hidden_size),
            nn.ReLU(),
            nn.Linear(fusion_hidden_size, 1),
        )
        final = self.fusion[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def encode_local(
        self, x_role: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        batch, _, units, _, _ = x_role.shape
        states = []
        for role_index in self.active_role_indices:
            feature_count = self.role_feature_counts[role_index]
            sequence = (
                x_role[:, :, :, role_index, :feature_count]
                .permute(0, 2, 1, 3)
                .reshape(batch * units, x_role.shape[1], feature_count)
            )
            states.append(
                self.role_encoders[str(role_index)](sequence).reshape(
                    batch, units, -1
                )
            )
        context_state = self.context_encoder(context)
        states.append(context_state[:, None, :].expand(batch, units, -1))
        return torch.cat(states, dim=-1)

    def predict_from_state(
        self,
        local_state: torch.Tensor,
        y_current: torch.Tensor,
        adjacency: torch.Tensor,
        *,
        normalization_adjacency: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if normalization_adjacency is None:
            normalized = normalize_by_incoming_degree(adjacency)
        else:
            denominator = normalization_adjacency.float().sum(
                dim=-1, keepdim=True
            ).clamp_min(1.0)
            normalized = adjacency.float() / denominator
        messages = self.message_projection(local_state)
        received = torch.bmm(normalized, messages)
        delta = self.fusion(local_state + received).squeeze(-1)
        return y_current + delta

    def forward(
        self,
        x_role: torch.Tensor,
        context: torch.Tensor,
        y_current: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        state = self.encode_local(x_role, context)
        return self.predict_from_state(state, y_current, adjacency), None


class RoleGraphGRU(nn.Module):
    """Shared role encoders plus adjacency-controlled unit communication."""

    def __init__(
        self,
        *,
        role_feature_counts: tuple[int, ...],
        active_roles: tuple[bool, ...],
        context_size: int,
        role_hidden_size: int,
        context_hidden_size: int,
        fusion_hidden_size: int,
    ) -> None:
        super().__init__()
        if len(role_feature_counts) != len(active_roles):
            raise ValueError("role feature counts and masks must align")
        self.active_role_indices = tuple(
            index for index, active in enumerate(active_roles) if active
        )
        self.role_feature_counts = role_feature_counts
        self.role_encoders = nn.ModuleDict(
            {
                str(index): nn.GRU(
                    role_feature_counts[index],
                    role_hidden_size,
                    batch_first=True,
                )
                for index in self.active_role_indices
            }
        )
        self.context_encoder = nn.GRU(
            context_size, context_hidden_size, batch_first=True
        )
        self.state_size = (
            len(self.active_role_indices) * role_hidden_size
            + context_hidden_size
        )
        self.message_projection = nn.Linear(
            self.state_size, self.state_size
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.state_size, fusion_hidden_size),
            nn.ReLU(),
            nn.Linear(fusion_hidden_size, 1),
        )
        final = self.fusion[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def encode_local(
        self,
        x_role: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        batch, steps, units, _, _ = x_role.shape
        role_states = []
        for role_index in self.active_role_indices:
            feature_count = self.role_feature_counts[role_index]
            role_sequence = (
                x_role[:, :, :, role_index, :feature_count]
                .permute(0, 2, 1, 3)
                .reshape(batch * units, steps, feature_count)
            )
            _, hidden = self.role_encoders[str(role_index)](role_sequence)
            role_states.append(hidden[-1].reshape(batch, units, -1))
        _, context_hidden = self.context_encoder(context)
        context_state = context_hidden[-1][:, None, :].expand(
            batch, units, context_hidden.shape[-1]
        )
        return torch.cat(role_states + [context_state], dim=-1)

    def predict_from_state(
        self,
        local_state: torch.Tensor,
        y_current: torch.Tensor,
        adjacency: torch.Tensor,
        *,
        normalization_adjacency: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if normalization_adjacency is None:
            normalized = normalize_by_incoming_degree(adjacency)
        else:
            denominator = (
                normalization_adjacency.float()
                .sum(dim=-1, keepdim=True)
                .clamp_min(1.0)
            )
            normalized = adjacency.float() / denominator
        sender_messages = self.message_projection(local_state)
        received = torch.bmm(normalized, sender_messages)
        delta = self.fusion(local_state + received).squeeze(-1)
        return y_current + delta

    def forward(
        self,
        x_role: torch.Tensor,
        context: torch.Tensor,
        y_current: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        local_state = self.encode_local(x_role, context)
        prediction = self.predict_from_state(
            local_state, y_current, adjacency
        )
        return prediction, None


class GatedRoleGraphGRU(nn.Module):
    """Learn soft gates inside a fixed wide candidate graph."""

    def __init__(
        self,
        *,
        role_feature_counts: tuple[int, ...],
        active_roles: tuple[bool, ...],
        context_size: int,
        role_hidden_size: int,
        context_hidden_size: int,
        fusion_hidden_size: int,
        gate_hidden_size: int,
        coordinates_xy: np.ndarray,
        use_physical_features: bool,
        wind_context_indices: tuple[int, int, int],
        initial_logit_bias: float,
    ) -> None:
        super().__init__()
        self.backbone = RoleGraphGRU(
            role_feature_counts=role_feature_counts,
            active_roles=active_roles,
            context_size=context_size,
            role_hidden_size=role_hidden_size,
            context_hidden_size=context_hidden_size,
            fusion_hidden_size=fusion_hidden_size,
        )
        self.use_physical_features = bool(use_physical_features)
        self.wind_context_indices = wind_context_indices
        coordinates = np.asarray(coordinates_xy, dtype=np.float32)
        relative = coordinates[:, None, :] - coordinates[None, :, :]
        distance = np.sqrt(np.square(relative).sum(axis=-1, keepdims=True))
        maximum_distance = float(distance.max())
        if maximum_distance <= 0:
            raise ValueError("coordinates must contain distinct units")
        relative /= maximum_distance
        distance /= maximum_distance
        physical_edge_features = np.concatenate(
            [relative, distance], axis=-1
        )
        self.register_buffer(
            "physical_edge_features",
            torch.from_numpy(physical_edge_features),
        )
        extra_size = 6 if self.use_physical_features else 0
        gate_input_size = 2 * self.backbone.state_size + extra_size
        self.gate_network = nn.Sequential(
            nn.Linear(gate_input_size, gate_hidden_size),
            nn.ReLU(),
            nn.Linear(gate_hidden_size, 1),
        )
        final = self.gate_network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.constant_(final.bias, initial_logit_bias)

    def forward(
        self,
        x_role: torch.Tensor,
        context: torch.Tensor,
        y_current: torch.Tensor,
        candidate_adjacency: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_state = self.backbone.encode_local(x_role, context)
        batch, units, state_size = local_state.shape
        receiver = local_state[:, :, None, :].expand(
            batch, units, units, state_size
        )
        sender = local_state[:, None, :, :].expand(
            batch, units, units, state_size
        )
        gate_parts = [receiver, sender]
        if self.use_physical_features:
            edge_features = self.physical_edge_features[None].expand(
                batch, units, units, 3
            )
            wind_endpoint = context[:, -1, list(self.wind_context_indices)]
            wind_features = wind_endpoint[:, None, None, :].expand(
                batch, units, units, 3
            )
            gate_parts.extend([edge_features, wind_features])
        gate_logits = self.gate_network(torch.cat(gate_parts, dim=-1))
        gates = torch.sigmoid(gate_logits.squeeze(-1))
        candidate = candidate_adjacency.float()
        masked_gates = gates * candidate
        prediction = self.backbone.predict_from_state(
            local_state,
            y_current,
            masked_gates,
            normalization_adjacency=candidate,
        )
        return prediction, masked_gates


class HardTopKRoleGraphGRU(nn.Module):
    """Select exactly ``top_k`` incoming senders per unit and sample.

    The forward pass uses a binary graph, so unselected messages are not
    evaluated by the graph aggregation.  During training a straight-through
    estimator supplies gradients to the selector scores.
    """

    def __init__(
        self,
        *,
        role_feature_counts: tuple[int, ...],
        active_roles: tuple[bool, ...],
        context_size: int,
        role_hidden_size: int,
        context_hidden_size: int,
        fusion_hidden_size: int,
        gate_hidden_size: int,
        coordinates_xy: np.ndarray,
        use_physical_features: bool,
        wind_context_indices: tuple[int, int, int],
        top_k: int,
        temperature: float,
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone or RoleGraphGRU(
                role_feature_counts=role_feature_counts,
                active_roles=active_roles,
                context_size=context_size,
                role_hidden_size=role_hidden_size,
                context_hidden_size=context_hidden_size,
                fusion_hidden_size=fusion_hidden_size,
            )
        self.use_physical_features = bool(use_physical_features)
        self.wind_context_indices = wind_context_indices
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

        coordinates = np.asarray(coordinates_xy, dtype=np.float32)
        relative = coordinates[:, None, :] - coordinates[None, :, :]
        distance = np.sqrt(np.square(relative).sum(axis=-1, keepdims=True))
        maximum_distance = float(distance.max())
        if maximum_distance <= 0:
            raise ValueError("coordinates must contain distinct units")
        relative /= maximum_distance
        distance /= maximum_distance
        self.register_buffer(
            "physical_edge_features",
            torch.from_numpy(np.concatenate([relative, distance], axis=-1)),
        )

        extra_size = 6 if self.use_physical_features else 0
        selector_input_size = 2 * self.backbone.state_size + extra_size
        self.selector = nn.Sequential(
            nn.Linear(selector_input_size, gate_hidden_size),
            nn.ReLU(),
            nn.Linear(gate_hidden_size, 1),
        )

    def _selector_logits(
        self,
        local_state: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        batch, units, state_size = local_state.shape
        receiver = local_state[:, :, None, :].expand(
            batch, units, units, state_size
        )
        sender = local_state[:, None, :, :].expand(
            batch, units, units, state_size
        )
        parts = [receiver, sender]
        if self.use_physical_features:
            edge_features = self.physical_edge_features[None].expand(
                batch, units, units, 3
            )
            wind_endpoint = context[:, -1, list(self.wind_context_indices)]
            wind_features = wind_endpoint[:, None, None, :].expand(
                batch, units, units, 3
            )
            parts.extend([edge_features, wind_features])
        return self.selector(torch.cat(parts, dim=-1)).squeeze(-1)

    def select_edges(
        self,
        logits: torch.Tensor,
        candidate_adjacency: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        candidate = candidate_adjacency.bool()
        if candidate.ndim != 3 or candidate.shape != logits.shape:
            raise ValueError("candidate adjacency and logits must align")
        available = candidate.sum(dim=-1)
        if torch.any(available < self.top_k):
            raise ValueError("each receiver needs at least top_k candidates")

        ranked_logits = logits.masked_fill(~candidate, float("-inf"))
        indices = ranked_logits.topk(self.top_k, dim=-1).indices
        hard_edges = torch.zeros_like(logits).scatter(-1, indices, 1.0)
        hard_edges = hard_edges * candidate.float()

        probabilities = torch.sigmoid(logits / self.temperature)
        probabilities = probabilities * candidate.float()
        if self.training:
            selected = hard_edges + probabilities - probabilities.detach()
        else:
            selected = hard_edges
        return selected, hard_edges

    def forward(
        self,
        x_role: torch.Tensor,
        context: torch.Tensor,
        y_current: torch.Tensor,
        candidate_adjacency: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_state = self.backbone.encode_local(x_role, context)
        logits = self._selector_logits(local_state, context)
        selected, hard_edges = self.select_edges(logits, candidate_adjacency)
        prediction = self.backbone.predict_from_state(
            local_state,
            y_current,
            selected,
        )
        return prediction, hard_edges


class HardTopKRoleGraphITransformer(HardTopKRoleGraphGRU):
    """Hard Top-k communication with an iTransformer local encoder."""

    def __init__(
        self,
        *,
        history_steps: int,
        role_feature_counts: tuple[int, ...],
        active_roles: tuple[bool, ...],
        context_size: int,
        d_model: int,
        n_heads: int,
        layers: int,
        feedforward_size: int,
        dropout: float,
        fusion_hidden_size: int,
        gate_hidden_size: int,
        coordinates_xy: np.ndarray,
        wind_context_indices: tuple[int, int, int],
        top_k: int,
        temperature: float,
    ) -> None:
        backbone = RoleGraphITransformer(
            history_steps=history_steps,
            role_feature_counts=role_feature_counts,
            active_roles=active_roles,
            context_size=context_size,
            d_model=d_model,
            n_heads=n_heads,
            layers=layers,
            feedforward_size=feedforward_size,
            dropout=dropout,
            fusion_hidden_size=fusion_hidden_size,
        )
        super().__init__(
            role_feature_counts=role_feature_counts,
            active_roles=active_roles,
            context_size=context_size,
            role_hidden_size=d_model,
            context_hidden_size=d_model,
            fusion_hidden_size=fusion_hidden_size,
            gate_hidden_size=gate_hidden_size,
            coordinates_xy=coordinates_xy,
            use_physical_features=False,
            wind_context_indices=wind_context_indices,
            top_k=top_k,
            temperature=temperature,
            backbone=backbone,
        )
