"""Reproducible external forecasting baselines for SMARTEOLE.

The power-only variants preserve the core input/output convention of the
published models.  Full-information variants are explicitly named
``*_fullinfo_adapted`` because their input adapters are project-specific.
"""

from __future__ import annotations

import torch
from torch import nn


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd number")
        self.kernel_size = int(kernel_size)
        self.pool = nn.AvgPool1d(
            kernel_size=self.kernel_size, stride=1, padding=0
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        # sequence: [batch, time, channels]
        padding = (self.kernel_size - 1) // 2
        front = sequence[:, :1].expand(-1, padding, -1)
        end = sequence[:, -1:].expand(-1, padding, -1)
        padded = torch.cat([front, sequence, end], dim=1)
        return self.pool(padded.transpose(1, 2)).transpose(1, 2)


class DLinearPower(nn.Module):
    """DLinear-style decomposition with one output per power series."""

    def __init__(self, history_steps: int, kernel_size: int = 25) -> None:
        super().__init__()
        self.moving_average = MovingAverage(kernel_size)
        self.seasonal = nn.Linear(history_steps, 1)
        self.trend = nn.Linear(history_steps, 1)

    def forward(
        self,
        *,
        power_history: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        trend = self.moving_average(power_history)
        seasonal = power_history - trend
        prediction = self.seasonal(seasonal.transpose(1, 2))
        prediction += self.trend(trend.transpose(1, 2))
        return prediction.squeeze(-1)


class TimeMixerPDM(nn.Module):
    """Past-Decomposable-Mixing block from TimeMixer (ICLR 2024)."""

    def __init__(self, *, lengths: tuple[int, ...], d_model: int,
                 d_ff: int, moving_average_kernel: int,
                 dropout: float) -> None:
        super().__init__()
        self.decomposition = MovingAverage(moving_average_kernel)
        self.season_down = nn.ModuleList([
            nn.Sequential(nn.Linear(source, target), nn.GELU(),
                          nn.Linear(target, target))
            for source, target in zip(lengths[:-1], lengths[1:])
        ])
        self.trend_up = nn.ModuleList([
            nn.Sequential(nn.Linear(source, target), nn.GELU(),
                          nn.Linear(target, target))
            for source, target in zip(
                reversed(lengths[1:]), reversed(lengths[:-1])
            )
        ])
        self.feedforward = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                          nn.Linear(d_ff, d_model))
            for _ in lengths
        ])

    def forward(self, scales: list[torch.Tensor]) -> list[torch.Tensor]:
        seasonal, trend = [], []
        for scale in scales:
            moving = self.decomposition(scale)
            seasonal.append((scale - moving).transpose(1, 2))
            trend.append(moving.transpose(1, 2))

        mixed_seasonal = [seasonal[0]]
        current = seasonal[0]
        for layer, coarser in zip(self.season_down, seasonal[1:]):
            current = coarser + layer(current)
            mixed_seasonal.append(current)

        reverse_trend = list(reversed(trend))
        mixed_trend_reverse = [reverse_trend[0]]
        current = reverse_trend[0]
        for layer, finer in zip(self.trend_up, reverse_trend[1:]):
            current = finer + layer(current)
            mixed_trend_reverse.append(current)
        mixed_trend = list(reversed(mixed_trend_reverse))

        outputs = []
        for original, season, moving, feedforward in zip(
            scales, mixed_seasonal, mixed_trend, self.feedforward
        ):
            mixed = (season + moving).transpose(1, 2)
            outputs.append(original + feedforward(mixed))
        return outputs


class TimeMixerPower(nn.Module):
    """Power-only TimeMixer adapted to the frozen future-mean target.

    PDM and Future-Multipredictor-Mixing are retained. Each unit is processed
    independently, matching the official channel-independence protocol.
    """

    def __init__(self, *, history_steps: int, d_model: int, d_ff: int,
                 layers: int, down_sampling_layers: int,
                 down_sampling_window: int, moving_average_kernel: int,
                 dropout: float) -> None:
        super().__init__()
        lengths = [int(history_steps)]
        for _ in range(int(down_sampling_layers)):
            lengths.append(lengths[-1] // int(down_sampling_window))
        if min(lengths) < 2:
            raise ValueError("too many TimeMixer down-sampling layers")
        self.lengths = tuple(lengths)
        self.down_sampling_window = int(down_sampling_window)
        self.embedding = nn.Linear(1, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        self.pdm_blocks = nn.ModuleList([
            TimeMixerPDM(
                lengths=self.lengths, d_model=d_model, d_ff=d_ff,
                moving_average_kernel=moving_average_kernel, dropout=dropout,
            )
            for _ in range(int(layers))
        ])
        self.predictors = nn.ModuleList([
            nn.Linear(length, 1) for length in self.lengths
        ])
        self.projection = nn.Linear(d_model, 1)

    def _multiscale_inputs(
        self, normalized: torch.Tensor
    ) -> list[torch.Tensor]:
        """Return channel-independent [B*U, L_i, 1] scale sequences."""
        batch, _, units = normalized.shape
        sampled = normalized.transpose(1, 2).contiguous()
        scales = []
        for index in range(len(self.lengths)):
            scales.append(
                sampled.reshape(batch * units, sampled.shape[-1], 1)
            )
            if index + 1 < len(self.lengths):
                sampled = torch.nn.functional.avg_pool1d(
                    sampled, kernel_size=self.down_sampling_window,
                    stride=self.down_sampling_window,
                )
        return scales

    def forward(self, *, power_history: torch.Tensor,
                **_: torch.Tensor) -> torch.Tensor:
        batch, _, units = power_history.shape
        mean = power_history.mean(dim=1, keepdim=True).detach()
        centered = power_history - mean
        scale = (centered.var(dim=1, keepdim=True, unbiased=False) + 1e-5
                 ).sqrt().detach()
        raw_scales = self._multiscale_inputs(centered / scale)
        embedded = [
            self.embedding_dropout(self.embedding(raw))
            for raw in raw_scales
        ]
        for block in self.pdm_blocks:
            embedded = block(embedded)
        predictions = []
        for encoded, predictor in zip(embedded, self.predictors):
            temporal = predictor(encoded.transpose(1, 2)).transpose(1, 2)
            predictions.append(self.projection(temporal).squeeze(-1))
        prediction = torch.stack(predictions, dim=0).sum(dim=0)
        prediction = prediction.reshape(batch, units)
        return prediction * scale.squeeze(1) + mean.squeeze(1)


class ITransformerPower(nn.Module):
    """Compact iTransformer preserving variates-as-tokens semantics."""

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
        self.embedding = nn.Linear(history_steps, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=feedforward_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=layers, enable_nested_tensor=False
        )
        self.head = nn.Linear(d_model, 1)

    def forward(
        self,
        *,
        power_history: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        mean = power_history.mean(dim=1, keepdim=True).detach()
        variance = power_history.var(dim=1, keepdim=True, unbiased=False)
        scale = (variance + 1e-5).sqrt().detach()
        normalized = (power_history - mean) / scale
        tokens = self.embedding(normalized.transpose(1, 2))
        prediction = self.head(self.encoder(tokens)).squeeze(-1)
        return prediction * scale.squeeze(1) + mean.squeeze(1)


class FullInformationAdapter(nn.Module):
    """Build per-unit sequences from valid role slots and global context."""

    def __init__(
        self,
        *,
        role_feature_counts: tuple[int, ...],
        active_roles: tuple[bool, ...],
    ) -> None:
        super().__init__()
        if len(role_feature_counts) != len(active_roles):
            raise ValueError("role counts and active mask must align")
        self.role_feature_counts = tuple(int(v) for v in role_feature_counts)
        self.active_role_indices = tuple(
            i for i, active in enumerate(active_roles) if active
        )

    @property
    def local_feature_count(self) -> int:
        return sum(
            self.role_feature_counts[i] for i in self.active_role_indices
        )

    def forward(
        self, x_role: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        parts = [
            x_role[:, :, :, role, : self.role_feature_counts[role]]
            for role in self.active_role_indices
        ]
        local = torch.cat(parts, dim=-1)
        global_context = context[:, :, None, :].expand(
            -1, -1, local.shape[2], -1
        )
        return torch.cat([local, global_context], dim=-1)


class DLinearFullInfoAdapted(nn.Module):
    """DLinear decomposition per unit using all role/context features."""

    def __init__(
        self,
        *,
        history_steps: int,
        role_feature_counts: tuple[int, ...],
        active_roles: tuple[bool, ...],
        context_size: int,
        kernel_size: int = 25,
    ) -> None:
        super().__init__()
        self.adapter = FullInformationAdapter(
            role_feature_counts=role_feature_counts,
            active_roles=active_roles,
        )
        feature_count = self.adapter.local_feature_count + context_size
        self.moving_average = MovingAverage(kernel_size)
        self.seasonal = nn.Linear(history_steps, 1)
        self.trend = nn.Linear(history_steps, 1)
        self.feature_head = nn.Linear(feature_count, 1)

    def forward(
        self,
        *,
        x_role: torch.Tensor,
        context: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        # [B, L, U, F] -> [B*U, L, F]
        sequence = self.adapter(x_role, context)
        batch, steps, units, features = sequence.shape
        sequence = sequence.permute(0, 2, 1, 3).reshape(
            batch * units, steps, features
        )
        trend = self.moving_average(sequence)
        seasonal = sequence - trend
        encoded = self.seasonal(seasonal.transpose(1, 2))
        encoded += self.trend(trend.transpose(1, 2))
        return self.feature_head(encoded.squeeze(-1)).reshape(batch, units)


class ITransformerFullInfoAdapted(nn.Module):
    """Use each turbine as a token containing its full role/context history."""

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
    ) -> None:
        super().__init__()
        self.adapter = FullInformationAdapter(
            role_feature_counts=role_feature_counts,
            active_roles=active_roles,
        )
        feature_count = self.adapter.local_feature_count + context_size
        self.embedding = nn.Linear(history_steps * feature_count, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=feedforward_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=layers, enable_nested_tensor=False
        )
        self.head = nn.Linear(d_model, 1)

    def forward(
        self,
        *,
        x_role: torch.Tensor,
        context: torch.Tensor,
        y_current: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        sequence = self.adapter(x_role, context)
        batch, steps, units, features = sequence.shape
        tokens = sequence.permute(0, 2, 1, 3).reshape(
            batch, units, steps * features
        )
        # A residual head matches the project's mean-power target and begins
        # from the leakage-safe persistence predictor.
        delta = self.head(self.encoder(self.embedding(tokens))).squeeze(-1)
        return y_current + delta


class STAR(nn.Module):
    """Series-Core Aggregate-Redistribute module from SOFTS.

    This clean-room integration follows the authors' MIT-licensed reference
    implementation while keeping MASPGE's frozen data/evaluation pipeline.
    """

    def __init__(self, d_series: int, d_core: int) -> None:
        super().__init__()
        self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(d_series, d_core)
        self.gen3 = nn.Linear(d_series + d_core, d_series)
        self.gen4 = nn.Linear(d_series, d_series)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, channels, _ = tokens.shape
        core_candidates = torch.nn.functional.gelu(self.gen1(tokens))
        core_candidates = self.gen2(core_candidates)
        if self.training:
            probabilities = torch.softmax(core_candidates, dim=1)
            probabilities = probabilities.permute(0, 2, 1).reshape(
                -1, channels
            )
            indices = torch.multinomial(probabilities, 1)
            indices = indices.view(batch, -1, 1).permute(0, 2, 1)
            core = torch.gather(core_candidates, 1, indices)
            core = core.repeat(1, channels, 1)
        else:
            weights = torch.softmax(core_candidates, dim=1)
            core = (core_candidates * weights).sum(dim=1, keepdim=True)
            core = core.repeat(1, channels, 1)
        fused = torch.cat([tokens, core], dim=-1)
        fused = torch.nn.functional.gelu(self.gen3(fused))
        return self.gen4(fused)


class SOFTSEncoderLayer(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        d_core: int,
        feedforward_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.star = STAR(d_model, d_core)
        self.conv1 = nn.Conv1d(d_model, feedforward_size, kernel_size=1)
        self.conv2 = nn.Conv1d(feedforward_size, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.dropout(self.star(tokens))
        normalized = self.norm1(tokens)
        feedforward = torch.nn.functional.gelu(
            self.conv1(normalized.transpose(1, 2))
        )
        feedforward = self.dropout(feedforward)
        feedforward = self.dropout(
            self.conv2(feedforward).transpose(1, 2)
        )
        return self.norm2(normalized + feedforward)


class SOFTSPower(nn.Module):
    """SOFTS using seven turbine-power series and a one-step mean target."""

    def __init__(
        self,
        *,
        history_steps: int,
        d_model: int,
        d_core: int,
        layers: int,
        feedforward_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Linear(history_steps, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                SOFTSEncoderLayer(
                    d_model=d_model,
                    d_core=d_core,
                    feedforward_size=feedforward_size,
                    dropout=dropout,
                )
                for _ in range(layers)
            ]
        )
        self.projection = nn.Linear(d_model, 1)

    def forward(
        self,
        *,
        power_history: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        mean = power_history.mean(dim=1, keepdim=True).detach()
        centered = power_history - mean
        scale = (
            centered.var(dim=1, keepdim=True, unbiased=False) + 1e-5
        ).sqrt()
        normalized = centered / scale
        tokens = self.embedding_dropout(
            self.embedding(normalized.transpose(1, 2))
        )
        for layer in self.layers:
            tokens = layer(tokens)
        prediction = self.projection(tokens).squeeze(-1)
        return prediction * scale.squeeze(1) + mean.squeeze(1)


class FourierGNNPower(nn.Module):
    """Official-code-faithful FourierGNN for seven power series.

    Device placement is intentionally left to the caller instead of using the
    official repository's hard-coded ``cuda:0``. The released diagonal einsum
    convention is retained for reproducibility.
    """

    def __init__(
        self,
        *,
        history_steps: int,
        embed_size: int,
        hidden_size: int,
        sparsity_threshold: float,
    ) -> None:
        super().__init__()
        self.history_steps = int(history_steps)
        self.embed_size = int(embed_size)
        self.sparsity_threshold = float(sparsity_threshold)
        scale = 0.02
        self.token_embedding = nn.Parameter(torch.randn(1, self.embed_size))
        self.w1 = nn.Parameter(
            scale * torch.randn(2, self.embed_size, self.embed_size)
        )
        self.b1 = nn.Parameter(scale * torch.randn(2, self.embed_size))
        self.w2 = nn.Parameter(
            scale * torch.randn(2, self.embed_size, self.embed_size)
        )
        self.b2 = nn.Parameter(scale * torch.randn(2, self.embed_size))
        self.w3 = nn.Parameter(
            scale * torch.randn(2, self.embed_size, self.embed_size)
        )
        self.b3 = nn.Parameter(scale * torch.randn(2, self.embed_size))
        self.temporal_projection = nn.Parameter(
            torch.randn(self.history_steps, 8)
        )
        self.head = nn.Sequential(
            nn.Linear(self.embed_size * 8, 64),
            nn.LeakyReLU(),
            nn.Linear(64, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1),
        )

    @staticmethod
    def _complex_diagonal_layer(
        values: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        real = torch.relu(
            torch.einsum("bli,ii->bli", values.real, weight[0])
            - torch.einsum("bli,ii->bli", values.imag, weight[1])
            + bias[0]
        )
        imaginary = torch.relu(
            torch.einsum("bli,ii->bli", values.imag, weight[0])
            + torch.einsum("bli,ii->bli", values.real, weight[1])
            + bias[1]
        )
        return real, imaginary

    def _shrink_complex(
        self, real: torch.Tensor, imaginary: torch.Tensor
    ) -> torch.Tensor:
        stacked = torch.stack([real, imaginary], dim=-1)
        shrunk = torch.nn.functional.softshrink(
            stacked, lambd=self.sparsity_threshold
        )
        return torch.view_as_complex(shrunk)

    def forward(
        self,
        *,
        power_history: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        batch, steps, units = power_history.shape
        if steps != self.history_steps:
            raise ValueError("unexpected FourierGNN history length")
        flattened = power_history.transpose(1, 2).reshape(batch, units * steps)
        embedded = flattened.unsqueeze(-1) * self.token_embedding
        spectrum = torch.fft.rfft(embedded, dim=1, norm="ortho")
        residual = spectrum

        real1, imag1 = self._complex_diagonal_layer(
            spectrum, self.w1, self.b1
        )
        layer1 = self._shrink_complex(real1, imag1)
        real2, imag2 = self._complex_diagonal_layer(
            torch.complex(real1, imag1), self.w2, self.b2
        )
        layer2 = self._shrink_complex(real2, imag2) + layer1
        real3, imag3 = self._complex_diagonal_layer(
            torch.complex(real2, imag2), self.w3, self.b3
        )
        layer3 = self._shrink_complex(real3, imag3) + layer2

        reconstructed = torch.fft.irfft(
            layer3 + residual, n=units * steps, dim=1, norm="ortho"
        )
        reconstructed = reconstructed.reshape(
            batch, units, steps, self.embed_size
        ).permute(0, 1, 3, 2)
        compressed = torch.matmul(reconstructed, self.temporal_projection)
        prediction = self.head(compressed.reshape(batch, units, -1))
        return prediction.squeeze(-1)


def build_external_baseline(
    mode: str,
    *,
    config: dict,
    role_feature_counts: tuple[int, ...],
    active_roles: tuple[bool, ...],
    context_size: int,
) -> nn.Module:
    model = config["model"]
    history_steps = int(config["dataset"]["history_steps"])
    if mode == "dlinear_power":
        return DLinearPower(
            history_steps, kernel_size=int(model["moving_average_kernel"])
        )
    if mode == "timemixer_power":
        timemixer = model["timemixer"]
        return TimeMixerPower(
            history_steps=history_steps,
            d_model=int(timemixer["d_model"]),
            d_ff=int(timemixer["d_ff"]),
            layers=int(timemixer["layers"]),
            down_sampling_layers=int(timemixer["down_sampling_layers"]),
            down_sampling_window=int(timemixer["down_sampling_window"]),
            moving_average_kernel=int(timemixer["moving_average_kernel"]),
            dropout=float(timemixer["dropout"]),
        )
    if mode.startswith("softs_power_"):
        softs = model["softs"][mode]
        return SOFTSPower(
            history_steps=history_steps,
            d_model=int(softs["d_model"]),
            d_core=int(softs["d_core"]),
            layers=int(softs["layers"]),
            feedforward_size=int(softs["feedforward_size"]),
            dropout=float(softs["dropout"]),
        )
    if mode.startswith("fouriergnn_power_"):
        fourier = model["fouriergnn"][mode]
        return FourierGNNPower(
            history_steps=history_steps,
            embed_size=int(fourier["embed_size"]),
            hidden_size=int(fourier["hidden_size"]),
            sparsity_threshold=float(fourier["sparsity_threshold"]),
        )
    if mode == "dlinear_fullinfo_adapted":
        return DLinearFullInfoAdapted(
            history_steps=history_steps,
            role_feature_counts=role_feature_counts,
            active_roles=active_roles,
            context_size=context_size,
            kernel_size=int(model["moving_average_kernel"]),
        )
    transformer = {
        "history_steps": history_steps,
        "d_model": int(model["d_model"]),
        "n_heads": int(model["n_heads"]),
        "layers": int(model["layers"]),
        "feedforward_size": int(model["feedforward_size"]),
        "dropout": float(model["dropout"]),
    }
    if mode == "itransformer_power":
        return ITransformerPower(**transformer)
    if mode == "itransformer_fullinfo_adapted":
        return ITransformerFullInfoAdapted(
            **transformer,
            role_feature_counts=role_feature_counts,
            active_roles=active_roles,
            context_size=context_size,
        )
    raise ValueError(f"Unknown external baseline: {mode}")
