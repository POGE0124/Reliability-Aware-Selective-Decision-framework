from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import sinusoidal_time


def _resolve_input_snr(input_snr: torch.Tensor | None = None, target_snr: torch.Tensor | None = None) -> torch.Tensor | None:
    return input_snr if input_snr is not None else target_snr


class ConvStem(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Conv1d(256, dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).transpose(1, 2)


class SpectralTemporalTeacher(nn.Module):
    def __init__(
        self,
        num_classes: int,
        dim: int = 512,
        z_dim: int = 512,
        depth: int = 8,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.05,
        max_tokens: int = 512,
    ) -> None:
        super().__init__()
        self.time_stem = ConvStem(dim, dropout)
        self.freq_stem = ConvStem(dim, dropout)
        self.branch_embed = nn.Parameter(torch.zeros(2, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(layer, num_layers=depth)
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, z_dim), nn.LayerNorm(z_dim))
        self.classifier = nn.Linear(z_dim, num_classes)
        nn.init.trunc_normal_(self.branch_embed, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    @staticmethod
    def fft_iq(x: torch.Tensor) -> torch.Tensor:
        s = torch.complex(x[:, 0], x[:, 1])
        f = torch.fft.fft(s, dim=-1, norm="ortho")
        return torch.stack([f.real, f.imag], dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        t = self.time_stem(x) + self.branch_embed[0].unsqueeze(0)
        f = self.freq_stem(self.fft_iq(x)) + self.branch_embed[1].unsqueeze(0)
        tokens = torch.cat([t, f], dim=1)
        if tokens.shape[1] > self.pos_embed.shape[1]:
            raise ValueError(f"Token length {tokens.shape[1]} exceeds max_tokens={self.pos_embed.shape[1]}")
        tokens = tokens + self.pos_embed[:, : tokens.shape[1]]
        h = self.backbone(tokens).mean(dim=1)
        return self.proj(h)

    def forward(self, x: torch.Tensor) -> dict:
        z = self.encode(x)
        return {"z": z, "logits": self.classifier(z)}


class MLPAlignment(nn.Module):
    def __init__(self, z_dim: int = 512, snr_dim: int = 128) -> None:
        super().__init__()
        self.snr_dim = snr_dim
        self.snr_embed = SNRCondition(snr_dim)
        self.net = nn.Sequential(
            nn.Linear(z_dim + snr_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, z_dim),
        )

    def restore(self, z: torch.Tensor, input_snr: torch.Tensor | None = None, *, target_snr: torch.Tensor | None = None) -> torch.Tensor:
        input_snr = _resolve_input_snr(input_snr, target_snr)
        return self.net(torch.cat([z, self.snr_embed(input_snr, z)], dim=-1))

    def forward(self, z: torch.Tensor, input_snr: torch.Tensor | None = None, *, target_snr: torch.Tensor | None = None) -> torch.Tensor:
        return self.restore(z, input_snr, target_snr=target_snr)


class FeatureDAE(nn.Module):
    def __init__(self, z_dim: int = 512, snr_dim: int = 128) -> None:
        super().__init__()
        self.snr_dim = snr_dim
        self.snr_embed = SNRCondition(snr_dim)
        self.encoder = nn.Sequential(nn.Linear(z_dim + snr_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Linear(512, 256), nn.GELU())
        self.decoder = nn.Sequential(nn.Linear(256 + snr_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Linear(512, z_dim))

    def restore(self, z: torch.Tensor, input_snr: torch.Tensor | None = None, *, target_snr: torch.Tensor | None = None) -> torch.Tensor:
        input_snr = _resolve_input_snr(input_snr, target_snr)
        cond = self.snr_embed(input_snr, z)
        h = self.encoder(torch.cat([z, cond], dim=-1))
        return self.decoder(torch.cat([h, cond], dim=-1))

    def forward(self, z: torch.Tensor, input_snr: torch.Tensor | None = None, *, target_snr: torch.Tensor | None = None) -> torch.Tensor:
        return self.restore(z, input_snr, target_snr=target_snr)


class ResidualAlignment(nn.Module):
    def __init__(self, z_dim: int = 512, snr_dim: int = 128) -> None:
        super().__init__()
        self.snr_embed = SNRCondition(snr_dim)
        self.residual_net = nn.Sequential(
            nn.Linear(z_dim + snr_dim, 1536),
            nn.LayerNorm(1536),
            nn.GELU(),
            nn.Linear(1536, 1536),
            nn.LayerNorm(1536),
            nn.GELU(),
            nn.Linear(1536, 1536),
            nn.LayerNorm(1536),
            nn.GELU(),
            nn.Linear(1536, z_dim),
        )

    @staticmethod
    def residual_gate(input_snr: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor:
        if input_snr is None:
            return ref.new_ones((ref.shape[0], 1))
        snr = input_snr.to(device=ref.device, dtype=ref.dtype).view(-1, 1)
        return 0.05 + 1.20 * torch.sigmoid((2.0 - snr) / 4.0)

    def residual_restore(self, z: torch.Tensor, input_snr: torch.Tensor | None = None, *, target_snr: torch.Tensor | None = None) -> torch.Tensor:
        input_snr = _resolve_input_snr(input_snr, target_snr)
        cond = self.snr_embed(input_snr, z)
        candidate = self.residual_net(torch.cat([z, cond], dim=-1))
        gate = self.residual_gate(input_snr, z).clamp(0.0, 1.0)
        return gate * candidate + (1.0 - gate) * z

    def restore(self, z: torch.Tensor, input_snr: torch.Tensor | None = None, *, target_snr: torch.Tensor | None = None) -> torch.Tensor:
        return self.residual_restore(z, input_snr, target_snr=target_snr)

    def forward(self, z: torch.Tensor, input_snr: torch.Tensor | None = None, *, target_snr: torch.Tensor | None = None) -> torch.Tensor:
        return self.restore(z, input_snr, target_snr=target_snr)


class FeatureFlow(nn.Module):
    def __init__(self, z_dim: int = 512, time_dim: int = 128, snr_dim: int = 128) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.time_dim = time_dim
        self.snr_dim = snr_dim
        self.snr_embed = SNRCondition(snr_dim)
        self.velocity_net = nn.Sequential(
            nn.Linear(z_dim + time_dim + snr_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, z_dim),
        )
        self.residual_net = nn.Sequential(
            nn.Linear(z_dim + snr_dim, 1536),
            nn.LayerNorm(1536),
            nn.GELU(),
            nn.Linear(1536, 1536),
            nn.LayerNorm(1536),
            nn.GELU(),
            nn.Linear(1536, 1536),
            nn.LayerNorm(1536),
            nn.GELU(),
            nn.Linear(1536, z_dim),
        )

    def velocity(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        input_snr: torch.Tensor | None = None,
        *,
        target_snr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_snr = _resolve_input_snr(input_snr, target_snr)
        emb = sinusoidal_time(t, self.time_dim)
        cond = self.snr_embed(input_snr, z_t)
        return self.velocity_net(torch.cat([z_t, emb, cond], dim=-1))

    def residual_restore(self, z: torch.Tensor, input_snr: torch.Tensor | None = None, *, target_snr: torch.Tensor | None = None) -> torch.Tensor:
        input_snr = _resolve_input_snr(input_snr, target_snr)
        cond = self.snr_embed(input_snr, z)
        candidate = self.residual_net(torch.cat([z, cond], dim=-1))
        gate = self.residual_gate(input_snr, z).clamp(0.0, 1.0)
        return gate * candidate + (1.0 - gate) * z

    @staticmethod
    def residual_gate(input_snr: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor:
        if input_snr is None:
            return ref.new_ones((ref.shape[0], 1))
        snr = input_snr.to(device=ref.device, dtype=ref.dtype).view(-1, 1)
        return 0.05 + 1.20 * torch.sigmoid((2.0 - snr) / 4.0)

    def ode_restore(self, z: torch.Tensor, input_snr: torch.Tensor | None = None, steps: int = 8, *, target_snr: torch.Tensor | None = None) -> torch.Tensor:
        input_snr = _resolve_input_snr(input_snr, target_snr)
        z_t = z.clone()
        dt = 1.0 / float(steps)
        for i in range(steps):
            t = z.new_full((z.shape[0],), (i + 0.5) * dt)
            z_t = z_t + dt * self.velocity(z_t, t, input_snr)
        return z_t

    def restore(
        self,
        z: torch.Tensor,
        input_snr: torch.Tensor | None = None,
        steps: int = 8,
        mode: str = "ode",
        *,
        target_snr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_snr = _resolve_input_snr(input_snr, target_snr)
        if mode == "ode":
            return self.ode_restore(z, input_snr, steps)
        if mode == "residual":
            return self.residual_restore(z, input_snr)
        if mode == "ode_residual":
            return self.residual_restore(self.ode_restore(z, input_snr, steps), input_snr)
        raise ValueError(f"Unknown FeatureFlow restore mode: {mode}")

    def forward(
        self,
        z_noisy: torch.Tensor,
        z_high: torch.Tensor | None = None,
        input_snr: torch.Tensor | None = None,
        steps: int = 8,
        restore_mode: str = "ode",
        *,
        target_snr: torch.Tensor | None = None,
    ) -> dict:
        input_snr = _resolve_input_snr(input_snr, target_snr)
        if z_high is None:
            z_restored = self.restore(z_noisy, input_snr, steps=steps, mode=restore_mode)
            return {"z_restored": z_restored}
        t = torch.rand(z_noisy.shape[0], device=z_noisy.device, dtype=z_noisy.dtype)
        z_t = (1.0 - t[:, None]) * z_noisy + t[:, None] * z_high
        target = z_high - z_noisy
        v = self.velocity(z_t, t, input_snr)
        z_restored = self.restore(z_noisy, input_snr, steps=steps, mode=restore_mode)
        return {"velocity": v, "target": target, "z_restored": z_restored}


class FeatureMeanFlow(nn.Module):
    """Interval-conditioned feature flow for staged teacher-guided restoration."""

    def __init__(self, z_dim: int = 512, time_dim: int = 128, snr_dim: int = 128, hidden: int = 1536) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.time_dim = time_dim
        self.snr_dim = snr_dim
        self.snr_embed = SNRCondition(snr_dim)
        self.net = nn.Sequential(
            nn.Linear(z_dim + 2 * time_dim + snr_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, z_dim),
        )

    def mean_velocity(
        self,
        z_r: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        input_snr: torch.Tensor | None = None,
        *,
        target_snr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_snr = _resolve_input_snr(input_snr, target_snr)
        r_emb = sinusoidal_time(r, self.time_dim)
        t_emb = sinusoidal_time(t, self.time_dim)
        cond = self.snr_embed(input_snr, z_r)
        return self.net(torch.cat([z_r, r_emb, t_emb, cond], dim=-1))

    @staticmethod
    def sample_interval(ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = 0.05 + 0.95 * torch.rand(ref.shape[0], device=ref.device, dtype=ref.dtype)
        r = t * torch.rand(ref.shape[0], device=ref.device, dtype=ref.dtype)
        return r, t

    def restore(self, z: torch.Tensor, input_snr: torch.Tensor | None = None, steps: int = 8, *, target_snr: torch.Tensor | None = None) -> torch.Tensor:
        input_snr = _resolve_input_snr(input_snr, target_snr)
        z_r = z.clone()
        steps = max(1, int(steps))
        dt = 1.0 / float(steps)
        for i in range(steps):
            r = z.new_full((z.shape[0],), i * dt)
            t = z.new_full((z.shape[0],), (i + 1) * dt)
            z_r = z_r + dt * self.mean_velocity(z_r, r, t, input_snr)
        return z_r

    def forward(
        self,
        z_noisy: torch.Tensor,
        z_high: torch.Tensor | None = None,
        input_snr: torch.Tensor | None = None,
        steps: int = 8,
        *,
        target_snr: torch.Tensor | None = None,
    ) -> dict:
        input_snr = _resolve_input_snr(input_snr, target_snr)
        if z_high is None:
            return {"z_restored": self.restore(z_noisy, input_snr, steps=steps)}
        r, t = self.sample_interval(z_noisy)
        z_r = (1.0 - r[:, None]) * z_noisy + r[:, None] * z_high
        z_t = (1.0 - t[:, None]) * z_noisy + t[:, None] * z_high
        target = (z_t - z_r) / (t - r).clamp_min(1.0e-4)[:, None]
        velocity = self.mean_velocity(z_r, r, t, input_snr)
        z_restored = self.restore(z_noisy, input_snr, steps=steps)
        return {"velocity": velocity, "target": target, "z_restored": z_restored}


class SNRCondition(nn.Module):
    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, input_snr: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor:
        """Embed input degradation SNR; the restoration target is always the high-SNR teacher manifold."""
        if input_snr is None:
            input_snr = ref.new_zeros((ref.shape[0],))
        input_snr = input_snr.to(device=ref.device, dtype=ref.dtype).view(-1)
        snr01 = ((input_snr + 20.0) / 50.0).clamp(0.0, 1.0)
        feats = torch.stack([snr01, torch.sin(input_snr / 10.0), torch.cos(input_snr / 10.0)], dim=-1)
        return self.net(feats)


def build_teacher(cfg: dict) -> SpectralTemporalTeacher:
    tc = cfg["teacher"]
    return SpectralTemporalTeacher(
        num_classes=int(cfg["data"]["num_classes"]),
        dim=int(tc["dim"]),
        z_dim=int(tc["z_dim"]),
        depth=int(tc["depth"]),
        heads=int(tc["heads"]),
        mlp_ratio=float(tc["mlp_ratio"]),
        dropout=float(tc["dropout"]),
        max_tokens=int(tc.get("max_tokens", 512)),
    )


def build_alignment(method: str, z_dim: int = 512) -> nn.Module:
    if method == "mlp":
        return MLPAlignment(z_dim)
    if method == "dae":
        return FeatureDAE(z_dim)
    if method == "residual":
        return ResidualAlignment(z_dim)
    if method == "flow":
        return FeatureFlow(z_dim)
    if method == "ifm_flow":
        return FeatureMeanFlow(z_dim)
    raise ValueError(f"Unknown alignment method: {method}")


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    per_sample = (pred - target).pow(2).mean(dim=-1)
    if weight is None:
        return per_sample.mean()
    weight = weight.to(device=per_sample.device, dtype=per_sample.dtype).view(-1)
    return (per_sample * weight).sum() / weight.sum().clamp_min(1.0)


def weighted_ce(logits: torch.Tensor, y: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    per_sample = F.cross_entropy(logits, y, reduction="none")
    if weight is None:
        return per_sample.mean()
    weight = weight.to(device=per_sample.device, dtype=per_sample.dtype).view(-1)
    return (per_sample * weight).sum() / weight.sum().clamp_min(1.0)


def batch_prototype_ce(z_restored: torch.Tensor, z_high: torch.Tensor, y: torch.Tensor, num_classes: int, temperature: float) -> torch.Tensor:
    proto = z_high.detach().new_zeros((num_classes, z_high.shape[-1]))
    counts = z_high.detach().new_zeros((num_classes, 1))
    proto.index_add_(0, y, z_high.detach())
    counts.index_add_(0, y, torch.ones((y.shape[0], 1), device=z_high.device, dtype=z_high.dtype))
    valid = counts.view(-1) > 0
    proto = proto / counts.clamp_min(1.0)
    logits = F.normalize(z_restored, dim=-1) @ F.normalize(proto, dim=-1).T
    logits = logits / float(temperature)
    logits = logits.masked_fill(~valid.view(1, -1), -1.0e4)
    return F.cross_entropy(logits, y)


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    return x[~torch.eye(n, dtype=torch.bool, device=x.device)].view(n, n - 1)


def relational_distillation_loss(
    z_student: torch.Tensor,
    z_teacher: torch.Tensor,
    temperature: float = 1.0,
    detach_teacher: bool = True,
    remove_diag: bool = True,
    mode: str = "mse",
) -> torch.Tensor:
    if z_student.shape[0] < 2:
        return z_student.new_zeros(())
    temp = max(float(temperature), 1.0e-6)
    s_student = F.normalize(z_student, dim=-1) @ F.normalize(z_student, dim=-1).T
    s_teacher = F.normalize(z_teacher, dim=-1) @ F.normalize(z_teacher, dim=-1).T
    if detach_teacher:
        s_teacher = s_teacher.detach()
    if remove_diag:
        s_student = _off_diagonal(s_student)
        s_teacher = _off_diagonal(s_teacher)
    if mode == "mse":
        return F.mse_loss(s_student / temp, s_teacher / temp)
    if mode == "smooth_l1":
        return F.smooth_l1_loss(s_student / temp, s_teacher / temp)
    if mode == "kl":
        return F.kl_div(F.log_softmax(s_student / temp, dim=-1), F.softmax(s_teacher / temp, dim=-1), reduction="batchmean")
    raise ValueError(f"Unknown relational loss mode: {mode}")


def class_aware_relational_loss(
    z_student: torch.Tensor,
    z_teacher: torch.Tensor,
    labels: torch.Tensor,
    same_class_weight: float = 1.0,
    diff_class_weight: float = 0.5,
    temperature: float = 1.0,
    detach_teacher: bool = True,
    remove_diag: bool = True,
    mode: str = "mse",
) -> torch.Tensor:
    if z_student.shape[0] < 2:
        return z_student.new_zeros(())
    temp = max(float(temperature), 1.0e-6)
    s_student = F.normalize(z_student, dim=-1) @ F.normalize(z_student, dim=-1).T
    s_teacher = F.normalize(z_teacher, dim=-1) @ F.normalize(z_teacher, dim=-1).T
    if detach_teacher:
        s_teacher = s_teacher.detach()
    labels = labels.view(-1)
    weights = torch.where(
        labels[:, None].eq(labels[None, :]),
        s_student.new_full(s_student.shape, float(same_class_weight)),
        s_student.new_full(s_student.shape, float(diff_class_weight)),
    )
    if remove_diag:
        s_student = _off_diagonal(s_student)
        s_teacher = _off_diagonal(s_teacher)
        weights = _off_diagonal(weights)
    if mode == "mse":
        loss = (s_student / temp - s_teacher / temp).pow(2)
    elif mode == "smooth_l1":
        loss = F.smooth_l1_loss(s_student / temp, s_teacher / temp, reduction="none")
    elif mode == "kl":
        log_student = F.log_softmax(s_student / temp, dim=-1)
        teacher_prob = F.softmax(s_teacher / temp, dim=-1)
        log_teacher = torch.log(teacher_prob.clamp_min(1.0e-8))
        loss = teacher_prob * (log_teacher - log_student)
    else:
        raise ValueError(f"Unknown relational loss mode: {mode}")
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def alignment_loss(
    model: nn.Module,
    method: str,
    classifier: nn.Module,
    batch: dict,
    lambda_cls: float,
    lambda_feat: float,
    flow_steps: int,
    flow_velocity_weight: float = 1.0,
    flow_direct_weight: float = 1.0,
    flow_lambda_cls: float | None = None,
    flow_lambda_feat: float | None = None,
    flow_low_snr_weight: float = 1.0,
    flow_low_snr_threshold: float = 0.0,
    flow_proto_weight: float = 0.0,
    flow_proto_temperature: float = 0.1,
    relational_weight: float = 0.0,
    relational_temperature: float = 1.0,
    relational_mode: str = "mse",
    relational_remove_diag: bool = True,
    relational_class_aware: bool = False,
    relational_same_class_weight: float = 1.0,
    relational_diff_class_weight: float = 0.5,
    identity_weight: float = 0.0,
    identity_snr: float = 30.0,
) -> tuple[torch.Tensor, dict]:
    z_noisy = batch["z_noisy"]
    z_high = batch["z_high"]
    z_target = batch.get("z_target", z_high)
    y = batch["label"]
    input_snr = batch.get("input_snr", batch.get("target_snr"))
    if input_snr is None:
        sample_weight = None
    else:
        sample_weight = torch.where(
            input_snr <= float(flow_low_snr_threshold),
            input_snr.new_full(input_snr.shape, float(flow_low_snr_weight)),
            input_snr.new_ones(input_snr.shape),
        )

    l_velocity = z_noisy.new_zeros(())
    l_direct = z_noisy.new_zeros(())
    l_identity = z_noisy.new_zeros(())
    if method in {"flow", "ifm_flow"}:
        if method == "flow":
            out = model(z_noisy, z_target, input_snr, steps=flow_steps, restore_mode="ode")
        else:
            out = model(z_noisy, z_target, input_snr, steps=flow_steps)
        z_restored = out["z_restored"]
        l_velocity = weighted_mse(out["velocity"], out["target"], sample_weight)
        l_direct = weighted_mse(z_restored, z_target, sample_weight)
        l_main = float(flow_velocity_weight) * l_velocity + float(flow_direct_weight) * l_direct
        lambda_cls = float(lambda_cls if flow_lambda_cls is None else flow_lambda_cls)
        lambda_feat = float(lambda_feat if flow_lambda_feat is None else flow_lambda_feat)
    elif method == "residual":
        z_restored = model.residual_restore(z_noisy, input_snr)
        l_direct = weighted_mse(z_restored, z_target, sample_weight)
        l_main = l_direct
    elif method in {"mlp", "dae"}:
        z_restored = model.restore(z_noisy, input_snr)
        l_direct = weighted_mse(z_restored, z_target, sample_weight)
        l_main = l_direct
    else:
        raise ValueError(f"Unknown alignment loss method: {method}")

    if float(identity_weight) > 0.0:
        identity_snr_tensor = z_high.new_full((z_high.shape[0],), float(identity_snr))
        if method in {"flow", "ifm_flow"}:
            z_identity = model.restore(z_high, identity_snr_tensor, steps=flow_steps)
        elif method == "residual":
            z_identity = model.residual_restore(z_high, identity_snr_tensor)
        else:
            z_identity = model.restore(z_high, identity_snr_tensor)
        l_identity = weighted_mse(z_identity, z_high, sample_weight)

    logits = classifier(z_restored)
    l_cls = weighted_ce(logits, y, sample_weight)
    l_feat = weighted_mse(z_restored, z_target, sample_weight)
    if float(flow_proto_weight) > 0.0:
        l_proto = batch_prototype_ce(z_restored, z_target, y, classifier.out_features, float(flow_proto_temperature))
    else:
        l_proto = z_restored.new_zeros(())
    if float(relational_weight) > 0.0:
        if relational_class_aware:
            l_rel = class_aware_relational_loss(
                z_restored,
                z_target,
                y,
                same_class_weight=float(relational_same_class_weight),
                diff_class_weight=float(relational_diff_class_weight),
                temperature=float(relational_temperature),
                detach_teacher=True,
                remove_diag=bool(relational_remove_diag),
                mode=str(relational_mode),
            )
        else:
            l_rel = relational_distillation_loss(
                z_restored,
                z_target,
                temperature=float(relational_temperature),
                detach_teacher=True,
                remove_diag=bool(relational_remove_diag),
                mode=str(relational_mode),
            )
    else:
        l_rel = z_restored.new_zeros(())
    loss = (
        l_main
        + float(lambda_cls) * l_cls
        + float(lambda_feat) * l_feat
        + float(flow_proto_weight) * l_proto
        + float(relational_weight) * l_rel
        + float(identity_weight) * l_identity
    )
    return loss, {
        "total": float(loss.detach().cpu()),
        "main": float(l_main.detach().cpu()),
        "velocity": float(l_velocity.detach().cpu()),
        "direct": float(l_direct.detach().cpu()),
        "cls": float(l_cls.detach().cpu()),
        "feat": float(l_feat.detach().cpu()),
        "proto": float(l_proto.detach().cpu()),
        "rel": float(l_rel.detach().cpu()),
        "identity": float(l_identity.detach().cpu()),
    }
