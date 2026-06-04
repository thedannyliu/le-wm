import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time
    
class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x

class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, dim, frequency_dim=64):
        super().__init__()
        self.frequency_dim = frequency_dim
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        """
        t: (B,) or (B, 1)
        """
        if t.ndim == 2:
            t = t[:, 0]
        half = self.frequency_dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=t.device, dtype=t.dtype)
            * torch.log(torch.tensor(10000.0, device=t.device, dtype=t.dtype))
            / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if emb.size(-1) < self.frequency_dim:
            emb = F.pad(emb, (0, self.frequency_dim - emb.size(-1)))
        return self.mlp(emb)


class ConditionalFlowPredictor(nn.Module):
    """Conditional flow-matching predictor for next latent embeddings."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        time_dim=64,
        sample_steps=8,
        stochastic_sample=False,
    ):
        super().__init__()
        output_dim = output_dim or input_dim
        self.output_dim = output_dim
        self.sample_steps = sample_steps
        self.stochastic_sample = stochastic_sample
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.time_embed = TimestepEmbedder(input_dim, frequency_dim=time_dim)
        self.cond_proj = nn.Sequential(
            nn.LayerNorm(input_dim * 3),
            nn.Linear(input_dim * 3, input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, input_dim),
        )
        self.noisy_proj = (
            nn.Linear(output_dim, input_dim)
            if output_dim != input_dim
            else nn.Identity()
        )
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def _condition(self, ctx, act, t):
        T = ctx.size(1)
        t_emb = self.time_embed(t).unsqueeze(1).expand(-1, T, -1)
        return self.cond_proj(torch.cat([ctx, act, t_emb], dim=-1))

    def vector_field(self, ctx, act, noisy_target, t):
        """
        ctx: (B, T, D)
        act: (B, T, D)
        noisy_target: (B, T, D)
        t: (B,) or (B, 1)
        """
        T = noisy_target.size(1)
        x = self.noisy_proj(noisy_target) + self.pos_embedding[:, :T]
        x = self.dropout(x)
        c = self._condition(ctx, act, t)
        return self.transformer(x, c)

    def flow_loss(self, ctx, act, target):
        B = target.size(0)
        t = torch.rand(B, device=target.device, dtype=target.dtype)
        noise = torch.randn_like(target)
        path_t = t.view(B, 1, 1)
        noisy = (1.0 - path_t) * noise + path_t * target
        velocity_target = target - noise
        velocity_pred = self.vector_field(ctx, act, noisy, t)
        return F.mse_loss(velocity_pred, velocity_target)

    @torch.no_grad()
    def sample(self, ctx, act, steps=None, stochastic=None):
        steps = steps or self.sample_steps
        stochastic = self.stochastic_sample if stochastic is None else stochastic
        x = torch.randn_like(ctx) if stochastic else torch.zeros_like(ctx)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full(
                (ctx.size(0),),
                i / steps,
                device=ctx.device,
                dtype=ctx.dtype,
            )
            x = x + dt * self.vector_field(ctx, act, x, t)
        return x

    def forward(self, ctx, act):
        return self.sample(ctx, act)


class ConditionalActionFlow(nn.Module):
    """Flow-matching model for goal-conditioned action chunk proposals."""

    def __init__(
        self,
        *,
        condition_dim,
        action_dim,
        horizon,
        hidden_dim=256,
        depth=4,
        time_dim=64,
        sample_steps=8,
        stochastic_sample=True,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.sample_steps = sample_steps
        self.stochastic_sample = stochastic_sample
        self.time_embed = TimestepEmbedder(hidden_dim, frequency_dim=time_dim)
        input_dim = action_dim + condition_dim + hidden_dim
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
        ]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def _expand_condition(self, condition, horizon):
        if condition.ndim == 2:
            condition = condition.unsqueeze(1)
        if condition.size(1) == 1:
            condition = condition.expand(-1, horizon, -1)
        return condition

    def vector_field(self, condition, noisy_action, t):
        horizon = noisy_action.size(1)
        condition = self._expand_condition(condition, horizon)
        t_emb = self.time_embed(t).unsqueeze(1).expand(-1, horizon, -1)
        x = torch.cat([noisy_action, condition, t_emb], dim=-1)
        return self.net(x)

    def flow_loss(self, condition, target_action):
        B = target_action.size(0)
        t = torch.rand(B, device=target_action.device, dtype=target_action.dtype)
        noise = torch.randn_like(target_action)
        path_t = t.view(B, 1, 1)
        noisy = (1.0 - path_t) * noise + path_t * target_action
        velocity_target = target_action - noise
        velocity_pred = self.vector_field(condition, noisy, t)
        return F.mse_loss(velocity_pred, velocity_target)

    @torch.no_grad()
    def sample(self, condition, num_samples=1, steps=None, stochastic=None):
        steps = steps or self.sample_steps
        stochastic = self.stochastic_sample if stochastic is None else stochastic
        if condition.ndim == 2:
            condition = condition.unsqueeze(1)
        B = condition.size(0)
        condition = condition[:, None].expand(B, num_samples, *condition.shape[1:])
        condition = rearrange(condition, "b s t d -> (b s) t d")
        x_shape = (B * num_samples, self.horizon, self.action_dim)
        x = (
            torch.randn(x_shape, device=condition.device, dtype=condition.dtype)
            if stochastic
            else torch.zeros(x_shape, device=condition.device, dtype=condition.dtype)
        )
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((x.size(0),), i / steps, device=x.device, dtype=x.dtype)
            x = x + dt * self.vector_field(condition, x, t)
        return rearrange(x, "(b s) t d -> b s t d", b=B, s=num_samples)
