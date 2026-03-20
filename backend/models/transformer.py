"""
transformer.py
--------------
150M-parameter transformer model for space debris collision risk prediction.

Architecture:
  - Input embedding: Linear(22, 1024)
  - Sinusoidal positional encoding
  - 10 × TransformerEncoderLayer (d_model=1024, nhead=16, d_ff=4096)
  - Pairwise interaction module
  - Evidential Deep Learning output head (3 Dirichlet concentration params)

Input:  [batch, sequence_length=168, input_features=22]
Output: [batch, 3]  # evidence for [LOW, MEDIUM, HIGH]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Vaswani et al. 2017).
    Adds a fixed position signal to each token embedding.
    """

    def __init__(self, d_model: int, max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create [max_len, d_model] positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            x + positional encoding, shape [batch, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PairwiseInteractionModule(nn.Module):
    """
    Models interactions between primary satellite trajectory features and
    the global context vector. Learns to weight the importance of each
    time step relative to the aggregate encounter features.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj   = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        sequence_repr: torch.Tensor,  # [batch, seq_len, d_model]
        global_context: torch.Tensor, # [batch, d_model]
    ) -> torch.Tensor:
        """
        Cross-attention: global_context queries the sequence.
        Returns: [batch, d_model]
        """
        B, T, D = sequence_repr.shape

        # Global context as query [batch, 1, d_model]
        q = self.query_proj(global_context).unsqueeze(1)
        k = self.key_proj(sequence_repr)    # [batch, T, d_model]
        v = self.value_proj(sequence_repr)  # [batch, T, d_model]

        # Scaled dot-product attention
        attn_weights = torch.bmm(q, k.transpose(1, 2)) / self.scale  # [batch, 1, T]
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum [batch, 1, d_model] → [batch, d_model]
        attended = torch.bmm(attn_weights, v).squeeze(1)
        out = self.out_norm(attended + global_context)
        return out


class EvidentialOutputHead(nn.Module):
    """
    Evidential Deep Learning output head.
    Maps from flattened representation to Dirichlet concentration parameters
    (evidence) for 3 risk classes (LOW, MEDIUM, HIGH).

    Based on: "Evidential Deep Learning to Quantify Classification Uncertainty"
    (Sensoy et al., NeurIPS 2018).
    """

    def __init__(self, d_model: int, n_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model // 4, n_classes),
        )
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, d_model]

        Returns:
            evidence:   [batch, n_classes] — raw Dirichlet concentration - 1
            alpha:      [batch, n_classes] — Dirichlet parameters (evidence + 1)
            uncertainty:[batch] — total uncertainty = n_classes / sum(alpha)
        """
        logits = self.head(x)                       # [batch, n_classes]
        evidence = F.softplus(logits)               # ensure evidence >= 0
        alpha = evidence + 1.0                      # Dirichlet params
        S = alpha.sum(dim=-1, keepdim=True)         # [batch, 1]
        uncertainty = self.n_classes / S.squeeze(1) # [batch]
        return evidence, alpha, uncertainty


class CollisionRiskTransformer(nn.Module):
    """
    150M-parameter transformer for space debris collision risk prediction.

    Input:  [batch, 168, 22]   (trajectory sequence)
    Output: (evidence, alpha, uncertainty, prob)
        evidence:    [batch, 3]  — Dirichlet evidence
        alpha:       [batch, 3]  — Dirichlet alpha params
        uncertainty: [batch]     — total predictive uncertainty
        prob:        [batch, 3]  — predicted class probabilities
    """

    def __init__(
        self,
        input_features: int = 22,
        d_model: int = 1024,
        n_heads: int = 16,
        n_encoder_layers: int = 10,
        d_feedforward: int = 4096,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        n_classes: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes

        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(input_features, d_model),
            nn.LayerNorm(d_model),
        )

        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,   # Pre-LayerNorm for training stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_encoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Global pooling: learnable CLS-style aggregation + mean
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.global_pool_norm = nn.LayerNorm(d_model)

        # Pairwise interaction module
        self.pairwise = PairwiseInteractionModule(d_model, dropout=dropout)

        # Evidential output head
        self.output_head = EvidentialOutputHead(d_model, n_classes=n_classes, dropout=dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,                           # [batch, seq_len, input_features]
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input trajectory tensor [batch, T, input_features]
            src_key_padding_mask: Optional mask [batch, T+1] (True = ignore)

        Returns:
            evidence, alpha, uncertainty, prob
        """
        B, T, _ = x.shape

        # 1. Input embedding: [batch, T, d_model]
        x_emb = self.input_embedding(x)

        # 2. Prepend [CLS] token: [batch, T+1, d_model]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_emb = torch.cat([cls_tokens, x_emb], dim=1)

        # 3. Positional encoding
        x_emb = self.pos_encoding(x_emb)

        # 4. Transformer encoder
        encoded = self.encoder(x_emb, src_key_padding_mask=src_key_padding_mask)  # [batch, T+1, d_model]

        # 5. Extract CLS token as global context
        cls_out = encoded[:, 0, :]  # [batch, d_model]
        seq_out = encoded[:, 1:, :] # [batch, T, d_model]

        # 6. Pairwise interaction module
        global_repr = self.pairwise(seq_out, cls_out)  # [batch, d_model]
        global_repr = self.global_pool_norm(global_repr)

        # 7. Evidential output
        evidence, alpha, uncertainty = self.output_head(global_repr)

        # 8. Predicted probabilities via normalized alpha (expected value of Dirichlet)
        S = alpha.sum(dim=-1, keepdim=True)
        prob = alpha / S  # [batch, n_classes]

        return evidence, alpha, uncertainty, prob

    def predict(self, x: torch.Tensor) -> dict:
        """
        Convenience method for inference.
        Returns a dict with all outputs plus epistemic/aleatoric uncertainty estimates.
        """
        with torch.no_grad():
            evidence, alpha, uncertainty, prob = self.forward(x)

        S = alpha.sum(dim=-1)

        # Epistemic uncertainty: variance of Dirichlet
        epistemic = (alpha * (S.unsqueeze(1) - alpha)) / (S.unsqueeze(1) ** 2 * (S.unsqueeze(1) + 1))
        epistemic_total = epistemic.sum(dim=-1)

        # Aleatoric uncertainty: entropy of expected distribution
        p_log_p = -torch.sum(prob * torch.log(prob.clamp(min=1e-8)), dim=-1)

        return {
            "probability": prob.cpu().numpy(),
            "evidence": evidence.cpu().numpy(),
            "alpha": alpha.cpu().numpy(),
            "uncertainty_total": uncertainty.cpu().numpy(),
            "uncertainty_epistemic": epistemic_total.cpu().numpy(),
            "uncertainty_aleatoric": p_log_p.cpu().numpy(),
            "predicted_class": prob.argmax(dim=-1).cpu().numpy(),
        }

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(config: dict | None = None) -> CollisionRiskTransformer:
    """Build model from config dict or use defaults (PRD specs)."""
    defaults = {
        "input_features": 22,
        "d_model": 1024,
        "n_heads": 16,
        "n_encoder_layers": 10,
        "d_feedforward": 4096,
        "dropout": 0.1,
        "max_seq_len": 256,
        "n_classes": 3,
    }
    if config:
        model_cfg = config.get("model", {})
        # Remap aliased keys from config.yaml
        key_map = {
            "output_classes": "n_classes",
            "nhead": "n_heads",
            "num_encoder_layers": "n_encoder_layers",
        }
        # Valid constructor params
        valid_keys = set(defaults.keys())
        for k, v in model_cfg.items():
            mapped = key_map.get(k, k)
            if mapped in valid_keys:
                defaults[mapped] = v
            # skip unknown keys like sequence_length silently
    return CollisionRiskTransformer(**defaults)


if __name__ == "__main__":
    """Quick sanity check for the model architecture."""
    import sys

    print("Building model...")
    model = build_model()
    n_params = model.count_parameters()
    print(f"  Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    # Test forward pass
    batch_size = 2
    seq_len    = 168
    n_feats    = 22
    x = torch.randn(batch_size, seq_len, n_feats)

    model.eval()
    evidence, alpha, uncertainty, prob = model(x)
    print(f"  Input shape:       {x.shape}")
    print(f"  Evidence shape:    {evidence.shape}")
    print(f"  Alpha shape:       {alpha.shape}")
    print(f"  Uncertainty shape: {uncertainty.shape}")
    print(f"  Probability shape: {prob.shape}")
    print(f"  Prob sum (should≈1): {prob.sum(dim=-1)}")
    print(f"  Uncertainty range: {uncertainty.min():.4f} – {uncertainty.max():.4f}")

    # Check gradient flow
    loss = evidence.sum()
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    print(f"  Gradient norms: min={min(grad_norms):.2e}, max={max(grad_norms):.2e}")
    print(f"  Grad flow OK: {all(g > 0 for g in grad_norms)}")

    # GPU test
    if torch.cuda.is_available():
        print(f"\n  Moving to GPU ({torch.cuda.get_device_name(0)})...")
        model = model.cuda()
        x_gpu = x.cuda()
        with torch.cuda.amp.autocast():
            evidence, alpha, uncertainty, prob = model(x_gpu)
        mem_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        print(f"  GPU memory used: {mem_gb:.2f} GB")
        print("  ✅ GPU forward pass OK")
    else:
        print("  ⚠️  CUDA not available, running on CPU only.")

    print("\n✅ Model architecture verified.")
