# transformer_htis_demo.py

"""
This script runs an end-to-end demo of a transformer block using
Hierarchical Token Importance Scoring (HTIS). The focus is on
demonstrating how token importance can be learned and applied
during the forward pass to reduce computation or selectively focus attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# Core attention logic using scaled dot-product
class HTISAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.scale = d_model ** 0.5

    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, V), weights


# Scores each token based on three perspectives
class TokenImportanceScorer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.att = nn.Linear(d_model, 1)
        self.conv = nn.Conv1d(d_model, 1, kernel_size=3, padding=1)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        att_score = self.att(x)
        conv_score = self.conv(x.transpose(1, 2)).transpose(1, 2)
        ffn_score = self.ffn(x)
        avg_score = (att_score + conv_score + ffn_score) / 3
        return torch.sigmoid(avg_score)


# Transformer block with HTIS
class TransformerBlockHTIS(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.attn = HTISAttention(d_model)
        self.token_scorer = TokenImportanceScorer(d_model)
        self.output = nn.Linear(d_model, d_model)

    def forward(self, x):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        attn_out, weights = self.attn(Q, K, V)

        importance = self.token_scorer(x)               # shape: [B, T, 1]
        gated = attn_out * importance                   # suppress low-importance tokens
        return self.output(gated), weights, importance


# Simple model wrapper
class MiniHTISModel(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.block = TransformerBlockHTIS(d_model)
        self.final = nn.Linear(d_model, d_model)

    def forward(self, x):
        out, attn, importance = self.block(x)
        return self.final(out), attn, importance


# Run the model and visualize token importance
def run_demo():
    torch.manual_seed(0)

    batch = 2
    seq_len = 8
    d_model = 32

    x = torch.randn(batch, seq_len, d_model)
    model = MiniHTISModel(d_model)

    output, attn, importance = model(x)

    print("Output shape:", output.shape)
    print("Attention shape:", attn.shape)
    print("Importance shape:", importance.shape)

    # Show token importance for one sample
    plt.imshow(importance[0].detach().numpy(), cmap='viridis', aspect='auto')
    plt.title("Token Importance Scores")
    plt.xlabel("Hidden Dim (compressed to 1)")
    plt.ylabel("Tokens")
    plt.colorbar()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run_demo()

