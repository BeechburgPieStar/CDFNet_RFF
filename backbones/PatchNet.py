import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, seq_len, patch_size, in_chans, embed_dim, stride=None):
        super().__init__()
        if stride is None:
            stride = patch_size // 2
        if seq_len < patch_size:
            raise ValueError(f"seq_len({seq_len}) must be >= patch_size({patch_size}).")
        if stride <= 0:
            raise ValueError(f"stride({stride}) must be > 0.")
        num_patches = (seq_len - patch_size) // stride + 1
        self.num_patches = num_patches

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        # x: [B, C, L]
        x_out = self.proj(x)          # [B, embed_dim, N]
        x_out = x_out.transpose(1, 2) # [B, N, embed_dim]
        return x_out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dr):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dr),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dr)
        )

    def forward(self, x):
        return self.net(x)


class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_ratio, dr):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, hidden_dim, dr)

    def forward(self, x):
        # x: [B, N, D]
        x = x + self.ffn(self.norm(x))
        return x


class PatchNet(nn.Module):
    def __init__(self,
                 seq_len=256,
                 in_chans=2,
                 patch_size=64,
                 embed_dim=128,
                 num_classes=6,
                 mlp_ratio=4.0,
                 dr=0.5,
                 stride=None):
        super().__init__()
        self.patch_embedding = PatchEmbed(seq_len, patch_size, in_chans, embed_dim, stride=stride)
        self.bottleneck = MLP(embed_dim, mlp_ratio=mlp_ratio, dr=dr)
        self.cls_head = nn.Linear(embed_dim, num_classes)

    def forward_features(self, x):
        x_patch = self.patch_embedding(x)  # [B, N, D]
        x_neck = self.bottleneck(x_patch)  # [B, N, D]
        x_mean = x_neck.mean(dim=1)        # [B, D]
        return x_mean

    def forward(self, x, return_features=False):
        x_mean = self.forward_features(x)
        logit = self.cls_head(x_mean)
        if return_features:
            return logit, x_mean
        return logit