import torch
import torch.nn as nn

from einops import rearrange


# An Image Is Worth 16x16 Words: Transformers For Image Recognition At Scale
# https://openreview.net/pdf?id=YicbFdNTTy


class VisionTransformer(nn.Module):
    def __init__(self, num_classes, C=3, H=224, W=224, P=16):
        super().__init__()

        D = 768
        M = (H * W) // (P * P)

        # Linear projection of flattened patches
        self.patch_projection = nn.Linear(P * P * C, D)

        # Patch position embedding
        self.positional_embedding = nn.Embedding(M + 1, D)

        # Extra learnable classification token
        self.class_token = nn.Parameter(torch.rand(D))

        # Standard transformer encoder, baseline defaults
        self.enc = nn.TransformerEncoder(num_layers=12,
                encoder_layer=nn.TransformerEncoderLayer(D, nhead=12, dim_feedforward=3072))

        # Final classification head
        self.final = nn.Sequential(
                nn.Linear(D, D * 4),
                nn.GELU(),
                nn.Linear(D * 4, num_classes))


    def forward(self, x):
        # x is N-sized batch of M HxW patches; splitting images into
        # patches must happen in transforms to not block the mainloop

        N, C, M, H, W = x.size()

        # Flatten and project the patches
        x = rearrange(x, "n c m h w -> n m (c h w)")
        x = self.patch_projection(x)

        # Learnable token to classify
        c = self.class_token.expand(N, 1, -1)
        x = torch.cat([c, x], dim=1)

        # Augment with patch positions
        p = torch.arange(0, M + 1).repeat(N, 1)
        p = self.positional_embedding(p)

        x = x + p

        # Feed through transformer encoder
        x = self.enc(x)
        x = x[:, 0, :]

        # Token output classification head
        x = self.final(x)

        return x
