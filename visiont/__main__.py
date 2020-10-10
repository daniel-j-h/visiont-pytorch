import sys

import torch

from .visiont import VisionTransformer


def main():
    net = VisionTransformer(num_classes=10)

    # Dummy batch with 14 * 14 patches of
    # size 16x16 px for 224x224 px images
    N, C, M, H, W = 1, 3, 14 * 14, 16, 16

    batch = torch.rand(N, C, M, H, W)

    out = net(batch)

    print("It works!", file=sys.stderr)


if __name__ == "__main__":
    main()
