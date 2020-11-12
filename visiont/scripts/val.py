import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
    Resize,
    RandomCrop,
    RandomHorizontalFlip,
    RandomApply,
    RandomGrayscale,
    ColorJitter,
    GaussianBlur,
    RandomRotation,
    RandomErasing,
)
from einops.layers.torch import Rearrange
from PIL import Image
from tqdm import tqdm

from visiont.models import VisionTransformer


# Simple directory image loader, applying two transforms
# on each loaded image and returning both transformations.
class ImageDirectory(Dataset):
    def __init__(self, root, transform1=None, transform2=None):
        super().__init__()

        self.paths = [p for p in root.iterdir() if p.is_file()]

        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = str(self.paths[i])

        # TODO: don't copy the image itself to save on resources
        image1 = Image.open(path)
        image2 = image1.copy()

        if self.transform1 is not None:
            image1 = self.transform1(image1)

        if self.transform2 is not None:
            image2 = self.transform2(image2)

        return image1, image2


# Transforms an image into a collection of
# the image's patches of a specific size.
class ToPatches:
    def __init__(self, size):
        self.rearrange = Rearrange(
            "c (h p1) (w p2) -> (h w) (p1 p2 c)", p1=size, p2=size
        )

    def __call__(self, x):
        return self.rearrange(x)


# Transforms an image's mode, see
# PIL's image modes e.g. "RGB".
class Convert:
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, x):
        return x.convert(self.mode)


# Updates a destination network's weights
# with a linear combination of
#   r * destination + (1 - r) * source
# Requires networks of same architectures.
def update(dst, src, r):
    assert 0 < r < 1

    for dp, sp in zip(dst.parameters(), src.parameters()):
        dp.data.copy_(r * dp.data + (1 - r) * sp.data)


# Simple MLP following the transformer
# architecture's choices of layers.
def mlp(fin, fmid, fout):
    return nn.Sequential(
        nn.Linear(fin, fmid), nn.LayerNorm(fmid), nn.GELU(), nn.Linear(fmid, fout)
    )


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # ImageNet stats for now
    # mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    transform = Compose(
        [
            Convert("RGB"),
            Resize(300),
            RandomCrop(224),
            RandomHorizontalFlip(p=0.5),
            RandomApply(
                [ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)],
                p=0.5,
            ),
            RandomApply([GaussianBlur((3, 3), (1.5, 1.5))], p=0.5),
            RandomGrayscale(p=0.5),
            RandomRotation(degrees=(-10, 10)),
            ToTensor(),
            RandomErasing(p=0.5),
            ToPatches(16),
        ]
    )

    # Applying the same transform twice will give us
    # different transformations on the same image,
    # because the transformation's rng state changes.
    dataset = ImageDirectory(args.dataset, transform, transform)

    # TODO: hard coded for now, works on my 2x Titan RTX machine
    loader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=40,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    # We will chop off the final layer anyway,
    # therefore num_classes doesn't matter here.
    model = VisionTransformer(num_classes=1, C=3, H=224, W=224, P=16)
    model.final = nn.Identity()
    # Target network does not learn on its own.
    # Gets average of online network's weights.

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.weights.as_posix()))
    model = model.to(device)
    model.eval()

    progress = tqdm(loader, unit="batch")

    def scores(x, y):
        x = x.view(x.shape[0], -1)
        y = y.view(y.shape[0], -1)
        x = nn.functional.normalize(x, dim=-1)
        y = nn.functional.normalize(y, dim=-1)
        return 2 - 2 * torch.matmul(x, torch.transpose(y, 0, 1))

    expected = torch.arange(0, 32).to(device)

    for (inputs1, inputs2) in progress:
        assert inputs1.size() == inputs2.size()

        # Overlap data transfers to gpus, pinned memory
        inputs1 = inputs1.to(device, non_blocking=True)
        inputs2 = inputs2.to(device, non_blocking=True)

        with torch.no_grad():
            feat_0 = model(inputs1).detach()
            feat_1 = model(inputs2).detach()
            distances = scores(feat_0, feat_1)
            import pdb

            pdb.set_trace()
            miss = distances.argsort(axis=1)[:, 0] - expected

        progress.set_description(f"mean rank {miss.mean()}")
