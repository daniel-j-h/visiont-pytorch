from tqdm import tqdm
import numpy as np

import torch
from einops.layers.torch import Rearrange
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (
    Compose,
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
from PIL import Image
from skimage.io import imsave
from visiont.scripts.train import Convert

rearrange = Rearrange("c h w -> h w c")


class ImageDirectory(Dataset):
    def __init__(self, root, base=None, patches=None):
        super().__init__()

        self.paths = [p for p in root.iterdir() if p.is_file()]

        self.base = base
        self.patches = patches

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = str(self.paths[i])

        # TODO: don't copy the image itself to save on resources
        image1 = Image.open(path)
        image2 = image1.copy()

        if self.base is not None:
            image1 = self.base(image1)
            image2 = self.base(image2)

        if self.patches is not None:
            patches1 = self.patches(image1)
            patches2 = self.patches(image2)

        return image1, image2, patches1, patches2


class ToPatches:
    def __init__(self, size):
        self.rearrange = Rearrange("c (h p1) (w p2) -> (h w) p1 p2 c", p1=size, p2=size)

    def __call__(self, x):
        return self.rearrange(x)


def save_samples(image1, image2, patch1, patch2, idx, path):
    def save_tensor_to_image(fpath, img):
        arr = img.cpu().numpy()
        arr = (arr * 255).astype(np.uint8)
        imsave(fpath.as_posix(), arr)

    image1 = rearrange(image1)
    image2 = rearrange(image2)

    save_tensor_to_image(path.joinpath(f"1-{idx:06}.jpg"), image1)
    save_tensor_to_image(path.joinpath(f"2-{idx:06}.jpg"), image2)

    for idy, img in enumerate(patch1):
        save_tensor_to_image(path.joinpath(f"1-{idx:06}-{idy:03}.jpg"), img)
    for idy, img in enumerate(patch2):
        save_tensor_to_image(path.joinpath(f"2-{idx:06}-{idy:03}.jpg"), img)


def main(args):

    # Just a list to try out different augmentation ideas here
    # Bring learnings into visiont.script.train.main
    base = [
        Convert("RGB"),
        Resize(1024),
        RandomCrop(800),
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
    ]
    base_transform = Compose(base)
    patch_transform = Compose([ToPatches(100)])

    dataset = ImageDirectory(args.dataset, base_transform, patch_transform)

    # TODO: hard coded for now, works on my 2x Titan RTX machine
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    progress = tqdm(loader, ascii=False, unit="samples", total=args.num)

    for idx, (img1, img2, patch1, patch2) in enumerate(progress):
        progress.set_description(f"sample : {idx}")
        save_samples(img1[0], img2[0], patch1[0], patch2[0], idx, args.location)

        if idx > args.num:
            break
