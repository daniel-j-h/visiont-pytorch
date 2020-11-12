from tqdm import tqdm
import numpy as np

from einops.layers.torch import Rearrange
from torch.utils.data import DataLoader
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
from skimage.io import imsave

from visiont.scripts.train import ImageDirectory, Convert


class ToPatches:
    def __init__(self, size):
        self.rearrange = Rearrange("c (h p1) (w p2) -> (h w) p1 p2 c", p1=size, p2=size)

    def __call__(self, x):
        return self.rearrange(x)


def save_samples(image1, image2, idx, path):
    def save_tensor_to_image(fpath, img):
        arr = img.cpu().numpy()
        arr = (arr * 255).astype(np.uint8)
        imsave(fpath.as_posix(), arr)

    image1 = image1.squeeze(0)
    image2 = image2.squeeze(0)

    for idy, img in enumerate(image1):
        save_tensor_to_image(path.joinpath(f"1-{idx:06}-{idy:03}.jpg"), img)
    for idy, img in enumerate(image2):
        save_tensor_to_image(path.joinpath(f"2-{idx:06}-{idy:03}.jpg"), img)


def main(args):

    # Just a list to try out different augmentation ideas here
    # Bring learnings into visiont.script.train.main
    transform = Compose(
        [
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
            ToPatches(80),
        ]
    )

    dataset = ImageDirectory(args.dataset, transform, transform)

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

    for idx, (img1, img2) in enumerate(progress):
        progress.set_description(f"sample : {idx}")
        save_samples(img1, img2, idx, args.location)

        if idx > args.num:
            break
