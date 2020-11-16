from PIL import Image

from torch.utils.data import Dataset


# Simple directory image loader, applying two transforms
# on each loaded image and returning both transformations.
class ImageDirectory(Dataset):
    def __init__(self, root, transform1=None, transform2=None):
        super().__init__()

        self.paths = sorted([p for p in root.iterdir() if p.is_file()])

        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = str(self.paths[i])

        image1 = Image.open(path)
        image2 = image1.copy()

        if self.transform1 is not None:
            image1 = self.transform1(image1)

        if self.transform2 is not None:
            image2 = self.transform2(image2)

        return image1, image2
