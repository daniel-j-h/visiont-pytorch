from einops.layers.torch import Rearrange


# Transforms an image into a collection of
# the image's patches of a specific size.
class ToPatches:
    def __init__(self, size):
        self.rearrange = Rearrange("c (h p1) (w p2) -> (h w) (p1 p2 c)", p1=size, p2=size)

    def __call__(self, x):
        return self.rearrange(x)


# Transforms an image's mode, see
# PIL's image modes e.g. "RGB".
class ToImageMode:
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, x):
        return x.convert(self.mode)
