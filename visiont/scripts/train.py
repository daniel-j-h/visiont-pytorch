import torch
import torch.nn as nn
import torch.optim
import torch.cuda.amp
import torch.backends.cudnn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, RandomCrop, RandomHorizontalFlip, RandomApply, RandomGrayscale, ColorJitter, GaussianBlur
from tqdm import tqdm

from visiont.models import VisionTransformer
from visiont.datasets import ImageDirectory
from visiont.transforms import ToImageMode, ToPatches


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
            nn.Linear(fin, fmid),
            nn.LayerNorm(fmid),
            nn.GELU(),
            nn.Linear(fmid, fout))


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # ImageNet stats for now
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    transform = Compose([
        ToImageMode("RGB"),
        Resize(1024),
        RandomCrop(224),
        RandomHorizontalFlip(p=0.5),
        RandomApply([ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)], p=0.8),
        RandomApply([GaussianBlur((3, 3), (1.5, 1.5))], p=0.1),
        RandomGrayscale(p=0.2),
        ToTensor(),
        Normalize(mean=mean, std=std),
        ToPatches(16)])

    # Applying the same transform twice will give us
    # different transformations on the same image,
    # because the transformation's rng state changes.
    dataset = ImageDirectory(args.dataset, transform, transform)

    # TODO: hard coded for now, works on my 2x Titan RTX machine
    loader = DataLoader(dataset, batch_size=160, num_workers=40, shuffle=True, pin_memory=True, drop_last=True)

    # We will chop off the final layer anyway,
    # therefore num_classes doesn't matter here.
    online = VisionTransformer(num_classes=1, C=3, H=224, W=224, P=16)
    target = VisionTransformer(num_classes=1, C=3, H=224, W=224, P=16)

    # Projection heads for both networks
    #online.final = mlp(768, 4096, 256)
    #target.final = mlp(768, 4096, 256)
    online.final = nn.Identity()
    target.final = nn.Identity()

    # Target network does not learn on its own.
    # Gets average of online network's weights.

    online.train()
    target.eval()

    for param in target.parameters():
        param.requires_grad = False

    def update_target():
        update(target, online, 0.99)

    # In addition to projection heads,
    # The online network has predictor.
    #predictor = mlp(256, 4096, 256)
    predictor = mlp(768, 4096, 768)

    # Move everything to devices

    online = online.to(device)
    online = nn.DataParallel(online)

    predictor = predictor.to(device)
    predictor = nn.DataParallel(predictor)

    target = target.to(device)
    target = nn.DataParallel(target)

    def criterion(x, y):
        x = nn.functional.normalize(x, dim=-1)
        y = nn.functional.normalize(y, dim=-1)
        return 2 - 2 * (x * y).sum(dim=-1)

    lr = 1e-4

    # Online and predictor learns, target gets assigned moving average of online network's weights.
    #optimizer = torch.optim.Adam(list(online.parameters()) + list(predictor.parameters()), lr=lr)
    optimizer = torch.optim.SGD(list(online.parameters()) + list(predictor.parameters()), lr=lr)

    # Warmup Adam, he cold
    def adjust_learning_rate(optimizer, step, lr):
        for g in optimizer.param_groups:
            g["lr"] = min(1, (step + 1) / 1000) * lr

    scaler = torch.cuda.amp.GradScaler()

    step = 0
    running = 0

    for epoch in range(100):

        progress = tqdm(loader, desc=f"Epoch {epoch+1}", unit="batch")

        for inputs1, inputs2 in progress:
            assert inputs1.size() == inputs2.size()

            # Overlap data transfers to gpus, pinned memory
            inputs1 = inputs1.to(device, non_blocking=True)
            inputs2 = inputs2.to(device, non_blocking=True)

            adjust_learning_rate(optimizer, step, lr)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                # Target network is in eval mode and does not
                # require grads, forward no grad ctx to be sure
                with torch.no_grad():
                    labels1 = target(inputs1).detach()
                    labels2 = target(inputs2).detach()

                outputs1 = predictor(online(inputs1))
                outputs2 = predictor(online(inputs2))

                # Symmetrize the loss, both transformations
                # go through both networks, one at a time
                loss = criterion(outputs1, labels2) + criterion(outputs2, labels1)
                loss = loss.mean()

            scaler.scale(loss).backward()

            # Transformers need their nails clipped
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(online.parameters(), 1)
            nn.utils.clip_grad_norm_(predictor.parameters(), 1)

            scaler.step(optimizer)
            scaler.update()

            # After training the online network, we transfer
            # a weighted average of the weights to the target
            update_target()

            running += loss.item() * inputs1.size(0)

            if step % 100 == 0:
                progress.write(f"loss: {running / 100}")
                running = 0

            step += 1

        torch.save(online.state_dict(), f"vt-{epoch + 1:03d}.pth")
