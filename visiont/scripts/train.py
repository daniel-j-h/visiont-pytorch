import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, CenterCrop, RandomHorizontalFlip
from einops.layers.torch import Rearrange

from tqdm import tqdm

from visiont.models import VisionTransformer


class ConvStem(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.resnet = resnet50(pretrained=pretrained)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        #x = self.resnet.maxpool(x)
        return x


class Model(nn.Module):
    def __init__(self, num_classes, pretrained=False, C=3, H=224, W=224, P=16):
        super().__init__()

        self.stem = ConvStem(pretrained=pretrained)

        self.reshape = Rearrange("n c (h n1) (w n2) -> n (n1 n2) c h w",
                                 n1=(H // 2) // (P // 2),
                                 n2=(W // 2) // (P // 2))

        self.vt = VisionTransformer(num_classes=num_classes,
                                    C=64, H=H // 2, W=W // 2, P=P // 2)

    def forward(self, x):
        x = self.stem(x)
        x = self.reshape(x)
        x = self.vt(x)
        return x


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # ImageNet statistics (because we use pre-trained model)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    train_transform = Compose([
        Resize(256),
        CenterCrop(224),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=mean, std=std)])

    val_transform = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=mean, std=std)])

    train_dataset = ImageFolder(args.dataset / "train", transform=train_transform)
    val_dataset = ImageFolder(args.dataset / "val", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=256, num_workers=4,
                              shuffle=True, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=256, num_workers=4,
                            shuffle=False, pin_memory=True)

    model = Model(num_classes=10, pretrained=True)

    model.stem.eval()

    for param in model.stem.parameters():
        param.requires_grad = False

    model = model.to(device)
    model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam([{"params": model.module.vt.parameters(), "lr": 1e-4}])

    for epoch in range(100):
        loss = train(model, criterion, optimizer, device, train_dataset, train_loader)
        print(f"train loss: {loss:.4f}")

        loss, acc, precision, recall = validate(model, criterion, optimizer, device, val_dataset, val_loader)
        print(f"validate loss: {loss:.4f} acc: {acc:.4f} precision: {precision:.4f} recall: {recall:.4f}")


def train(model, criterion, optimizer, device, dataset, loader):
    model.train()

    model.module.stem.eval()  #

    running_loss = 0.0

    for inputs, labels in tqdm(loader, desc="train"):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(dataset)


def validate(model, criterion, optimizer, device, dataset, loader):
    model.eval()

    running_loss = 0.0
    tn, fn, tp, fp = 0, 0, 0, 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="validate"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            confusion = preds.float() / labels.float()
            tn += torch.sum(torch.isnan(confusion)).item()
            fn += torch.sum(confusion == float("inf")).item()
            tp += torch.sum(confusion == 1).item()
            fp += torch.sum(confusion == 0).item()

        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)

        return running_loss / len(dataset), accuracy, precision, recall
