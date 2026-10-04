# Adapted from haidog-yaqub/MeanFlow, MIT; see third_party/MeanFlow.LICENSE.
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms as T
from torchvision.datasets import ImageFolder


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def build_dataset(cfg, train=True, augment=True):
    if cfg["dataset"] == "cifar10":
        return torchvision.datasets.CIFAR10(
            root=cfg["data_root"],
            train=train,
            download=True,
            transform=T.Compose([T.ToTensor()] + ([T.RandomHorizontalFlip()] if train and augment else [])),
        )

    if cfg["dataset"] == "mnist":
        return torchvision.datasets.MNIST(
            root=cfg["data_root"],
            train=train,
            download=True,
            transform=T.Compose([
                T.Resize((cfg["image_size"], cfg["image_size"])),
                T.ToTensor(),
            ]),
        )

    if cfg["dataset"] == "imagenet":
        if not train:
            raise ValueError("A separate ImageNet validation loader has not been configured")
        transform = T.Compose([
            T.Lambda(lambda pil_image: center_crop_arr(pil_image, cfg["image_size"])),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ])
        return ImageFolder(cfg["data_root"], transform=transform)

    raise ValueError(f"Unknown dataset: {cfg['dataset']}")


def cycle(iterable):
    while True:
        for item in iterable:
            yield item


class ResidentCIFARBatcher:
    """Keep CIFAR tensors on the GPU; checkpoint permutation and cursor for resume."""
    def __init__(self, dataset, batch_size, device, state=None):
        self.images = torch.as_tensor(dataset.data, device=device).permute(0, 3, 1, 2).float().div_(255)
        self.labels = torch.as_tensor(dataset.targets, device=device, dtype=torch.long)
        self.batch_size = batch_size
        self.position = len(self.images)
        self.permutation = torch.arange(len(self.images), device=device)
        if state is not None:
            self.position = state["position"]
            self.permutation = state["permutation"].to(device)

    def __iter__(self):
        return self

    def __next__(self):
        if self.position + self.batch_size > len(self.images):
            self.permutation = torch.randperm(len(self.images), device=self.images.device)
            self.position = 0
        indices = self.permutation[self.position:self.position + self.batch_size]
        self.position += self.batch_size
        images = self.images[indices]
        # Per-image flips avoid correlating the augmentation across the whole batch.
        flipped = torch.rand(len(images), device=images.device) < 0.5
        images = torch.where(flipped[:, None, None, None], images.flip(-1), images)
        return images.contiguous(memory_format=torch.channels_last), self.labels[indices]

    def state_dict(self):
        return dict(position=self.position, permutation=self.permutation.cpu())
