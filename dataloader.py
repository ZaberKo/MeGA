import torch
import torch.distributed as dist
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def load_cifar100(path: str, train_batch_size: int, val_batch_size: int, num_workers: int = 0, is_distributed=False):
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409),
                             (0.2673, 0.2564, 0.2762)),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409),
                             (0.2673, 0.2564, 0.2762)),
    ])
    train_dataset = datasets.CIFAR100(
        root=path, train=True, download=True, transform=transform_train)
    val_dataset = datasets.CIFAR100(
        root=path, train=False, download=True, transform=transform_val)
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            sampler=train_sampler
        )

        val_sampler = DistributedSampler(val_dataset)
        val_loader = DataLoader(
            train_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            sampler=val_sampler
        )
    else:
        train_loader=DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        val_loader=DataLoader(
            val_dataset,
            batch_size=train_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader


# if __name__ == "__main__":
#     train_loader, val_loader=load_cifar100('./dataset',100,10)
#     loader=iter(train_loader)
#     images,labels=next(loader)
#     print(images.shape)
#     print(labels.shape)