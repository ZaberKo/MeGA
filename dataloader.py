import torch
import torch.distributed as dist
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import os


def load_cifar100(path: str, train_batch_size: int, val_batch_size: int, num_workers: int = 0,data_augment=False, big_size=False,is_distributed=True):
    if data_augment:
        transform_train = transforms.Compose([
            transforms.Resize(224 if big_size else 32),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05),
            transforms.RandomAffine(degrees=30, translate=(0.05,0.05), scale=(0.8,1.2), shear=30, fillcolor=0),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409),
                                (0.2673, 0.2564, 0.2762)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize(224 if big_size else 32),
            transforms.RandomCrop(224 if big_size else 32, padding=224//32*4 if big_size else 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409),
                                (0.2673, 0.2564, 0.2762)),
        ])

    transform_val = transforms.Compose([
        transforms.Resize(224 if big_size else 32),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409),
                             (0.2673, 0.2564, 0.2762)),
    ])
    train_dataset = datasets.CIFAR100(
        root=path, train=True, download=True, transform=transform_train)
    val_dataset = datasets.CIFAR100(
        root=path, train=False, download=True, transform=transform_val)
    train_sampler = None
    val_sampler = None
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=train_sampler is None,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=val_sampler
    )

    return train_loader, val_loader





def load_ImageNet(path: str, train_batch_size: int, val_batch_size: int, num_workers: int = 0, is_distributed=True):
    crop_size = 224
    val_size = 256
    traindir = os.path.join(path, 'train')
    valdir = os.path.join(path, 'val')

    # Note: ToTensor & Normalize steps are put into data_prefetcher
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip()
    ])

    transform_val = transforms.Compose([
        transforms.Resize(val_size),
        transforms.CenterCrop(crop_size),
    ])
    train_dataset = datasets.ImageFolder(
        root=traindir, transform=transform_train)
    val_dataset = datasets.ImageFolder(
        root=valdir,  transform=transform_val)
    train_sampler = None
    val_sampler = None
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=train_sampler is None,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
        collate_fn=fast_collate
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=val_sampler,
        collate_fn=fast_collate
    )

    return train_loader, val_loader

def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)

    return tensor, targets

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor(
            [0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor(
            [0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target

    def __next__(self):
        data = self.next()
        if data is None:
            raise StopIteration
        return data

    def __iter__(self):
        return self


# if __name__ == "__main__":
#     train_loader, val_loader=load_cifar100('./dataset',100,10)
#     loader=iter(train_loader)
#     images,labels=next(loader)
#     print(images.shape)
#     print(labels.shape)
