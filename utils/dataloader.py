import numpy
import torch
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms


def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w, h = imgs[0].size
    tensor = torch.zeros((len(imgs), 3, w, h), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = numpy.asarray(img, dtype=numpy.uint8)
        if (nump_array.ndim < 3):
            nump_array = numpy.expand_dims(nump_array, axis=-1)
        nump_array = numpy.rollaxis(nump_array, 2)
        tensor[i] = torch.from_numpy(nump_array)
    return tensor.contiguous(), targets


def folder_loader(traindir, valdir, batch_size):
    train_dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip()
        ]))
    train_loader = Data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=24,
                                   shuffle=True,
                                   drop_last=True,
                                   collate_fn=fast_collate)

    val_dataset = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224)]))
    val_loader = Data.DataLoader(val_dataset,
                                 batch_size=batch_size * 3,
                                 num_workers=24,
                                 collate_fn=fast_collate)
    return train_loader, val_loader
