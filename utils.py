import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def get_train_loader(set_path, dataset_name, img_size, batch_size):
    transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()], )
    if dataset_name == "STL":
        data = datasets.STL10(root=set_path, split='train', transform=transform, download=True)

    elif dataset_name == "CIFAR":
        data = datasets.CIFAR10(root=set_path, train=True, transform=transform, download=True)

    else:
        data = datasets.ImageFolder(root=set_path, transform=transform)

    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True), len(data.classes)


def get_test_loader(set_path, dataset_name, img_size, batch_size=1):
    transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()], )
    if dataset_name == "STL":
        data = datasets.STL10(root=set_path, split='test', transform=transform, download=True)

    elif dataset_name == "CIFAR":
        data = datasets.CIFAR10(root=set_path, train=False, transform=transform, download=True)

    else:
        data = datasets.ImageFolder(root=set_path, transform=transform)

    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True), len(data.classes)