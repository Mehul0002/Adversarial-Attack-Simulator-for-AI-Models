import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_mnist_data(batch_size=64, train=True):
    """
    Load MNIST dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(root='./data', train=train, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return dataloader

def get_sample_image(dataloader):
    """
    Get a sample image and label from the dataloader.
    """
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    return images[0].unsqueeze(0), labels[0]
