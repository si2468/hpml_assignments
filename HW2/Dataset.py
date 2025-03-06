import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
    
def create_data_loader(data_path, batch_size=128, num_workers=2):
    variances = [0.2023, 0.1994, 0.2010]
    stds = [x ** 0.5 for x in variances]
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=stds)
    ])

    train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    return DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
