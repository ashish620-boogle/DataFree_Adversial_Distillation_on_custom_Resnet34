from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader, random_split

def get_dataloader(args):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])

    # Load the full CIFAR100 training dataset
    full_dataset = datasets.CIFAR100(args.data_root, train=True, download=True, transform=transform)

    # Calculate sizes for split
    total_len = len(full_dataset)  # 50000
    train_len = int(0.9 * total_len)  # 45000
    test_len = total_len - train_len  # 5000

    # Split into 90% train and 10% test
    train_dataset, test_dataset = random_split(full_dataset, [train_len, test_len])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    return train_loader, test_loader
