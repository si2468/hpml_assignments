import argparse
import torch
from Resnet18 import ResNet18
from Dataset import create_data_loader
from train import train_model  # Import the training function

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--disable_cuda', action='store_true', help='Disable CUDA if available')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to dataset')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of dataloader workers')
    parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer type (sgd, adam, sgdnesterov, adagrad, adadelta)')
    args = parser.parse_args()
    
    # Set device
    device = "cpu"
    if not args.disable_cuda and torch.cuda.is_available():
        device = "cuda"

    # Load dataset
    train_loader = create_data_loader(args.data_path, batch_size=128, num_workers=args.num_workers)
    
    # Initialize model
    model = ResNet18(num_classes=10).to(device)

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Create optimizer
    lr = 0.1
    optimizers = {
        "adam": torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4),
        "adagrad": torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=5e-4),
        "adadelta": torch.optim.Adadelta(model.parameters(), lr=lr, weight_decay=5e-4),
        "sgdnesterov": torch.optim.SGD(model.parameters(), lr=lr, weight_decay=5e-4, nesterov=True, momentum=0.9),
        "sgd": torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4),
    }
    
    optimizer = optimizers.get(args.optimizer.lower(), optimizers["sgd"])

    # Train model
    train_model(model, train_loader, optimizer, criterion, device, epochs=5)

if __name__ == '__main__':
    main()
