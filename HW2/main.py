import argparse
from Resnet18 import ResNet18
import torch
from Dataset import create_data_loader
from train import train_model
import torch.nn as nn
from torchsummary import summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--disable_cuda', action='store_true', help='Disable CUDA if available')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to dataset')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of dataloader workers')
    parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer type (sgd, adam, sgdnesterov, adagrad, adadelta)')
    parser.add_argument('--disable_batch_normalization', action='store_true', help='Disable batch normaliaztion in Resnet')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument(
        '--torch_compile', 
        choices=['default', 'reduce-overhead', 'max-autotune', 'none'],
        default='none',
        help='Choose the Torch Compile mode: default, reduce-overhead, or max-autotune (default: default)'
    )
    parser.add_argument("--profile_code", action="store_true", help="Profile certain parts of the code")
    parser.add_argument('--profile_directory', type=str, default="Name of directory where this will be created - will automatically be put in ./profile_experiments_<profile_directory>/")

    args = parser.parse_args()
    
    # Set device
    device = "cpu"
    if not args.disable_cuda and torch.cuda.is_available():
        device = "cuda"

    # Load dataset
    train_loader = create_data_loader(args.data_path, batch_size=128, num_workers=args.num_workers)
    
    torch_compile_mode = args.torch_compile.lower()
    model = ResNet18(num_classes=10, disable_bn=args.disable_batch_normalization, compile_mode=torch_compile_mode).to(device)

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Create optimizer
    lr = 0.1
    optimizers = {
        "adam": torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4),
        "adagrad": torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=5e-4),
        "adadelta": torch.optim.Adadelta(model.parameters(), lr=lr, weight_decay=5e-4, rho=0.9),
        "sgdnesterov": torch.optim.SGD(model.parameters(), lr=lr, weight_decay=5e-4, nesterov=True, momentum=0.9),
        "sgd": torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4),
    }
    
    optimizer = optimizers.get(args.optimizer.lower(), optimizers["sgd"])

    # Train model
    train_model(model, train_loader, optimizer, criterion, device, args.num_epochs, args.profile_code, args.profile_directory)

if __name__ == '__main__':
    main()
