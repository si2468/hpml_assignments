import argparse
from Resnet18 import ResNet18
from Resnet18torchcompiled import ResNet18TorchCompiled
import torch
from Dataset import create_data_loader
from train import train_model  # Import the training function
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
    parser.add_argument('--torch_compile', action="store_true", help="Compile forward pass of Resnet")
    args = parser.parse_args()
    
    # Set device
    device = "cpu"
    if not args.disable_cuda and torch.cuda.is_available():
        device = "cuda"

    # Load dataset
    train_loader = create_data_loader(args.data_path, batch_size=128, num_workers=args.num_workers)
    
    # Initialize model
    if args.torch_compile:
        model = ResNet18(num_classes=10, disable_bn=args.disable_batch_normalization, use_compile=args.torch_compile).to(device)
    else:
        model = ResNet18TorchCompiled(num_classes=10, disable_bn=args.disable_batch_normalization, use_compile=args.torch_compile).to(device)

    def count_conv_layers(model):
        return sum(1 for layer in model.modules() if isinstance(layer, nn.Conv2d))

    # Count convolutional layers
    num_conv_layers = count_conv_layers(model)
    #print(num_conv_layers)

    #print(model)
    #print(model.fc)

    for name, param in model.named_parameters():
        print(f"{name}: {param.size()}")
    # Use summary function to display model architecture
    summary(model, input_size=(3, 32, 32))  # 3 channels, 32x32 image size

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Number of trainable parameters: {num_trainable_params}")

    # Count the number of parameters that require gradients
    num_gradients = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Number of gradients: {num_gradients}")

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
    train_model(model, train_loader, optimizer, criterion, device, epochs=args.num_epochs)

if __name__ == '__main__':
    main()
