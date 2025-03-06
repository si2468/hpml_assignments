import argparse
from Resnet18 import ResNet18
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from Dataset import create_data_loader
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to dataset')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of dataloader workers')
    parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer type (sgd or adam)')
    args = parser.parse_args()
    

    # set the device according to cuda availability and args.cuda
    device = "cpu"
    if args.cuda and torch.cuda.is_available():
        device = "cuda"

    # load the CIFAR-10 dataset and create the dataloader object
    train_loader = create_data_loader(args.data_path, batch_size=128, num_workers=args.num_workers)
    
    model = ResNet18(num_classes=10)
    criterion = nn.CrossEntropyLoss()

    # create the optimizer
    lr = 0.001

    optimizer = None
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    elif args.optimizer.lower() == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=5e-4)
    elif args.optimizer.lower() == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=5e-4)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    
    # hyperparameters
    epochs = 5

    model.to(device)
    model.train()
    for epoch in range(epochs):
        start_epoch = time.perf_counter()
        running_loss = 0.0
        correct = 0
        total = 0

        data_loading_time = 0.0
        training_time = 0.0

        for _ in range(len(train_loader)):  # To accurately measure data retrieval
            start_data = time.perf_counter()
            inputs, labels = next(iter(train_loader))  # Fetch a batch from the DataLoader
            end_data = time.perf_counter()
            data_loading_time += end_data - start_data  # Accumulate data loading time

            start_train = time.perf_counter()
            inputs, labels = inputs.to(device), labels.to(device)  # Move to device AFTER timing retrieval
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            end_train = time.perf_counter()
            training_time += end_train - start_train  # Accumulate training time

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total

        end_epoch = time.perf_counter()

        print(f'Epoch {epoch+1}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        print(f"Epoch {epoch+1}: Data Loading Time: {data_loading_time:.4f}s")
        print(f"Epoch {epoch+1} took {training_time:.4f}s to train")
        print(f"Epoch {epoch+1} took {end_epoch - start_epoch:.4f}s")
    
if __name__ == '__main__':
    main()