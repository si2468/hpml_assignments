import time
import torch
from tqdm.auto import tqdm

def train_model(model, train_loader, optimizer, criterion, device, epochs=5):
    model.train()
    
    for epoch in range(epochs):
        start_epoch = time.perf_counter()
        running_loss = 0.0
        correct = 0
        total = 0

        data_loading_time = 0.0
        training_time = 0.0

        training_iterator = iter(train_loader)

        progress_bar = tqdm(range(len(train_loader)), desc=f"Epoch {epoch+1}", leave=False)

        for _ in progress_bar:
            start_data = time.perf_counter()
            inputs, labels = next(training_iterator)
            end_data = time.perf_counter()
            data_loading_time += end_data - start_data

            start_train = time.perf_counter()
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            end_train = time.perf_counter()
            training_time += end_train - start_train

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix(loss=loss.item(), acc=100. * correct / total)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        end_epoch = time.perf_counter()

        print()
        print(f'Epoch {epoch+1}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        print(f"Epoch {epoch+1}: Data Loading Time: {data_loading_time:.4f}s")
        print(f"Epoch {epoch+1}: Training time: {training_time:.4f}s")
        print(f"Epoch {epoch+1}: Total time: {end_epoch - start_epoch:.4f}s")
