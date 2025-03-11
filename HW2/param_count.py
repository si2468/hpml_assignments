import torch.nn as nn
from Resnet18 import ResNet18   

model = ResNet18(num_classes=10)

def count_conv_layers(model):
    return sum(1 for layer in model.modules() if isinstance(layer, nn.Conv2d))

# Count convolutional layers
num_conv_layers = sum(1 for layer in model.modules() if isinstance(layer, nn.Conv2d))
print(num_conv_layers)

for name, param in model.named_parameters():
    print(f"{name}: {param.size()}")

#summary(model, input_size=(3, 32, 32))  # 3 channels, 32x32 image size

num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Number of trainable parameters: {num_trainable_params}")

# Count the number of parameters that require gradients
num_gradients = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Number of gradients: {num_gradients}")


