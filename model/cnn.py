import torch
from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    CNN with 3 conv layers + 2 FC layers. Designed for small images (e.g. CIFAR-10).
    
    Args:
        input_shape (tuple): (C, H, W)
        n_classes (int): number of output classes
    """
    def __init__(self, input_shape=(3,32,32), n_classes=10):
        super().__init__()
        C, H, W = input_shape
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(C, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        #self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Use dummy forward to infer FC input size
        with torch.no_grad():
            x = torch.zeros(1, C, H, W)  # batch size 1
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = F.relu(self.conv3(x))
            x = self.pool(x)
            #x = F.relu(self.conv4(x))
            #x = self.pool(x)
            fc_input_size = x.shape[1] * x.shape[2] * x.shape[3]
        
        # Fully connected layers
        self.fc1 = nn.Linear(fc_input_size, 256)
        self.fc2 = nn.Linear(256, n_classes)
    
    def forward(self, x):
        # Conv layers with ReLU + pooling
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        #x = F.relu(self.conv4(x))
        #x = self.pool(x)
        
        # Flatten and fully connected
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
