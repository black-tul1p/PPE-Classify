import torch
import torch.nn as nn

class NNet(nn.Module):
    def __init__(self) -> None:
        super(NNet, self).__init__()

        # Add convolutional layers to extract features
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        
        # Add batch normalization for convolutional layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        # Add pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Add dropout layer to reduce overfitting
        self.dropout = nn.Dropout(p=0.5)
        
        # Fully connected layers to classify the images
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 5)

    def forward(self, x):
        # Apply batch-normalized convolutional layers with ReLU activation and max pooling
        x = self.bn1(nn.functional.relu(self.conv1(x)))
        x = self.pool(self.bn2(nn.functional.relu(self.conv2(x))))
        x = self.pool(self.bn3(nn.functional.relu(self.conv3(x))))
        x = self.pool(self.bn4(nn.functional.relu(self.conv4(x))))

        # Flatten feature maps
        x = x.view(-1, 256 * 8 * 8)

        # Drop layers
        x = self.dropout(nn.functional.relu(self.fc1(x)))
        x = self.dropout(nn.functional.relu(self.fc2(x)))

        # Apply output layer with softmax activation
        x = self.fc3(x)
        return x

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))
        self.eval()