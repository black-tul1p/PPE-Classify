import os
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from ScrapedDataset import ScrapedDataset
import torchvision.transforms as transforms


class NNet(nn.Module):
    def __init__(self) -> None:
        super(NNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))
        self.eval()

class Trainer:
    def __init__(self, net, batch_size=32, shuffle=True):
        self.net = net
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") # type: ignore

        # Define the dataset and data loaders
        transform = transforms.Compose(
            [transforms.Resize((64, 64)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        # Create the dataset
        dataset = ScrapedDataset(transform=transform)

        # Determine the sizes of the train and test sets
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size

        # Split the dataset into train and test sets
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        # Create the data loaders
        self.trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        self.testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Initialize empty lists for train and test accuracy
        self.train_accuracy = []
        self.test_accuracy = []


    def train(self, criterion, optimizer, epochs=10):
        # Set up the model and optimizer
        self.net.to(self.device)

        # Train the model
        for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0

            # Calculate train and test accuracy after each epoch
            train_acc = self._accuracy(self.trainloader)
            test_acc = self._accuracy(self.testloader)
            self.train_accuracy.append(train_acc)
            self.test_accuracy.append(test_acc)

    def eval(self):
        # Evaluate the model on the test set
        accuracy = self._accuracy(self.testloader)
        print('Accuracy of the network on the test images: %.2f%%' % accuracy)
        return accuracy

    def save_model(self):
        torch.save(self.net.state_dict(), 'model.pth')
        print('Model saved.')

    def _accuracy(self, dataloader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in dataloader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = (100 * correct / total)
        return accuracy
    

    def plot_accuracy(self):
        plt.plot(range(1, len(self.train_accuracy) + 1), self.train_accuracy, label='Train')
        plt.plot(range(1, len(self.test_accuracy) + 1), self.test_accuracy, label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy')
        plt.legend()
        # plt.show()
        folder_path = os.path.join('.', 'Plots')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plt.savefig(os.path.join(folder_path, f"accuracy_{datetime.now().strftime('%H_%M_%S')}"))
