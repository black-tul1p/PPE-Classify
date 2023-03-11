import torch
import torch.nn as nn
from tqdm import tqdm
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

        print('Finished Training')

    def eval(self):
        # Evaluate the model on the test set
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = (100 * correct / total)
        print('Accuracy of the network on the test images: %.2f%%' % accuracy)
        return accuracy

    def save_model(self):
        torch.save(self.net.state_dict(), 'model.pth')
        print('Model saved.')
