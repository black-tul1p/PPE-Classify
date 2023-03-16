import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from ScrapedDataset import ScrapedDataset
import torchvision.transforms as transforms

class Trainer:
    def __init__(self, model, batch_size=32, shuffle=False, learning_rate=0.001):
        self.model = model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") # type: ignore
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.train_loss = []
        self.train_accuracy = []
        self.test_loss = []
        self.test_accuracy = []

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
        self.train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        self.test_data = DataLoader(test_dataset, batch_size=batch_size)


    def train(self, epochs):
        self.model.to(self.device)
        for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
            running_loss = 0.0
            correct = 0
            total = 0
            for data in self.train_data:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_loss = running_loss / len(self.train_data)
            train_accuracy = 100 * correct / total
            self.train_loss.append(train_loss)
            self.train_accuracy.append(train_accuracy)

            test_loss, test_accuracy = self.test()
            self.test_loss.append(test_loss)
            self.test_accuracy.append(test_accuracy)

            # print('Epoch [%d], Train Loss: %.4f, Train Accuracy: %.2f%%, Test Loss: %.4f, Test Accuracy: %.2f%%'
            #       % (epoch + 1, train_loss, train_accuracy, test_loss, test_accuracy))


    def test(self):
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test_data:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss /= len(self.test_data)
        test_accuracy = 100 * correct / total
        return test_loss, test_accuracy


    def plot_loss(self, show=False):
        plt.plot(range(1, len(self.train_loss) + 1), self.train_loss, label='Train')
        plt.plot(range(1, len(self.test_loss) + 1), self.test_loss, label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Model Loss')
        plt.legend()
        folder_path = os.path.join('.', 'Plots')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plt.savefig(os.path.join(folder_path, f"loss_{datetime.now().strftime('%H_%M_%S')}"))
        if show:
            plt.show()


    def plot_accuracy(self, show=False):
        plt.plot(self.train_accuracy, label='Train')
        plt.plot(self.test_accuracy, label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy')
        plt.ylim(0, 110)
        plt.legend()
        folder_path = os.path.join('.', 'Plots')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plt.savefig(os.path.join(folder_path, f"accuracy_{datetime.now().strftime('%H_%M_%S')}"))
        if show:
            plt.show()

    def save_model(self, file_path="model.pth"):
        if not os.path.isdir('Checkpoints'):
            os.mkdir('Checkpoints')
        path = os.path.join('Checkpoints', file_path)
        torch.save(self.model.state_dict(), path)
        print(f"Model parameters saved to {path}")