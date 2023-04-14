import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from Utils.scrapedDataset import scrapedDataset
import torchvision.transforms as transforms

# Get root folder absolute path
abs_path = os.path.abspath('.')
root_dir = '/'.join(abs_path.split('/')[:-1])

class Trainer:
    def __init__(self, model, transform, batch_size=32, shuffle=False, learning_rate=0.001, decay=0.0001):
         # Define the dataset and data loaders
        self.model = model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.decay = decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") # type: ignore
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.decay)
        self.train_loss = []
        self.train_accuracy = []
        self.test_loss = []
        self.test_accuracy = []

        # Create the dataset and set model transform
        dataset = scrapedDataset(transform=transform)

        # Get classes and initialize arrays to store accuracy information
        self.classes = dataset.get_classes()
        self.class_test_correct = {i: 0 for i in range(len(self.classes))}
        self.class_test_total = {i: 0 for i in range(len(self.classes))}
        self.class_train_correct = {i: 0 for i in range(len(self.classes))}
        self.class_train_total = {i: 0 for i in range(len(self.classes))}
        self.class_train_acc = {i: [] for i in range(len(self.classes))}
        self.class_test_acc = {i: [] for i in range(len(self.classes))}

        # Determine the sizes of the train and test sets
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size

        # Split the dataset into train and test sets
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        self.train_data = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=shuffle, drop_last=True)
        self.test_data = DataLoader(test_dataset, batch_size=self.batch_size, drop_last=True)


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

                # Calculate class train correct/total
                for i in range(self.batch_size):
                    label = labels[i]
                    self.class_train_correct[int(label)] += int(predicted[i] == label)
                    self.class_train_total[int(label)] += 1

            # Update class train accuracy
            for j in range(len(self.classes)):
                class_acc = self.class_train_correct[j] / self.class_train_total[j] * 100
                self.class_train_acc[j].append(class_acc)

            train_loss = running_loss / len(self.train_data)
            train_accuracy = 100 * correct / total
            self.train_loss.append(train_loss)
            self.train_accuracy.append(train_accuracy)

            test_loss, test_accuracy = self.test()
            self.test_loss.append(test_loss)
            self.test_accuracy.append(test_accuracy)


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

                # Calculate class test correct/total
                for i in range(self.batch_size):
                    label = labels[i]
                    self.class_test_correct[int(label)] += int(predicted[i] == label)
                    self.class_test_total[int(label)] += 1

            # Update class test accuracy
            for i in range(len(self.classes)):
                class_acc = self.class_test_correct[i] / self.class_test_total[i] * 100
                self.class_test_acc[i].append(class_acc)

        test_loss /= len(self.test_data)
        test_accuracy = 100 * correct / total
        return test_loss, test_accuracy


    def plot_loss(self, show=False):
        plt.close('all')
        plt.plot(range(1, len(self.train_loss) + 1), self.train_loss, label='Train')
        plt.plot(range(1, len(self.test_loss) + 1), self.test_loss, label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Model Loss')
        plt.legend()

        # Save to folder
        folder_path = os.path.join(root_dir, 'Plots')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plt.savefig(os.path.join(folder_path, f"{datetime.now().strftime('%y_%m_%d-%H_%M_%S')}_loss"))
        if show:
            plt.show()


    def plot_accuracy(self, show=False):
        plt.close('all')
        plt.plot(self.train_accuracy, label='Train')
        plt.plot(self.test_accuracy, label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy')
        plt.ylim(0, 110)
        plt.legend()

        # Save to folder
        folder_path = os.path.join(root_dir, 'Plots')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plt.savefig(os.path.join(folder_path, f"{datetime.now().strftime('%y_%m_%d-%H_%M_%S')}_accuracy"))
        if show:
            plt.show()

    def plot_class_acc_bar(self, show=False):
        plt.close('all')
        # Extract class accuracy values
        train_acc = [self.class_train_correct[i] / self.class_train_total[i] * 100 for i in range(len(self.classes))]
        test_acc = [self.class_test_correct[i] / self.class_test_total[i] * 100 for i in range(len(self.classes))]

        # Set up bar plot data
        bar_width = 0.35
        r1 = np.arange(len(self.classes))
        r2 = [x + bar_width for x in r1]

        # Create bar plot
        plt.bar(r1, train_acc, color='blue', width=bar_width, label='Train Accuracy')
        plt.bar(r2, test_acc, color='orange', width=bar_width, label='Test Accuracy')
        plt.xticks([r + bar_width / 2 for r in range(len(self.classes))], self.classes)
        plt.ylabel('Accuracy (%)')
        plt.legend()

        # Add accuracy values above bars
        for i, (train, test) in enumerate(zip(train_acc, test_acc)):
            plt.text(r1[i] - bar_width/2, train+1, f"{train:.2f}%", color='black', fontweight='bold')
            plt.text(r2[i] - bar_width/2, test+1, f"{test:.2f}%", color='black', fontweight='bold')

        # Save to folder
        folder_path = os.path.join(root_dir, 'Plots')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plt.savefig(os.path.join(folder_path, f"{datetime.now().strftime('%y_%m_%d-%H_%M_%S')}_class_accuracy_bar"))
        if show:
            plt.show()

    def plot_class_acc_line(self, show=False):
        plt.close('all')
        # Create a figure with a subplot for each class
        fig, axs = plt.subplots(len(self.classes), figsize=(6, 10), sharex=True, sharey=True)

        # Iterate over the classes
        for i, class_name in enumerate(self.classes):
            # Plot the class train and test accuracy on the subplot
            axs[i].plot(self.class_train_acc[i], label='train')
            axs[i].plot(self.class_test_acc[i], label='test')
            axs[i].set_title(class_name)
            axs[i].set_xlabel('Epoch')
            axs[i].set_ylabel('Accuracy (%)')
            axs[i].set_ylim(0, 100)
            axs[i].legend()

         # Add a title for the whole figure
        fig.suptitle("Class-wise Train/Test Accuracy", fontsize=16)

        plt.tight_layout()
        
        # Save to folder
        folder_path = os.path.join(root_dir, 'Plots')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plt.savefig(os.path.join(folder_path, f"{datetime.now().strftime('%y_%m_%d-%H_%M_%S')}_class_accuracy_line"))
        if show:
            plt.show()

    def save_model(self, file_path="model.pth"):
        check_path = os.path.join(root_dir, 'Checkpoints')
        if not os.path.isdir(check_path):
            os.makedirs(check_path)
        path = os.path.join(check_path, file_path)
        torch.save(self.model.state_dict(), path)
        print(f"Model parameters saved to {path}")