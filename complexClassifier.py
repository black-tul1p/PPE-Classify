import torch
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms

class NNet(nn.Module):
    def __init__(self, params=None) -> None:
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

        # Handle saved model parameter loading
        self.params = params
        if self.params is not None:
            self.load_model()

        # Store output labels
        self.labels = {
            0: 'face_shield',
            1: 'gloves',
            2: 'goggles',
            3: 'lab_coat',
            4: 'scrub_cap'
        }

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
    
    def predict_img(self, image_path: str):
        if self.params is not None:
            # Load and preprocess the image
            image = Image.open(image_path)
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            input_tensor = transform(image)
            input_batch = input_tensor.unsqueeze(0) # type: ignore

            # Pass the image through the model
            with torch.no_grad():
                output = self(input_batch)

            # Get the predicted label
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            predicted_idx = torch.argmax(probabilities).item()

            return self.labels[int(predicted_idx)]
        else: return "No parameters supplied"

    def load_model(self):
        if isinstance(self.params, str):
            self.load_state_dict(torch.load(self.params))
            self.eval()
        else:
            print("Incorrect path or format.")