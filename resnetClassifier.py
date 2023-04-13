import torch
from PIL import Image
import torch.nn as nn
import torchvision.models as models
from Utils.scrapedDataset import scrapedDataset

class ResNetClassifier(nn.Module):
    def __init__(self, transform):
        super(ResNetClassifier, self).__init__()
        # Initialize transforms
        self.transform = transform
        
        # Store output labels and number of classes
        self.labels = scrapedDataset(get_class=True).get_labels()
        self.num_classes = len(self.labels)

        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x
    
    def predict_img(self, image_path: str):
        if self.params is not None:
            # Load and preprocess the image
            image = Image.open(image_path)
            input_tensor = self.transform(image)
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