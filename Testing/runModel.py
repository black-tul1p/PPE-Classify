import os, sys

# Adding root folder to the system path
abs_path = os.path.abspath('.')
root_dir = '/'.join(abs_path.split('/')[:-1])
sys.path.insert(0, root_dir)

# Model imports
# from simpleClassifier import NNet
from complexClassifier import NNet
from resnetClassifier import ResNetClassifier
from vggClassifier import VGGClassifier
from Utils.train import Trainer
import torchvision.transforms as transforms

# Globals
predict = False
train = True
# model_name = 'simple.pth'  # simpleCLassifier
model_name = 'vggnet.pth' # complexClassifier
model_path = os.path.join('..', 'Checkpoints', model_name)
image_path = 'test.jpg'


# Set transform
transform = transforms.Compose(
    [transforms.Resize((224, 244)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

# Training model
if train:
    # Define the hyperparameters
    epochs = 100
    batch_size = 16
    learning_rate = 0.001
    decay = 0.000

    # Initialize model and define device
    # model = ResNetClassifier()
    model = NNet(transform=transform)

    # Initialize complex trainer
    trainer = Trainer(model=model,
                      transform=transform,
                      batch_size=batch_size, 
                      shuffle=True, 
                      learning_rate=learning_rate, 
                      decay=decay)
    trainer.train(epochs)
    trainer.plot_loss()
    trainer.plot_accuracy()
    trainer.plot_class_acc_bar()
    trainer.plot_class_acc_line()

    # Save model
    bool = input("Save model? [Y/N]: ").lower() == 'y'
    if (bool):
        trainer.save_model(model_name)
        print("Model saved.")

# Image classification
if predict:
    model = NNet(transform=transform, params=model_path)
    label = model.predict_img(image_path=image_path)
    print(f"This image contains a {label}")