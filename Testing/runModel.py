import os, sys

# Adding root folder to the system path
abs_path = os.path.abspath('.')
root_dir = '/'.join(abs_path.split('/')[:-1])
sys.path.insert(0, root_dir)

# Model imports
from complexClassifier import NNet
from resnetClassifier import ResNetClassifier
from Utils.train import Trainer

# Globals
predict = False
train = True
model_path = os.path.join('..', 'Checkpoints', 'complex2.pth')
image_path = 'test.jpg'

# Training model
if train:
    # Define the hyperparameters
    epochs = 50
    batch_size = 32
    learning_rate = 0.001

    # Initialize model and define device
    # model = ResNetClassifier(num_classes=3)
    model = NNet()
    # model.load_model(model_path='model.pth')

    # Initialize complex trainer
    trainer = Trainer(model, batch_size=batch_size, shuffle=True, learning_rate=learning_rate)
    trainer.train(epochs)
    trainer.plot_loss()
    trainer.plot_accuracy()
    # trainer.save_model("complex.pth")

# Image classification
if predict:
    model = NNet(params=model_path)
    label = model.predict_img(image_path=image_path)
    print(f"This image contains a {label}")