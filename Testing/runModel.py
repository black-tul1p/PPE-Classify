import os, sys

# Adding root folder to the system path
abs_path = os.path.abspath('.')
root_dir = '/'.join(abs_path.split('/')[:-1])
sys.path.insert(0, root_dir)

# Model imports
# from simpleClassifier import NNet
from complexClassifier import NNet
from resnetClassifier import ResNetClassifier
from Utils.train import Trainer

# Globals
predict = True
train = False
# model_name = 'simple.pth'  # simpleCLassifier
model_name = 'complex.pth' # complexClassifier
model_path = os.path.join('..', 'Checkpoints', model_name)
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

    # Initialize complex trainer
    trainer = Trainer(model, batch_size=batch_size, shuffle=True, learning_rate=learning_rate)
    trainer.train(epochs)
    trainer.plot_loss()
    trainer.plot_accuracy()
    trainer.save_model(model_name)

# Image classification
if predict:
    model = NNet(params=model_path)
    label = model.predict_img(image_path=image_path)
    print(f"This image contains a {label}")