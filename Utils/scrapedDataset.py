import os
from PIL import Image
from torch.utils.data import Dataset

# Get root folder absolute path
abs_path = os.path.abspath('.')
root_dir = '/'.join(abs_path.split('/')[:-1])

class scrapedDataset(Dataset):
    def __init__(self, root_dir=os.path.join(root_dir, 'Images'), transform=None, get_class=False):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.imgs = []
        if not get_class:
            for c in self.classes:
                class_path = os.path.join(self.root_dir, c)
                for img_file in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_file)
                    if img_file.endswith('.jpg') or img_file.endswith('.jpeg') or img_file.endswith('.png'):
                        self.imgs.append((img_path, self.class_to_idx[c]))
            print(f'Training model on classes: {self.classes}')

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_classes(self):
        return self.classes
    
    def get_labels(self):
        return {i: self.classes[i] for i in range(len(self.classes))}

    def __len__(self):
        return len(self.imgs)