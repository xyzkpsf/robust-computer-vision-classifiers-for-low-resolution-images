import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import pandas as pd
import numpy as np
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
from PIL import Image
from livelossplot import PlotLosses
import time
import os
import copy
import sys


class CustomImageDataset(Dataset):
    def __init__(self, val_label_dir, img_dir, train_path, transform=None, target_transform=None):
        super(CustomImageDataset, self).__init__()
        self.val_label_file = pd.read_csv(val_label_dir, delimiter="\t", names=[
                                          "pics", "labels", "_1", "_2", "_3", "_4"])
        self.img_labels = self.val_label_file[["pics", 'labels']]
        self.img_dir = img_dir
        self.transform = transform
        self.classes, self.class_to_idx = self._find_classes(train_path)

    def __len__(self):
        return len(self.img_labels)

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(
                dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def pad(self, img):
        padding = np.ones((64, 64, 2))
        img = img.reshape((64, 64, 1))
        img = np.concatenate((img, padding), axis=2)
        return img.astype(np.float32)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        img = Image.open(img_path)
        img = copy.deepcopy(np.asarray(img))
        # if it has less than 3 channels
        if img.shape != (64, 64, 3):
            img = self.pad(img)
        # print(img.shape)
        label = self.img_labels.iloc[idx, 1]
        label = self.class_to_idx[label]
        if self.transform:
            img = self.transform(img)
        #sample = {"image": img, "label": label}
        return img, label


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(degrees=10),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(p=0.1),
        transforms.ColorJitter(
            brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
    ]),
}

data_dir = '/content/tiny-imagenet-200'

val_label_dir = '/content/tiny-imagenet-200/val/val_annotations.txt'

image_datasets = {}

image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                               data_transforms['train'])

image_datasets['val'] = CustomImageDataset(val_label_dir, data_dir+'/val/images',
                                           os.path.join(data_dir, 'train'),
                                           transform=data_transforms['val'])

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=128,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=30):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()
    liveloss = PlotLosses()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                print("\rIteration: {}/{}, Loss: {}.".format(i+1,
                                                             len(dataloaders[phase]), loss.item() * inputs.size(0)), end="")
                sys.stdout.flush()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == 'train':
                avg_loss = epoch_loss
                t_acc = epoch_acc
            else:
                val_loss = epoch_loss
                val_acc = epoch_acc

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        liveloss.update({
            'log loss': avg_loss,
            'val_log loss': val_loss,
            'accuracy': t_acc,
            'val_accuracy': val_acc
        })

        liveloss.draw()
        print('Train Loss: {:.4f} Acc: {:.4f}'.format(avg_loss, t_acc))
        print('Val Loss: {:.4f} Acc: {:.4f}'.format(val_loss, val_acc))
        print('Best Val Accuracy: {}'.format(best_acc))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


model = models.resnet50(pretrained=True)
# (N, 3, 64, 64) => (N, 64, 64, 64)  Remain the input dimension
model.conv1 = nn.Conv2d(3, 64, kernel_size=(
    3, 3), stride=(1, 1), padding=(1, 1))
# Remain size after maxpool (N, 64, 64, 64)
model.maxpool = nn.MaxPool2d(
    kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
model.fc = nn.Linear(2048, 200)

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, exp_lr_scheduler,
                    num_epochs=20)

model_save_path = "/content/drive/MyDrive/Project/Models/step_20_Epoch20-40_72_baseline.pt"
torch.save(model, model_save_path)
