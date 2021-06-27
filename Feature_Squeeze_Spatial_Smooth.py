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
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from art.defences.preprocessor import FeatureSqueezing
from art.defences.preprocessor import SpatialSmoothing


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=30):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()
    liveloss = PlotLosses()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    fs = FeatureSqueezing(clip_values=[0, 255])
    ss = SpatialSmoothing()
    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 64, 64),
        nb_classes=200,
        preprocessing_defences=[fs]
    )
    attack = FastGradientMethod(estimator=classifier, eps=0.02)
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
                # classifier = PyTorchClassifier(
                #     model=model,
                #     loss=criterion,
                #     optimizer=optimizer,
                #     input_shape=(3, 64, 64),
                #     nb_classes=200
                #     )
                # attack = FastGradientMethod(estimator=classifier, eps=0.2)
                if phase == 'val':
                    inputs = torch.from_numpy(
                        attack.generate(x=inputs.numpy()))
                    # add adversarial
                # inputs, _ = fs(inputs, labels)
                # inputs = torch.from_numpy(attack.generate(x=inputs.numpy()))
                # inputs, _ = fs(inputs, labels)
                inputs, _ = ss(inputs, labels)
                inputs = torch.from_numpy(inputs)
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
