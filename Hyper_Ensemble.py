import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import BasicIterativeMethod
from art.defences.preprocessor import FeatureSqueezing
from art.defences.preprocessor import SpatialSmoothing

# Load Modela
model1 = torch.load('/content/drive/MyDrive/Project/Models/resnet_101_ss.pt')
model1 = model1.to(device)

model2 = torch.load('/content/drive/MyDrive/Project/Models/resnet_101.pt')
model2 = model2.to(device)

model3 = torch.load(
    '/content/drive/MyDrive/Project/Models/step_20_Epoch20-40_72_baseline.pt')
model3 = model3.to(device)


class MyEnsemble(nn.Module):
    # w1, w2, w3 are Weights of models when predict
    def __init__(self, model1, model2, model3, w1, w2, w3):
        super(MyEnsemble, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.softmax = nn.Softmax(dim=1)
        self.ss = SpatialSmoothing()

    def forward(self, x):
        #inputs = torch.from_numpy(self.ss(x.clone().cpu())[0]).to(device)
        inputs = torch.from_numpy(self.ss(x.clone().cpu())[0]).to(device)
        x1 = self.model1(inputs)
        x1 = self.softmax(x1)

        x2 = self.model2(x.clone())
        x2 = self.softmax(x2)

        x3 = self.model3(x)
        x3 = self.softmax(x3)

        x = (self.w1 * x1 + self.w2 * x2 + self.w3 * x3)

        return x


def hyper_ensemble(model1, model2, model3):
    best_top1 = 0
    a, b, c = 0, 0, 0
    for i in np.arange(0.2, 1, 0.1):
        for j in np.arange(0.1, 1, 0.1):
            i = round(i, 2)
            j = round(j, 2)
            if (i + j < 1):
                model = MyEnsemble(model1, model2, model3, i, j, 1-i-j)
                model = model.to(device)
                model.eval()
                print(f"w1: {i}, w2: {j}, w3: {round(1-i-j, 2)}", end=': ')
                curr_top1 = adversarial_validation(
                    model, dataloaders, [1, 5, 10])
                print(' ')
            if (curr_top1 > best_top1):
                best_top1 = curr_top1
                a, b, c = i, j, round(1-i-j, 2)
    print(f"Best hyper: w1: {a}, w2: {b}, w3: {c}, best top1: {best_top1}")

    hyper_ensemble(model1, model2, model3)
