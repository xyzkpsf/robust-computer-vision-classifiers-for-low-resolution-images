import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
import pandas as pd
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from PIL import Image

import pathlib
import time
import copy
import os
import sys

from art.attacks.evasion import FastGradientMethod, DeepFool
from art.estimators.classification import PyTorchClassifier
from art.defences.preprocessor import FeatureSqueezing
from art.defences.preprocessor import SpatialSmoothing
from art.defences.preprocessor import ThermometerEncoding

# Uncomment the following line if needed
#CLASSES = {0: 'n01443537', 1: 'n01629819', 2: 'n01641577', 3: 'n01644900', 4: 'n01698640', 5: 'n01742172', 6: 'n01768244', 7: 'n01770393', 8: 'n01774384', 9: 'n01774750', 10: 'n01784675', 11: 'n01855672', 12: 'n01882714', 13: 'n01910747', 14: 'n01917289', 15: 'n01944390', 16: 'n01945685', 17: 'n01950731', 18: 'n01983481', 19: 'n01984695', 20: 'n02002724', 21: 'n02056570', 22: 'n02058221', 23: 'n02074367', 24: 'n02085620', 25: 'n02094433', 26: 'n02099601', 27: 'n02099712', 28: 'n02106662', 29: 'n02113799', 30: 'n02123045', 31: 'n02123394', 32: 'n02124075', 33: 'n02125311', 34: 'n02129165', 35: 'n02132136', 36: 'n02165456', 37: 'n02190166', 38: 'n02206856', 39: 'n02226429', 40: 'n02231487', 41: 'n02233338', 42: 'n02236044', 43: 'n02268443', 44: 'n02279972', 45: 'n02281406', 46: 'n02321529', 47: 'n02364673', 48: 'n02395406', 49: 'n02403003', 50: 'n02410509', 51: 'n02415577', 52: 'n02423022', 53: 'n02437312', 54: 'n02480495', 55: 'n02481823', 56: 'n02486410', 57: 'n02504458', 58: 'n02509815', 59: 'n02666196', 60: 'n02669723', 61: 'n02699494', 62: 'n02730930', 63: 'n02769748', 64: 'n02788148', 65: 'n02791270', 66: 'n02793495', 67: 'n02795169', 68: 'n02802426', 69: 'n02808440', 70: 'n02814533', 71: 'n02814860', 72: 'n02815834', 73: 'n02823428', 74: 'n02837789', 75: 'n02841315', 76: 'n02843684', 77: 'n02883205', 78: 'n02892201', 79: 'n02906734', 80: 'n02909870', 81: 'n02917067', 82: 'n02927161', 83: 'n02948072', 84: 'n02950826', 85: 'n02963159', 86: 'n02977058', 87: 'n02988304', 88: 'n02999410', 89: 'n03014705', 90: 'n03026506', 91: 'n03042490', 92: 'n03085013', 93: 'n03089624', 94: 'n03100240', 95: 'n03126707', 96: 'n03160309', 97: 'n03179701', 98: 'n03201208', 99: 'n03250847', 100: 'n03255030', 101: 'n03355925', 102: 'n03388043', 103: 'n03393912', 104: 'n03400231', 105: 'n03404251', 106: 'n03424325', 107: 'n03444034', 108: 'n03447447', 109: 'n03544143', 110: 'n03584254', 111: 'n03599486', 112: 'n03617480', 113: 'n03637318', 114: 'n03649909', 115: 'n03662601', 116: 'n03670208', 117: 'n03706229', 118: 'n03733131', 119: 'n03763968', 120: 'n03770439', 121: 'n03796401', 122: 'n03804744', 123: 'n03814639', 124: 'n03837869', 125: 'n03838899', 126: 'n03854065', 127: 'n03891332', 128: 'n03902125', 129: 'n03930313', 130: 'n03937543', 131: 'n03970156', 132: 'n03976657', 133: 'n03977966', 134: 'n03980874', 135: 'n03983396', 136: 'n03992509', 137: 'n04008634', 138: 'n04023962', 139: 'n04067472', 140: 'n04070727', 141: 'n04074963', 142: 'n04099969', 143: 'n04118538', 144: 'n04133789', 145: 'n04146614', 146: 'n04149813', 147: 'n04179913', 148: 'n04251144', 149: 'n04254777', 150: 'n04259630', 151: 'n04265275', 152: 'n04275548', 153: 'n04285008', 154: 'n04311004', 155: 'n04328186', 156: 'n04356056', 157: 'n04366367', 158: 'n04371430', 159: 'n04376876', 160: 'n04398044', 161: 'n04399382', 162: 'n04417672', 163: 'n04456115', 164: 'n04465501', 165: 'n04486054', 166: 'n04487081', 167: 'n04501370', 168: 'n04507155', 169: 'n04532106', 170: 'n04532670', 171: 'n04540053', 172: 'n04560804', 173: 'n04562935', 174: 'n04596742', 175: 'n04597913', 176: 'n06596364', 177: 'n07579787', 178: 'n07583066', 179: 'n07614500', 180: 'n07615774', 181: 'n07695742', 182: 'n07711569', 183: 'n07715103', 184: 'n07720875', 185: 'n07734744', 186: 'n07747607', 187: 'n07749582', 188: 'n07753592', 189: 'n07768694', 190: 'n07871810', 191: 'n07873807', 192: 'n07875152', 193: 'n07920052', 194: 'n09193705', 195: 'n09246464', 196: 'n09256479', 197: 'n09332890', 198: 'n09428293', 199: 'n12267677'}
data_dir = pathlib.Path('./data/tiny-imagenet-200/train/')
CLASSES = sorted([item.name for item in data_dir.glob('*')])
model1_path = './model1.pt'
model2_path = './model2.pt' 
model3_path = './model3.pt'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MyEnsemble(nn.Module):
    def __init__(self, model1, model2, model3):
          super(MyEnsemble,self).__init__()
          self.model1 = model1 
          self.model2 = model2 
          self.model3 = model3  
          self.softmax = nn.Softmax(dim=1)
          
          self.ss = SpatialSmoothing()

    def forward(self, x):
        inputs = torch.from_numpy(self.ss(x.clone().cpu())[0]).to(device)
        
        x1 = self.model1(inputs)
        x1 = self.softmax(x1)

        x2 = self.model2(x.clone())
        x2 = self.softmax(x2)

        x3 = self.model3(x)
        x3 = self.softmax(x3)
        
        x = (0.3 * x1 + 0.2 * x2 + 0.5 * x3) / 3

        return x

def main():
    model1 = torch.load(model1_path)
    model2 = torch.load(model2_path)
    model3 = torch.load(model3_path)
    model = MyEnsemble(model1, model2, model3)
    model = model.to(device)
    model.eval()
    
    data_transforms = transforms.Compose([
		transforms.ToTensor(),
	])
    with open('eval_classified.csv', 'w') as eval_output_file:
        for line in pathlib.Path(sys.argv[1]).open(): 
            image_id, image_path, image_height, image_width, image_channels = line.strip().split(
                ',') 

            print(image_id, image_path, image_height, image_width, image_channels)
            with open(image_path, 'rb') as f:
                img = Image.open(f).convert('RGB')
            img = data_transforms(img)[None, :]
            img = img.to(device)
            
            outputs = model(img)
            _, predicted = outputs.max(1)
            #print(predicted, predicted.item())

            eval_output_file.write('{},{}\n'.format(image_id, CLASSES[predicted.item()]))

if __name__ == '__main__':
	main()