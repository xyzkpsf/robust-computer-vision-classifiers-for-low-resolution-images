import torch
import torch.nn as nn
import numpy as np
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
import torch.optim as optim

model = torch.load(
    '/content/drive/MyDrive/Project/Models/step_20_Epoch20-40_72_baseline.pt')
model = model.to(device)


def validation(model, dataloaders, topk=[1]):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    correct = np.zeros((len(topk)))
    maxk = max(topk)
    total = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            batch_size = inputs.size(0)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            #_, preds = torch.max(outputs, 1)
            _, pred = outputs.topk(maxk, 1, True, True)
            pred = pred.t()

            correct_batch = pred.eq(labels.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct_batch[:k].reshape(
                    -1).float().sum(0, keepdim=True).item()
                res.append(correct_k)
            res = np.array(res)
            total += labels.size(0)
            correct += res

            if i % 10 == 0:
                print('\rIteration: {}/{}'.format(i+1,
                                                  len(dataloaders['val'])), end='. ')
                for j in range(len(topk)):
                    print(
                        f"Top {topk[j]} val acc: {(correct[j]/total):.4f}", end='. ')
    print("\n\nTotal Sample numbers: {}".format(total))
    for j in range(len(topk)):
        print(f"Top {topk[j]} val acc: {(correct[j]/total):.4f}")


def adversarial_validation(model, dataloaders, topk=[1]):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    correct = np.zeros((len(topk)))
    maxk = max(topk)
    total = 0

    classifier = PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=optim.SGD(model.parameters(), lr=0.001, momentum=0.9),
        input_shape=(3, 64, 64),
        nb_classes=200
    )
    attack = FastGradientMethod(estimator=classifier, eps=0.02)

    for i, (inputs, labels) in enumerate(dataloaders['val']):
        batch_size = inputs.size(0)

        inputs = torch.from_numpy(attack.generate(x=inputs.numpy()))
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)

        #_, preds = torch.max(outputs, 1)
        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()

        correct_batch = pred.eq(labels.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct_batch[:k].reshape(
                -1).float().sum(0, keepdim=True).item()
            res.append(correct_k)
        res = np.array(res)
        total += labels.size(0)
        correct += res

        if i % 10 == 0:
            print('\rIteration: {}/{}'.format(i+1,
                                              len(dataloaders['val'])), end='.')
            for j in range(len(topk)):
                print(
                    f"Top {topk[j]} val acc: {(correct[j]/total):.4f}", end=' ')
    print("\n\nTotal adversarial Attacked Sample numbers: {}".format(total))
    for j in range(len(topk)):
        print(f"Top {topk[j]} val acc: {(correct[j]/total):.4f}")
