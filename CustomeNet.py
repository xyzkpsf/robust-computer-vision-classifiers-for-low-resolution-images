import torch
import torch.nn as nn

# A Custome Wide and shallow net to fit the chracteristics of the low resolution data.


class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # for layer 0
        self.conv = nn.Conv2d(3, 32, kernel_size=3, stride=1,
                              padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

        # for layer 1
        self.conv_1 = nn.Conv2d(32, 128, kernel_size=3, stride=1,
                                padding=1, bias=False)
        # for layer 2 - 4
        self.conv_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1,
                                padding=1, bias=False)

        # for layer 6
        self.conv_3 = nn.Conv2d(160, 256, kernel_size=3, stride=1,
                                padding=1, bias=False)
        # for layer 7 - 9
        self.conv_4 = nn.Conv2d(256, 256, kernel_size=3, stride=1,
                                padding=1, bias=False)
        # for layer 11
        self.conv_5 = nn.Conv2d(416, 512, kernel_size=3, stride=1,
                                padding=1, bias=False)
        # for layer 12 - 14
        self.conv_6 = nn.Conv2d(512, 512, kernel_size=3, stride=1,
                                padding=1, bias=False)

        # for layer 16
        self.conv_7 = nn.Conv2d(928, 200, kernel_size=1, stride=1,
                                padding=1, bias=False)

        self.bn_32 = nn.BatchNorm2d(32)
        self.bn_128 = nn.BatchNorm2d(128)
        self.bn_160 = nn.BatchNorm2d(160)
        self.bn_256 = nn.BatchNorm2d(256)
        self.bn_416 = nn.BatchNorm2d(416)
        self.bn_512 = nn.BatchNorm2d(512)
        self.bn_928 = nn.BatchNorm2d(928)

        self.max_pool = nn.MaxPool2d(2, stride=2)

        self.avg_pool = nn.AvgPool2d(8)

    def forward(self, x):
        # block 1, layer 0
        layer0 = self.conv(x)
        layer0 = self.bn_32(layer0)
        layer0 = self.relu(layer0)
        skip_connection_1 = layer0

        # block 2, layer 1 - 5
        # layer 1
        layer1 = self.conv_1(layer0)
        layer1 = self.bn_128(layer1)
        layer1 = self.relu(layer1)
        # layer 2 - 4
        out = layer1
        for _ in range(3):
            out = self.conv_2(out)
            out = self.bn_128(out)
            out = self.relu(out)
        layer4 = out
        # layer 5
        layer5 = torch.cat((skip_connection_1, layer4), 1)
        layer5 = self.bn_160(layer5)
        layer5 = self.relu(layer5)
        layer5 = self.max_pool(layer5)
        skip_connection_2 = layer5

        # block 3, layer 6 - 10
        # layer 6
        layer6 = self.conv_3(layer5)
        layer6 = self.bn_256(layer6)
        layer6 = self.relu(layer6)
        # layer 7 - 9
        out = layer6
        for _ in range(3):
            out = self.conv_4(out)
            out = self.bn_256(out)
            out = self.relu(out)
        layer9 = out
        # layer 10
        layer10 = torch.cat((skip_connection_2, layer9), 1)
        layer10 = self.bn_416(layer10)
        layer10 = self.relu(layer10)
        layer10 = self.max_pool(layer10)
        skip_connection_3 = layer10

        # block 4, layer 11 - 15
        # layer 11
        layer11 = self.conv_5(layer10)
        layer11 = self.bn_512(layer11)
        layer11 = self.relu(layer11)
        # layer 12 - 14
        out = layer11
        for _ in range(3):
            out = self.conv_6(out)
            out = self.bn_512(out)
            out = self.relu(out)
        layer14 = out
        # layer 15
        layer15 = torch.cat((skip_connection_3, layer14), 1)
        layer15 = self.bn_928(layer15)
        layer15 = self.relu(layer15)
        layer15 = self.max_pool(layer15)

        # layer 16 and output
        layer16 = self.conv_7(layer15)
        out = self.avg_pool(layer16)
        out = out.view(out.size(0), -1)
        return out
