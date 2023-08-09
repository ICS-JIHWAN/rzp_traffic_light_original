import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score

# Basic Conv block [conv2d + batch normalization + relu]
class ConvBlock(nn.Module):
    """ Basic Convolution layer with Conv2d, Batch normalization and Activation function """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# Residual Block of ResNet [basic block or bottleneck block]
class ResidualBlock(nn.Module):
    def __init__(self, channels, bottle=False):
        super(ResidualBlock, self).__init__()
        self.identity_block = ConvBlock(channels, channels, kernel_size=1, stride=1, padding=0)
        if not bottle:
            self.residual = nn.Sequential(
                ConvBlock(channels, channels, kernel_size=3, stride=1, padding=1), # down sampling
                ConvBlock(channels, channels, kernel_size=3, stride=1, padding=1)
            )
        else:
            self.residual = nn.Sequential(
                ConvBlock(channels, channels//2, kernel_size=1, stride=1, padding=0),
                ConvBlock(channels//2, channels//2, kernel_size=3, stride=1, padding=1), # down sampling
                ConvBlock(channels//2, channels, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, x):
        # When identity mapping, after identity mapping apply the relu.
        fx = self.identity_block(x)
        out = fx + self.residual(x)
        return out

# The pyramid architecture extracts feature maps
class PyramidArchitecture(nn.Module):
    # This class define the feature pyramid architecture.
    #   C5     -       (conv1x1)->   P5_x    -       _
    #   C4    ---      (conv1x1)->   P4_x   ---       | feature size=256 (dimension of output)
    #   C3   -----     (conv1x1)->   P3_x  -----      | You must use 3x3 Conv layer(s=1, padding=1),
    #   C2  -------    (conv1x1)->   P2_x -------    _| when design the desired output channel(number of classes).
    #     [backbone][lateral connection]   [fpn]

    def __init__(self, C2_sizes, C3_sizes, C4_sizes, C5_sizes, feature_size=256):
        super(PyramidArchitecture, self).__init__()
        self.P5_1 = ConvBlock(C5_sizes, feature_size, kernel_size=1)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

        self.P4_1 = ConvBlock(C4_sizes, feature_size, kernel_size=1)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

        self.P3_1 = ConvBlock(C3_sizes, feature_size, kernel_size=1)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

        self.P2_1 = ConvBlock(C2_sizes, feature_size, kernel_size=1)

    def forward(self, inputs):
        C2, C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P4_x + P5_upsampled_x
        P4_upsampled_x = self.P4_upsampled(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_upsampled_x = self.P3_upsampled(P3_x)

        P2_x = self.P2_1(C2)
        P2_x = P2_x + P3_upsampled_x

        return [P2_x, P3_x, P4_x, P5_x]

# The feature pyramid model consist of resnet and pyramid architecture
class FPN(nn.Module):
    """ This class define the backbone network. """
    def __init__(self, layers, bottle=False):
        super(FPN, self).__init__()
        self.conv1 = ConvBlock(3, 64, 7, 2, 3)
        self.max_pool1 = nn.MaxPool2d(3, 2, 1)

        if not bottle: # ResNet18, ResNet34
            self.layer2 = self._make_residual_block(64, layers[0])
            self.layer3 = self._make_residual_block(128, layers[1])
            self.layer4 = self._make_residual_block(256, layers[2])
            self.layer5 = self._make_residual_block(512, layers[3])
        else:          # ResNet50, ResNet101, ResNet152
            self.layer2 = self._make_residual_block(64, layers[0], bottle)
            self.layer3 = self._make_residual_block(128, layers[1], bottle)
            self.layer4 = self._make_residual_block(256, layers[2], bottle)
            self.layer5 = self._make_residual_block(512, layers[3], bottle)

        self.conv2 = ConvBlock(64, 128, kernel_size=1, stride=2, padding=0)
        self.conv3 = ConvBlock(128, 256, kernel_size=1, stride=2, padding=0)
        self.conv4 = ConvBlock(256, 512, kernel_size=1, stride=2, padding=0)

        fpn_sizes = [self.layer2[layers[0]-1].residual[-1].conv[0].out_channels,
                     self.layer3[layers[1]-1].residual[-1].conv[0].out_channels,
                     self.layer4[layers[2]-1].residual[-1].conv[0].out_channels,
                     self.layer5[layers[3]-1].residual[-1].conv[0].out_channels]

        self.fpn = PyramidArchitecture(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2], fpn_sizes[3])

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)

        c2_1 = self.layer2(x)
        c2_2 = self.conv2(c2_1)

        c3_1 = self.layer3(c2_2)
        c3_2 = self.conv3(c3_1)

        c4_1 = self.layer4(c3_2)
        c4_2 = self.conv4(c4_1)

        c5 = self.layer5(c4_2)

        features = self.fpn([c2_1, c3_1, c4_1, c5])

        return features

    def _make_residual_block(self, in_channels, num_block, bottle=False):
        blocks = []
        for i in range(num_block):
            blocks.append(ResidualBlock(in_channels, bottle))
        return nn.Sequential(*blocks)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        print('=> Complete initializing weights')

# The classification model classifiers the imagenet dataset of 1000 classes
class Classification(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv_final5 = ConvBlock(256, num_classes*2, kernel_size=1)
        self.conv_final4 = ConvBlock(256, num_classes*2, kernel_size=1)
        self.conv_final3 = ConvBlock(256, num_classes*2, kernel_size=1)
        self.conv_final2 = ConvBlock(256, num_classes*2, kernel_size=1)

        self.GAP = nn.AdaptiveAvgPool2d(1)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(num_classes*2, num_classes)

    def forward(self, features):
        conv_final5 = self.conv_final5(features[3])
        conv_final5 = self.GAP(conv_final5)

        conv_final4 = self.conv_final4(features[2])
        conv_final4 = self.GAP(conv_final4)

        conv_final3 = self.conv_final3(features[1])
        conv_final3 = self.GAP(conv_final3)

        conv_final2 = self.conv_final2(features[0])
        conv_final2 = self.GAP(conv_final2)

        conv_final = conv_final5 + conv_final4 + conv_final3 + conv_final2
        conv_final = self.flatten(conv_final)
        classifier_output = self.linear(conv_final)

        return classifier_output

    def cal_accuracy(self, prediction, target):
        softmax = nn.Softmax()
        prediction_classes = softmax(prediction).argmax(dim=1)
        target_classes = target.argmax(dim=1)
        return accuracy_score(target_classes.to('cpu'), prediction_classes.to('cpu'))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        print('=> Complete initializing weights')


if __name__ == '__main__':
    backbone = FPN(layers=[3, 8, 36, 3])
    classifier = Classification(num_classes=1000)

    input = torch.rand([1, 3, 224, 224])

    output = backbone(input)
    output = classifier(output)