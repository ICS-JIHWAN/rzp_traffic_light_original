import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score

# Basic Conv block [conv2d + batch normalization + relu]
class ConvBlock(nn.Module):
    """ Basic Convolution layer with Conv2d, Batch normalization and Activation function """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=True):
        super(ConvBlock, self).__init__()
        if activation:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return self.conv(x)

# Residual Block of ResNet [basic block or bottleneck block]
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, bottle=False):
        super(ResidualBlock, self).__init__()
        self.relu = nn.ReLU()
        self.downsample = downsample

        if not bottle:
            self.residual = nn.Sequential(
                ConvBlock(in_channels, out_channels, kernel_size=3, stride=stride, padding=1), # down sampling
                ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=False)
            )
        else:
            self.residual = nn.Sequential(
                ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                ConvBlock(out_channels, out_channels, kernel_size=3, stride=stride, padding=1), # down sampling
                ConvBlock(out_channels, out_channels * 4, kernel_size=1, stride=1, padding=0, activation=False)
            )

    def forward(self, x):
        # When identity mapping, after identity mapping apply the relu.
        fx = x
        if self.downsample is not None:
            fx = self.downsample(x)
        out = fx + self.residual(x)
        out = self.relu(out)
        return out

# The feature pyramid model consist of resnet and pyramid architecture
class FPN(nn.Module):
    """ This class define the backbone network. """
    def __init__(self, layers, bottle=False):
        super(FPN, self).__init__()
        self.inplanes = 64
        self.conv1 = ConvBlock(3, 64, 7, 2, 3)
        self.max_pool1 = nn.MaxPool2d(3, 2, 1)

        if not bottle: # ResNet18, ResNet34
            self.expansion = 1
            self.layer2 = self._make_residual_block(64, layers[0])
            self.layer3 = self._make_residual_block(128, layers[1], stride=2)
            self.layer4 = self._make_residual_block(256, layers[2], stride=2)
            self.layer5 = self._make_residual_block(512, layers[3], stride=2)
        else:          # ResNet50, ResNet101, ResNet152
            self.expansion = 4
            self.layer2 = self._make_residual_block(64, layers[0], bottle=bottle)
            self.layer3 = self._make_residual_block(128, layers[1], stride=2, bottle=bottle)
            self.layer4 = self._make_residual_block(256, layers[2], stride=2, bottle=bottle)
            self.layer5 = self._make_residual_block(512, layers[3], stride=2, bottle=bottle)

        fpn_sizes = [self.layer3[layers[1]-1].residual[-1].conv[0].out_channels,
                     self.layer4[layers[2]-1].residual[-1].conv[0].out_channels,
                     self.layer5[layers[3]-1].residual[-1].conv[0].out_channels]

        self.fpn = PyramidArchitecture(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)

        c2 = self.layer2(x)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)

        features = self.fpn([c3, c4, c5])

        return features

    def _make_residual_block(self, planes, num_block, stride=1, bottle=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * self.expansion: # If you want downs sampling...
            downsample = nn.Sequential(
                ConvBlock(self.inplanes, planes * self.expansion,
                          kernel_size=1, stride=stride, activation=False)
            )

        blocks = [ResidualBlock(self.inplanes, planes, stride, downsample, bottle)]
        self.inplanes = planes * self.expansion
        for i in range(1, num_block):
            blocks.append(ResidualBlock(self.inplanes, planes, bottle=bottle))

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

# The pyramid architecture extracts feature maps
class PyramidArchitecture(nn.Module):
    # This class define the feature pyramid architecture.
    #                       |-(conv3x3)->   P6_x  ->(conv3x3)+(relu)->  P7_x  _
    #   layer5 C5     -    _|-(conv1x1)->   P5_x    -      (conv3x3)->  P5_x   | feature size=256 (dimension of output)
    #   layer4 C4    ---      (conv1x1)->   P4_x   ---     (conv3x3)->  P4_x   | Output Feature {P3, P4, P5, P6, P7}
    #   layer3 C3   -----     (conv1x1)->   P3_x  -----    (conv3x3)->  P3_x  _|
    #   layer2 C2  -------
    #            [backbone][lateral connection]   [fpn]

    def __init__(self, C3_sizes, C4_sizes, C5_sizes, feature_size=256):
        """ Among the contents of the FPN paper...
        'The anchors to have areas of {32, 64, 128, 256, 512}(each elements)^2 pixels on {P2, P3, P4, P5, P6} respectively'
        """
        super(PyramidArchitecture, self).__init__()
        self.P5_1 = nn.Conv2d(C5_sizes, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P4_1 = nn.Conv2d(C4_sizes, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P3_1 = nn.Conv2d(C3_sizes, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # P6 is obtained via a 3x3 stride-2 conv on C5
        self.P6_1 = nn.Conv2d(C5_sizes, feature_size, kernel_size=3, stride=2, padding=1)

        # P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6
        self.P7_1 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P4_x + P5_upsampled_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P5_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P5_2(P3_x)

        P6_x = self.P6_1(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.relu(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]

# The classification model classifiers the imagenet dataset of 1000 classes
class Classification(nn.Module):
    def __init__(self, num_classes=1000):
        #   P7_x(c:256) -> (conv3x3) -> P7_y(c:n_c) -> (bn+relu) -
        #   P6_x(c:256) -> (conv3x3) -> P6_y(c:n_c) -> (bn+relu)  |
        #   P5_x(c:256) -> (conv3x3) -> P5_y(c:n_c) -> (bn+relu)  |-> sum(GAP) -> (FCL)
        #   P4_x(c:256) -> (conv3x3) -> P4_y(c:n_c) -> (bn+relu)  |
        #   P3_x(c:256) -> (conv3x3) -> P3_y(c:n_c) -> (bn+relu) _|
        super(Classification, self).__init__()
        self.conv_final7 = nn.Conv2d(256, num_classes, kernel_size=3, stride=1, padding=0)
        self.conv_final6 = nn.Conv2d(256, num_classes, kernel_size=3, stride=1, padding=0)
        self.conv_final5 = nn.Conv2d(256, num_classes, kernel_size=3, stride=1, padding=0)
        self.conv_final4 = nn.Conv2d(256, num_classes, kernel_size=3, stride=1, padding=0)
        self.conv_final3 = nn.Conv2d(256, num_classes, kernel_size=3, stride=1, padding=0)

        self.bn = nn.BatchNorm2d(num_classes)
        self.relu = nn.ReLU()

        self.GAP = nn.AdaptiveAvgPool2d(1)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(num_classes, num_classes)

    def forward(self, features):
        P7_y = self.conv_final7(features[4])
        P7_y = self.relu(self.bn(P7_y))
        P7_y = self.GAP(P7_y)

        P6_y = self.conv_final6(features[3])
        P6_y = self.relu(self.bn(P6_y))
        P6_y = self.GAP(P6_y)

        P5_y = self.conv_final5(features[2])
        P5_y = self.relu(self.bn(P5_y))
        P5_y = self.GAP(P5_y)

        P4_y = self.conv_final4(features[1])
        P4_y = self.relu(self.bn(P4_y))
        P4_y = self.GAP(P4_y)

        P3_y = self.conv_final3(features[0])
        P3_y = self.relu(self.bn(P3_y))
        P3_y = self.GAP(P3_y)

        P_y = P7_y + P6_y + P5_y + P4_y + P3_y
        P_y = self.flatten(P_y)
        classifier_output = self.linear(P_y)

        return classifier_output

    def cal_accuracy(self, prediction, target):
        prediction_classes = nn.Softmax(dim=1)(prediction).argmax(dim=1)
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
    from pytorch_model_summary import summary
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    def summary_model(model, input):
        print(summary(model, input, show_input=True))

    layer_dict = {
        'resnet18': [2, 2, 2, 2],
        'resnet34': [3, 4, 6, 3],
        'resnet50': [3, 4, 6, 3],
        'resnet101': [3, 4, 23, 3],
        'resnet152': [3, 8, 36, 3]
    }

    import time

    backbone = FPN(layers=layer_dict['resnet18'], bottle=False)
    input = torch.rand([2, 3, 416, 416])

    print(count_parameters(backbone))
    print(summary_model(backbone, input))

    device = torch.device('cpu')
    backbone.to(device).eval()
    total_time = 0
    for i in range(3):
        start = time.time()
        output = backbone(input.to(device))
        end = time.time()
        total_time += end - start
    print(f'{total_time/3: .5f} sec')


