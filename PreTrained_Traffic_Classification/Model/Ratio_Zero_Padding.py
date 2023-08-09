import torch
import torch.nn as nn

class RatioZeroPadding(nn.Module):
    def __init__(self, in_channels, f_size, n_class, stride=1):
        super(RatioZeroPadding, self).__init__()
        self.conv = nn.Conv2d(in_channels, n_class, f_size, stride=stride, padding=0)
        self.relu = nn.ReLU()

        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.GAP(x)
        x = self.flatten(x)

        return x

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


class RatioZeroPaddingFPN(nn.Module):
    def __init__(self, Iw, Ih, n_classes):
        super(RatioZeroPaddingFPN, self).__init__()
        self.layer1 = nn.Conv2d(3, 16, 3, stride=2, padding=0)
        self.layer2 = nn.Conv2d(16, 32, 3, stride=2, padding=0)
        self.layer3 = nn.Conv2d(32, 64, 3, stride=2, padding=0)

        filter_sizes = self.cal_filter_sizes(Iw, Ih)

        self.RZP1 = RatioZeroPadding(16, filter_sizes[0], n_classes)
        self.RZP2 = RatioZeroPadding(32, filter_sizes[1], n_classes)
        self.RZP3 = RatioZeroPadding(64, filter_sizes[2], n_classes)

        self.fl = nn.Linear(n_classes*3, n_classes)

    def forward(self, x):
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)

        out1 = self.RZP1(f1)
        out2 = self.RZP2(f2)
        out3 = self.RZP3(f3)

        out = torch.concat((out1, out2, out3), dim=1)
        out = self.fl(out)

        return out

    def cal_filter_sizes(self, w, h):
        size_f1 = self.layer1(torch.rand([16, 3, w, h]))
        size_f2 = self.layer2(size_f1)
        size_f3 = self.layer3(size_f2)

        return [min(size_f1.shape), min(size_f2.shape), min(size_f3.shape)]

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

    # original size padding
    input = torch.rand([16, 3, 20, 100])
    RZP_model = RatioZeroPadding(3, 20, 5)

    RZP_model_output = RZP_model(input)
    print(count_parameters(RZP_model))

    # Ratio zero padding using FPN model
    RZP_FPN_model = RatioZeroPaddingFPN(20, 100, 5)

    RZP_FPN_model_output = RZP_FPN_model(input)
    print(count_parameters(RZP_FPN_model))
