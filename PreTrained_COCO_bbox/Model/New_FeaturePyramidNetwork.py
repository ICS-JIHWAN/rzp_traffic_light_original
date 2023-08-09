import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
import torch.nn as nn
from Model import detection_fn as fn
from Model.anchors import AnchorBox

class BoxRegression(nn.Module):
    def __init__(self, anchors, image_size: int):
        super().__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.image_size = image_size
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.ignore_thres = 0.5
        self.obj_scale = 100
        self.no_obj_scale = 1
        self.metrics = {}

    def forward(self, x, targets):
        # x: output of a network
        # targets: ground truth

        device = x.device if x.is_cuda else torch.device('cpu')

        num_batches = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_batches, self.num_anchors, 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2).contiguous()
        )  # shape: (batch, num_anchors, gird_size, grid_size, 5 + num_classes)

        # Get output
        cx = torch.sigmoid(prediction[..., 0])  # offset of x
        cy = torch.sigmoid(prediction[..., 1])  # offset of y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Object confidence (objectness)

        # Calculate offsets for each grid
        stride = self.image_size / grid_size  #
        grid_x = torch.arange(grid_size, dtype=torch.float, device=device). \
            repeat(grid_size, 1).view([1, 1, grid_size, grid_size])
        grid_y = torch.arange(grid_size, dtype=torch.float, device=device). \
            repeat(grid_size, 1).t().view([1, 1, grid_size, grid_size])
        # a_w: the width of an anchor relative to the original image
        # a_w / stride: the width of an anchor relative to the feature map 13x13, 26x26, or 52x52.
        scaled_anchors = torch.as_tensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors],
                                         dtype=torch.float, device=device)
        anchor_w = scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

        # Add offset and scale with anchors
        pred_boxes = torch.zeros_like(prediction[..., :4], device=device)  # x, y, w, h
        pred_boxes[..., 0] = grid_x + cx
        pred_boxes[..., 1] = grid_y + cy
        pred_boxes[..., 2] = torch.exp(w) * anchor_w
        pred_boxes[..., 3] = torch.exp(h) * anchor_h

        # pred_boxes.shape = (batch, num_anchors, grid_size, grid_size, 4)
        pred = (pred_boxes.view(num_batches, -1, 4) * stride,
                pred_conf.view(num_batches, -1, 1),
                )
        output = torch.cat(pred, -1)  # shape for a 13x13 feature pyramid level: [batch, 13*13*num_anchors, 5 + num_classes]

        if targets is None:
            return output, 0

        iou_scores, obj_mask, no_obj_mask, tx, ty, tw, th, tconf = fn.build_targets(
            pred_boxes=pred_boxes,
            target=targets,
            anchors=scaled_anchors,
            ignore_thres=self.ignore_thres,
            device=device
        )

        # Loss: Mask outputs to ignore non-existing objects (except with conf. loss)
        loss_x = self.mse_loss(cx[obj_mask], tx[obj_mask])
        loss_y = self.mse_loss(cy[obj_mask], ty[obj_mask])
        loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
        loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
        loss_bbox = loss_x + loss_y + loss_w + loss_h
        #
        loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
        loss_conf_no_obj = self.bce_loss(pred_conf[no_obj_mask], tconf[no_obj_mask])
        loss_conf = self.obj_scale * loss_conf_obj + self.no_obj_scale * loss_conf_no_obj
        loss_layer = loss_bbox + loss_conf

        # Metrics
        conf50 = (pred_conf > 0.5).float()
        iou50 = (iou_scores > 0.5).float()
        iou75 = (iou_scores > 0.75).float()
        detected_mask = conf50 * tconf
        conf_obj = pred_conf[obj_mask].mean()
        conf_no_obj = pred_conf[no_obj_mask].mean()
        precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
        recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
        recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

        # Write loss and metrics
        self.metrics = {
            "loss_x": loss_x.detach().cpu().item(),
            "loss_y": loss_y.detach().cpu().item(),
            "loss_w": loss_w.detach().cpu().item(),
            "loss_h": loss_h.detach().cpu().item(),
            "loss_bbox": loss_bbox.detach().cpu().item(),
            "loss_conf": loss_conf.detach().cpu().item(),
            "loss_layer": loss_layer.detach().cpu().item(),
            "conf_obj": conf_obj.detach().cpu().item(),
            "conf_no_obj": conf_no_obj.detach().cpu().item(),
            "precision": precision.detach().cpu().item(),
            "recall50": recall50.detach().cpu().item(),
            "recall75": recall75.detach().cpu().item()
        }

        return output, loss_layer


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
                ConvBlock(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),  # down sampling
                ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=False)
            )
        else:
            self.residual = nn.Sequential(
                ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                ConvBlock(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),  # down sampling
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

        if not bottle:  # ResNet18, ResNet34
            self.expansion = 1
            self.layer2 = self._make_residual_block(64, layers[0])
            self.layer3 = self._make_residual_block(128, layers[1], stride=2)
            self.layer4 = self._make_residual_block(256, layers[2], stride=2)
            self.layer5 = self._make_residual_block(512, layers[3], stride=2)
        else:  # ResNet50, ResNet101, ResNet152
            self.expansion = 4
            self.layer2 = self._make_residual_block(64, layers[0], bottle=bottle)
            self.layer3 = self._make_residual_block(128, layers[1], stride=2, bottle=bottle)
            self.layer4 = self._make_residual_block(256, layers[2], stride=2, bottle=bottle)
            self.layer5 = self._make_residual_block(512, layers[3], stride=2, bottle=bottle)

        fpn_sizes = [self.layer3[layers[1] - 1].residual[-1].conv[0].out_channels,
                     self.layer4[layers[2] - 1].residual[-1].conv[0].out_channels,
                     self.layer5[layers[3] - 1].residual[-1].conv[0].out_channels]

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
        if stride != 1 or self.inplanes != planes * self.expansion:  # If you want downs sampling...
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


class RetinaNet(nn.Module):
    def __init__(self, img_size: int):
        super(RetinaNet, self).__init__()
        self.anchors = AnchorBox().get_anchors_hr()
        self.box_head = self.build_head(9 * (4 + 1))  # anchor * (box[cx, vy, w, h], object)

        self.fpn_p3 = BoxRegression(anchors=self.anchors['P3'], image_size=img_size)
        self.fpn_p4 = BoxRegression(anchors=self.anchors['P4'], image_size=img_size)
        self.fpn_p5 = BoxRegression(anchors=self.anchors['P5'], image_size=img_size)
        self.fpn_p6 = BoxRegression(anchors=self.anchors['P6'], image_size=img_size)
        self.fpn_p7 = BoxRegression(anchors=self.anchors['P7'], image_size=img_size)

    def forward(self, features, targets):
        loss = 0
        pred = []

        layer_pred, layer_loss = self.fpn_p3(self.box_head(features[0]), targets)
        pred.append(layer_pred)
        loss += layer_loss
        #
        layer_pred, layer_loss = self.fpn_p4(self.box_head(features[1]), targets)
        pred.append(layer_pred)
        loss += layer_loss
        #
        layer_pred, layer_loss = self.fpn_p5(self.box_head(features[2]), targets)
        pred.append(layer_pred)
        loss += layer_loss
        #
        layer_pred, layer_loss = self.fpn_p6(self.box_head(features[3]), targets)
        pred.append(layer_pred)
        loss += layer_loss
        #
        layer_pred, layer_loss = self.fpn_p7(self.box_head(features[4]), targets)
        pred.append(layer_pred)
        loss += layer_loss

        return torch.cat(pred, dim=1), loss

    def build_head(self, out_channel):
        head = []
        for _ in range(4):
            head.append(ConvBlock(256, 256, 3, stride=1, padding=1, activation=True))
        head.append(nn.Conv2d(256, out_channel, 3, stride=1, padding=1, bias=False))

        return nn.Sequential(*head)

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

    backbone = FPN(layers=layer_dict['resnet34'])
    retina = RetinaNet(img_size=416)
    input = torch.rand([3, 3, 416, 416])
    targets = torch.rand([10, 6])

    # print(count_parameters(backbone))
    # print(summary_model(backbone, input))
    output = backbone(input)
    output = retina(output, targets)

# def forward(self, features):
#     box_outputs = dict()
#     for i, feature in enumerate(features):
#         # box_outputs.append(
#         #     torch.reshape(self.box_head(feature), [-1, feature.shape[2] * feature.shape[3] * 9, 5])
#         # )
#     # box_outputs = torch.concat(box_outputs, dim=1)
#     return box_outputs
