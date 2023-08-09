import os
import itertools

import torch

from tqdm import tqdm

from options.utils import Config, load_weights, save_weights, count_parameters, summary_model
from Data_load.coco_dataloader import CocoDataset
from Model.detection_fn import non_max_suppression
from Model.New_FeaturePyramidNetwork import FPN, RetinaNet

layer_dict = {
    'resnet18': [2, 2, 2, 2],
    'resnet34': [3, 4, 6, 3],
    'resnet50': [3, 4, 6, 3],
    'resnet101': [3, 4, 23, 3],
    'resnet152': [3, 8, 36, 3]
}

if __name__ == '__main__':
    # Configurations
    # Configurations
    CFG = Config()
    CFG.print_options()
    config = CFG.opt

    if config.gpu_id is not None:
        if torch.cuda.is_available():
            device = torch.device('cuda:{}'.format(config.gpu_id))
            print('Using GPU!! ==> GPU_id : {}\n'.format(config.gpu_id))
        else:
            device = torch.device('cpu')
            print('Using CPU!!\n')
    else:
        device = torch.device('cpu')
        print('Using CPU!!\n')

    # dataset_object_test = CocoDataset(config.coco_dir_path, 'val2017')
    dataset_object_test = CocoDataset(config.coco_dir_path, set_name='val2017')
    test_loader = torch.utils.data.DataLoader(
        dataset_object_test,
        batch_size=config.batch_size,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
        collate_fn=dataset_object_test.collate_fn
    )

    # Model
    if config.backbone_network == 'resnet18' or config.backbone_network == 'resnet34':
        """ Basic residual block of ResNet model """
        backbone_model = FPN(layers=layer_dict[config.backbone_network]).to(device)
    elif config.backbone_network == 'resnet50' or config.backbone_network == 'resnet101' or config.backbone_network == 'resnet152':
        """ Bottle neck block of ResNet model """""
        backbone_model = FPN(layers=layer_dict[config.backbone_network], bottle=True).to(device)
    else:
        raise ValueError('Unsupported model name, must be one of 18, 34, 50, 101, 152')
    regression_model = RetinaNet(img_size=config.input_size[0]).to(device)

    backbone_model = load_weights(device, config.check_point_path, 'fpn_resnet18_0.4700486377462164.pth',
                                  net=backbone_model, optimizer=None
                                  )
    regression_model = load_weights(device, config.check_point_path, 'regression_resnet18_0.4700486377462164.pth',
                                    net=regression_model, optimizer=None
                                    )

    # Train
    loop_test = tqdm(test_loader, leave=True)
    for batch_idx, data in enumerate(loop_test):
        img, labels = data

        output_FPN = backbone_model(img.to(device))
        output_pred, loss = regression_model(output_FPN, labels.to(device))
        output = non_max_suppression(output_pred, conf_thres=0.7, nms_thres=0.8)

        # For debugging
        # import cv2
        # import numpy as np
        # import matplotlib.pyplot as plt
        # image_index = 2
        # img_out = (np.array(img[image_index].permute(1, 2, 0).cpu()) * 255).astype(np.uint8).copy()
        # out_bboxes = output[image_index].cpu().detach().numpy()
        # for obj_index in range(len(out_bboxes)):
        #     out_bbox = out_bboxes[obj_index]
        #     img_out = cv2.rectangle(img_out, (out_bbox[0].astype(np.int).item(), out_bbox[1].astype(np.int).item()),
        #                             (out_bbox[2].astype(np.int).item(), out_bbox[3].astype(np.int).item()), (255, 0, 0),
        #                             3)
        # plt.imshow(img_out)
        # plt.show()

