import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from tqdm import tqdm

from options.utils import Config, load_weights, count_parameters, save_weights
from Data_Load.Traffic_lights_datas_RZP import traffic_lights
from Model.Ratio_Zero_Padding import RatioZeroPaddingFPN, RatioZeroPadding

if __name__ == '__main__':
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

    # Dataset
    train_img_path_list = open('/home/jhchoi/PycharmProjects/traffic_lights/Data_set/correct_img_train_path.txt', 'r').read().split('\n')[:-1]
    train_cls_path_list = open('/home/jhchoi/PycharmProjects/traffic_lights/Data_set/correct_cls_train_path.txt', 'r').read().split('\n')[:-1]
    test_img_path_list = open('/home/jhchoi/PycharmProjects/traffic_lights/Data_set/correct_img_test_path.txt', 'r').read().split('\n')[:-1]
    test_cls_path_list = open('/home/jhchoi/PycharmProjects/traffic_lights/Data_set/correct_cls_test_path.txt', 'r').read().split('\n')[:-1]

    dataset_object_train = traffic_lights(train_img_path_list, train_cls_path_list,
                                          config.num_classes, config.img_height, config.ratio)
    train_loader = torch.utils.data.DataLoader(
        dataset_object_train,
        batch_size=config.batch_size,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        collate_fn=dataset_object_train.collate_fn
    )
    dataset_object_test = traffic_lights(test_img_path_list, test_cls_path_list,
                                         config.num_classes, config.img_height, config.ratio)
    test_loader = torch.utils.data.DataLoader(
        dataset_object_test,
        batch_size=config.batch_size,
        pin_memory=True,    # Dataloader put on the fixed Cuda memory
        shuffle=True,
        drop_last=True,     # last batch drop when not same the batch size and last batch size
        collate_fn=dataset_object_test.collate_fn
    )

    # Model
    model1 = RatioZeroPadding(3, config.img_height, config.num_classes)
    model2 = RatioZeroPaddingFPN(config.img_height, config.img_height*config.ratio, config.num_classes)

    print('The number of parameters, where full RZP model : {}'.format(count_parameters(model1)))
    print('The number of parameters, where FPN RZP model : {}'.format(count_parameters(model2)))

    # Hyperparameter
    optimizer1 = optim.Adam(model1.parameters(), lr=config.lr, weight_decay=0)
    optimizer2 = optim.Adam(model2.parameters(), lr=config.lr, weight_decay=0)
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=20, eta_min=config.lr_min)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=20, eta_min=config.lr_min)
    fn_loss = nn.BCEWithLogitsLoss(reduction='mean')

    # Fine tuning
    if not config.continue_train:
        model1.init_weights()
        model2.init_weights()
    else:
        model1, optimizer1 = load_weights(device, config.check_point_path, config.RZP_check_point_file_name,
                                          net=model1, optimizer=optimizer1)
        model2, optimizer2 = load_weights(device, config.check_point_path, config.RZP_FPN_check_point_file_name,
                                          net=model2, optimizer=optimizer2)

    # Train
    for curr_epoch in range(config.epochs):
        print('\033[95m' + '==== epoch {} starts ===='.format(curr_epoch + 1) + '\033[0m')

        loop_train = tqdm(train_loader, leave=True)
        mean_loss1 = []
        mean_loss2 = []
        for batch_idx, (images, targets) in enumerate(loop_train):
            inputs = images
            labels = targets

            # Forward
            output1 = model1(inputs)
            output2 = model2(inputs)

            # Update
            loss1 = fn_loss(output1, targets)
            loss2 = fn_loss(output2, targets)

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss1.backward()
            loss2.backward()
            optimizer1.step()
            optimizer2.step()

            mean_loss1.append(loss1.item())
            mean_loss2.append(loss2.item())
            loop_train.set_postfix(loss1=loss1.item(), loss2=loss2.item())

        curr_loss1 = sum(mean_loss1) / len(mean_loss1)
        curr_loss2 = sum(mean_loss2) / len(mean_loss2)
        print('\033[94m' + f"Mean loss1 was [{curr_loss1}] \t Mean Loss2 was [{curr_loss2}]" + '\033[0m')

        # Save Model
        try:
            if not os.path.exists(config.check_point_path):
                os.makedirs(config.check_point_path)
        except OSError:
            print('Error: Failed to create the directory.')

        save_weights(
            {'net': model1.state_dict(), 'opt': model1.state_dict()},
            os.path.join(config.check_point_path, 'RZP_{}.pth'.format(curr_loss1))
        )
        save_weights(
            {'net': model2.state_dict(), 'opt': model2.state_dict()},
            os.path.join(config.check_point_path, 'RZP_FPN_{}.pth'.format(curr_loss2))
        )

        loop_test = tqdm(test_loader, leave=True)
        for batch_idx, (images, targets) in enumerate(loop_test):
            inputs = images
            labels = targets

            # Forward
            output1 = model1(inputs)
            output2 = model2(inputs)

            output1 = torch.where(torch.sigmoid(output1) > 0.5, 1, 0)
            output2 = torch.where(torch.sigmoid(output2) > 0.5, 1, 0)

            # Precision - RZP
            TP1 = torch.count_nonzero(targets * output1, dim=1)
            FP1 = torch.count_nonzero(torch.where((output1 - targets) > 0, 1, 0), dim=1)
            precision1 = TP1 / (TP1 + FP1)
            P1 = torch.sum(precision1) / precision1.shape[0]
            # Recall - RZP
            FN1 = torch.count_nonzero(torch.where((targets - output1) > 0, 1, 0), dim=1)
            recall1 = TP1 / (TP1 + FN1)
            R1 = torch.sum(recall1) / recall1.shape[0]

            # Precision - RZP&FPN
            TP2 = torch.count_nonzero(targets * output1, dim=1)
            FP2 = torch.count_nonzero(torch.where((output2 - targets) > 0, 1, 0), dim=1)
            precision2 = TP2 / (TP2 + FP2)
            P2 = torch.sum(precision2) / precision2.shape[0]
            # Recall - RZP&FPN
            FN2 = torch.count_nonzero(torch.where((targets - output2) > 0, 1, 0), dim=1)
            recall2 = TP2 / (TP2 + FN2)
            R2 = torch.sum(recall2) / recall2.shape[0]

            loop_test.set_postfix(Precision1=P1.item(), Recall1=R1.item(),
                                  Precision2=P2.item(), Recall2=R2.item())

        print('\033[95m' + '==== epoch {} ends ====\n'.format(curr_epoch + 1) + '\033[0m')
