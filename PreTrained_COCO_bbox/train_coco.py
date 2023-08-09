import os
import itertools

import torch
import torch.optim as optim

from tqdm import tqdm

from options.utils import Config, load_weights, save_weights, count_parameters, summary_model
from Data_load.coco_dataloader import CocoDataset
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

    dataset_object_train = CocoDataset(config.coco_dir_path)
    train_loader = torch.utils.data.DataLoader(
        dataset_object_train,
        batch_size=config.batch_size,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        collate_fn=dataset_object_train.collate_fn
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

    print('The number of parameters, where backbone model : {}'.format(count_parameters(backbone_model)))
    print('The number of parameters, where regression model : {}'.format(count_parameters(regression_model)))

    # Hyperparameter
    optimizer = optim.Adam(itertools.chain(backbone_model.parameters(), regression_model.parameters()), lr=config.lr,
                           weight_decay=0)
    if config.scheduler == 'steplr':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif config.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=config.lr_min)
    else:
        NotImplementedError('scheduler not implemented {}'.format(config.scheduler))

    # Fine tuning
    if not config.continue_FPN_train:
        backbone_model.init_weights()
    else:
        backbone_model = load_weights(device, config.check_point_path, config.fpn_check_point_file_name,
                                                 net=backbone_model, optimizer=None)
    if not config.continue_REG_train:
        regression_model.init_weights()
    else:
        regression_model = load_weights(device, config.check_point_path, config.regression_check_point_file_name,
                                        net=regression_model, optimizer=None)

    # Train
    for curr_epoch in range(config.epochs):
        print('\033[95m' + '==== epoch {} starts ===='.format(curr_epoch + 1) + '\033[0m')

        loop_train = tqdm(train_loader, leave=True)
        mean_loss = []
        for batch_idx, data in enumerate(loop_train):
            img, labels = data

            # Forward
            output_FPN = backbone_model(img.to(device))
            output_pred, loss = regression_model(output_FPN, labels.to(device))

            # Update
            mean_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop_train.set_postfix(loss=loss.item())

        curr_loss = sum(mean_loss) / len(mean_loss)
        print('\033[94m' + f"Mean loss was {curr_loss}" + '\033[0m')

        # Save Model
        try:
            if not os.path.exists(config.check_point_path):
                os.makedirs(config.check_point_path)
        except OSError:
            print('Error: Failed to create the directory.')

        scheduler.step()
        print('learning rate : {}'.format(scheduler.get_lr()))

        save_weights(
            {'net': backbone_model.state_dict(), 'opt': optimizer.state_dict()},
            os.path.join(config.check_point_path, 'fpn_{}_{}.pth'.format(config.backbone_network, round(curr_loss, 4)))
        )
        save_weights(
            {'net': regression_model.state_dict(), 'opt': optimizer.state_dict()},
            os.path.join(config.check_point_path, 'regression_{}_{}.pth'.format(config.backbone_network, round(curr_loss, 4)))
        )
        print('\033[95m' + '==== epoch {} ends ====\n'.format(curr_epoch + 1) + '\033[0m')
