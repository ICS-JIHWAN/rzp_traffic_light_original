import torch

from tqdm import tqdm

from options.utils import Config, load_weights
from Data_Load.imagenet_dataloader import ImagenetDataset
from Model.New_FeaturePyramidNetwork import FPN, Classification

layer_dict = {
    'resnet18': [2, 2, 2, 2],
    'resnet34': [3, 4, 6, 3],
    'resnet50': [3, 4, 6, 3],
    'resnet101': [3, 4, 23, 3],
    'resnet152': [3, 8, 36, 3]
}

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
    dataset_object_test = ImagenetDataset(
        config.imagenet_dir_path,
        config.is_train,
        config.input_size,
        num_classes=config.num_classes,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_object_test,
        batch_size=config.batch_size,
        shuffle=True
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
    classifier_model = Classification(num_classes=config.num_classes).to(device)

    backbone_model = load_weights(device, config.check_point_path, config.fpn_check_point_file_name,
                                  net=backbone_model, optimizer=None)
    classifier_model = load_weights(device, config.check_point_path, config.classifier_check_point_file_name,
                                    net=classifier_model, optimizer=None)

    print('\033[95m' + '==== Test starts ====' + '\033[0m')

    loop_test = tqdm(test_loader, leave=True)
    total_acc = []
    for batch_idx, data in enumerate(loop_test):
        img, labels = data['img'].type('torch.FloatTensor').to(device), data['target'].to(device)

        output_FPN = backbone_model(img)
        output_predict = classifier_model(output_FPN)

        acc = classifier_model.cal_accuracy(output_predict, labels)
        total_acc.append(acc)

        loop_test.set_postfix(acc=acc)

    print(f'Total accuracy {sum(total_acc) / len(total_acc)}')
