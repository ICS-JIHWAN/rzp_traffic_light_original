import os
import torch
import argparse

from pytorch_model_summary import summary

class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument('--imagenet_dir_path', type=str, default='/storage/jhchoi/imagenet/imagenet-object-localization-challenge/ILSVRC', help='Imagenet directory path')
        self.parser.add_argument('--is_train', type=str, default='val', help='The present state of things') # [train|val|test]
        self.parser.add_argument('--continue_FPN_train', type=bool, default=False, help='Whether initialize or load the model')
        self.parser.add_argument('--continue_CLS_train', type=bool, default=False, help='Whether initialize or load the model')

        self.parser.add_argument('--check_point_path', type=str, default='./Model/Check_point', help='Check point directory path')
        self.parser.add_argument('--fpn_check_point_file_name', type=str, default=None, help='FPN check point file name')
        self.parser.add_argument('--classifier_check_point_file_name', type=str, default=None, help='Classifier check point file name')

        self.parser.add_argument('--input_size', type=list, default=[416, 416], help='Width and Height of input image')
        self.parser.add_argument('--num_classes', type=int, default=1000, help='number of classes')

        self.parser.add_argument('--backbone_network', type=str, default='resnet34', help='Name of the backbone network')
        self.parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
        self.parser.add_argument('--batch_size', type=int, default=8, help='Number of batch')

        self.parser.add_argument('--scheduler', type=str, default='steplr')
        self.parser.add_argument('--lr', type=float, default=1e-4)
        self.parser.add_argument('--lr_min', type=float, default=1e-7)
        self.parser.add_argument('--gpu_id', type=int, default=1, help='GPU process number')

        self.opt, _ = self.parser.parse_known_args()

    def print_options(self):
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(self.opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)


def save_weights(state, file_path):
    print('=> Complete saving weights')
    torch.save(state, file_path)


def load_weights(device, check_point_path, check_point_file, **kwargs):
    state_dict = torch.load(os.path.join(check_point_path, check_point_file), map_location=str(device))
    kwargs['net'].load_state_dict(state_dict['net'], strict=True)
    if kwargs['optimizer'] is not None:
        kwargs['optimizer'].load_state_dict(state_dict['opt'])
        print('=> Complete loading weights')
        return kwargs['net'], kwargs['optimizer']
    else:
        print('=> Complete loading weights')
        return kwargs['net']

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def summary_model(model):
    print(summary(model, torch.zeros((16, 3, 416, 416)), show_input=True))