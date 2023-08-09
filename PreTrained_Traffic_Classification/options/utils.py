import os
import argparse
import torch

class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument('--continue_train', type=bool, default=False, help='Whether initialize or load the model')

        self.parser.add_argument('--check_point_path', type=str, default='./Model/Check_point', help='Check point directory path')
        self.parser.add_argument('--RZP_check_point_file_name', type=str, default=None)
        self.parser.add_argument('--RZP_FPN_check_point_file_name', type=str, default=None)

        self.parser.add_argument('--img_height', type=int, default=20, help='height size of traffic light image')
        self.parser.add_argument('--ratio', type=list, default=5, help='ratio of input image')
        self.parser.add_argument('--num_classes', type=int, default=9, help='number of classes')

        self.parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
        self.parser.add_argument('--batch_size', type=int, default=16, help='Number of batch')

        self.parser.add_argument('--lr', type=float, default=1e-4)
        self.parser.add_argument('--lr_min', type=float, default=1e-7)
        self.parser.add_argument('--gpu_id', type=int, default=None, help='GPU process number')

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
    kwargs['net'].load_state_dict(state_dict['net'], strict=False)
    if kwargs['optimizer'] is not None:
        kwargs['optimizer'].load_state_dict(state_dict['opt'])
        print('=> Complete loading weights')
        return kwargs['net'], kwargs['optimizer']
    else:
        print('=> Complete loading weights')
        return kwargs['net']

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
