import os
import glob
import csv
import torch
import torchvision.transforms as T

import xml.etree.ElementTree as ET
from PIL import Image

class ImagenetDataset():
    def __init__(self, dir_path, is_train, input_size, num_classes=1000):
        self.is_train = is_train
        self.w, self.h = input_size[0], input_size[1]
        self.transform = self._toTensor()
        self.n_classes = num_classes

        self.dir_path = dir_path
        self.ann_path = os.path.join(self.dir_path, 'Annotations/CLS-LOC', is_train)
        self.img_path = os.path.join(self.dir_path, 'Data/CLS-LOC', is_train)
        self.LOC_path = os.path.join(self.dir_path, 'ImageSets/CLS-LOC', is_train)

        self.one_hot_encoding()

        if is_train == 'train':
            CLS_LOC_path = os.path.join(self.dir_path, self.LOC_path + '_loc.txt')
        else:
            CLS_LOC_path = os.path.join(self.dir_path, self.LOC_path + '.txt')

        with open(CLS_LOC_path) as f:
            lines = f.read().splitlines()
        self.all_data = [line[:line.index(' ')] for line in lines]

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        # 55311
        data_path = os.path.join(self.all_data[idx])

        image = self.transform(Image.open(os.path.join(self.dir_path, self.img_path, data_path) + '.JPEG').convert('RGB'))
        if self.is_train == 'train' or self.is_train == 'val':
            target = self._make_target(data_path)
            return {'img': image, 'target': target}
        else:
            return {'img': image}


    def one_hot_encoding(self):
        # name to label dictionary --> self.class_dict
        # label to name dictionary --> self.class_dict_reverse
        os.path.join(self.dir_path[:-7], 'LOC_synset_mapping.txt')
        with open(os.path.join(self.dir_path[:-7], 'LOC_synset_mapping.txt'), 'r') as f:
            lines = f.read().splitlines()
        classes = sorted([line[:9] for line in lines])

        # one-hot encoding
        self.class_dict = dict()
        self.class_dict_reverse = dict()
        for idx, c in enumerate(classes):
            self.class_dict[c] = idx
            self.class_dict_reverse[idx] = c

    def _make_target(self, data_path):
        tree = ET.parse(os.path.join(self.dir_path, self.ann_path, data_path) + '.xml')
        root = tree.getroot()

        labels = list(set([book.find('name').text for book in root.findall('object')]))

        target = torch.zeros([self.n_classes])
        for label in labels:
            target[self.class_dict[label]] = 1

        return target

    def _toTensor(self):
        toTensor = T.Compose([
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),  # Random value +0.5 ~ -0.5
            T.GaussianBlur(kernel_size=(3, 3)),
            # T.AugMix(),
            T.TrivialAugmentWide(),
            T.Resize((self.w, self.h)),
            T.ToTensor()
        ])

        return toTensor

if __name__ == '__main__':
    dataset = ImagenetDataset('/storage/jhchoi/imagenet/imagenet-object-localization-challenge/ILSVRC',
                              'train',
                              [416, 416])

    a = dataset.__getitem__(55311)
    print(dataset.__len__())
