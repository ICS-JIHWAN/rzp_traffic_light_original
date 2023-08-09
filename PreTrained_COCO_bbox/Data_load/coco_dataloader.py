import os
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from skimage import io, color
from pycocotools.coco import COCO
from PIL import Image

class CocoDataset():
    def __init__(self, dir_path, set_name='train2017'):
        self.dir_path = dir_path
        self.set_name = set_name
        self.n_classes = 80

        self.toTensor = self._toTensor()

        self.coco = COCO(os.path.join(dir_path, 'annotations', 'instances_{}.json'.format(self.set_name)))

        self.ImgIds = self.coco.getImgIds()

        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.Cat2Lab = dict()
        self.Lab2Cat = dict()
        self.Idx2Cat = dict()
        for idx, c in enumerate(categories):
            self.Cat2Lab[c['name']] = idx
            self.Lab2Cat[idx] = c['name']
            self.Idx2Cat[c['id']] = c['name']

    def __len__(self):
        return len(self.ImgIds)

    def __getitem__(self, index):
        img_id = self.ImgIds[index]
        img_dict = self.coco.loadImgs(img_id)[0]

        # Image
        img = io.imread(os.path.join(self.dir_path, 'images', self.set_name, img_dict['file_name']))

        if len(img.shape) == 2:
            img = color.gray2rgb(img)

        h_factor, w_factor, _ = img.shape

        img = img.astype(np.float32) / 255.0
        img = self.toTensor(img)

        # Labels
        ann_id = self.coco.getAnnIds(imgIds=img_id)
        ann_dict = self.coco.loadAnns(ann_id)

        if len(ann_dict) == 0:
            targets = torch.zeros(0, 6)
        else:
            labels = []
            for ann in ann_dict:
                if ann['bbox'][2] < 1 or ann['bbox'][3] < 1:
                    continue
                if len(ann['bbox']) < 4:
                    continue
                cls = [self.Cat2Lab[self.Idx2Cat[ann['category_id']]]]
                bbox = ann['bbox']

                labels.append(cls + bbox)

            boxes = torch.tensor(labels)
            # annotation : (point of left top) & (width and height)
            # targets    : (point of center) & (width and height) -> adjust normalization
            x1 = boxes[..., 1]
            y1 = boxes[..., 2]
            x2 = boxes[..., 1] + boxes[..., 3]
            y2 = boxes[..., 2] + boxes[..., 4]

            boxes[..., 1] = ((x1 + x2) / 2) / w_factor  # center x
            boxes[..., 2] = ((y1 + y2) / 2) / h_factor  # center y
            boxes[..., 3] /= w_factor                   # width
            boxes[..., 4] /= h_factor                   # height

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        return img, targets

    def _toTensor(self):
        totensor = transforms.Compose([transforms.ToTensor(),
                                       transforms.Resize((416, 416))])
        return totensor

    def collate_fn(self, batch):
        images, targets = list(zip(*batch))

        # Image stack [#, c, w, h]
        images = torch.stack([img for img in images])

        # Target stack [number of bboxes in one batch, information of bbox]
        targets = [boxes for boxes in targets if boxes is not None]

        for i, boxes in enumerate(targets):
            boxes[:, 0] = i

        try:
            targets = torch.cat(targets, 0)
        except RuntimeError:
            targets = None

        return images, targets

if __name__ == '__main__':
    coco_dir_path = '/storage/jhchoi/coco'
    coco_set_name = 'train2017'
    dataset_object_train = CocoDataset(coco_dir_path, coco_set_name)
    train_loader = torch.utils.data.DataLoader(
        dataset_object_train,
        batch_size=16,
        collate_fn=dataset_object_train.collate_fn
    )

    for batch_idx, data in enumerate(train_loader):
        img = data[0]
        targets = data[1]

