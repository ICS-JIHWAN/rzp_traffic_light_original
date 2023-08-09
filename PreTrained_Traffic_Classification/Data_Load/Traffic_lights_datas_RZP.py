import torch
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
def _toTensor(h, w, r_h):
    H, W = int(h*(r_h/h)), int(w*(r_h/h))
    if H < 20:
        H = 20
    if W > 100:
        W = 100

    totensor = transforms.Compose([transforms.ToTensor(),
                                   transforms.Resize((H, W))])
    return totensor


class traffic_lights():
    def __init__(self, img_paths, cls_paths, n_cls=9, img_height=20, ratio=5):
        self.n_cls = n_cls
        self.resize_height = img_height
        self.ratio = ratio

        self.img_path_list = img_paths
        self.cls_path_list = cls_paths


    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        cls_path = self.cls_path_list[index]

        ori_image = Image.open(img_path)
        ori_image = np.array(ori_image)

        tr_img = []
        labels = []
        with open(cls_path) as f:
            for l in f.readlines():
                x1, y1, x2, y2, cls = [
                    int(x) for x in l.replace('\n', '').split()
                ]
                img = ori_image[y1:y2, x1:x2]
                toTensor = _toTensor(img.shape[0], img.shape[1], self.resize_height)
                img = toTensor(img)
                RZP_image = torch.zeros((3, self.resize_height, self.resize_height*self.ratio))
                RZP_image[..., :img.shape[2]] = img

                tr_img.append(RZP_image)
                labels.append(self.one_hot_encoding(cls))

        return tr_img, labels

    def one_hot_encoding(self, cls):
        label = torch.zeros(self.n_cls)
        if cls == 1300 or cls == 1400 or cls == 1405 or cls == 1406 or cls == 1502: # blue
            label[2] = 1
        if cls == 1301 or cls == 1303 or cls == 1401 or cls == 1403 or cls == 1404 or cls == 1501: # red
            label[0] = 1
        if cls == 1302 or cls == 1306 or cls == 1402 or cls == 1404 or cls == 1406 or cls == 1407: # yellow
            label[1] = 1
        if cls == 1303 or cls == 1305 or cls == 1403 or cls == 1405: # left
            label[3] = 1
        if cls == 1407 or cls == 1408 or cls == 1409: # right top
            label[4] = 1
        if cls == 1409 or cls == 1502: # left top
            label[5] = 1
        if cls == 1502: # left down
            label[6] = 1
        if cls == 1700: # blue of a pedestrian traffic light
            label[7] = 1
        if cls == 1701: # red of a pedestrian traffic light
            label[8] = 1
        return label

    def collate_fn(self, batch):
        images, targets = list(zip(*batch))

        images = [torch.stack(img, 0) for img in images]
        images = torch.cat(images, 0)

        targets = [torch.stack(gt, 0) for gt in targets]
        targets = torch.cat(targets, 0)

        return images, targets

if __name__ == '__main__':
    from tqdm import tqdm

    img_path_list = open('/home/jhchoi/PycharmProjects/traffic_lights/Data_set/correct_img_train_path.txt',
                         'r').read().split('\n')[:-1]
    cls_path_list = open('/home/jhchoi/PycharmProjects/traffic_lights/Data_set/correct_cls_train_path.txt',
                         'r').read().split('\n')[:-1]

    traffic = traffic_lights(img_path_list, cls_path_list)
    data_loader = torch.utils.data.DataLoader(
        traffic,
        batch_size=1
    )

    w_h = []
    ratio = []
    loop = tqdm(data_loader, leave=True)
    for batch_idx, data in enumerate(loop):
        ori_img, labels = data['img'], data['label']
        for label in labels:
            img = ori_img[:, :, int(label[3]):int(label[4]), int(label[1]):int(label[2])]
            w_h.append([img.shape[2], img.shape[3]])
            ratio.append(round(max(img.shape[2:])/min(img.shape[2:])))

    # ratio : {2, 3, 4, 5}
    # number of ratio : {3142, 6528, 1560, 37} -> total 11267
    # percentage of ratio : {28%, 58%, 14%, 0.32%}
    #
    # width Max : 330 | height Max : 86
    # width Min : 19  | height Min : 9
    print('1')