import os
import glob
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
def cross_validation():
    """
    Random Cross Validation rating 0.7 : 0.3
    You want to change rating value, just change the numer in variable 'correct_train_img' !
    """
    data_path = '/storage/jhchoi/Traffic_Light/Traffic Light/01 Traffic light (Sample)/*'
    list_data_paths = sorted(glob.glob(data_path))

    for a in list_data_paths:
        file_list_img = sorted(glob.glob(a + '/JPEGImages_mosaic/*'))
        file_list_cls = sorted(glob.glob(a + '/labels_class_5/*'))

        correct_img = []
        correct_cls = []
        for idx, file_list_path in zip(range(len(file_list_cls)), file_list_cls):
            example = []
            with open(file_list_path, 'r') as f:
                classes = f.readlines()
                for label in classes:
                    example += [x for x in label.replace('\n', '').split()]

            if '0000' in example:
                if len(example) == 5:
                    continue
                else:
                    with open(file_list_path, 'w') as f:
                        for label in classes:
                            if '0000' not in label.strip('\n'):
                                f.write(label)

            if len(example) % 5 == 0 and len(example) != 0:
                correct_img.append(file_list_img[idx])
                correct_cls.append(file_list_cls[idx])

        correct_train_img = sorted(random.sample(correct_img, int(len(correct_img) * 0.7)))
        correct_test_img = sorted(list(set(correct_img) - set(correct_train_img)))
        correct_train_cls = sorted([correct_cls[correct_img.index(i)] for i in correct_train_img])
        correct_test_cls = sorted(list(set(correct_cls) - set(correct_train_cls)))

        correct_img_train_path = 'correct_img_train_path.txt'
        correct_img_test_path = 'correct_img_test_path.txt'
        correct_cls_train_path = 'correct_cls_train_path.txt'
        correct_cls_test_path = 'correct_cls_test_path.txt'
        # correct image path train and test Cross validation : 0.7
        # train set
        if not os.path.exists(correct_img_train_path):
            with open(correct_img_train_path, 'w', encoding='UTF-8') as f:
                for name in correct_train_img:
                    f.write(name + '\n')
        else:
            with open(correct_img_train_path, 'a', encoding='UTF-8') as f:
                for name in correct_train_img:
                    f.write(name + '\n')
        # test set
        if not os.path.exists(correct_img_test_path):
            with open(correct_img_test_path, 'w', encoding='UTF-8') as f:
                for name in correct_test_img:
                    f.write(name + '\n')
        else:
            with open(correct_img_test_path, 'a', encoding='UTF-8') as f:
                for name in correct_test_img:
                    f.write(name + '\n')

        # correct class path train and test Cross validation : 0.7
        # train set
        if not os.path.exists(correct_cls_train_path):
            with open(correct_cls_train_path, 'w', encoding='UTF-8') as f:
                for name in correct_train_cls:
                    f.write(name + '\n')
        else:
            with open(correct_cls_train_path, 'a', encoding='UTF-8') as f:
                for name in correct_train_cls:
                    f.write(name + '\n')
        # test set
        if not os.path.exists(correct_cls_test_path):
            with open(correct_cls_test_path, 'w', encoding='UTF-8') as f:
                for name in correct_test_cls:
                    f.write(name + '\n')
        else:
            with open(correct_cls_test_path, 'a', encoding='UTF-8') as f:
                for name in correct_test_cls:
                    f.write(name + '\n')

        print("end")

def conunt_width_and_height():
    import matplotlib.pyplot as plt
    from PIL import Image

    img_path_list = open('/home/jhchoi/PycharmProjects/traffic_lights/Data_set/correct_img_train_path.txt', 'r').read().split('\n')[:-1]
    cls_path_list = open('/home/jhchoi/PycharmProjects/traffic_lights/Data_set/correct_cls_train_path.txt', 'r').read().split('\n')[:-1]

    width_dict = dict()
    height_dict = dict()

    for img_path in img_path_list:
        img = Image.open(img_path)
        img = np.array(img)

        width = img.shape[0]
        height = img.shape[1]

        if width_dict.get('{}'.format(width), 0) == 0:
            width_dict['{}'.format(width)] = 1
        else:
            width_dict['{}'.format(width)] += 1

        if height_dict.get('{}'.format(height), 0) == 0:
            height_dict['{}'.format(height)] = 1
        else:
            height_dict['{}'.format(height)] += 1

    print('end')


# method txt file read
# with open(file_list_cls[39], 'r') as f:
#     example = f.read()
#     example = '\t'.join(example.split('\n')).split('\t')[:-1]
#     if len(example) % 5 == 0:
#         example = np.array(example)
#         example = example.reshape(-1, 5)
