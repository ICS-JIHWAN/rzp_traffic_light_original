import os
import time

dir_path = '/storage/jhchoi/imagenet/imagenet-object-localization-challenge/ILSVRC'
ann_path = 'Annotations/CLS-LOC/train'
img_path = 'Data/CLS-LOC/train'
LOC_path = 'ImageSets/CLS-LOC/train'

start = time.time()
with open(os.path.join(dir_path, LOC_path+'_loc.txt')) as f:
    lines = f.read().splitlines()
end = time.time()
print(f'{end - start:.5f} sec')

start = time.time()
classes = [p[-9:] for p in glob.glob(os.path.join(dir_path, ann_path, '*'))]
end = time.time()
print(f'{end - start:.5f} sec')

start = time.time()
all_path = [line[:line.index(' ')] for line in lines]
end = time.time()
print(f'{end - start:.5f} sec')
