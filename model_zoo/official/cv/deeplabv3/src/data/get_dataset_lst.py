import argparse
import os

import numpy as np
import scipy.io
from PIL import Image

parser = argparse.ArgumentParser('dataset list generator')
parser.add_argument("--data_dir", type=str, default='./', help='where dataset stored.')

args, _ = parser.parse_known_args()

data_dir = args.data_dir
print("Data dir is:", data_dir)

#
VOC_IMG_DIR = os.path.join(data_dir, 'VOCdevkit/VOC2012/JPEGImages')
VOC_ANNO_DIR = os.path.join(data_dir, 'VOCdevkit/VOC2012/SegmentationClass')
VOC_ANNO_GRAY_DIR = os.path.join(data_dir, 'VOCdevkit/VOC2012/SegmentationClassGray')
VOC_TRAIN_TXT = os.path.join(data_dir, 'VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt')
VOC_VAL_TXT = os.path.join(data_dir, 'VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt')

SBD_ANNO_DIR = os.path.join(data_dir, 'benchmark_RELEASE/dataset/cls')
SBD_IMG_DIR = os.path.join(data_dir, 'benchmark_RELEASE/dataset/img')
SBD_ANNO_PNG_DIR = os.path.join(data_dir, 'benchmark_RELEASE/dataset/cls_png')
SBD_ANNO_GRAY_DIR = os.path.join(data_dir, 'benchmark_RELEASE/dataset/cls_png_gray')
SBD_TRAIN_TXT = os.path.join(data_dir, 'benchmark_RELEASE/dataset/train.txt')
SBD_VAL_TXT = os.path.join(data_dir, 'benchmark_RELEASE/dataset/val.txt')

VOC_TRAIN_LST_TXT = os.path.join(data_dir, 'voc_train_lst.txt')
VOC_VAL_LST_TXT = os.path.join(data_dir, 'voc_val_lst.txt')
VOC_AUG_TRAIN_LST_TXT = os.path.join(data_dir, 'vocaug_train_lst.txt')


def __get_data_list(data_list_file):
    with open(data_list_file, mode='r') as f:
        return f.readlines()


def conv_voc_colorpng_to_graypng():
    if not os.path.exists(VOC_ANNO_GRAY_DIR):
        os.makedirs(VOC_ANNO_GRAY_DIR)

    for ann in os.listdir(VOC_ANNO_DIR):
        ann_im = Image.open(os.path.join(VOC_ANNO_DIR, ann))
        ann_im = Image.fromarray(np.array(ann_im))
        ann_im.save(os.path.join(VOC_ANNO_GRAY_DIR, ann))


def __gen_palette(cls_nums=256):
    palette = np.zeros((cls_nums, 3), dtype=np.uint8)
    for i in range(cls_nums):
        lbl = i
        j = 0
        while lbl:
            palette[i, 0] |= (((lbl >> 0) & 1) << (7 - j))
            palette[i, 1] |= (((lbl >> 1) & 1) << (7 - j))
            palette[i, 2] |= (((lbl >> 2) & 1) << (7 - j))
            lbl >>= 3
            j += 1
    return palette.flatten()


def conv_sbd_mat_to_png():
    if not os.path.exists(SBD_ANNO_PNG_DIR):
        os.makedirs(SBD_ANNO_PNG_DIR)
    if not os.path.exists(SBD_ANNO_GRAY_DIR):
        os.makedirs(SBD_ANNO_GRAY_DIR)

    palette = __gen_palette()
    for an in os.listdir(SBD_ANNO_DIR):
        img_id = an[:-4]
        mat = scipy.io.loadmat(os.path.join(SBD_ANNO_DIR, an))
        anno = mat['GTcls'][0]['Segmentation'][0].astype(np.uint8)
        anno_png = Image.fromarray(anno)
        # save to gray png
        anno_png.save(os.path.join(SBD_ANNO_GRAY_DIR, img_id + '.png'))
        # save to color png use palette
        anno_png.putpalette(palette)
        anno_png.save(os.path.join(SBD_ANNO_PNG_DIR, img_id + '.png'))


def create_voc_train_lst_txt():
    voc_train_data_lst = __get_data_list(VOC_TRAIN_TXT)
    with open(VOC_TRAIN_LST_TXT, mode='w') as f:
        for id_ in voc_train_data_lst:
            id_ = id_.strip()
            img_ = os.path.join(VOC_IMG_DIR, id_ + '.jpg')
            anno_ = os.path.join(VOC_ANNO_GRAY_DIR, id_ + '.png')
            f.write(img_ + ' ' + anno_ + '\n')


def create_voc_val_lst_txt():
    voc_val_data_lst = __get_data_list(VOC_VAL_TXT)
    with open(VOC_VAL_LST_TXT, mode='w') as f:
        for id_ in voc_val_data_lst:
            id_ = id_.strip()
            img_ = os.path.join(VOC_IMG_DIR, id_ + '.jpg')
            anno_ = os.path.join(VOC_ANNO_GRAY_DIR, id_ + '.png')
            f.write(img_ + ' ' + anno_ + '\n')


def create_voc_train_aug_lst_txt():
    voc_train_data_lst = __get_data_list(VOC_TRAIN_TXT)
    voc_val_data_lst = __get_data_list(VOC_VAL_TXT)

    sbd_train_data_lst = __get_data_list(SBD_TRAIN_TXT)
    sbd_val_data_lst = __get_data_list(SBD_VAL_TXT)

    with open(VOC_AUG_TRAIN_LST_TXT, mode='w') as f:
        for id_ in sbd_train_data_lst + sbd_val_data_lst:
            if id_ in voc_train_data_lst + voc_val_data_lst:
                continue
            id_ = id_.strip()
            img_ = os.path.join(SBD_IMG_DIR, id_ + '.jpg')
            anno_ = os.path.join(SBD_ANNO_GRAY_DIR, id_ + '.png')
            f.write(img_ + ' ' + anno_ + '\n')

        for id_ in voc_train_data_lst:
            id_ = id_.strip()
            img_ = os.path.join(VOC_IMG_DIR, id_ + '.jpg')
            anno_ = os.path.join(VOC_ANNO_GRAY_DIR, id_ + '.png')
            f.write(img_ + ' ' + anno_ + '\n')


if __name__ == '__main__':
    print('converting voc color png to gray png ...')
    conv_voc_colorpng_to_graypng()
    print('converting done.')

    create_voc_train_lst_txt()
    print('generating voc train list success.')

    create_voc_val_lst_txt()
    print('generating voc val list success.')

    print('converting sbd annotations to png ...')
    conv_sbd_mat_to_png()
    print('converting done')

    create_voc_train_aug_lst_txt()
    print('generating voc train aug list success.')
