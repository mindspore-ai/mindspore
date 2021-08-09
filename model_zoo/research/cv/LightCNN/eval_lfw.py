# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""eval lfw"""
import os
import argparse
import cv2
import numpy as np
import scipy.io

import mindspore.common.dtype as mstype
from mindspore import context, load_param_into_net, load_checkpoint, Tensor
from sklearn.metrics import roc_curve

from src.lightcnn import lightCNN_9Layers4Test
from src.config import lightcnn_cfg as cfg

parser = argparse.ArgumentParser(description='PyTorch ImageNet Feature Extracting')
parser.add_argument('--device_target', default='Ascend', choices=['Ascend', 'GPU', 'CPU'], type=str)
parser.add_argument('--device_id', default=0, type=int)
parser.add_argument('--resume', default='',
                    type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--num_classes', default=79077, type=int,  # !!!
                    metavar='N', help='number of classes (default: 79077)')

args = parser.parse_args()


def extract_feature(img_list):
    """extra features"""
    model = lightCNN_9Layers4Test(num_classes=args.num_classes)
    model.set_train(False)

    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        params_dict = load_checkpoint(args.resume)
        load_param_into_net(model, params_dict)
    else:
        print("=> ERROR: No checkpoint found at '{}'".format(args.resume))
        exit(0)

    features_shape = (len(img_list), 256)
    features = np.empty(features_shape, dtype='float32', order='C')

    for idx, img_name in enumerate(img_list):
        print('%d images processed' % (idx + 1,))
        img = cv2.imread(os.path.join(cfg.root_path, img_name), cv2.IMREAD_GRAYSCALE)
        if img.shape != (128, 128):
            img = cv2.resize(img, (128, 128))
        img = np.reshape(img, (1, 1, 128, 128))
        inputs = img.astype(np.float32) / 255.0
        inputs = Tensor(inputs, mstype.float32)
        _, feature = model(inputs)
        features[idx:idx + 1, :] = feature.asnumpy()

    return features


def load_image_list(img_dir, list_file_name):
    """load image list"""
    img_dir_cp = img_dir.replace('/image', '')
    list_file_path = os.path.join(img_dir_cp, list_file_name)
    f = open(list_file_path, 'r')
    image_list = []
    labels = []
    for line in f:
        items = line.split()
        image_list.append(items[0].strip())
        labels.append(items[1].strip())
    return labels, image_list


def labels_list_to_int(labels):
    """convert type of labels to integer"""
    int_labels = []
    for e in labels:
        try:
            inte = int(e)
        except ValueError:
            print('Labels are not int numbers. A mapping will be used.')
            break
        int_labels.append(inte)
    if len(int_labels) == len(labels):
        return int_labels
    return None


def string_list_to_cells(lst):
    """
    Uses numpy.ndarray with dtype=object. Convert list to np.ndarray().
    """
    cells = np.ndarray(len(lst), dtype='object')
    for idx, ele in enumerate(lst):
        cells[idx] = ele
    return cells


def extract_features_to_dict(image_dir, list_file):
    """extract features and save them with dictionary"""
    labels, img_list = load_image_list(image_dir, list_file)
    ftr = extract_feature(img_list)
    integer_labels = labels_list_to_int(labels)
    feature_dict = {'features': ftr,
                    'labels': integer_labels,
                    'labels_original': string_list_to_cells(labels),
                    'image_path': string_list_to_cells(img_list)}
    return feature_dict


def compute_cosine_score(feature1, feature2):
    """compute cosine score"""
    feature1_norm = np.linalg.norm(feature1)
    feature2_norm = np.linalg.norm(feature2)
    score = np.dot(feature1, feature2) / (feature1_norm * feature2_norm)
    return score


def lfw_eval(lightcnn_result, lfw_pairs_mat_path):
    """eval lfw"""
    features = lightcnn_result['features']
    lfw_pairs_mat = scipy.io.loadmat(lfw_pairs_mat_path)
    pos_pair = lfw_pairs_mat['pos_pair']
    neg_pair = lfw_pairs_mat['neg_pair']

    pos_scores = np.zeros(len(pos_pair[1]))

    for idx, _ in enumerate(pos_pair[1]):
        feat1 = features[pos_pair[0, idx] - 1, :]
        feat2 = features[pos_pair[1, idx] - 1, :]
        pos_scores[idx] = compute_cosine_score(feat1, feat2)
    pos_label = np.ones(len(pos_pair[1]))

    neg_scores = np.zeros(len(neg_pair[1]))
    for idx, _ in enumerate(neg_pair[1]):
        feat1 = features[neg_pair[0, idx] - 1, :]
        feat2 = features[neg_pair[1, idx] - 1, :]
        neg_scores[idx] = compute_cosine_score(feat1, feat2)
    neg_label = -1 * np.ones(len(neg_pair[1]))

    scores = np.concatenate((pos_scores, neg_scores), axis=0)
    label = np.concatenate((pos_label, neg_label), axis=0)

    fpr, tpr, _ = roc_curve(label, scores, pos_label=1)
    res = tpr - (1 - fpr)

    eer = tpr[np.squeeze(np.where(res >= 0))[0]] * 100
    far_10 = tpr[np.squeeze(np.where(fpr <= 0.01))[-1]] * 100
    far_01 = tpr[np.squeeze(np.where(fpr <= 0.001))[-1]] * 100
    far_00 = tpr[np.squeeze(np.where(fpr <= 0.0))[-1]] * 100

    print('100%eer:      ', round(eer, 2))
    print('tpr@far=1%:   ', round(far_10, 2))
    print('tpr@far=0.1%: ', round(far_01, 2))
    print('tpr@far=0%:   ', round(far_00, 2))


if __name__ == '__main__':
    device_id = int(os.getenv('DEVICE_ID'))
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=device_id)
    dic = extract_features_to_dict(image_dir=cfg.root_path, list_file=cfg.lfw_img_list)
    lfw_eval(dic, lfw_pairs_mat_path=cfg.lfw_pairs_mat_path)
