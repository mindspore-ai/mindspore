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
'''
evaluation of lfw, calfw, cfp_fp, agedb_30, cplfw
'''

import datetime
import os
import pickle
import argparse
from io import BytesIO

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from scipy import interpolate
import mindspore as ms
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import context

from src.iresnet import iresnet100


class LFold:
    '''
    LFold
    '''
    def __init__(self, n_splits=2, shuffle=False):
        self.n_splits = n_splits
        if self.n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

    def split(self, indices):
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        return [(indices, indices)]


def calculate_roc(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  nrof_folds=10,
                  pca=0):
    '''
    calculate_roc
    '''
    assert embeddings1.shape[0] == embeddings2.shape[0]
    assert embeddings1.shape[1] == embeddings2.shape[1]
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if pca > 0:
            print('doing pca on', fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(
                threshold, dist[test_set],
                actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index], dist[test_set],
            actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    '''calculate_acc
    '''
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(
        np.logical_and(np.logical_not(predict_issame),
                       np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  far_target,
                  nrof_folds=10):
    '''
    calculate_val
    '''
    assert embeddings1.shape[0] == embeddings2.shape[0]
    assert embeddings1.shape[1] == embeddings2.shape[1]
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(
                threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(
            threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    '''calculate_val_far
    '''
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(
        np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def evaluate(embeddings, actual_issame, nrof_folds=10, pca=0):
    '''evaluate
    '''
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds,
                                       embeddings1,
                                       embeddings2,
                                       np.asarray(actual_issame),
                                       nrof_folds=nrof_folds,
                                       pca=pca)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds,
                                      embeddings1,
                                      embeddings2,
                                      np.asarray(actual_issame),
                                      1e-3,
                                      nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far


def load_bin(path, image_size):
    '''load evalset of .bin
    '''
    try:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f)  # py2
    except UnicodeDecodeError as _:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes')  # py3
    data_list = []
    for _ in [0, 1]:
        data = np.zeros(
            (len(issame_list) * 2, 3, image_size[0], image_size[1]))
        data_list.append(data)
    for idx in range(len(issame_list) * 2):
        _bin = bins[idx]
        img = plt.imread(BytesIO(_bin), "jpg")
        if img.shape[1] != image_size[0]:
            img = mx.image.resize_short(img, image_size[0])
        img = np.transpose(img, axes=(2, 0, 1))
        for flip in [0, 1]:
            if flip == 1:
                img = np.flip(img, axis=2)
            data_list[flip][idx][:] = img
    return data_list, issame_list


def test(data_set, backbone, batch_size, nfolds=10):
    '''test
    '''
    print('testing verification..')
    data_list = data_set[0]
    issame_list = data_set[1]
    embeddings_list = []
    time_consumed = 0.0
    for data in data_list:
        embeddings = None
        ba = 0
        while ba < data.shape[0]:
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba
            _data = data[bb - batch_size: bb]

            time0 = datetime.datetime.now()
            img = ((_data / 255) - 0.5) / 0.5
            net_out = backbone(ms.Tensor(img, ms.float32))
            _embeddings = net_out.asnumpy()
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
            ba = bb
        embeddings_list.append(embeddings)
    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm = np.linalg.norm(_em)
            _xnorm += _norm
            _xnorm_cnt += 1
    _xnorm /= _xnorm_cnt

    embeddings = embeddings_list[0].copy()
    embeddings = sklearn.preprocessing.normalize(embeddings)
    _, _, acc, _, _, _ = evaluate(embeddings, issame_list, nrof_folds=nfolds)
    acc1 = np.mean(acc)
    std1 = np.std(acc)
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print(embeddings.shape)
    print('infer time', time_consumed)
    _, _, accuracy, _, _, _ = evaluate(
        embeddings, issame_list, nrof_folds=nfolds)
    acc2, std2 = np.mean(accuracy), np.std(accuracy)
    return acc1, std1, acc2, std2, _xnorm, embeddings_list


def main():
    '''r
    main function
    '''
    parser = argparse.ArgumentParser(description='do verification')
    # general
    parser.add_argument(
        '--eval_url', default='/opt_data/zjc/arcface/', help='')
    parser.add_argument('--device_id', default=0, type=int, help='device id')
    parser.add_argument('--target',
                        default='lfw,cfp_fp,agedb_30',
                        help='test targets.')
    parser.add_argument(
        '--ckpt_url', default="/cache/ArcFace--25_11372.ckpt", type=str, help='ckpt path')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--batch-size', default=64, type=int, help='')
    parser.add_argument('--max', default='', type=str, help='')
    parser.add_argument('--nfolds', default=10, type=int, help='')
    args = parser.parse_args()
    context.set_context(device_id=args.device_id, mode=context.GRAPH_MODE)
    image_size = [112, 112]
    time0 = datetime.datetime.now()
    model = iresnet100()
    param_dict = load_checkpoint(args.ckpt_url)
    load_param_into_net(model, param_dict)
    time_now = datetime.datetime.now()
    diff = time_now - time0
    print('model loading time', diff.total_seconds())

    ver_list = []
    ver_name_list = []
    for name in args.target.split(','):
        path = os.path.join(args.eval_url, name + ".bin")
        if os.path.exists(path):
            print('loading.. ', name)
            data_set = load_bin(path, image_size)
            ver_list.append(data_set)
            ver_name_list.append(name)

    length = len(ver_list)
    for i in range(length):
        acc1, std1, acc2, std2, xnorm, _ = test(
            ver_list[i], model, args.batch_size, args.nfolds)
        print('[%s]XNorm: %f' % (ver_name_list[i], xnorm))
        print('[%s]Accuracy: %1.5f+-%1.5f' % (ver_name_list[i], acc1, std1))
        print('[%s]Accuracy-Flip: %1.5f+-%1.5f' %
              (ver_name_list[i], acc2, std2))


if __name__ == '__main__':
    main()
