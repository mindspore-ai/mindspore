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
"""eval blfur"""
import os
import argparse
import cv2
import numpy as np
import scipy.io
# from numba import jit

from mindspore import context, load_param_into_net, load_checkpoint, Tensor
from mindspore.common import dtype as mstype

from src.lightcnn import lightCNN_9Layers4Test
from src.config import lightcnn_cfg as cfg

parser = argparse.ArgumentParser(description='PyTorch ImageNet Feature Extracting')
parser.add_argument('--device_target', default='Ascend', choices=['Ascend', 'GPU', 'CPU'], type=str)
parser.add_argument('--device_id', default=0, type=int)
parser.add_argument('--resume', default='',
                    type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--num_classes', default=79077, type=int,
                    metavar='N', help='number of classes (default: 79077)')

args = parser.parse_args()


def extract_feature(img_list):
    """extract features from model's predictions"""
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
    """get image list"""
    img_dir_cp = img_dir.replace('/image', '')
    list_file_path = os.path.join(img_dir_cp, list_file_name)
    f = open(list_file_path, 'r')
    image_list = []
    for line in f:
        img_name = line[:-4]
        person_name = line[:img_name.rfind('_')]
        path = person_name + '/' + img_name + 'bmp'
        image_list.append(path)
    return image_list


def string_list_to_cells(lst):
    """
    Uses numpy.ndarray with dtype=object. Convert list to np.ndarray().
    """
    cells = np.ndarray(len(lst), dtype='object')
    for idx, ele in enumerate(lst):
        cells[idx] = ele
    return cells


def extract_features_to_dic(image_dir, list_file):
    """extract features and save them in dict"""
    img_list = load_image_list(image_dir, list_file)
    ftr = extract_feature(img_list)
    dic = {'Descriptors': ftr}
    return dic


def compute_cosine_score(feature1, feature2):
    """compute cosine score"""
    feature1_norm = np.linalg.norm(feature1)
    feature2_norm = np.linalg.norm(feature2)
    score = np.dot(feature1, feature2) / (feature1_norm * feature2_norm)
    return score


def normr(data):
    """compute normr"""
    ratio = np.sqrt(np.sum(np.power(data, 2)))
    return data / ratio


# @jit(nopython=True)
def bsxfun_eq(galLabels, probLabels, binaryLabels):
    """get bsxfun_eq"""
    for idx1, ele1 in enumerate(galLabels):
        for idx2, ele2 in enumerate(probLabels):
            binaryLabels[idx1, idx2] = 1 if ele1 == ele2 else 0
    return binaryLabels


# @jit(nopython=True)
def bsxfun_eq2(galLabels, probLabels, binaryLabels):
    """get bsxfun_eq2"""
    for i, _ in enumerate(galLabels):
        for j, ele in enumerate(probLabels):
            binaryLabels[i, j] = 1 if galLabels[i, j] == ele else 0
    return binaryLabels


# @jit(nopython=True)
def bsxfun_ge(genScore, thresholds):
    """get bsxfun_ge"""
    temp = np.zeros((len(genScore), len(thresholds)))
    for i, ele1 in enumerate(genScore):
        for j, ele2 in enumerate(thresholds):
            temp[i, j] = 1 if ele1 >= ele2 else 0
    return temp


# @jit(nopython=True)
def bsxfun_le(genScore, thresholds):
    """get bsxfun_le"""
    temp = np.zeros((len(genScore), len(thresholds)))
    for i, ele1 in enumerate(genScore):
        for j, ele2 in enumerate(thresholds):
            temp[i, j] = 1 if ele1 <= ele2 else 0
    return temp


# @jit(nopython=True)
def bsxfun_and(T1, T2):
    """get bsxfun_and"""
    temp = np.zeros((T2.shape[0], T2.shape[1], T1.shape[1]))
    for i in range(temp.shape[0]):
        for j in range(temp.shape[1]):
            for k in range(temp.shape[2]):
                temp[i, j, k] = 1 if T1[i, k] * T2[i, j] != 0 else 0
    return temp


def ismember(a, b):
    """get bsxfun_and"""
    tf = np.in1d(a, b)
    index = np.array([(np.where(b == i))[0][-1] if t else 0 for i, t in zip(a, tf)])
    return tf, index


def EvalROC(score, galLabels, farPoints):
    """eval ROC"""
    probLabels = galLabels
    scoreMask = np.tril(np.ones_like(score), k=-1)
    binaryLabels = np.zeros_like(score)
    binaryLabels = bsxfun_eq(galLabels, probLabels, binaryLabels)

    score_ = score[scoreMask == 1]
    binaryLabels_ = binaryLabels[scoreMask == 1]

    genScore = score_[binaryLabels_ == 1]
    impScore = score_[binaryLabels_ == 0]
    del score, score_, binaryLabels, binaryLabels_

    Nimp = len(impScore)
    falseAlarms = np.round(farPoints * Nimp)

    impScore = np.sort(impScore)
    impScore = impScore[::-1]

    isZeroFAR = np.zeros_like(falseAlarms)
    isZeroFAR[np.squeeze(np.where(falseAlarms == 0))] = 1

    isOneFAR = np.zeros_like(falseAlarms)
    isOneFAR[np.squeeze(np.where(falseAlarms == Nimp))] = 1

    thresholds = np.zeros_like(falseAlarms)
    for i, _ in enumerate(isZeroFAR):
        thresholds[i] = impScore[int(falseAlarms[i]) - 1] if isZeroFAR[i] != 1 and isOneFAR[i] != 1 else 0

    highGenScore = genScore[genScore > impScore[0]]
    eps = 1.490116119384766e-08
    if highGenScore.size:
        thresholds[isZeroFAR == 1] = (impScore[0] + np.min(highGenScore)) / 2
    else:
        thresholds[isZeroFAR == 1] = impScore[0] + eps

    thresholds[isOneFAR == 1] = np.minimum(impScore[-1], np.min(genScore)) - np.sqrt(eps)

    FAR = falseAlarms / Nimp
    VR = np.mean(bsxfun_ge(genScore, thresholds), axis=0)

    return VR, FAR


def OpenSetROC(score, galLabels, probLabels, farPoints):
    """open set ROC"""
    rankPoints = np.zeros(19)
    for i in range(10):
        rankPoints[i] = i + 1
        rankPoints[i + 9] = (i + 1) * 10
    probLabels = probLabels.T

    binaryLabels = np.zeros_like(score)
    binaryLabels = bsxfun_eq(galLabels, probLabels, binaryLabels)

    t = np.any(binaryLabels, axis=0)
    genProbIndex = np.squeeze(np.where(t))
    impProbIndex = np.squeeze(np.where(~t))
    # Ngen = len(genProbIndex)
    Nimp = len(impProbIndex)
    falseAlarms = np.round(farPoints * Nimp)

    # get detection scores and matching ranks of each probe
    impScore = [np.max(score[:, i]) for i in impProbIndex]
    impScore = np.sort(impScore)
    impScore = impScore[::-1]

    S = np.zeros((score.shape[0], len(genProbIndex)))
    for i, ele in enumerate(genProbIndex):
        S[:, i] = score[:, ele]
    sortedIndex = np.argsort(S, axis=0)
    sortedIndex = np.flipud(sortedIndex)
    M = np.zeros((binaryLabels.shape[0], len(genProbIndex)))
    for i, ele in enumerate(genProbIndex):
        M[:, i] = binaryLabels[:, ele]
    del binaryLabels
    S[M == 0] = -np.Inf
    del M
    genScore, genGalIndex = np.max(S, axis=0), np.argmax(S, axis=0)
    del S
    temp = np.zeros_like(sortedIndex)
    temp = bsxfun_eq2(sortedIndex, genGalIndex, temp)
    probRanks = (temp != 0).argmax(axis=0)
    del sortedIndex

    # compute thresholds
    isZeroFAR = np.zeros_like(falseAlarms)
    isZeroFAR[np.squeeze(np.where(falseAlarms == 0))] = 1

    isOneFAR = np.zeros_like(falseAlarms)
    isOneFAR[np.squeeze(np.where(falseAlarms == Nimp))] = 1

    thresholds = np.zeros_like(falseAlarms)
    for i, _ in enumerate(isZeroFAR):
        thresholds[i] = impScore[int(falseAlarms[i]) - 1] if isZeroFAR[i] != 1 and isOneFAR[i] != 1 else 0

    highGenScore = genScore[genScore > impScore[0]]
    eps = 1.490116119384766e-08
    if highGenScore.size:
        thresholds[isZeroFAR == 1] = (impScore[0] + np.min(highGenScore)) / 2
    else:
        thresholds[isZeroFAR == 1] = impScore[0] + eps

    thresholds[isOneFAR == 1] = np.minimum(impScore[-1], np.min(genScore)) - np.sqrt(eps)

    # evaluate
    genScore = genScore.T
    T1 = bsxfun_ge(genScore, thresholds)
    T2 = bsxfun_le(probRanks, rankPoints)
    T = bsxfun_and(T1, T2)
    DIR = np.squeeze(np.mean(T, axis=0))
    FAR = falseAlarms / Nimp
    return DIR, FAR


def blufr_eval(lightcnn_result, config_file_path):
    """eval blufr"""
    Descriptors = lightcnn_result['Descriptors']
    config_file = scipy.io.loadmat(config_file_path)
    testIndex = config_file['testIndex']
    galIndex = config_file['galIndex']
    probIndex = config_file['probIndex']
    labels = config_file['labels']

    veriFarPoints = [0]
    for i in range(1, 9):
        for j in range(1, 10):
            veriFarPoints.append(round(j * pow(10, i - 9), 9 - i))
    veriFarPoints.append(1)
    veriFarPoints = np.array(veriFarPoints)

    osiFarPoints = [0]
    for i in range(1, 5):
        for j in range(1, 10):
            osiFarPoints.append(round(j * pow(10, i - 5), 5 - i))
    osiFarPoints.append(1)
    osiFarPoints = np.array(osiFarPoints)

    rankPoints = []
    for i in range(0, 2):
        for j in range(1, 10):
            rankPoints.append(j * pow(10, i))
    rankPoints.append(100)
    rankPoints = np.array(rankPoints)

    reportVeriFar = 0.001
    reportOsiFar = 0.01
    reportRank = 1

    numTrials = len(testIndex)
    numVeriFarPoints = len(veriFarPoints)

    VR = np.zeros((numTrials, numVeriFarPoints))
    veriFAR = np.zeros((numTrials, numVeriFarPoints))

    numOsiFarPoints = len(osiFarPoints)
    numRanks = len(rankPoints)

    DIR = np.zeros((numRanks, numOsiFarPoints, numTrials))
    osiFAR = np.zeros((numTrials, numOsiFarPoints))

    veriFarIndex = np.squeeze(np.where(veriFarPoints == reportVeriFar))
    osiFarIndex = np.squeeze(np.where(osiFarPoints == reportOsiFar))
    rankIndex = np.squeeze(np.where(rankPoints == reportRank))

    for t in range(numTrials):
        print('Processing with trail %s ...' % str(t + 1))
        idx_list = testIndex[t][0]
        X = np.zeros((len(idx_list), 256))
        for k, ele in enumerate(idx_list):
            data = Descriptors[np.squeeze(ele) - 1, :]
            X[k, :] = normr(data)
        score = np.dot(X, X.T)

        testLabels = np.zeros(len(idx_list), dtype=np.int)
        for k, ele in enumerate(idx_list):
            testLabels[k] = labels[np.squeeze(ele) - 1]

        VR[t, :], veriFAR[t, :] = EvalROC(score, testLabels, veriFarPoints)

        _, gIdx = ismember(galIndex[t][0], testIndex[t][0])
        _, pIdx = ismember(probIndex[t][0], testIndex[t][0])

        score_sub = np.zeros((len(gIdx), len(pIdx)))
        for i, ele1 in enumerate(gIdx):
            for j, ele2 in enumerate(pIdx):
                score_sub[i, j] = score[ele1, ele2]

        testLabels_gIdx = np.zeros(len(gIdx), dtype=np.int)
        for i, ele in enumerate(gIdx):
            testLabels_gIdx[i] = testLabels[ele]

        testLabels_pIdx = np.zeros(len(pIdx), dtype=np.int)
        for i, ele in enumerate(pIdx):
            testLabels_pIdx[i] = testLabels[ele]

        DIR[:, :, t], osiFAR[t, :] = OpenSetROC(score_sub, testLabels_gIdx, testLabels_pIdx, osiFarPoints)

        print('Verification:')
        print('\t@ FAR = %s%%: VR = %.4f%%' % (reportVeriFar * 100, VR[t, veriFarIndex] * 100))

        print('Open-set Identification:')
        print('\t@ Rank = %d, FAR = %s%%: DIR = %.4f%%\n'
              % (reportRank, reportOsiFar * 100, DIR[rankIndex, osiFarIndex, t] * 100))

        del X, score

    # meanVerFAR = np.mean(veriFAR, axis=0)
    meanVR = np.mean(VR, axis=0)
    stdVR = np.std(VR, axis=0)
    reportMeanVR = meanVR[veriFarIndex]
    reportStdVR = stdVR[veriFarIndex]

    # meanOsiFAR = np.mean(osiFAR, axis=0)
    meanDIR = np.mean(DIR, axis=2)
    stdDIR = np.std(DIR, axis=2)
    reportMeanDIR = meanDIR[rankIndex, osiFarIndex]
    reportStdDIR = stdDIR[rankIndex, osiFarIndex]

    # Get the mu - sigma performance measures
    # fusedVR = (meanVR - stdVR) * 100
    reportVR = (reportMeanVR - reportStdVR) * 100
    # fusedDIR = (meanDIR - stdDIR) * 100
    reportDIR = (reportMeanDIR - reportStdDIR) * 100

    # Display the benchmark performance
    print('Verification:')
    print('\t@ FAR = %s%%: VR = %.2f%%' % (reportVeriFar * 100, reportVR))
    print('\t@ Rank = %d, FAR = %s%%: DIR = %.2f%%.' % (reportRank, reportOsiFar * 100, reportDIR))


if __name__ == '__main__':
    device_id = int(os.getenv('DEVICE_ID'))
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=device_id)
    feature_dict = extract_features_to_dic(image_dir=cfg.root_path, list_file=cfg.blufr_img_list)
    blufr_eval(feature_dict, config_file_path=cfg.blufr_config_mat_path)
