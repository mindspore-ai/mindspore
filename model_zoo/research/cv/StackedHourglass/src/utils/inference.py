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
"""
inference module
"""
import copy
import os

import cv2
import h5py
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
import tqdm

import src.dataset.MPIIDataLoader as ds
import src.utils.img
from src.config import parse_args

args = parse_args()


def match_format(dic):
    """
    get match format
    """
    loc = dic["loc_k"][0, :, 0, :]
    val = dic["val_k"][0, :, :]
    ans = np.hstack((loc, val))
    ans = np.expand_dims(ans, axis=0)
    ret = []
    ret.append(ans)
    return ret


class MaxPool2dFilter(nn.Cell):
    """
    maxpool 2d for filter
    """

    def __init__(self):
        super(MaxPool2dFilter, self).__init__()
        self.pool = nn.MaxPool2d(3, 1, "same")
        self.eq = ops.Equal()

    def construct(self, x):
        """
        forward
        """
        maxm = self.pool(x)
        return self.eq(maxm, x)


class HeatmapParser:
    """
    parse heatmap
    """

    def __init__(self):
        self.maxpool2dfilter = MaxPool2dFilter().to_float(mindspore.float16)  # avoid float error
        self.topk = ops.TopK(sorted=True)
        self.stack = ops.Stack(axis=3)

    def nms(self, det):
        """
        keep max in 3x3
        """
        maxm = self.maxpool2dfilter(det)
        det = det * maxm
        return det

    def calc(self, det):
        """
        calc distance
        """
        det = self.nms(det)
        w = det.shape[3]
        det = det.view(det.shape[0], det.shape[1], -1)
        val_k, ind = self.topk(det, 1)

        x = ind % w
        y = (ind.astype(mindspore.float32) / w).astype(mindspore.int32)
        ind_k = self.stack((x, y))
        ans = {"loc_k": ind_k, "val_k": val_k}
        return {key: ans[key].asnumpy() for key in ans}

    def adjust(self, ans, det):
        """
        adjust for joint
        """
        for people in ans:
            for i in people:
                for joint_id, joint in enumerate(i):
                    if joint[2] > 0:
                        y, x = joint[0:2]
                        xx, yy = int(x), int(y)
                        tmp = det[0][joint_id]
                        if tmp[xx, min(yy + 1, tmp.shape[1] - 1)] > tmp[xx, max(yy - 1, 0)]:
                            y += 0.25
                        else:
                            y -= 0.25

                        if tmp[min(xx + 1, tmp.shape[0] - 1), yy] > tmp[max(0, xx - 1), yy]:
                            x += 0.25
                        else:
                            x -= 0.25
                        ans[0][0, joint_id, 0:2] = (y + 0.5, x + 0.5)
        return ans

    def parse(self, det, adjust=True):
        """
        parse heatmap
        """
        ans = match_format(self.calc(det))
        if adjust:
            ans = self.adjust(ans, det)
        return ans


parser = HeatmapParser()


def post_process(det, mat_, trainval, c=None, s=None, resolution=None):
    """
    post process for parser
    """
    mat = np.linalg.pinv(np.array(mat_).tolist() + [[0, 0, 1]])[:2]
    cropped_preds = parser.parse(mindspore.Tensor(np.float32([det])))[0]

    if cropped_preds.size > 0:
        cropped_preds[:, :, :2] = src.utils.img.kpt_affine(cropped_preds[:, :, :2] * 4, mat)  # size 1x16x3

    preds = np.copy(cropped_preds)
    # revert to origin image
    if trainval != "cropped":
        for j in range(preds.shape[1]):
            preds[0, j, :2] = src.utils.img.transform(preds[0, j, :2], c, s, resolution, invert=1)
    return preds


def inference(img, net, c, s):
    """
    forward pass at test time
    calls post_process to post process results
    """

    height, width = img.shape[0:2]
    center = (width / 2, height / 2)
    scale = max(height, width) / 200
    res = (args.input_res, args.input_res)

    mat_ = src.utils.img.get_transform(center, scale, res)[:2]
    inp = img / 255

    tmp1 = net(mindspore.Tensor([inp], dtype=mindspore.float32)).asnumpy()
    tmp2 = net(mindspore.Tensor([inp[:, ::-1]], dtype=mindspore.float32)).asnumpy()

    tmp = np.concatenate((tmp1, tmp2), axis=0)

    det = tmp[0, -1] + tmp[1, -1, :, :, ::-1][ds.flipped_parts["mpii"]]

    if det is None:
        return [], []
    det = det / 2

    det = np.minimum(det, 1)

    return post_process(det, mat_, "valid", c, s, res)


class MPIIEval:
    """
    eval for MPII dataset
    """

    template = {
        "all": {
            "total": 0,
            "ankle": 0,
            "knee": 0,
            "hip": 0,
            "pelvis": 0,
            "thorax": 0,
            "neck": 0,
            "head": 0,
            "wrist": 0,
            "elbow": 0,
            "shoulder": 0,
        },
        "visible": {
            "total": 0,
            "ankle": 0,
            "knee": 0,
            "hip": 0,
            "pelvis": 0,
            "thorax": 0,
            "neck": 0,
            "head": 0,
            "wrist": 0,
            "elbow": 0,
            "shoulder": 0,
        },
        "not visible": {
            "total": 0,
            "ankle": 0,
            "knee": 0,
            "hip": 0,
            "pelvis": 0,
            "thorax": 0,
            "neck": 0,
            "head": 0,
            "wrist": 0,
            "elbow": 0,
            "shoulder": 0,
        },
    }

    joint_map = [
        "ankle",
        "knee",
        "hip",
        "hip",
        "knee",
        "ankle",
        "pelvis",
        "thorax",
        "neck",
        "head",
        "wrist",
        "elbow",
        "shoulder",
        "shoulder",
        "elbow",
        "wrist",
    ]

    def __init__(self):
        self.correct = copy.deepcopy(self.template)
        self.count = copy.deepcopy(self.template)
        self.correct_train = copy.deepcopy(self.template)
        self.count_train = copy.deepcopy(self.template)

    def eval(self, pred, gt, normalizing, num_train, bound=0.5):
        """
        use PCK with threshold of .5 of normalized distance (presumably head size)
        """
        idx = 0
        for p, g, normalize in zip(pred, gt, normalizing):
            for j in range(g.shape[1]):
                vis = "visible"
                if g[0, j, 0] == 0:  # Not in image
                    continue
                if g[0, j, 2] == 0:
                    vis = "not visible"
                joint = self.joint_map[j]

                if idx >= num_train:
                    self.count["all"]["total"] += 1
                    self.count["all"][joint] += 1
                    self.count[vis]["total"] += 1
                    self.count[vis][joint] += 1
                else:
                    self.count_train["all"]["total"] += 1
                    self.count_train["all"][joint] += 1
                    self.count_train[vis]["total"] += 1
                    self.count_train[vis][joint] += 1
                error = np.linalg.norm(p[0]["keypoints"][j, :2] - g[0, j, :2]) / normalize
                if idx >= num_train:
                    if bound > error:
                        self.correct["all"]["total"] += 1
                        self.correct["all"][joint] += 1
                        self.correct[vis]["total"] += 1
                        self.correct[vis][joint] += 1
                else:
                    if bound > error:
                        self.correct_train["all"]["total"] += 1
                        self.correct_train["all"][joint] += 1
                        self.correct_train[vis]["total"] += 1
                        self.correct_train[vis][joint] += 1
            idx += 1

        self.output_result(bound)

    def output_result(self, bound):
        """
        output split via train/valid
        """
        for k in self.correct:
            print(k, ":")
            for key in self.correct[k]:
                print(
                    "Val PCK @,",
                    bound,
                    ",",
                    key,
                    ":",
                    round(self.correct[k][key] / max(self.count[k][key], 1), 3),
                    ", count:",
                    self.count[k][key],
                )
                print(
                    "Tra PCK @,",
                    bound,
                    ",",
                    key,
                    ":",
                    round(self.correct_train[k][key] / max(self.count_train[k][key], 1), 3),
                    ", count:",
                    self.count_train[k][key],
                )
            print("\n")


def get_img(num_eval=2958, num_train=300):
    """
    load validation and training images
    """
    input_res = args.input_res
    val_f = h5py.File(os.path.join(args.annot_dir, "valid.h5"), "r")

    tr = tqdm.tqdm(range(0, num_train), total=num_train)
    # Train set
    train_f = h5py.File(os.path.join(args.annot_dir, "train.h5"), "r")
    for i in tr:
        path_t = "%s/%s" % (args.img_dir, train_f["imgname"][i].decode("UTF-8"))

        orig_img = cv2.imread(path_t)[:, :, ::-1]
        c = train_f["center"][i]
        s = train_f["scale"][i]
        im = src.utils.img.crop(orig_img, c, s, (input_res, input_res))

        kp = train_f["part"][i]
        vis = train_f["visible"][i]
        kp2 = np.insert(kp, 2, vis, axis=1)
        kps = np.zeros((1, 16, 3))
        kps[0] = kp2

        n = train_f["normalize"][i]

        yield kps, im, c, s, n

    tr2 = tqdm.tqdm(range(0, num_eval), total=num_eval)
    # Valid
    for i in tr2:
        path_t = "%s/%s" % (args.img_dir, val_f["imgname"][i].decode("UTF-8"))

        orig_img = cv2.imread(path_t)[:, :, ::-1]
        c = val_f["center"][i]
        s = val_f["scale"][i]
        im = src.utils.img.crop(orig_img, c, s, (input_res, input_res))

        kp = val_f["part"][i]
        vis = val_f["visible"][i]
        kp2 = np.insert(kp, 2, vis, axis=1)
        kps = np.zeros((1, 16, 3))
        kps[0] = kp2

        n = val_f["normalize"][i]

        yield kps, im, c, s, n
