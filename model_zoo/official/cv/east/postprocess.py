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
import os
import time
import argparse
import shutil
import math
import subprocess
from PIL import ImageDraw
import numpy as np

import mindspore.ops as P

import lanms


parser = argparse.ArgumentParser(description="east inference")
parser.add_argument("--result_path", type=str, required=True, help="result files path.")
args = parser.parse_args()


def get_rotate_mat(theta):
    '''positive theta value means rotate clockwise'''
    return np.array([[math.cos(theta), -math.sin(theta)],
                     [math.sin(theta), math.cos(theta)]])


def is_valid_poly(res, score_shape, scale):
    """check if the poly in image scope
        Input:
                res        : restored poly in original image
                score_shape: score map shape
                scale      : feature map -> image
        Output:
                True if valid
        """
    cnt = 0
    for i in range(res.shape[1]):
        if res[0, i] < 0 or res[0, i] >= score_shape[1] * scale or \
                res[1, i] < 0 or res[1, i] >= score_shape[0] * scale:
            cnt += 1
    return cnt <= 1


def restore_polys(valid_pos, valid_geo, score_shape, scale=4):
    """restore polys from feature maps in given positions
        Input:
                valid_pos  : potential text positions <numpy.ndarray, (n,2)>
                valid_geo  : geometry in valid_pos <numpy.ndarray, (5,n)>
                score_shape: shape of score map
                scale      : image / feature map
        Output:
                restored polys <numpy.ndarray, (n,8)>, index
    """
    polys = []
    index = []
    valid_pos *= scale
    d = valid_geo[:4, :]  # 4 x N
    angle = valid_geo[4, :]  # N,

    for i in range(valid_pos.shape[0]):
        x = valid_pos[i, 0]
        y = valid_pos[i, 1]
        y_min = y - d[0, i]
        y_max = y + d[1, i]
        x_min = x - d[2, i]
        x_max = x + d[3, i]
        rotate_mat = get_rotate_mat(-angle[i])

        temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x
        temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y
        coordidates = np.concatenate((temp_x, temp_y), axis=0)
        res = np.dot(rotate_mat, coordidates)
        res[0, :] += x
        res[1, :] += y

        if is_valid_poly(res, score_shape, scale):
            index.append(i)
            polys.append([res[0, 0], res[1, 0], res[0, 1], res[1, 1],
                          res[0, 2], res[1, 2], res[0, 3], res[1, 3]])
    return np.array(polys), index


def get_boxes(score, geo, score_thresh=0.9, nms_thresh=0.2):
    """get boxes from feature map
        Input:
                score       : score map from model <numpy.ndarray, (1,row,col)>
                geo         : geo map from model <numpy.ndarray, (5,row,col)>
                score_thresh: threshold to segment score map
                nms_thresh  : threshold in nms
        Output:
                boxes       : final polys <numpy.ndarray, (n,9)>
        """
    score = score[0, :, :]
    xy_text = np.argwhere(score > score_thresh)  # n x 2, format is [r, c]
    if xy_text.size == 0:
        return None

    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    valid_pos = xy_text[:, ::-1].copy()  # n x 2, [x, y]
    valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]]  # 5 x n
    polys_restored, index = restore_polys(valid_pos, valid_geo, score.shape)
    if polys_restored.size == 0:
        return None

    boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = polys_restored
    boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)
    return boxes


def adjust_ratio(boxes, ratio_w, ratio_h):
    """refine boxes
        Input:
                boxes  : detected polys <numpy.ndarray, (n,9)>
                ratio_w: ratio of width
                ratio_h: ratio of height
        Output:
                refined boxes
        """
    if boxes is None or boxes.size == 0:
        return None
    boxes[:, [0, 2, 4, 6]] /= ratio_w
    boxes[:, [1, 3, 5, 7]] /= ratio_h
    return np.around(boxes)


def detect(img):
    """detect text regions of img using model
        Input:
                img   : PIL Image
                model : detection model
                device: gpu if gpu is available
        Output:
                detected polys
        """
    img, ratio_h, ratio_w = resize_img(img)
    score, geo = model(load_pil(img))
    score = P.Squeeze(0)(score)
    geo = P.Squeeze(0)(geo)
    boxes = get_boxes(score.asnumpy(), geo.asnumpy())
    return adjust_ratio(boxes, ratio_w, ratio_h)


def plot_boxes(img, boxes):
    """plot boxes on image
        """
    if boxes is None:
        return img

    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.polygon([box[0], box[1], box[2], box[3], box[4],
                      box[5], box[6], box[7]], outline=(0, 255, 0))
    return img

def detect_dataset(result_path, submit_path):
    """detection on whole dataset, save .txt results in submit_path
        Input:
                model        : detection model
                device       : gpu if gpu is available
                test_img_path: dataset path
                submit_path  : submit result for evaluation
        """
    img_files = os.listdir(result_path)
    img_files = sorted([os.path.join(result_path, img_file)
                        for img_file in img_files])
    n = len(img_files)

    for i in range(0, n, 2):
        print('evaluating {} image'.format(i/2), end='\r')

        score = np.fromfile(img_files[i], dtype=np.float32).reshape(1, 176, 320)
        geo = np.fromfile(img_files[i+1], dtype=np.float32).reshape(5, 176, 320)

        boxes = get_boxes(score, geo)
        boxes = adjust_ratio(boxes, 1, 0.97777777777)
        seq = []
        if boxes is not None:
            seq.extend([','.join([str(int(b))
                                  for b in box[:-1]]) + '\n' for box in boxes])
        with open(os.path.join(submit_path, 'res_' +
                               os.path.basename(img_files[i]).replace('_0.bin', '.txt')), 'w') as f:
            f.writelines(seq)

def eval_model(img_path, submit, save_flag=True):
    if os.path.exists(submit):
        shutil.rmtree(submit)
    os.mkdir(submit)

    start_time = time.time()
    detect_dataset(img_path, submit)
    os.chdir(submit)
    res = subprocess.getoutput('zip -q submit.zip *.txt')
    res = subprocess.getoutput('mv submit.zip ../')
    os.chdir('../')
    res = subprocess.getoutput(
        'python ./evaluate/script.py -g=./evaluate/gt.zip -s=./submit.zip')
    print(res)
    os.remove('./submit.zip')
    print('eval time is {}'.format(time.time() - start_time))

    if not save_flag:
        shutil.rmtree(submit)


if __name__ == '__main__':
    eval_model(args.result_path, './submit')
