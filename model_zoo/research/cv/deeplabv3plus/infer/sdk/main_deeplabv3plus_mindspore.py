# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# less required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""the main sdk infer file"""
import argparse
import base64
import json
import os

import cv2
import numpy as np
from StreamManagerApi import MxDataInput
from StreamManagerApi import StreamManagerApi

from get_dataset_colormap import label_to_color_image

PIPELINE_PATH = "./config/deeplabv3plus_mindspore.pipeline"
INFER_RESULT_DIR = "./result"


def _parse_args():
    parser = argparse.ArgumentParser('mindspore deeplabv3plus eval')
    parser.add_argument('--data_root', type=str, default='',
                        help='root path of val data')
    parser.add_argument('--data_lst', type=str, default='',
                        help='list of val data')
    parser.add_argument('--num_classes', type=int, default=21,
                        help='number of classes')
    args, _ = parser.parse_known_args()
    return args


def _cal_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(
        n * a[k].astype(np.int32) + b[k], minlength=n ** 2).reshape(n, n)


def _init_stream(pipeline_path):
    """
    initial sdk stream before inference

    Returns:
        stream manager api
    """
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        raise RuntimeError(f"Failed to init stream manager, ret={ret}")

    with open(pipeline_path, 'rb') as f:
        pipeline_str = f.read()

        ret = stream_manager_api.CreateMultipleStreams(pipeline_str)
        if ret != 0:
            raise RuntimeError(f"Failed to create stream, ret={ret}")
        return stream_manager_api


def _do_infer(stream_manager_api, data_input):
    """
    send images into stream to do infer

    Returns:
        infer result, numpy array
    """
    stream_name = b'segmentation'
    unique_id = stream_manager_api.SendDataWithUniqueId(
        stream_name, 0, data_input)
    if unique_id < 0:
        raise RuntimeError("Failed to send data to stream.")

    timeout = 3000
    infer_result = stream_manager_api.GetResultWithUniqueId(
        stream_name, unique_id, timeout)
    if infer_result.errorCode != 0:
        raise RuntimeError(
            "GetResultWithUniqueId error, errorCode=%d, errorMsg=%s" % (
                infer_result.errorCode, infer_result.data.decode()))

    load_dict = json.loads(infer_result.data.decode())
    image_mask = load_dict["MxpiImageMask"][0]
    data_str = base64.b64decode(image_mask['dataStr'])
    shape = image_mask['shape']
    return np.frombuffer(data_str, dtype=np.uint8).reshape(shape)


def main():
    args = _parse_args()

    stream_manager_api = _init_stream(PIPELINE_PATH)
    if not stream_manager_api:
        exit(1)

    with open(args.data_lst) as f:
        img_lst = f.readlines()

        os.makedirs(INFER_RESULT_DIR, exist_ok=True)
        data_input = MxDataInput()
        hist = np.zeros((args.num_classes, args.num_classes))
        for _, line in enumerate(img_lst):
            img_path, msk_path = line.strip().split(' ')
            img_path = os.path.join(args.data_root, img_path)
            msk_path = os.path.join(args.data_root, msk_path)
            print("img_path:", img_path)
            msk_ = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
            with open(img_path, 'rb') as f:
                data_input.data = f.read()
            each_array = _do_infer(stream_manager_api, data_input)

            hist += _cal_hist(
                msk_.flatten(), each_array.flatten(), args.num_classes)
            color_mask_res = label_to_color_image(each_array)
            color_mask_res = cv2.cvtColor(color_mask_res.astype(np.uint8),
                                          cv2.COLOR_RGBA2BGR)
            result_path = os.path.join(
                INFER_RESULT_DIR,
                f"{img_path.split('/')[-1].split('.')[0]}.png")
            cv2.imwrite(result_path, color_mask_res)
        iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        print("per-class IoU", iou)
        print("mean IoU", np.nanmean(iou))

    stream_manager_api.DestroyAllStreams()


if __name__ == '__main__':
    main()
