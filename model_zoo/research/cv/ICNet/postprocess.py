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
"""Evaluate mIou and Pixacc"""
import os
import time
import sys
import argparse
import yaml
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description="ICNet Evaluation")
parser.add_argument("--dataset_path", type=str, default="/home/dataset",
                    help="dataset path for evaluation")
parser.add_argument("--project_path", type=str, default='/home/ICNet',
                    help="project_path")
parser.add_argument("--device_id", type=int, default=5, help="Device id, default is 5.")
parser.add_argument("--result_path", type=str, default="", help="Image path.")

args_opt = parser.parse_args()


class Evaluator:
    """evaluate"""

    def __init__(self, config):
        self.cfg = config

        self.mask_folder = '/home/data'

        # evaluation metrics
        self.metric = SegmentationMetric(19)

    def eval(self):
        """evaluate"""
        self.metric.reset()

        list_time = []

        for root, _, files in os.walk(args_opt.dataset_path):
            for filename in files:
                if filename.endswith('.png'):
                    img_path = os.path.join(root, filename)
                    file_name = filename.split('.')[0]
                    output_file = os.path.join(args_opt.result_path, file_name + "_0.bin")
                    output = np.fromfile(output_file, dtype=np.float32).reshape(1, 19, 1024, 2048)
                    folder_name = os.path.basename(os.path.dirname(img_path))
                    mask_name = filename.replace('leftImg8bit', 'gtFine_labelIds')
                    mask_file = os.path.join(self.mask_folder, folder_name, mask_name)
                    mask = Image.open(mask_file)  # mask shape: (W,H)

                    mask = self._mask_transform(mask)  # mask shape: (H,w)

                    start_time = time.time()
                    end_time = time.time()
                    step_time = end_time - start_time

                    mask = np.expand_dims(mask, axis=0)
                    self.metric.update(output, mask)
                    list_time.append(step_time)

        mIoU, pixAcc = self.metric.get()

        average_time = sum(list_time) / len(list_time)

        print("avgmiou", mIoU)
        print("avg_pixacc", pixAcc)
        print("avgtime", average_time)

    def _mask_transform(self, mask):
        mask = self._class_to_index(np.array(mask).astype('int32'))
        return np.array(mask).astype('int32')

    def _class_to_index(self, mask):
        """assert the value"""
        values = np.unique(mask)
        self._key = np.array([-1, -1, -1, -1, -1, -1,
                              -1, -1, 0, 1, -1, -1,
                              2, 3, 4, -1, -1, -1,
                              5, -1, 6, 7, 8, 9,
                              10, 11, 12, 13, 14, 15,
                              -1, -1, 16, 17, 18])
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32')
        for value in values:
            assert value in self._mapping
        # Get the index of each pixel value in the mask corresponding to _mapping
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        # According to the above index index, according to _key, the corresponding mask image is obtained
        return self._key[index].reshape(mask.shape)


if __name__ == '__main__':
    sys.path.append(args_opt.project_path)
    from src.metric import SegmentationMetric
    from src.logger import SetupLogger
    # Set config file
    config_file = "src/model_utils/icnet.yaml"
    config_path = os.path.join(args_opt.project_path, config_file)
    with open(config_path, "r") as yaml_file:
        cfg = yaml.load(yaml_file.read())
    logger = SetupLogger(name="semantic_segmentation",
                         save_dir=cfg["train"]["ckpt_dir"],
                         distributed_rank=0,
                         filename='{}_{}_evaluate_log.txt'.format(cfg["model"]["name"], cfg["model"]["backbone"]))

    evaluator = Evaluator(cfg)
    evaluator.eval()
