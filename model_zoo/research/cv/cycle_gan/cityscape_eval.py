# Copyright 2020 Huawei Technologies Co., Ltd
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
"""Eval use cityscape dataset."""
import os
import argparse
import numpy as np

from src.dataset import make_dataset
from src.utils import CityScapes, fast_hist, get_scores

parser = argparse.ArgumentParser()
parser.add_argument("--cityscapes_dir", type=str, required=True, help="Path to the original cityscapes dataset")
parser.add_argument("--result_dir", type=str, required=True, help="Path to the generated images to be evaluated")
args = parser.parse_args()


def main():
    CS = CityScapes()
    cityscapes = make_dataset(args.cityscapes_dir)
    hist_perframe = np.zeros((CS.class_num, CS.class_num))
    for i, img_path in enumerate(cityscapes):
        if i % 100 == 0:
            print('Evaluating: %d/%d' % (i, len(cityscapes)))
        img_name = os.path.split(img_path)[1]
        ids1 = CS.get_id(os.path.join(args.cityscapes_dir, img_name))
        ids2 = CS.get_id(os.path.join(args.result_dir, img_name))
        hist_perframe += fast_hist(ids1.flatten(), ids2.flatten(), CS.class_num)

    mean_pixel_acc, mean_class_acc, mean_class_iou, per_class_acc, per_class_iou = get_scores(hist_perframe)
    print(f"mean_pixel_acc: {mean_pixel_acc}, mean_class_acc: {mean_class_acc}, mean_class_iou: {mean_class_iou}")
    with open('./evaluation_results.txt', 'w') as f:
        f.write('Mean pixel accuracy: %f\n' % mean_pixel_acc)
        f.write('Mean class accuracy: %f\n' % mean_class_acc)
        f.write('Mean class IoU: %f\n' % mean_class_iou)
        f.write('************ Per class numbers below ************\n')
        for i, cl in enumerate(CS.classes):
            while len(cl) < 15:
                cl = cl + ' '
            f.write('%s: acc = %f, iou = %f\n' % (cl, per_class_acc[i], per_class_iou[i]))

if __name__ == '__main__':
    main()
