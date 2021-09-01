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
'''pre process for 310 inference'''
import os

from PIL import Image
import numpy as np

from src.utils import modelarts_pre_process
from src.dataset import FolderImagePair, AUG_DICT
from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper

MAX_HR_SIZE = 2040


def padding(img, target_shape):
    h, w = target_shape[0], target_shape[1]
    img_h, img_w, _ = img.shape
    dh, dw = h - img_h, w - img_w
    if dh < 0 or dw < 0:
        raise RuntimeError(f"target_shape is bigger than img.shape, {target_shape} > {img.shape}")
    if dh != 0 or dw != 0:
        img = np.pad(img, ((0, dh), (0, dw), (0, 0)), "constant")
    return img


def get_lr_dataset(cfg):
    """
    get lr dataset
    """
    dataset_path = cfg.data_path
    lr_scale = cfg.scale
    lr_type = cfg.lr_type
    dataset_type = "valid"
    self_ensemble = "_self_ensemble" if cfg.self_ensemble else ""

    # get LR_PATH/X2/*x2.png, LR_PATH/X3/*x3.png, LR_PATH/X4/*x4.png
    lrs_pattern = []
    dir_lr = os.path.join(dataset_path, f"DIV2K_{dataset_type}_LR_{lr_type}", f"X{lr_scale}")
    lr_pattern = os.path.join(dir_lr, f"*x{lr_scale}.png")
    lrs_pattern.append(lr_pattern)
    save_dir = os.path.join(dataset_path, f"DIV2K_{dataset_type}_LR_{lr_type}_AUG{self_ensemble}", f"X{lr_scale}")
    os.makedirs(save_dir, exist_ok=True)
    save_format = os.path.join(save_dir, "{}" + f"x{lr_scale}" + "_{}.png")

    # make dataset
    dataset = FolderImagePair(lrs_pattern)

    return dataset, save_format


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_pre_process():
    """
    run pre process
    """
    print(config)
    cfg = config

    aug_dict = AUG_DICT
    if not cfg.self_ensemble:
        aug_dict = {"0": AUG_DICT["0"]}

    dataset, save_format = get_lr_dataset(cfg)
    for i in range(len(dataset)):
        img_key = dataset.get_key(i)
        org_img = None
        for a_key, aug in aug_dict.items():
            save_path = save_format.format(img_key, a_key)
            if os.path.isfile(save_path):
                continue
            if org_img is None:
                _, lr = dataset[i]
                target_shape = [MAX_HR_SIZE // cfg.scale, MAX_HR_SIZE // cfg.scale]
                org_img = padding(lr, target_shape)
            img = org_img.copy()
            for a in aug:
                img = a(img)
            Image.fromarray(img).save(save_path)
            print(f"[{i+1}/{len(dataset)}]\tsave {save_path}\tshape = {img.shape}", flush=True)

    print("pre_process success", flush=True)

if __name__ == "__main__":
    run_pre_process()
