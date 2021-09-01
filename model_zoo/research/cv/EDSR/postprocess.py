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
'''post process for 310 inference'''
import os
import math

from PIL import Image
import numpy as np
from mindspore import Tensor

from src.utils import init_env, modelarts_pre_process
from src.dataset import FolderImagePair, AUG_DICT
from src.metric import PSNR
from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper


def read_bin(bin_path):
    img = np.fromfile(bin_path, dtype=np.float32)
    num_pix = img.size
    img_shape = int(math.sqrt(num_pix // 3))
    if 1 * 3 * img_shape * img_shape != num_pix:
        raise RuntimeError(f'bin file error, it not output from edsr network, {bin_path}')
    img = img.reshape(1, 3, img_shape, img_shape)
    return img


def read_bin_as_hwc(bin_path):
    nchw_img = read_bin(bin_path)
    chw_img = np.squeeze(nchw_img)
    hwc_img = chw_img.transpose(1, 2, 0)
    return hwc_img


def unpadding(img, target_shape):
    h, w = target_shape[0], target_shape[1]
    img_h, img_w, _ = img.shape
    if img_h > h:
        img = img[:h, :, :]
    if img_w > w:
        img = img[:, :w, :]
    return img


def img_to_tensor(img):
    img = np.array([img.transpose(2, 0, 1)], np.float32)
    img = Tensor(img)
    return img


def float_to_uint8(img):
    clip_img = np.clip(img, 0, 255)
    round_img = np.round(clip_img)
    uint8_img = round_img.astype(np.uint8)
    return uint8_img


def bin_to_png(cfg):
    """
    bin from ascend310_infer outputs will be covert to png
    """
    dataset_path = cfg.data_path
    dataset_type = "valid"
    aug_keys = list(AUG_DICT.keys())
    lr_scale = cfg.scale

    if cfg.self_ensemble:
        dir_sr_bin = os.path.join(dataset_path, f"DIV2K_{dataset_type}_SR_bin", f"X{lr_scale}")
        save_sr_se_dir = os.path.join(dataset_path, f"DIV2K_{dataset_type}_SR_self_ensemble", f"X{lr_scale}")
        if os.path.isdir(dir_sr_bin):
            os.makedirs(save_sr_se_dir, exist_ok=True)
            bin_patterns = [os.path.join(dir_sr_bin, f"*x{lr_scale}_{a_key}_0.bin") for a_key in  aug_keys]
            dataset = FolderImagePair(bin_patterns, reader=read_bin_as_hwc)
            for i in range(len(dataset)):
                img_key = dataset.get_key(i)
                sr_se_path = os.path.join(save_sr_se_dir, f"{img_key}x{lr_scale}.png")
                if os.path.isfile(sr_se_path):
                    continue
                data = dataset[i]
                img_key, sr_8 = data[0], data[1:]
                sr = np.zeros_like(sr_8[0], dtype=np.float64)
                for img, a_key in zip(sr_8, aug_keys):
                    aug = AUG_DICT[a_key]
                    for a in reversed(aug):
                        img = a(img)
                    sr += img
                sr /= len(sr_8)
                sr = float_to_uint8(sr)
                Image.fromarray(sr).save(sr_se_path)
                print(f"merge sr bin save to {sr_se_path}")
        return

    if not cfg.self_ensemble:
        dir_sr_bin = os.path.join(dataset_path, f"DIV2K_{dataset_type}_SR_bin", f"X{lr_scale}")
        save_sr_dir = os.path.join(dataset_path, f"DIV2K_{dataset_type}_SR", f"X{lr_scale}")
        if os.path.isdir(dir_sr_bin):
            os.makedirs(save_sr_dir, exist_ok=True)
            bin_patterns = [os.path.join(dir_sr_bin, f"*x{lr_scale}_0_0.bin")]
            dataset = FolderImagePair(bin_patterns, reader=read_bin_as_hwc)
            for i in range(len(dataset)):
                img_key = dataset.get_key(i)
                sr_path = os.path.join(save_sr_dir, f"{img_key}x{lr_scale}.png")
                if os.path.isfile(sr_path):
                    continue
                img_key, sr = dataset[i]
                sr = float_to_uint8(sr)
                Image.fromarray(sr).save(sr_path)
                print(f"merge sr bin save to {sr_path}")
        return


def get_hr_sr_dataset(cfg):
    """
    make hr sr dataset
    """
    dataset_path = cfg.data_path
    dataset_type = "valid"
    lr_scale = cfg.scale

    dir_patterns = []

    # get HR_PATH/*.png
    dir_hr = os.path.join(dataset_path, f"DIV2K_{dataset_type}_HR")
    hr_pattern = os.path.join(dir_hr, "*.png")
    dir_patterns.append(hr_pattern)

    # get LR_PATH/X2/*x2.png, LR_PATH/X3/*x3.png, LR_PATH/X4/*x4.png
    se = "_self_ensemble" if cfg.self_ensemble else ""

    dir_sr = os.path.join(dataset_path, f"DIV2K_{dataset_type}_SR" + se, f"X{lr_scale}")
    if not os.path.isdir(dir_sr):
        raise RuntimeError(f'{dir_sr} is not a dir for saving sr')
    sr_pattern = os.path.join(dir_sr, f"*x{lr_scale}.png")
    dir_patterns.append(sr_pattern)

    # make dataset
    dataset = FolderImagePair(dir_patterns)
    return dataset


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_post_process():
    """
    run post process
    """
    print(config)
    cfg = config
    lr_scale = cfg.scale

    init_env(cfg)

    print("begin to run bin_to_png...")
    bin_to_png(cfg)
    print("bin_to_png finish")

    dataset = get_hr_sr_dataset(cfg)

    metrics = {
        "psnr": PSNR(rgb_range=cfg.rgb_range, shave=6 + lr_scale),
    }

    total_step = len(dataset)
    setw = len(str(total_step))
    for i in range(len(dataset)):
        _, hr, sr = dataset[i]
        sr = unpadding(sr, hr.shape)
        sr = img_to_tensor(sr)
        hr = img_to_tensor(hr)
        _ = [m.update(sr, hr) for m in metrics.values()]
        result = {k: m.eval(sync=False) for k, m in metrics.items()}
        print(f"[{i+1:>{setw}}/{total_step:>{setw}}] result = {result}", flush=True)
    result = {k: m.eval(sync=False) for k, m in metrics.items()}
    print(f"evaluation result = {result}", flush=True)

    print("post_process success", flush=True)


if __name__ == "__main__":
    run_post_process()
