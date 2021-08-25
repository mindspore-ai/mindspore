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
#################evaluate EDSR example on DIV2K########################
"""
import os

import numpy as np
from mindspore.common import set_seed
from mindspore import Tensor, ops

from src.metric import SelfEnsembleWrapperNumpy, PSNR, SaveSrHr
from src.utils import init_env, init_dataset, init_net, modelarts_pre_process, do_eval
from src.dataset import get_rank_info, LrHrImages, hwc2chw, uint8_to_float32
from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper


set_seed(2021)


class HrCutter:
    """
    cut hr into correct shape, for evaluating benchmark
    """
    def __init__(self, lr_scale):
        self.lr_scale = lr_scale

    def __call__(self, lr, hr):
        lrh, lrw, _ = lr.shape
        hrh, hrw, _ = hr.shape
        h, w = lrh * self.lr_scale, lrw * self.lr_scale
        if hrh != h or hrw != w:
            hr = hr[0:h, 0:w, :]
        return lr, hr


class RepeatDataSet:
    """
    Repeat DataSet so that it can dist evaluate Set5
    """
    def __init__(self, dataset, repeat):
        self.dataset = dataset
        self.repeat = repeat

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]

    def __len__(self):
        return len(self.dataset) * self.repeat


def create_dataset_benchmark(dataset_path, scale):
    """
    create a train or eval benchmark dataset
    Args:
        dataset_path(string): the path of dataset.
        scale(int): lr scale, read data ordered by it, choices=(2,3,4)
    Returns:
        multi_datasets
    """
    lr_scale = scale

    multi_datasets = {}
    for dataset_name in ["Set5", "Set14", "B100", "Urban100"]:
        # get HR_PATH/*.png
        dir_hr = os.path.join(dataset_path, dataset_name, "HR")
        hr_pattern = os.path.join(dir_hr, "*.png")

        # get LR
        column_names = [f"lrx{lr_scale}", "hr"]
        dir_lr = os.path.join(dataset_path, dataset_name, "LR_bicubic", f"X{lr_scale}")
        lr_pattern = os.path.join(dir_lr, f"*x{lr_scale}.png")
        lrs_pattern = [lr_pattern]

        device_num, rank_id = get_rank_info()

        # make dataset
        dataset = LrHrImages(lr_pattern=lrs_pattern, hr_pattern=hr_pattern)
        if len(dataset) < device_num:
            dataset = RepeatDataSet(dataset, repeat=device_num // len(dataset) + 1)

        # make mindspore dataset
        if device_num == 1 or device_num is None:
            generator_dataset = ds.GeneratorDataset(dataset, column_names=column_names,
                                                    num_parallel_workers=3,
                                                    shuffle=False)
        else:
            sampler = ds.DistributedSampler(num_shards=device_num, shard_id=rank_id, shuffle=False, offset=0)
            generator_dataset = ds.GeneratorDataset(dataset, column_names=column_names,
                                                    num_parallel_workers=3,
                                                    sampler=sampler)

        # define map operations
        transform_img = [
            HrCutter(lr_scale),
            hwc2chw,
            uint8_to_float32,
        ]

        # pre-process hr lr
        generator_dataset = generator_dataset.map(input_columns=column_names,
                                                  output_columns=column_names,
                                                  column_order=column_names,
                                                  operations=transform_img)

        # apply batch operations
        generator_dataset = generator_dataset.batch(1, drop_remainder=False)

        multi_datasets[dataset_name] = generator_dataset
    return multi_datasets


class BenchmarkPSNR(PSNR):
    """
    eval psnr for Benchmark
    """
    def __init__(self, rgb_range, shave, channels_scale):
        super(BenchmarkPSNR, self).__init__(rgb_range=rgb_range, shave=shave)
        self.channels_scale = channels_scale
        self.c_scale = Tensor(np.array(self.channels_scale, dtype=np.float32).reshape((1, -1, 1, 1)))
        self.sum = ops.ReduceSum(keep_dims=True)

    def update(self, *inputs):
        if len(inputs) != 2:
            raise ValueError('PSNR need 2 inputs (sr, hr), but got {}'.format(len(inputs)))
        sr, hr = inputs
        sr = self.quantize(sr)
        diff = (sr - hr) / self.rgb_range
        diff = diff * self.c_scale
        valid = self.sum(diff, 1)
        if self.shave is not None and self.shave != 0:
            valid = valid[..., self.shave:(-self.shave), self.shave:(-self.shave)]
        mse_list = (valid ** 2).mean(axis=(1, 2, 3))
        mse_list = self._convert_data(mse_list).tolist()
        psnr_list = [float(1e32) if mse == 0 else(- 10.0 * math.log10(mse)) for mse in mse_list]
        self._accumulate(psnr_list)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_eval():
    """
    run eval
    """
    print(config)
    cfg = config

    init_env(cfg)
    net = init_net(cfg)
    eval_net = SelfEnsembleWrapperNumpy(net) if cfg.self_ensemble else net

    if cfg.dataset_name == "DIV2K":
        cfg.batch_size = 1
        cfg.patch_size = -1
        ds_val = init_dataset(cfg, "valid")
        metrics = {
            "psnr": PSNR(rgb_range=cfg.rgb_range, shave=6 + cfg.scale),
        }
        if config.save_sr:
            save_img_dir = os.path.join(cfg.output_path, "HrSr")
            os.makedirs(save_img_dir, exist_ok=True)
            metrics["num_sr"] = SaveSrHr(save_img_dir)
        do_eval(eval_net, ds_val, metrics)
        print("eval success", flush=True)

    elif cfg.dataset_name == "benchmark":
        multi_datasets = create_dataset_benchmark(dataset_path=cfg.data_path, scale=cfg.scale)
        result = {}
        for dname, ds_val in multi_datasets.items():
            dpnsr = f"{dname}_psnr"
            gray_coeffs = [65.738, 129.057, 25.064]
            channels_scale = [x / 256.0 for x in gray_coeffs]
            metrics = {
                dpnsr: BenchmarkPSNR(rgb_range=cfg.rgb_range, shave=cfg.scale, channels_scale=channels_scale)
            }
            if config.save_sr:
                save_img_dir = os.path.join(cfg.output_path, "HrSr", dname)
                os.makedirs(save_img_dir, exist_ok=True)
                metrics["num_sr"] = SaveSrHr(save_img_dir)
            result[dpnsr] = do_eval(eval_net, ds_val, metrics)[dpnsr]
        if get_rank_id() == 0:
            print(result, flush=True)
        print("eval success", flush=True)
    else:
        raise RuntimeError("Unsupported dataset.")

if __name__ == '__main__':
    run_eval()
