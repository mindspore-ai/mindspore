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
#################utils for train.py and eval.py########################
"""
import os
import time

from mindspore import context
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint

from model_utils.config import config
from model_utils.device_adapter import get_device_id, get_rank_id, get_device_num
from .dataset import create_dataset_DIV2K
from .edsr import EDSR


def init_env(cfg):
    """
    init env for mindspore
    """
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target)
    device_num = get_device_num()
    if cfg.device_target == "Ascend":
        context.set_context(device_id=get_device_id())
        if device_num > 1:
            init()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
    elif cfg.device_target == "GPU":
        context.set_context(enable_graph_kernel=True)
        if device_num > 1:
            init()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
    elif cfg.device_target == "CPU":
        pass
    else:
        raise ValueError("Unsupported platform.")


def init_dataset(cfg, dataset_type="train"):
    """
    init DIV2K dataset
    """
    ds_cfg = {
        "dataset_path": cfg.data_path,
        "scale": cfg.scale,
        "lr_type": cfg.lr_type,
        "batch_size": cfg.batch_size,
        "patch_size": cfg.patch_size,
    }
    if cfg.dataset_name == "DIV2K":
        dataset = create_dataset_DIV2K(config=ds_cfg,
                                       dataset_type=dataset_type,
                                       num_parallel_workers=10,
                                       shuffle=dataset_type == "Train")
    else:
        raise ValueError("Unsupported dataset.")
    return dataset


def init_net(cfg):
    """
    init edsr network
    """
    net = EDSR(scale=cfg.scale,
               n_feats=cfg.n_feats,
               kernel_size=cfg.kernel_size,
               n_resblocks=cfg.n_resblocks,
               n_colors=cfg.n_colors,
               res_scale=cfg.res_scale,
               rgb_range=cfg.rgb_range,
               rgb_mean=cfg.rgb_mean,
               rgb_std=cfg.rgb_std,)
    if cfg.pre_trained:
        pre_trained_path = os.path.join(cfg.output_path, cfg.pre_trained)
        if len(cfg.pre_trained) >= 5 and cfg.pre_trained[:5] == "s3://":
            pre_trained_path = cfg.pre_trained
            import moxing as mox
            mox.file.shift("os", "mox") # then system can read file from s3://
        elif os.path.isfile(cfg.pre_trained):
            pre_trained_path = cfg.pre_trained
        elif os.path.isfile(pre_trained_path):
            pass
        else:
            raise ValueError(f"pre_trained error: {cfg.pre_trained}")
        print(f"loading pre_trained = {pre_trained_path}", flush=True)
        param_dict = load_checkpoint(pre_trained_path)
        net.load_pre_trained_param_dict(param_dict, strict=False)
    return net


def modelarts_pre_process():
    '''modelarts pre process function.'''
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        zip_isexist = zipfile.is_zipfile(zip_file)
        zip_name = os.path.basename(zip_file)
        if zip_isexist:
            fz = zipfile.ZipFile(zip_file, 'r')
            data_num = len(fz.namelist())
            data_print = int(data_num / 4) if data_num > 4 else 1
            len_data_num = len(str(data_num))
            for i, _file in enumerate(fz.namelist()):
                if i % data_print == 0:
                    print("[{1:>{0}}/{2:>{0}}] {3:>2}%  const time: {4:0>2}:{5:0>2}  unzipping {6}".format(
                        len_data_num,
                        i,
                        data_num,
                        int(i / data_num * 100),
                        int((time.time() - s_time) / 60),
                        int(int(time.time() - s_time) % 60),
                        zip_name,
                        flush=True))
                fz.extract(_file, save_dir)
            print("       finish  const time: {:0>2}:{:0>2}  unzipping {}".format(
                int((time.time() - s_time) / 60),
                int(int(time.time() - s_time) % 60),
                zip_name,
                flush=True))
        else:
            print("{} is not zip.".format(zip_name), flush=True)

    if config.enable_modelarts and config.need_unzip_in_modelarts:
        sync_lock = "/tmp/unzip_sync.lock"
        # Each server contains 8 devices as most.
        if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
            for ufile in config.need_unzip_files:
                zip_file = os.path.join(config.data_path, ufile)
                save_dir = os.path.dirname(zip_file)
                unzip(zip_file, save_dir)
            print("===Finish extract data synchronization===", flush=True)
            try:
                os.mknod(sync_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(sync_lock):
                break
            time.sleep(1)

        print("Device: {}, Finish sync unzip data.".format(get_device_id()), flush=True)

    config.ckpt_save_dir = os.path.join(config.output_path, config.ckpt_save_dir)


def do_eval(eval_network, ds_val, metrics, cur_epoch=None):
    """
    do eval for psnr and save hr, sr
    """
    eval_network.set_train(False)
    total_step = ds_val.get_dataset_size()
    setw = len(str(total_step))
    begin = time.time()
    step_begin = time.time()
    rank_id = get_rank_id()
    for i, (lr, hr) in enumerate(ds_val):
        sr = eval_network(lr)
        _ = [m.update(sr, hr) for m in metrics.values()]
        result = {k: m.eval(sync=False) for k, m in metrics.items()}
        result["time"] = time.time() - step_begin
        step_begin = time.time()
        print(f"[{i+1:>{setw}}/{total_step:>{setw}}] rank = {rank_id} result = {result}", flush=True)
    result = {k: m.eval(sync=True) for k, m in metrics.items()}
    result["time"] = time.time() - begin
    if cur_epoch is not None:
        result["epoch"] = cur_epoch
    if rank_id == 0:
        print(f"evaluation result = {result}", flush=True)
    eval_network.set_train(True)
    return result
