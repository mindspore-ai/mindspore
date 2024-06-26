# Copyright 2024 Huawei Technologies Co., Ltd
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
import mindspore as ms
from mindspore.communication.management import init
from src.utils import logger
from src.utils.local_adapter import get_rank_id, get_device_num, get_local_device_num


def clear(cfg):
    if os.path.exists("/tmp/load.lock"):
        os.remove("/tmp/load.lock")
    if cfg.enable_modelarts:
        from src.utils.modelarts import sync_data

        sync_data(cfg.save_dir, cfg.train_url)


def cpu_affinity(rank_id, device_num):
    """Bind CPU cores according to rank_id and device_num."""
    import psutil

    cores = psutil.cpu_count()
    if cores < device_num:
        return
    process = psutil.Process()
    used_cpu_num = cores // device_num
    rank_id = rank_id % device_num
    used_cpu_list = [i for i in range(rank_id * used_cpu_num, (rank_id + 1) * used_cpu_num)]
    process.cpu_affinity(used_cpu_list)
    print(f"==== {rank_id}/{device_num} ==== bind cpu: {used_cpu_list}")


def init_env(cfg):
    seed = int(getattr(cfg, "seed", 1))
    device_target = getattr(cfg, "device_target", "Ascend")
    ms_mode = int(getattr(cfg, "ms_mode", 0))
    run_profilor = bool(getattr(cfg, "run_profilor", False))
    enable_modelarts = bool(getattr(cfg, "enable_modelarts", False))
    ms.set_seed(seed)
    # Set Context
    ms.set_context(mode=ms_mode, device_target=device_target, max_call_depth=2000, runtime_num_threads=15)
    if ms_mode == 1:
        ms.set_context(pynative_synchronize=True)
    elif ms_mode == 0:
        ms.set_context(jit_config={"jit_level": "O2"})
    if device_target != "CPU":
        device_id = int(os.environ.get("DEVICE_ID", "0"))
        ms.set_context(device_id=device_id)
    elif device_target == "GPU" and cfg.get("ms_enable_graph_kernel", False):
        ms.set_context(enable_graph_kernel=True)
    if run_profilor:
        ms.set_context(save_graphs=True, save_graphs_path="ir")

    # Set Parallel
    rank, rank_size = get_rank_id(), get_device_num()
    parallel_mode = ms.ParallelMode.STAND_ALONE
    if rank_size > 1:
        init()
        parallel_mode = ms.ParallelMode.DATA_PARALLEL
        cpu_affinity(rank, get_local_device_num())
        print(f"=== run distribute {rank}, {rank_size}, {parallel_mode}")
    ms.set_auto_parallel_context(device_num=rank_size, parallel_mode=parallel_mode, gradients_mean=True)
    cfg.rank, cfg.rank_size = rank, rank_size

    if cfg.rank % min(cfg.rank_size, 8) == 0:
        cfg.save_dir = os.path.abspath(cfg.save_dir)
        if not os.path.exists(cfg.save_dir):
            os.makedirs(cfg.save_dir, exist_ok=True)
            os.makedirs(os.path.join(cfg.save_dir, "checkpoints"), exist_ok=True)
    # Set Logger
    logger.setup_logging(logger_name="fasterrcnn", log_level="INFO", rank_id=rank, device_per_servers=rank_size)
    logger.setup_logging_file(log_dir=os.path.join(cfg.save_dir, "logs"))

    # Modelarts: Copy data, from the s3 bucket to the computing node; Reset dataset dir.
    if enable_modelarts:
        from src.utils.modelarts import sync_data

        print("==== data_dir", cfg.data_dir)
        print("==== save_dir", cfg.save_dir)
        print("==== train_url", cfg.train_url)
        print("==== data_url", cfg.data_url)
        print("==== ckpt_url", cfg.ckpt_url)
        print("==== dataset_dir", os.path.join(cfg.data_dir, cfg.data.dataset_dir))
        os.makedirs(cfg.data_dir, exist_ok=True)
        sync_data(cfg.data_url, cfg.data_dir)
        sync_data(cfg.save_dir, cfg.train_url)

        if cfg.ckpt_url:
            sync_data(cfg.ckpt_url, cfg.ckpt_dir)  # pretrain ckpt
        cfg.data.dataset_dir = os.path.join(cfg.data_dir, cfg.data.dataset_dir)
        cfg.pre_trained_ckpt = os.path.join(cfg.ckpt_dir, cfg.pre_trained_ema_ckpt) if cfg.pre_trained_ema_ckpt else ""
        cfg.pre_trained_ema_ckpt = (
            os.path.join(cfg.ckpt_dir, cfg.pre_trained_ema_ckpt) if cfg.pre_trained_ema_ckpt else ""
        )
        print("==== list", os.listdir(cfg.data_dir))


def save_checkpoint(cfg, network, epoch):
    os.makedirs(os.path.join(cfg.save_dir, "checkpoints"), exist_ok=True)
    save_path = os.path.join(
        cfg.save_dir, "checkpoints", f"{cfg.net}_{cfg.backbone.name}_epoch{epoch}_rank{cfg.rank}.ckpt"
    )
    ms.save_checkpoint(network, save_path)
    if cfg.enable_modelarts:
        from src.utils.modelarts import sync_data

        sync_data(
            save_path,
            os.path.join(
                cfg.train_url, "checkpoints", f"{cfg.net}_{cfg.backbone.name}_epoch{epoch}_rank{cfg.rank}.ckpt"
            ),
        )
