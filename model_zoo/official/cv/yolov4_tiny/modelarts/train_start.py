# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 3.0 (the "License");
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
"""train yolov4_tiny."""

import argparse
import os
import subprocess

_CACHE_DATA_URL = "/cache/data_url"
_CACHE_TRAIN_URL = "/cache/train_url"


def _parse_args():
    parser = argparse.ArgumentParser('mindspore yolov4_tiny training')
    parser.add_argument('--train_url', type=str, default='',
                        help='where training log and ckpts saved')

    # dataset
    parser.add_argument('--data_url', type=str, default='',
                        help='path of dataset')
    parser.add_argument('--per_batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--training_shape', type=int, default=416, help='training size')
    parser.add_argument('--num_classes', type=int, default=80,
                        help='number of classes')

    # optimizer
    parser.add_argument('--max_epoch', type=int, default=1, help='epoch')
    parser.add_argument('--lr_scheduler', type=str, default='cosine_annealing',
                        help='type of learning rate')
    parser.add_argument('--lr', type=float, default=0.012,
                        help='base learning rate')

    # model
    parser.add_argument('--pretrained_backbone', type=str, default='',
                        help='pretrained backbone')
    parser.add_argument('--resume_yolov4', action='store_true', help='resume')
    parser.add_argument('--pretrained_checkpoint', type=str, default='',
                        help='pretrained model')

    # train
    parser.add_argument('--device_target', type=str, default='Ascend',
                        choices=['Ascend', 'CPU'],
                        help='device where the code will be implemented. '
                             '(Default: Ascend)')
    parser.add_argument('--is_distributed', type=int, default=0, help='distributed training')
    parser.add_argument('--rank', type=int, default=0,
                        help='local rank of distributed')
    parser.add_argument('--group_size', type=int, default=1,
                        help='world size of distributed')
    parser.add_argument('--filter_weight', type=str, default=False,
                        help="filter weight")
    parser.add_argument('--file_name', type=str, default='yolov4_tiny', help='output air file name')

    args, _ = parser.parse_known_args()
    return args


def _train(args, train_url, data_url, pretrained_checkpoint):
    train_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              "train.py")
    cmd = ["python", train_file,
           f"--ckpt_path={os.path.abspath(train_url)}",
           f"--data_dir={os.path.abspath(data_url)}",
           f"--per_batch_size={args.per_batch_size}",
           f"--training_shape={args.training_shape}",
           f"--num_classes={args.num_classes}",
           f"--max_epoch={args.max_epoch}",
           f"--lr_scheduler={args.lr_scheduler}",
           f"--lr={args.lr}",
           f"--pretrained_checkpoint={pretrained_checkpoint}",
           f"--device_target={args.device_target}",
           f"--rank={args.rank}",
           f"--group_size={args.group_size}",
           f"--is_distributed={args.is_distributed}",
           f"--filter_weight={args.filter_weight}"]
    if args.is_distributed:
        cmd.append('--is_distributed')
    if args.filter_weight == "True":
        cmd.append('--filter_weight=True')
    print(' '.join(cmd))
    os.environ["DEVICE_ID"] = str(args.rank)
    process = subprocess.Popen(cmd, shell=False)
    return process.wait()


def _get_last_ckpt(ckpt_dir):
    file_dict = {}
    lists = os.listdir(ckpt_dir)
    for i in lists:
        ctime = os.stat(os.path.join(ckpt_dir, i)).st_ctime
        file_dict[ctime] = i
    max_ctime = max(file_dict.keys())
    ckpt_dir = os.path.join(ckpt_dir, file_dict[max_ctime], 'ckpt_0')
    ckpt_files = [ckpt_file for ckpt_file in os.listdir(ckpt_dir)
                  if ckpt_file.endswith('.ckpt')]
    if not ckpt_files:
        print("No ckpt file found.")
        return None

    return os.path.join(ckpt_dir, sorted(ckpt_files)[-1])


def _export_air(args, ckpt_dir):
    ckpt_file = _get_last_ckpt(ckpt_dir)
    if not ckpt_file:
        return

    export_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "export.py")
    cmd = ["python", export_file,
           f"--batch_size={args.per_batch_size}",
           f"--checkpoint={ckpt_file}",
           f"--num_classes={args.num_classes}",
           f"--file_name={os.path.join(_CACHE_TRAIN_URL, args.file_name)}"]
    print(f"Start exporting AIR, cmd = {' '.join(cmd)}.")
    process = subprocess.Popen(cmd, shell=False)
    process.wait()


def main():
    args = _parse_args()
    try:
        import moxing as mox
        os.makedirs(_CACHE_TRAIN_URL, exist_ok=True)
        os.makedirs(_CACHE_DATA_URL, exist_ok=True)
        mox.file.copy_parallel(args.data_url, _CACHE_DATA_URL)
        train_url = _CACHE_TRAIN_URL
        data_url = _CACHE_DATA_URL
        pretrained_checkpoint = os.path.join(_CACHE_DATA_URL,
                                             args.pretrained_checkpoint) if args.pretrained_checkpoint else ""
        ret = _train(args, train_url, data_url, pretrained_checkpoint)
        _export_air(args, train_url)
        mox.file.copy_parallel(_CACHE_TRAIN_URL, args.train_url)
    except ModuleNotFoundError:
        train_url = args.train_url
        data_url = args.data_url
        pretrained_checkpoint = args.pretrained_checkpoint
        ret = _train(args, train_url, data_url, pretrained_checkpoint)

    if ret != 0:
        exit(1)


if __name__ == '__main__':
    main()
