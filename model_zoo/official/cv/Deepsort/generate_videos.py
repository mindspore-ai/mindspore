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
import argparse
import ast
import show_results

from mindspore.train.model import ParallelMode
from mindspore.communication.management import init
from mindspore import context

def convert(filename_input, filename_output, ffmpeg_executable="ffmpeg"):
    import subprocess
    command = [ffmpeg_executable, "-i", filename_input, "-c:v", "libx264",
               "-preset", "slow", "-crf", "21", filename_output]
    subprocess.call(command)


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Siamese Tracking")
    parser.add_argument('--run_distribute', type=ast.literal_eval, default=False, help='Run distribute')
    parser.add_argument('--run_modelarts', type=ast.literal_eval, default=True, help='Run distribute')
    parser.add_argument("--device_id", type=int, default=4, help="Use which device.")
    parser.add_argument('--data_url', type=str, default='', help='Det directory.')
    parser.add_argument('--train_url', type=str, help='Folder to store the videos in')
    parser.add_argument(
        "--result_dir", help="Path to the folder with tracking output.", default="")
    parser.add_argument(
        "--convert_h264", help="If true, convert videos to libx264 (requires "
        "FFMPEG", default=False)
    parser.add_argument(
        "--update_ms", help="Time between consecutive frames in milliseconds. "
        "Defaults to the frame_rate specified in seqinfo.ini, if available.",
        default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False)
    if args.run_modelarts:
        import moxing as mox
        device_id = int(os.getenv('DEVICE_ID'))
        device_num = int(os.getenv('RANK_SIZE'))
        context.set_context(device_id=device_id)
        local_data_url = '/cache/data'
        local_train_url = '/cache/train'
        local_result_url = '/cache/result'
        mox.file.make_dirs(local_train_url)
        mox.file.copy_parallel(args.data_url, local_data_url)
        mox.file.copy_parallel(args.result_dir, local_result_url)
        if device_num > 1:
            init()
            context.set_auto_parallel_context(device_num=device_num,\
                 parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
        data_dir = local_data_url + '/'
        result_dir = local_result_url + '/'
    else:
        if args.run_distribute:
            device_id = int(os.getenv('DEVICE_ID'))
            device_num = int(os.getenv('RANK_SIZE'))
            context.set_context(device_id=device_id)
            init()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num,\
                 parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
        else:
            context.set_context(device_id=args.device_id)
            device_num = 1
            device_id = args.device_id
        data_dir = args.data_url
        local_train_url = args.train_url
        result_dir = args.result_dir


    os.makedirs(local_train_url, exist_ok=True)
    for sequence_txt in os.listdir(result_dir):
        sequence = os.path.splitext(sequence_txt)[0]
        sequence_dir = os.path.join(data_dir, sequence)
        if not os.path.exists(sequence_dir):
            continue
        result_file = os.path.join(result_dir, sequence_txt)
        update_ms = args.update_ms
        video_filename = os.path.join(local_train_url, "%s.avi" % sequence)

        print("Saving %s to %s." % (sequence_txt, video_filename))
        show_results.run(
            sequence_dir, result_file, False, None, update_ms, video_filename)

    if not args.convert_h264:
        import sys
        sys.exit()
    for sequence_txt in os.listdir(result_dir):
        sequence = os.path.splitext(sequence_txt)[0]
        sequence_dir = os.path.join(data_dir, sequence)
        if not os.path.exists(sequence_dir):
            continue
        filename_in = os.path.join(local_train_url, "%s.avi" % sequence)
        filename_out = os.path.join(local_train_url, "%s.mp4" % sequence)
        convert(filename_in, filename_out)
    if args.run_modelarts:
        mox.file.copy_parallel(src_url=local_train_url, dst_url=args.train_url)
