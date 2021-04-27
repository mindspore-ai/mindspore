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
import argparse
import os
import ast
import deep_sort_app
from mindspore.communication.management import init
from mindspore.train.model import ParallelMode
from mindspore import context


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="MOTChallenge evaluation")
    parser.add_argument('--run_modelarts', type=ast.literal_eval, default=True, help='Run distribute')
    parser.add_argument(
        "--mot_dir", help="Path to MOTChallenge directory (train or test)", default="../MOT16/train")
    parser.add_argument(
        "--detection_url", type=str, help="Path to detection files.")
    parser.add_argument(
        "--data_url", type=str, help="Path to image data.")
    parser.add_argument(
        "--train_url", type=str, help="Path to save result.")
    parser.add_argument(
        "--output_dir", help="Folder in which the results will be stored. Will "
        "be created if it does not exist.", default="results")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.0, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap", help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    context.set_context(mode=context.GRAPH_MODE, enable_auto_mixed_precision=True,\
         device_target="Ascend", save_graphs=False)
    if args.run_modelarts:
        import moxing as mox
        device_id = int(os.getenv('DEVICE_ID'))
        device_num = int(os.getenv('RANK_SIZE'))
        context.set_context(device_id=device_id)
        local_data_url = '/cache/data'
        local_result_url = '/cache/result'
        local_detection_url = '/cache/detection'
        mox.file.copy_parallel(args.detection_url, local_detection_url)
        mox.file.copy_parallel(args.data_url, local_data_url)
        mox.file.copy_parallel(args.train_url, local_result_url)
        if device_num > 1:
            init()
            context.set_auto_parallel_context(device_num=device_num,\
                 parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
        DATA_DIR = local_data_url + '/'
        detection_dir = local_detection_url + '/'
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
        detection_dir = args.detection_url
        DATA_DIR = args.data_url + '/'
        local_result_url = args.train_url
    os.makedirs(args.output_dir, exist_ok=True)
    sequences = os.listdir(DATA_DIR)
    for sequence in sequences:
        print("Running sequence %s" % sequence)
        sequence_dir = os.path.join(DATA_DIR, sequence)
        detection_file = os.path.join(detection_dir, "%s.npy" % sequence)
        output_file = os.path.join(local_result_url, "%s.txt" % sequence)
        deep_sort_app.run(
            sequence_dir, detection_file, output_file, args.min_confidence,
            args.nms_max_overlap, args.min_detection_height,
            args.max_cosine_distance, args.nn_budget, display=False)
    if args.run_modelarts:
        mox.file.copy_parallel(src_url=local_result_url, dst_url=args.train_url)
