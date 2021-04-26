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
import deep_sort_app

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="MOTChallenge evaluation")
    parser.add_argument(
        "--detection_url", type=str, help="Path to detection files.")
    parser.add_argument(
        "--data_url", type=str, help="Path to image data.")
    parser.add_argument(
        "--train_url", type=str, help="Path to save result.")
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
    detection_dir = args.detection_url
    DATA_DIR = args.data_url + '/'
    local_result_url = args.train_url

    if not os.path.exists(local_result_url):
        os.makedirs(local_result_url)
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
