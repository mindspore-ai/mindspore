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
import show_results

def convert(filename_input, filename_output, ffmpeg_executable="ffmpeg"):
    import subprocess
    command = [ffmpeg_executable, "-i", filename_input, "-c:v", "libx264",
               "-preset", "slow", "-crf", "21", filename_output]
    subprocess.call(command)


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Siamese Tracking")
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
