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

'''
postprocess script.
'''

import os
import argparse
import numpy as np
from src.config import config

parser = argparse.ArgumentParser(description="postprocess")
parser.add_argument("--label_dir", type=str, default="", help="label data dir")
parser.add_argument("--result_dir", type=str, default="./result_Files", help="infer result Files")

args, _ = parser.parse_known_args()

if __name__ == "__main__":
    file_name = os.listdir(args.label_dir)
    predictions = []
    target_sents = []
    for f in file_name:
        target_ids = np.fromfile(os.path.join(args.label_dir, f), np.int32)
        target_sents.append(target_ids.reshape(config.eval_batch_size, config.max_length))
        predicted_ids = np.fromfile(os.path.join(args.result_dir, f.split('.')[0] + '_0.bin'), np.int32)
        predictions.append(predicted_ids.reshape(config.eval_batch_size, config.max_length - 1))

    f_output = open(config.output_file, 'w')
    f_target = open(config.target_file, 'w')
    for batch_out, true_sentence in zip(predictions, target_sents):
        for i in range(config.eval_batch_size):
            target_ids = [str(x) for x in true_sentence[i].tolist()]
            f_target.write(" ".join(target_ids) + "\n")
            token_ids = [str(x) for x in batch_out[i].tolist()]
            f_output.write(" ".join(token_ids) + "\n")
    f_output.close()
    f_target.close()
