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
"""Sort."""
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Accuracy sort.")
    parser.add_argument("--result_path", type=str, default="/home/result.txt",
                        help='Text save address.')

    args_eval = parser.parse_args()
    result_path = args_eval.result_path
    if not os.path.exists(result_path):
        print("The result file not found!")

    with open(result_path, "r") as file:
        epoch = []
        result_dev = []
        result_test = []
        for i, line in enumerate(file):
            if i == 0:
                continue
            curLine = line.strip().split(" ")
            epoch.append(curLine[0])
            result_dev.append(curLine[1])
            result_test.append(curLine[2])
            print(epoch, " ", result_dev, ",", result_test)
        index_max_dev = result_dev.index(max(result_dev))
        acc_last = result_test[index_max_dev]
        print("++++ Then accuracy on the test dataset is:", acc_last)
