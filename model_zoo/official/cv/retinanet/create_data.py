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

"""create mindrecord for training retinanet."""

import argparse
from src.dataset import create_mindrecord

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="retinanet dataset create")
    parser.add_argument("--dataset", type=str, default="coco", help="Dataset, default is coco.")
    args_opt = parser.parse_args()
    mindrecord_file = create_mindrecord(args_opt.dataset, "retinanet.mindrecord", True)
