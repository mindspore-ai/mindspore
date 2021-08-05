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
"""start eval """
from __future__ import absolute_import
import argparse
import os
import sys
from got10k.experiments import ExperimentOTB
from mindspore import context
from src import SiamFCTracker

sys.path.append(os.getcwd())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='siamfc tracking')
    parser.add_argument('--device_id', type=int, default=0, help='device id of GPU or Ascend')
    parser.add_argument('--model_path', default='/root/SiamFC/models/siamfc_{}.ckpt/SiamFC-6650.ckpt'
                                                , type=str, help='eval one special video')
    parser.add_argument('--dataset_path', default='/root/datasets/OTB2013', type=str)

    args = parser.parse_args()
    context.set_context(
        mode=context.GRAPH_MODE,
        device_id=args.device_id,
        save_graphs=False,
        device_target='Ascend')
    tracker = SiamFCTracker(model_path=args.model_path)

    root_dir = os.path.abspath(args.dataset_path)
    e = ExperimentOTB(root_dir, version=2013)
    e.run(tracker, visualize=False)
    prec_score, succ_score, succ_rate = e.report([tracker.name])
    ss = '-prec_score:%.3f -succ_score:%.3f -succ_rate:%.3f' % (float(prec_score),
                                                                float(succ_score),
                                                                float(succ_rate))
    print(args.model_path.split('/')[-1], ss)
