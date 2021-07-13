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
"""
run model eval
"""
import os

from mindspore import context, load_checkpoint, load_param_into_net

from src.config import parse_args
from src.models.StackedHourglassNet import StackedHourglassNet
from src.utils.inference import MPIIEval, get_img, inference

args = parse_args()

if __name__ == "__main__":
    if not os.path.exists(args.ckpt_file):
        print("ckpt file not valid")
        exit()

    if not os.path.exists(args.img_dir) or not os.path.exists(args.annot_dir):
        print("Dataset not found.")
        exit()

    # Set context mode
    if args.context_mode == "GRAPH":
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, save_graphs=False)
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target)

    # Import net
    net = StackedHourglassNet(args.nstack, args.inp_dim, args.oup_dim)
    param_dict = load_checkpoint(args.ckpt_file)
    load_param_into_net(net, param_dict)

    gts = []
    preds = []
    normalizing = []

    num_eval = args.num_eval
    num_train = args.train_num_eval
    for anns, img, c, s, n in get_img(num_eval, num_train):
        gts.append(anns)
        ans = inference(img, net, c, s)
        if ans.size > 0:
            ans = ans[:, :, :3]

        # (num preds, joints, x/y/visible)
        pred = []
        for i in range(ans.shape[0]):
            pred.append({"keypoints": ans[i, :, :]})
        preds.append(pred)
        normalizing.append(n)

    mpii_eval = MPIIEval()
    mpii_eval.eval(preds, gts, normalizing, num_train)
