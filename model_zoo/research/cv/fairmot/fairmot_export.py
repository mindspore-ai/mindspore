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
"""export fairmot."""
import numpy as np
from mindspore import context, Tensor, export
from mindspore import dtype as mstype
from mindspore.train.serialization import load_checkpoint
from src.opts import Opts
from src.backbone_dla_conv import DLASegConv
from src.infer_net import InferNet
from src.fairmot_pose import WithNetCell


def fairmot_export(opt):
    """export fairmot to mindir or air."""
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target="Ascend",
        save_graphs=False,
        device_id=opt.id)
    backbone_net = DLASegConv(opt.heads,
                              down_ratio=4,
                              final_kernel=1,
                              last_level=5,
                              head_conv=256,
                              is_training=True)
    load_checkpoint(opt.load_model, net=backbone_net)
    infer_net = InferNet()
    net = WithNetCell(backbone_net, infer_net)
    net.set_train(False)
    input_data = Tensor(np.zeros([1, 3, 608, 1088]), mstype.float32)
    export(net, input_data, file_name='fairmot', file_format="MINDIR")


if __name__ == '__main__':
    opt_ = Opts().init()
    fairmot_export(opt_)
