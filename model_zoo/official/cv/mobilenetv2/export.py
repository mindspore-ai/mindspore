# Copyright 2020 Huawei Technologies Co., Ltd
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
mobilenetv2 export mindir.
"""
import numpy as np
from mindspore import Tensor, export
from src.config import set_config
from src.args import export_parse_args
from src.models import define_net, load_ckpt
from src.utils import set_context

if __name__ == '__main__':
    args_opt = export_parse_args()
    cfg = set_config(args_opt)
    set_context(cfg)
    _, _, net = define_net(cfg, args_opt.is_training)

    load_ckpt(net, args_opt.pretrain_ckpt)
    input_shp = [1, 3, cfg.image_height, cfg.image_width]
    input_array = Tensor(np.random.uniform(-1.0, 1.0, size=input_shp).astype(np.float32))
    export(net, input_array, file_name=cfg.export_file, file_format=cfg.export_format)
