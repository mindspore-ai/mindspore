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

"""Evaluation for CTPN"""
import os
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from src.ctpn import CTPN
from src.dataset import create_ctpn_dataset
from src.eval_utils import eval_for_ctpn
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id


set_seed(1)


context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=get_device_id())


def modelarts_pre_process():
    pass


def modelarts_post_process():
    local_path = os.path.join(config.modelarts_home, config.object_name)
    basename = os.path.basename(config.checkpoint_path)
    copy_label = 'cd {}&&zip submit_{}.zip ./submit *.txt'.format(local_path, basename)
    os.system(copy_label)
    os.system('cd {}&&sed -i "s/\r//" scripts/eval_res.sh'.format(local_path))
    os.system('cd {}&& sh scripts/eval_res.sh'.format(local_path))


@moxing_wrapper(pre_process=modelarts_pre_process, post_process=modelarts_post_process)
def ctpn_infer_test():
    config.feature_shapes = [config.img_height // 16, config.img_width // 16]
    config.num_bboxes = (config.img_height // 16) * (config.img_width // 16) * config.num_anchors
    config.num_step = config.img_width // 16
    config.rnn_batch_size = config.img_height // 16

    print("ckpt path is {}".format(config.checkpoint_path))
    ds = create_ctpn_dataset(config.dataset_path, batch_size=config.test_batch_size, repeat_num=1, is_training=False)
    total = ds.get_dataset_size()
    print("eval dataset size is {}".format(total))
    net = CTPN(config, batch_size=config.test_batch_size, is_training=False)
    param_dict = load_checkpoint(config.checkpoint_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)
    eval_for_ctpn(net, ds, config.img_dir)


if __name__ == '__main__':
    ctpn_infer_test()
