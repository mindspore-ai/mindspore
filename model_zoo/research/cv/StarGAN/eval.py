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
"""Evaluation for StarGAN"""
import os
import numpy as np
from PIL import Image

from mindspore import context
from mindspore import Tensor
from mindspore.train.serialization import load_param_into_net
from mindspore.common import dtype as mstype
import mindspore.ops as ops

from src.utils import resume_model, create_labels, denorm, get_network
from src.config import get_config
from src.dataset import dataloader


if __name__ == "__main__":

    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', device_id=1)
    config = get_config()
    G, D = get_network(config)
    para_g, _ = resume_model(config, G, D)
    load_param_into_net(G, para_g)

    if not os.path.exists(config.result_dir):
        os.mkdir(config.result_dir)
    # Define Dataset

    data_path = config.celeba_image_dir
    attr_path = config.attr_path

    dataset, length = dataloader(img_path=data_path,
                                 attr_path=attr_path,
                                 batch_size=4,
                                 selected_attr=config.selected_attrs,
                                 device_num=config.num_workers,
                                 dataset=config.dataset,
                                 mode=config.mode,
                                 shuffle=False)

    op = ops.Concat(axis=3)
    ds = dataset.create_dict_iterator()
    print(length)
    print('Start Evaluating!')
    for i, data in enumerate(ds):
        result_list = ()
        img_real = denorm(data['image'].asnumpy())
        x_real = Tensor(data['image'], mstype.float32)
        result_list += (x_real,)
        c_trg_list = create_labels(data['attr'].asnumpy(), selected_attrs=config.selected_attrs)
        c_trg_list = Tensor(c_trg_list, mstype.float32)
        x_fake_list = []

        for c_trg in c_trg_list:

            x_fake = G(x_real, c_trg)
            x = Tensor(x_fake.asnumpy().copy())

            result_list += (x,)

        x_fake_list = op(result_list)

        result = denorm(x_fake_list.asnumpy())
        result = np.reshape(result, (-1, 768, 3))

        im = Image.fromarray(np.uint8(result))
        im.save(config.result_dir + '/test_{}.jpg'.format(i))
        print('Successful save image in ' + config.result_dir + '/test_{}.jpg'.format(i))
