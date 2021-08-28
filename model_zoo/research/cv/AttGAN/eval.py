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
"""Entry point for testing AttGAN network"""

import argparse
import json
import math
import os
from os.path import join

import numpy as np
from PIL import Image

import mindspore.common.dtype as mstype
import mindspore.dataset as de
from mindspore import context, Tensor, ops
from mindspore.train.serialization import load_param_into_net

from src.attgan import Gen
from src.cell import init_weights
from src.data import check_attribute_conflict
from src.data import get_loader, Custom
from src.helpers import Progressbar
from src.utils import resume_generator, denorm

device_id = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend", save_graphs=False, device_id=device_id)


def parse(arg=None):
    """Define configuration of Evaluation"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', dest='experiment_name', required=True)
    parser.add_argument('--test_int', dest='test_int', type=float, default=1.0)
    parser.add_argument('--num_test', dest='num_test', type=int)
    parser.add_argument('--gen_ckpt_name', type=str, default='')
    parser.add_argument('--custom_img', action='store_true')
    parser.add_argument('--custom_data', type=str, default='../data/custom')
    parser.add_argument('--custom_attr', type=str, default='../data/list_attr_custom.txt')
    parser.add_argument('--shortcut_layers', dest='shortcut_layers', type=int, default=1)
    parser.add_argument('--inject_layers', dest='inject_layers', type=int, default=1)
    return parser.parse_args(arg)


args_ = parse()
print(args_)

with open(join('output', args_.experiment_name, 'setting.txt'), 'r') as f:
    args = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
args.test_int = args_.test_int
args.num_test = args_.num_test
args.gen_ckpt_name = args_.gen_ckpt_name
args.custom_img = args_.custom_img
args.custom_data = args_.custom_data
args.custom_attr = args_.custom_attr
args.shortcut_layers = args_.shortcut_layers
args.inject_layers = args_.inject_layers
args.n_attrs = len(args.attrs)
args.betas = (args.beta1, args.beta2)

print(args)

# Data loader
if args.custom_img:
    output_path = join("output", args.experiment_name, "custom_testing")
    os.makedirs(output_path, exist_ok=True)
    test_dataset = Custom(args.custom_data, args.custom_attr, args.attrs)
    test_len = len(test_dataset)
else:
    output_path = join("output", args.experiment_name, "sample_testing")
    os.makedirs(output_path, exist_ok=True)
    test_dataset = get_loader(args.data_path, args.attr_path,
                              selected_attrs=args.attrs,
                              mode="test"
                              )
    test_len = len(test_dataset)
dataset_column_names = ["image", "attr"]
num_parallel_workers = 8
ds = de.GeneratorDataset(test_dataset, column_names=dataset_column_names,
                         num_parallel_workers=min(32, num_parallel_workers))
ds = ds.batch(1, num_parallel_workers=min(8, num_parallel_workers), drop_remainder=False)
test_dataset_iter = ds.create_dict_iterator()

if args.num_test is None:
    print('Testing images:', test_len)
else:
    print('Testing images:', min(test_len, args.num_test))

# Model loader
gen = Gen(args.enc_dim, args.enc_layers, args.enc_norm, args.enc_acti, args.dec_dim, args.dec_layers, args.dec_norm,
          args.dec_acti, args.n_attrs, args.shortcut_layers, args.inject_layers, args.img_size, mode='test')

# Initialize network
init_weights(gen, 'KaimingUniform', math.sqrt(5))
para_gen = resume_generator(args, gen, args.gen_ckpt_name)
load_param_into_net(gen, para_gen)

print("Network initializes successfully.")

progressbar = Progressbar()
it = 0
for data in test_dataset_iter:
    img_a = data["image"]
    att_a = data["attr"]
    if args.num_test is not None and it == args.num_test:
        break

    att_a = Tensor(att_a, mstype.float32)
    att_b_list = [att_a]
    for i in range(args.n_attrs):
        clone = ops.Identity()
        tmp = clone(att_a)
        tmp[:, i] = 1 - tmp[:, i]
        tmp = check_attribute_conflict(tmp, args.attrs[i], args.attrs)
        att_b_list.append(tmp)

    samples = [img_a]

    for i, att_b in enumerate(att_b_list):
        att_b_ = (att_b * 2 - 1) * args.thres_int
        if i > 0:
            att_b_[..., i - 1] = att_b_[..., i - 1] * args.test_int / args.thres_int
        samples.append(gen(img_a, att_b_, mode="enc-dec"))
    cat = ops.Concat(axis=3)
    samples = cat(samples).asnumpy()
    result = denorm(samples)
    result = np.reshape(result, (128, -1, 3))
    im = Image.fromarray(np.uint8(result))
    if args.custom_img:
        out_file = test_dataset.images[it]
    else:
        out_file = "{:06d}.jpg".format(it + 182638)
    im.save(output_path + '/' + out_file)
    print('Successful save image in ' + output_path + '/' + out_file)
    it += 1
