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
"""Cycle GAN predict."""
import os
from mindspore import Tensor

from src.models import get_generator
from src.utils import get_args, load_ckpt, save_image, Reporter
from src.dataset import create_dataset

def predict():
    """Predict function."""
    args = get_args("predict")
    G_A = get_generator(args)
    G_B = get_generator(args)
    # Use BatchNorm2d with batchsize=1, affine=False, training=True instead of InstanceNorm2d
    # Use real mean and varance rather than moving_men and moving_varance in BatchNorm2d
    G_A.set_train(True)
    G_B.set_train(True)
    load_ckpt(args, G_A, G_B)

    imgs_out = os.path.join(args.outputs_dir, "predict")
    if not os.path.exists(imgs_out):
        os.makedirs(imgs_out)
    if not os.path.exists(os.path.join(imgs_out, "fake_A")):
        os.makedirs(os.path.join(imgs_out, "fake_A"))
    if not os.path.exists(os.path.join(imgs_out, "fake_B")):
        os.makedirs(os.path.join(imgs_out, "fake_B"))
    args.data_dir = 'testA'
    ds = create_dataset(args)
    reporter = Reporter(args)
    reporter.start_predict("A to B")
    for data in ds.create_dict_iterator(output_numpy=True):
        img_A = Tensor(data["image"])
        path_A = str(data["image_name"][0], encoding="utf-8")
        fake_B = G_A(img_A)
        save_image(fake_B, os.path.join(imgs_out, "fake_B", path_A))
    reporter.info('save fake_B at %s', os.path.join(imgs_out, "fake_B", path_A))
    reporter.end_predict()
    args.data_dir = 'testB'
    ds = create_dataset(args)
    reporter.dataset_size = args.dataset_size
    reporter.start_predict("B to A")
    for data in ds.create_dict_iterator(output_numpy=True):
        img_B = Tensor(data["image"])
        path_B = str(data["image_name"][0], encoding="utf-8")
        fake_A = G_B(img_B)
        save_image(fake_A, os.path.join(imgs_out, "fake_A", path_B))
    reporter.info('save fake_A at %s', os.path.join(imgs_out, "fake_A", path_B))
    reporter.end_predict()

if __name__ == "__main__":
    predict()
