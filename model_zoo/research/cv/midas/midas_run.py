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
"""export midas."""
import glob
import os
import cv2
from mindspore import context
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.train import serialization
import mindspore.ops as ops
from src.utils import transforms
import src.util as util
from src.config import config
from src.midas_net import MidasNet


def export():
    """export."""
    print("initialize")
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=2, save_graphs=False)
    net = MidasNet()
    param_dict = serialization.load_checkpoint(config.model_weights)
    serialization.load_param_into_net(net, param_dict)
    img_input_1 = transforms.Resize(config.img_width,
                                    config.img_height,
                                    resize_target=config.resize_target,
                                    keep_aspect_ratio=config.keep_aspect_ratio,
                                    ensure_multiple_of=config.ensure_multiple_of,
                                    resize_method=config.resize_method,
                                    image_interpolation_method=cv2.INTER_CUBIC)
    img_input_2 = transforms.NormalizeImage(mean=config.nm_img_mean, std=config.nm_img_std)
    img_input_3 = transforms.PrepareForNet()
    # get input
    img_names = glob.glob(os.path.join(config.input_path, "*"))
    num_images = len(img_names)
    # create output folder
    os.makedirs(config.output_path, exist_ok=True)

    print("start processing")
    expand_dims = ops.ExpandDims()
    resize_bilinear = ops.ResizeBilinear
    squeeze = ops.Squeeze()
    for ind, img_name in enumerate(img_names):
        print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))
        # input

        img = util.read_image(img_name)
        img_input = img_input_1({"image": img})
        img_input = img_input_2(img_input)
        img_input = img_input_3(img_input)["image"]
        sample = Tensor(img_input, mstype.float32)
        sample = expand_dims(sample, 0)
        prediction = net(sample)
        prediction = expand_dims(prediction, 1)
        prediction = resize_bilinear((img.shape[:2]))(prediction)
        prediction = squeeze(prediction).asnumpy()
        # output
        filename = os.path.join(
            config.output_path, os.path.splitext(os.path.basename(img_name))[0]
        )
        util.write_depth(filename, prediction, bits=2)

    print("finished")


if __name__ == "__main__":
    export()
