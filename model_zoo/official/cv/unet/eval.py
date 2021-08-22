# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

import logging
from mindspore import context, Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.data_loader import create_dataset, create_multi_class_dataset
from src.unet_medical import UNetMedical
from src.unet_nested import NestedUNet, UNet
from src.utils import UnetEval, TempLoss, dice_coeff
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id

@moxing_wrapper()
def test_net(data_dir,
             ckpt_path,
             cross_valid_ind=1):
    if config.model_name == 'unet_medical':
        net = UNetMedical(n_channels=config.num_channels, n_classes=config.num_classes)
    elif config.model_name == 'unet_nested':
        net = NestedUNet(in_channel=config.num_channels, n_class=config.num_classes, use_deconv=config.use_deconv,
                         use_bn=config.use_bn, use_ds=False)
    elif config.model_name == 'unet_simple':
        net = UNet(in_channel=config.num_channels, n_class=config.num_classes)
    else:
        raise ValueError("Unsupported model: {}".format(config.model_name))
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, param_dict)
    net = UnetEval(net, eval_activate=config.eval_activate.lower())
    if hasattr(config, "dataset") and config.dataset != "ISBI":
        split = config.split if hasattr(config, "split") else 0.8
        valid_dataset = create_multi_class_dataset(data_dir, config.image_size, 1, 1,
                                                   num_classes=config.num_classes, is_train=False,
                                                   eval_resize=config.eval_resize, split=split,
                                                   python_multiprocessing=False, shuffle=False)
    else:
        _, valid_dataset = create_dataset(data_dir, 1, 1, False, cross_valid_ind, False,
                                          do_crop=config.crop, img_size=config.image_size)
    model = Model(net, loss_fn=TempLoss(), metrics={"dice_coeff": dice_coeff(show_eval=config.show_eval)})

    print("============== Starting Evaluating ============")
    eval_score = model.eval(valid_dataset, dataset_sink_mode=False)["dice_coeff"]
    print("============== Cross valid dice coeff is:", eval_score[0])
    print("============== Cross valid IOU is:", eval_score[1])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False)
    if config.device_target == "Ascend":
        device_id = get_device_id()
        context.set_context(device_id=device_id)
    test_net(data_dir=config.data_path,
             ckpt_path=config.checkpoint_file_path,
             cross_valid_ind=config.cross_valid_ind)
