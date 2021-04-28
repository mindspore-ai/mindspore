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
"""train resnet."""
import os
import argparse
import numpy as np
from mindspore import Tensor
from mindspore import context
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint, load_param_into_net

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--net', type=str, default=None, help='Resnet Model, either resnet18, '
                                                          'resnet50 or resnet101')
parser.add_argument('--dataset', type=str, default=None, help='Dataset, imagenet2012')

parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint file path')
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
parser.add_argument('--device_target', type=str, default='Ascend', choices=("Ascend", "GPU", "CPU"),
                    help="Device target, support Ascend, GPU and CPU.")
args_opt = parser.parse_args()

set_seed(1)

if args_opt.dataset != "imagenet2012":
    raise ValueError("Currently only support imagenet2012 dataset format")
if args_opt.net in ("resnet18", "resnet50"):
    if args_opt.net == "resnet18":
        from src.resnet import resnet18 as resnet
    if args_opt.net == "resnet50":
        from src.resnet import resnet50 as resnet
    from src.config import config2 as config
    from src.dataset_infer import create_dataset

elif args_opt.net == "resnet101":
    from src.resnet import resnet101 as resnet
    from src.config import config3 as config
    from src.dataset_infer import create_dataset2 as create_dataset
else:
    from src.resnet import se_resnet50 as resnet
    from src.config import config4 as config
    from src.dataset_infer import create_dataset3 as create_dataset


def show_predict_info(label_list, prediction_list, filename_list, predict_ng):
    label_index = 0
    for label_index, predict_index, filename in zip(label_list, prediction_list, filename_list):
        filename = np.array(filename).tostring().decode('utf8')
        if label_index == -1:
            print("file: '{}' predict class id is: {}".format(filename, predict_index))
            continue
        if predict_index != label_index:
            predict_ng.append((filename, predict_index, label_index))
            print("file: '{}' predict wrong, predict class id is: {}, "
                  "label is {}".format(filename, predict_index, label_index))
    return predict_ng, label_index


if __name__ == '__main__':
    target = args_opt.device_target

    # init context
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)
    if target == "Ascend":
        device_id = int(os.getenv('DEVICE_ID'))
        context.set_context(device_id=device_id)

    # create dataset
    dataset = create_dataset(dataset_path=args_opt.dataset_path, do_train=False, batch_size=config.batch_size,
                             target=target)
    step_size = dataset.get_dataset_size()

    # define net
    net = resnet(class_num=config.class_num)

    # load checkpoint
    param_dict = load_checkpoint(args_opt.checkpoint_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    print("start infer")
    predict_negative = []
    total_sample = step_size * config.batch_size
    only_file = 0
    data_loader = dataset.create_dict_iterator(output_numpy=True, num_epochs=1)
    for i, data in enumerate(data_loader):
        images = data["image"]
        label = data["label"]
        file_name = data["filename"]
        res = net(Tensor(images))
        res = res.asnumpy()
        predict_id = np.argmax(res, axis=1)
        predict_negative, only_file = show_predict_info(label.tolist(), predict_id.tolist(),
                                                        file_name.tolist(), predict_negative)

    if only_file != -1:
        print(f"total {total_sample} data, top1 acc is {(total_sample - len(predict_negative)) * 1.0 / total_sample}")
    else:
        print("infer completed")
