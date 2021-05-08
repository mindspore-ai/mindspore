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
"""YoloV3_quant export coco bin."""
import os
import argparse
import datetime
import numpy as np
import mindspore as ms
from mindspore import Tensor
from mindspore.context import ParallelMode
from mindspore import context

from src.logger import get_logger
from src.yolo_dataset import create_yolo_dataset
from src.config import ConfigYOLOV3DarkNet53

def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser('mindspore coco export bin')
    parser.add_argument('--device_target', type=str, default="Ascend",
                        choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented (default: Ascend)')
    # dataset related
    parser.add_argument('--data_dir', type=str, default="", help='Eval data dir. Default: ""')
    parser.add_argument('--per_batch_size', default=1, type=int, help='Batch size for per device, Default: 1')

    # logging related
    parser.add_argument('--log_path', type=str, default="outputs/", help='Log save location, Default: "outputs/"')
    parser.add_argument('--save_path', type=str, default="", help='Bin file save location')

    # detect_related
    parser.add_argument('--nms_thresh', type=float, default=0.5, help='Threshold for NMS. Default: 0.5')
    parser.add_argument('--annFile', type=str, default="", help='The path to annotation. Default: ""')
    parser.add_argument('--testing_shape', type=str, default="", help='Shape for test. Default: ""')

    args_, _ = parser.parse_known_args()

    args_.data_root = os.path.join(args_.data_dir, 'val2014')
    args_.annFile = os.path.join(args_.data_dir, 'annotations/instances_val2014.json')

    return args_

def conver_testing_shape(args_org):
    """Convert testing shape to list."""
    testing_shape = [int(args_org.testing_shape), int(args_org.testing_shape)]
    return testing_shape

if __name__ == "__main__":
    args = parse_args()
    devid = int(os.getenv('DEVICE_ID')) if os.getenv('DEVICE_ID') else 0
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=devid)

    # logger
    args.outputs_dir = os.path.join(args.log_path,
                                    datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    rank_id = int(os.environ.get('RANK_ID')) if os.environ.get('RANK_ID') else 0
    args.logger = get_logger(args.outputs_dir, rank_id)

    context.reset_auto_parallel_context()
    parallel_mode = ParallelMode.STAND_ALONE
    context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=1)

    config = ConfigYOLOV3DarkNet53()
    if args.testing_shape:
        config.test_img_shape = conver_testing_shape(args)
    data_root = args.data_root
    ann_file = args.annFile

    ds, data_size = create_yolo_dataset(data_root, ann_file, is_training=False, batch_size=args.per_batch_size,
                                        max_epoch=1, device_num=1, rank=rank_id, shuffle=False,
                                        config=config)

    args.logger.info('testing shape : {}'.format(config.test_img_shape))
    args.logger.info('totol {} images to eval'.format(data_size))

    cur_dir = args.save_path
    save_folder = os.path.join(cur_dir, "yolov3_quant_coco_310_infer_data")
    image_folder = os.path.join(save_folder, "image_bin")
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    list_image_shape = []
    list_image_id = []

    input_shape = Tensor(tuple(config.test_img_shape), ms.float32)
    args.logger.info('Start inference....')
    for i, data in enumerate(ds.create_dict_iterator()):
        image = data["image"].asnumpy()
        image_shape = data["image_shape"]
        image_id = data["img_id"]
        file_name = "YoloV3-DarkNet_coco_bs_" + str(args.per_batch_size) + "_" + str(i) + ".bin"
        file_path = image_folder + "/" + file_name
        image.tofile(file_path)
        list_image_shape.append(image_shape.asnumpy())
        list_image_id.append(image_id.asnumpy())
    shapes = np.array(list_image_shape)
    ids = np.array(list_image_id)
    np.save(save_folder + "/image_shape.npy", shapes)
    np.save(save_folder + "/image_id.npy", ids)
