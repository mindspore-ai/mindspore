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

"""Evaluation for yolov3-resnet18"""
import os
import time
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.yolov3 import yolov3_resnet18, YoloWithEval
from src.dataset import create_yolo_dataset, data_to_mindrecord_byte_image
from src.config import ConfigYOLOV3ResNet18
from src.utils import metrics

from model_utils.config import config as default_config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_device_num

def yolo_eval(dataset_path, ckpt_path):
    """Yolov3 evaluation."""

    ds = create_yolo_dataset(dataset_path, is_training=False)
    config = ConfigYOLOV3ResNet18()
    net = yolov3_resnet18(config)
    eval_net = YoloWithEval(net, config)
    print("Load Checkpoint!")
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, param_dict)


    eval_net.set_train(False)
    i = 1.
    total = ds.get_dataset_size()
    start = time.time()
    pred_data = []
    print("\n========================================\n")
    print("total images num: ", total)
    print("Processing, please wait a moment.")
    for data in ds.create_dict_iterator(output_numpy=True, num_epochs=1):
        img_np = data['image']
        image_shape = data['image_shape']
        annotation = data['annotation']

        eval_net.set_train(False)
        output = eval_net(Tensor(img_np), Tensor(image_shape))
        for batch_idx in range(img_np.shape[0]):
            pred_data.append({"boxes": output[0].asnumpy()[batch_idx],
                              "box_scores": output[1].asnumpy()[batch_idx],
                              "annotation": annotation})
        percent = round(i / total * 100, 2)

        print('    %s [%d/%d]' % (str(percent) + '%', i, total), end='\r')
        i += 1
    print('    %s [%d/%d] cost %d ms' % (str(100.0) + '%', total, total, int((time.time() - start) * 1000)), end='\n')

    precisions, recalls = metrics(pred_data)
    print("\n========================================\n")
    for i in range(config.num_classes):
        print("class {} precision is {:.2f}%, recall is {:.2f}%".format(i, precisions[i] * 100, recalls[i] * 100))

def modelarts_pre_process():
    '''modelarts pre process function.'''
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, default_config.modelarts_dataset_unzip_name)):
            zip_isexist = zipfile.is_zipfile(zip_file)
            if zip_isexist:
                fz = zipfile.ZipFile(zip_file, 'r')
                data_num = len(fz.namelist())
                print("Extract Start...")
                print("unzip file num: {}".format(data_num))
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        print("unzip percent: {}%".format(int(i * 100 / data_num)), flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),
                                                     int(int(time.time() - s_time) % 60)))
                print("Extract Done.")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if default_config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(default_config.data_path, default_config.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(default_config.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most.
        if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
            print("Zip file path: ", zip_file_1)
            print("Unzip file save dir: ", save_dir_1)
            unzip(zip_file_1, save_dir_1)
            print("===Finish extract data synchronization===")
            try:
                os.mknod(sync_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(sync_lock):
                break
            time.sleep(1)

        print("Device: {}, Finish sync unzip data from {} to {}.".format(get_device_id(), zip_file_1, save_dir_1))


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_eval():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=get_device_id())

    # It will generate mindrecord file in default_config.eval_mindrecord_dir,
    # and the file name is yolo.mindrecord0, 1, ... file_num.
    if not os.path.isdir(default_config.eval_mindrecord_dir):
        os.makedirs(default_config.eval_mindrecord_dir)

    yolo_prefix = "yolo.mindrecord"
    mindrecord_file = os.path.join(default_config.eval_mindrecord_dir, yolo_prefix + "0")
    if not os.path.exists(mindrecord_file):
        if os.path.isdir(default_config.image_dir) and os.path.exists(default_config.anno_path):
            print("Create Mindrecord")
            data_to_mindrecord_byte_image(default_config.image_dir,
                                          default_config.anno_path,
                                          default_config.eval_mindrecord_dir,
                                          prefix=yolo_prefix,
                                          file_num=8)
            print("Create Mindrecord Done, at {}".format(default_config.eval_mindrecord_dir))
        else:
            print("image_dir or anno_path not exits")

    if not default_config.only_create_dataset:
        print("Start Eval!")
        yolo_eval(mindrecord_file, default_config.ckpt_path)


if __name__ == '__main__':
    run_eval()
