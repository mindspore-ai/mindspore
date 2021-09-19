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
######################## train YOLOv3_DARKNET53 example ########################
train YOLOv3 and get network model files(.ckpt) :
python train.py --image_dir /data --anno_path /data/coco/train_coco.txt --mindrecord_dir=/data/Mindrecord_train
If the mindrecord_dir is empty, it wil generate mindrecord file by image_dir and anno_path.
Note if mindrecord_dir isn't empty, it will use mindrecord_dir rather than image_dir and anno_path.
"""
import os
import time
import re
import pytest
import numpy as np

from mindspore import context, Tensor
from mindspore.common.initializer import initializer
from mindspore.train.callback import Callback
from mindspore.context import ParallelMode
from mindspore.nn.optim.momentum import Momentum
import mindspore as ms

from src.yolo import YOLOV3DarkNet53, YoloWithLossCell, TrainingWrapper
from src.util import AverageMeter, get_param_groups
from src.lr_scheduler import warmup_cosine_annealing_lr
from src.yolo_dataset import create_yolo_dataset
from src.initializer import default_recurisive_init
from src.config import ConfigYOLOV3DarkNet53
from tests.st.model_zoo_tests import utils

np.random.seed(1)
def get_lr(learning_rate, start_step, global_step, decay_step, decay_rate, steps=False):
    """Set learning rate."""
    lr_each_step = []
    for i in range(global_step):
        if steps:
            lr_each_step.append(learning_rate * (decay_rate ** (i // decay_step)))
        else:
            lr_each_step.append(learning_rate * (decay_rate ** (i / decay_step)))
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    lr_each_step = lr_each_step[start_step:]
    return lr_each_step

def init_net_param(network, init_value='ones'):
    """Init:wq the parameters in network."""
    params = network.trainable_params()
    for p in params:
        if isinstance(p.data, Tensor) and 'beta' not in p.name and 'gamma' not in p.name and 'bias' not in p.name:
            p.set_parameter_data(initializer(init_value, p.data.shape, p.data.dtype))

class ModelCallback(Callback):
    def __init__(self):
        super(ModelCallback, self).__init__()
        self.loss_list = []

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        self.loss_list.append(cb_params.net_outputs.asnumpy())
        print("epoch: {}, outputs are: {}".format(cb_params.cur_epoch_num, str(cb_params.net_outputs)))


class TimeMonitor(Callback):
    """Time Monitor."""
    def __init__(self, data_size):
        super(TimeMonitor, self).__init__()
        self.data_size = data_size
        self.epoch_mseconds_list = []
        self.per_step_mseconds_list = []

    def epoch_begin(self, run_context):
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        epoch_mseconds = (time.time() - self.epoch_time) * 1000
        self.epoch_mseconds_list.append(epoch_mseconds)
        self.per_step_mseconds_list.append(epoch_mseconds / self.data_size)


DATA_DIR = "/home/workspace/mindspore_dataset/coco/coco2014/"

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_yolov3_darknet53():
    devid = int(os.getenv('DEVICE_ID')) if os.getenv('DEVICE_ID') else 0
    context.set_context(mode=context.GRAPH_MODE, enable_auto_mixed_precision=True,
                        device_target="Ascend", device_id=devid)

    rank = 0
    device_num = 1
    lr_init = 0.001
    epoch_size = 3
    batch_size = 32
    loss_scale = 1024
    mindrecord_dir = DATA_DIR
    # It will generate mindrecord file in args_opt.mindrecord_dir,
    # and the file name is yolo.mindrecord0, 1, ... file_num.
    if not os.path.isdir(mindrecord_dir):
        raise KeyError("mindrecord path is not exist.")
    data_root = os.path.join(mindrecord_dir, 'train2014')
    annFile = os.path.join(mindrecord_dir, 'annotations/instances_train2014.json')
    # print("yolov3 mindrecord is ", mindrecord_file)
    if not os.path.exists(annFile):
        print("instances_train2014 file is not exist.")
        assert False
    loss_meter = AverageMeter('loss')
    context.reset_auto_parallel_context()
    parallel_mode = ParallelMode.STAND_ALONE
    context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=1)
    network = YOLOV3DarkNet53(is_training=True)
    # default is kaiming-normal
    default_recurisive_init(network)
    network = YoloWithLossCell(network)
    print('finish get network')

    config = ConfigYOLOV3DarkNet53()
    label_smooth = 0
    label_smooth_factor = 0.1
    config.label_smooth = label_smooth
    config.label_smooth_factor = label_smooth_factor
    # When create MindDataset, using the fitst mindrecord file, such as yolo.mindrecord0.
    print("Create dataset begin!")
    training_shape = [int(416), int(416)]
    config.multi_scale = [training_shape]
    num_samples = 256
    ds, data_size = create_yolo_dataset(image_dir=data_root, anno_path=annFile, is_training=True,
                                        batch_size=batch_size, max_epoch=epoch_size,
                                        device_num=device_num, rank=rank, config=config, num_samples=num_samples)
    print("Create dataset done!")
    per_batch_size = batch_size
    group_size = 1
    print("data_size:", data_size)
    steps_per_epoch = int(data_size / per_batch_size / group_size)
    print("steps_per_epoch:", steps_per_epoch)

    warmup_epochs = 0.
    max_epoch = epoch_size
    T_max = 1
    eta_min = 0
    lr = warmup_cosine_annealing_lr(lr_init,
                                    steps_per_epoch,
                                    warmup_epochs,
                                    max_epoch,
                                    T_max,
                                    eta_min)

    opt = Momentum(params=get_param_groups(network),
                   learning_rate=Tensor(lr),
                   momentum=0.9,
                   weight_decay=0.0005,
                   loss_scale=loss_scale)

    network = TrainingWrapper(network, opt)
    network.set_train()
    old_progress = -1
    t_end = time.time()
    data_loader = ds.create_dict_iterator(output_numpy=True)
    train_starttime = time.time()
    time_used_per_epoch = 0
    print("time:", time.time())
    for i, data in enumerate(data_loader):
        images = data["image"]
        input_shape = images.shape[2:4]
        print('iter[{}], shape{}'.format(i, input_shape[0]))
        images = Tensor.from_numpy(images)
        batch_y_true_0 = Tensor.from_numpy(data['bbox1'])
        batch_y_true_1 = Tensor.from_numpy(data['bbox2'])
        batch_y_true_2 = Tensor.from_numpy(data['bbox3'])
        batch_gt_box0 = Tensor.from_numpy(data['gt_box1'])
        batch_gt_box1 = Tensor.from_numpy(data['gt_box2'])
        batch_gt_box2 = Tensor.from_numpy(data['gt_box3'])
        input_shape = Tensor(tuple(input_shape[::-1]), ms.float32)
        loss = network(images, batch_y_true_0, batch_y_true_1, batch_y_true_2, batch_gt_box0, batch_gt_box1,
                       batch_gt_box2, input_shape)
        loss_meter.update(loss.asnumpy())
        if (i + 1) % steps_per_epoch == 0:
            time_used = time.time() - t_end
            epoch = int(i / steps_per_epoch)
            fps = per_batch_size * (i - old_progress) * group_size / time_used
            if rank == 0:
                print(
                    'epoch[{}], iter[{}], {}, {:.2f} imgs/sec, lr:{}, time_used:{}'.format(epoch,
                                                                                           i, loss_meter, fps, lr[i],
                                                                                           time_used))
            t_end = time.time()
            loss_meter.reset()
            old_progress = i
            time_used_per_epoch = time_used

    train_endtime = time.time() - train_starttime
    print('train_time_used:{}'.format(train_endtime))
    expect_loss_value = 3210.0
    loss_value = re.findall(r"\d+\.?\d*", str(loss_meter))
    print('loss_value:{}'.format(loss_value[0]))
    assert float(loss_value[0]) < expect_loss_value
    export_time_used = 20.0
    print('time_used_per_epoch:{}'.format(time_used_per_epoch))
    assert time_used_per_epoch < export_time_used
    print('==========test case passed===========')


@pytest.mark.level1
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_single
def test_yolov3_darknet_8p():
    cur_path = os.path.dirname(os.path.abspath(__file__))
    model_path = "{}/../../../../tests/models/official/cv".format(cur_path)
    model_name = "yolov3_darknet53"
    dataset_path = os.path.join(utils.data_root, "coco/coco2014/")
    ckpt_path = os.path.join(utils.ckpt_root, "yolov3_darknet/yolov3_darknet53_pretrain.ckpt")
    utils.copy_files(model_path, cur_path, model_name)
    cur_model_path = os.path.join(cur_path, model_name)
    train_file = os.path.join(cur_model_path, "train.py")
    old_list = ["--lr_scheduler=cosine_annealing"]
    new_list = ["--lr_scheduler=cosine_annealing --training_shape=416 --log_interval=10"]
    utils.exec_sed_command(old_list, new_list,
                           os.path.join(cur_model_path, "scripts/run_distribute_train.sh"))
    old_list = ["max_epoch=config.max_epoch"]
    new_list = ["max_epoch=1"]
    utils.exec_sed_command(old_list, new_list, train_file)
    old_list = ["sampler=distributed_sampler"]
    new_list = ["sampler=distributed_sampler, num_samples=100*batch_size"]
    utils.exec_sed_command(old_list, new_list, os.path.join(cur_model_path, "src/yolo_dataset.py"))
    exec_network_shell = "cd yolov3_darknet53/scripts; bash run_distribute_train.sh {0} {1} {2}"\
        .format(dataset_path, ckpt_path, utils.rank_table_path)
    os.system(exec_network_shell)
    cmd = "ps -ef | grep python | grep train.py | grep -v grep"
    ret = utils.process_check(120, cmd)
    assert ret
    train_log_file = os.path.join(cur_path, "yolov3_darknet53/scripts/train_parallel0/log.txt")
    pattern1 = r", *([\d\.]+) imgs/sec"
    pattern2 = r"loss:*([\d\.]+),"
    fps_list = utils.parse_log_file(pattern1, train_log_file)[1:]
    assert sum(fps_list) / len(fps_list) > 370
    loss_list = utils.parse_log_file(pattern2, train_log_file)
    assert loss_list[-1] < 280
