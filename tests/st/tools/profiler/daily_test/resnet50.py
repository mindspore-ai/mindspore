# Copyright 2025 Huawei Technologies Co., Ltd
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
"""resnet50"""
import mindspore as ms
from mindspore import nn
from mindspore.ops import operations as P
from mindspore.profiler import mstx
import mindspore.dataset as ds
from mindspore.profiler import ProfilerLevel, ProfilerActivity, AicoreMetrics, ExportType
from mindspore.profiler import HostSystem


class Config:
    """profiler config."""
    run_distribute = True

    # profiler相关
    profiler = False
    # profiler
    start_step = 1
    end_step = 10
    profiler_dir = "./profiler_output"
    profiler_level = ProfilerLevel.Level2
    activities = [ProfilerActivity.CPU, ProfilerActivity.NPU]
    aicore_metrics = AicoreMetrics.PipeUtilization
    with_stack = True
    profile_memory = True
    data_process = True
    parallel_strategy = True
    start_profile = True
    l2_cache = True
    hbm_ddr = True
    sys_interconnection = False
    sys_io = False
    pcie = True
    sync_enable = True
    data_simplification = False
    mstx = False
    dynamic_profiler = False
    add_metadata = True
    analyse_flag = True
    async_mode = False
    mstx_domain_exclude = []
    mstx_domain_include = []
    record_shapes = False
    host_sys = [HostSystem.CPU]
    wait = 0
    warmup = 0
    active = 1
    repeat = 1
    skip_first = 0
    dir_name = "./data_prof"
    worker_name = ""
    export_type = [ExportType.Text]
    cfg_paths = "./"


def create_dataset(data_dir, batch_size=32, train_image_size=224):
    """Build Cifar10 dataset with preprocessing and batching."""
    data_set = ds.Cifar10Dataset(data_dir, shuffle=True)
    trans = []
    trans += [
        ds.vision.Resize((train_image_size, train_image_size)),
        ds.vision.Rescale(1.0 / 255.0, 0.0),
        ds.vision.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ds.vision.HWC2CHW()
    ]
    type_cast_op = ds.transforms.transforms.TypeCast(ms.int32)
    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)
    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=8)
    data_set = data_set.batch(batch_size, drop_remainder=True)
    return data_set


def conv3x3(in_channels, out_channels, stride=1, padding=1, pad_mode='pad'):
    """3x3 convolution."""
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=3, stride=stride, padding=padding, pad_mode=pad_mode)


def conv1x1(in_channels, out_channels, stride=1, padding=0, pad_mode='pad'):
    """1x1 convolution."""
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=1, stride=stride, padding=padding, pad_mode=pad_mode)


class ResidualBlock(nn.Cell):
    """
    residual Block.
    """
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 down_sample=False):
        super().__init__()

        out_chls = out_channels // self.expansion
        self.conv1 = conv1x1(in_channels, out_chls, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_chls)

        self.conv2 = conv3x3(out_chls, out_chls, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_chls)

        self.conv3 = conv1x1(out_chls, out_channels, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
        self.downsample = down_sample

        self.conv_down_sample = conv1x1(in_channels, out_channels,
                                        stride=stride, padding=0)
        self.bn_down_sample = nn.BatchNorm2d(out_channels)
        self.add = P.Add()

    def construct(self, x):
        """
        :param x:
        :return:
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample:
            identity = self.conv_down_sample(identity)
            identity = self.bn_down_sample(identity)

        out = self.add(out, identity)
        out = self.relu(out)

        return out


class ResNet50(nn.Cell):
    """
    resnet nn.Cell
    """

    def __init__(self, block, num_classes=100):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, pad_mode='pad')
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid')

        self.layer1 = self.MakeLayer(
            block, 3, in_channels=64, out_channels=256, stride=1)
        self.layer2 = self.MakeLayer(
            block, 4, in_channels=256, out_channels=512, stride=2)
        self.layer3 = self.MakeLayer(
            block, 6, in_channels=512, out_channels=1024, stride=2)
        self.layer4 = self.MakeLayer(
            block, 3, in_channels=1024, out_channels=2048, stride=2)

        self.avgpool = nn.AvgPool2d(7, 1)
        self.flatten = P.Flatten()
        self.fc = nn.Dense(512 * block.expansion, num_classes)
        self.cs = ms.runtime.current_stream()

    def MakeLayer(self, block, layer_num, in_channels, out_channels, stride):
        """Build a sequence of residual blocks."""
        layers = []
        resblk = block(in_channels, out_channels,
                       stride=stride, down_sample=True)
        layers.append(resblk)

        for _ in range(1, layer_num):
            resblk = block(out_channels, out_channels, stride=1)
            layers.append(resblk)

        return nn.SequentialCell(layers)

    def construct(self, x):
        """
        :param x:
        :return:
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        tl_id3 = mstx.range_start('Reduce and Fullconnect', stream=self.cs, domain="first_domain")
        x = self.avgpool(x)
        mstx.mark('End reduce', stream=self.cs, domain="first2_domain")
        x = self.flatten(x)
        mstx.mark('Start fullconnect', stream=self.cs, domain="first2_domain")
        x = self.fc(x)
        mstx.range_end(tl_id3, domain="first_domain")

        return x


def create_resnet50():
    """Instantiate ResNet50 with ResidualBlock and 10 classes."""
    return ResNet50(ResidualBlock, 10)
