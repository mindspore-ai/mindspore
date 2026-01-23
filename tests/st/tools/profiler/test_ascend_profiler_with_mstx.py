# Copyright 2025-2025 Huawei Technologies Co., Ltd
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
"""test ascend profiler with env mstx."""
import csv
import glob
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import mindspore as ms
from mindspore import dataset as ds
from mindspore import context, Tensor, Profiler
from mindspore.profiler import ProfilerLevel, ProfilerActivity, mstx
from model_zoo import TinyAddNet
from tests.mark_utils import arg_mark
from file_check import FileChecker

DATASET_PATH = "/home/workspace/mindspore_dataset/mnist"


# pylint: disable=protected-access
def plot(imgs, first_origin=None):
    num_rows = 1
    num_cols = len(imgs)

    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for idx, img in enumerate(imgs):
        ax = axs[0, idx]
        ax.imshow(img.asnumpy())
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if first_origin:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    plt.tight_layout()


def train(add):
    """ Train add net"""
    x = np.random.randn(1, 3, 3, 4).astype(np.float32)
    y = np.random.randn(1, 3, 3, 4).astype(np.float32)
    add(Tensor(x), Tensor(y))


def check_result(result_path: str):
    expect_csv_header = ['Device_id', 'pid', 'tid', 'category', 'event_type', 'payload_type', 'payload_value',
                         'Start_time(us)', 'End_time(us)', 'message_type', 'message', 'domain', 'Device Start_time(us)',
                         'Device End_time(us)']
    expect_save_checkpoint_msg = 'save_checkpoint'
    expect_dataset_msg = 'dataloader'
    msprof_path = glob.glob(f'{result_path}/*_ascend_ms/PROF_*')[0]
    msproftx_csv_path = glob.glob(f'{msprof_path}/mindstudio_profiler_output/msprof_tx_*.csv')[0]
    current_result = []
    with open(msproftx_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            current_result.append(row)
    for i, value in enumerate(current_result):
        if i == 0:
            assert value == expect_csv_header
        elif i == 1:
            assert value[10] == expect_save_checkpoint_msg
        else:
            assert value[10] == expect_dataset_msg


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_mstx_profiler():
    """
    Feature: Ascend Profiler for mstx data
    Description: Test Ascend Profiler for mstx data.
    Expectation: The profiler successfully collects data and generates the expected files.
    """
    with tempfile.TemporaryDirectory(suffix="_mstx_profiler") as tmpdir:
        context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")

        mnist_ds = ds.MnistDataset(os.path.join(DATASET_PATH, "train"))

        net = TinyAddNet()
        prof = Profiler(activities=[ProfilerActivity.NPU],
                        mstx=True,
                        profiler_level=ProfilerLevel.LevelNone,
                        output_path=tmpdir,
                        data_simplification=False,
                        mstx_domain_exclude=["model_preparation"])

        ms.save_checkpoint(net, "./add.ckpt")  # to get save checkpoint tx data

        # to get dataset tx data
        images = []
        for image, _ in mnist_ds:
            images.append(image)
            if len(images) > 5:
                break

        plot(images)

        prof.analyse()
        check_result(tmpdir)
        # Check profiler.log
        profiler_log_paths = glob.glob(f"{tmpdir}/*_ascend_ms/"
                                       f"logs/profiler_*.log")
        for profiler_log_path in profiler_log_paths:
            FileChecker.check_file_for_keyword(profiler_log_path, "error")


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=1024.00)
def test_mstx_profiler_with_domain_include():
    """
    Feature: Ascend Profiler for mstx domain data
    Description: Test Ascend Profiler for mstx domain data.
    Expectation: The profiler successfully collects domain data and generates the expected files.
    """
    with tempfile.TemporaryDirectory(suffix="_mstx_profiler") as tmpdir:
        context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")
        net = TinyAddNet()
        experimental_config = ms.profiler._ExperimentalConfig(
            profiler_level=ProfilerLevel.LevelNone,
            mstx=True,
            mstx_domain_include=["default"]
        )
        with ms.profiler.profile(activities=[ProfilerActivity.NPU],
                                 schedule=ms.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
                                 on_trace_ready=ms.profiler.tensorboard_trace_handler(dir_name=tmpdir),
                                 experimental_config=experimental_config) as profiler:
            for _ in range(1):
                stream = ms.runtime.current_stream()
                range_id1 = mstx.range_start("range1", stream)
                range_id2 = mstx.range_start("range2", None, "1")
                mstx.mark("mark1", stream)
                mstx.mark("mark2", None, "1")
                train(net)
                mstx.range_end(range_id1)
                mstx.range_end(range_id2, "1")
                profiler.step()

        trace_view_json_path = os.path.join(
            glob.glob(f"{tmpdir}/*_ascend_ms")[0],
            "ASCEND_PROFILER_OUTPUT",
            "trace_view.json"
        )
        FileChecker.check_timeline_values(
            trace_view_json_path,
            "name",
            [
                "range1"
            ],
        )
        FileChecker.check_timeline_values(
            trace_view_json_path,
            "name",
            [
                "mark1"
            ],
        )


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=1024.00)
def test_mstx_profiler_with_domain_exclude():
    """
    Feature: Ascend Profiler for mstx domain data
    Description: Test Ascend Profiler for mstx domain data.
    Expectation: The profiler successfully collects domain data and generates the expected files.
    """
    with tempfile.TemporaryDirectory(suffix="_mstx_profiler") as tmpdir:
        context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")
        net = TinyAddNet()
        experimental_config = ms.profiler._ExperimentalConfig(
            profiler_level=ProfilerLevel.LevelNone,
            mstx=True,
            mstx_domain_exclude=["default", "model_preparation"]
        )
        with ms.profiler.profile(activities=[ProfilerActivity.NPU],
                                 schedule=ms.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
                                 on_trace_ready=ms.profiler.tensorboard_trace_handler(dir_name=tmpdir),
                                 experimental_config=experimental_config) as profiler:
            for _ in range(1):
                stream = ms.runtime.current_stream()
                range_id1 = mstx.range_start("range1", stream)
                range_id2 = mstx.range_start("range2", None, "1")
                mstx.mark("mark1", stream)
                mstx.mark("mark2", None, "1")
                train(net)
                mstx.range_end(range_id1)
                mstx.range_end(range_id2, "1")
                profiler.step()

        trace_view_json_path = os.path.join(
            glob.glob(f"{tmpdir}/*_ascend_ms")[0],
            "ASCEND_PROFILER_OUTPUT",
            "trace_view.json"
        )
        FileChecker.check_timeline_values(
            trace_view_json_path,
            "name",
            [
                "range2"
            ],
        )
        FileChecker.check_timeline_values(
            trace_view_json_path,
            "name",
            [
                "mark2"
            ],
        )


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_mstx_profiler_with_domain_include_model_preparation():
    """
    Feature: Profiler with model preparation
    Description: Test Profiler with model preparation.
    Expectation: The profiler successfully collects model_preparation data and generates the expected files.
    """
    with tempfile.TemporaryDirectory(suffix="_profiler_with_model_preparation") as tmpdir:
        context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")
        net = TinyAddNet()
        experimental_config = ms.profiler._ExperimentalConfig(
            profiler_level=ProfilerLevel.LevelNone,
            mstx=True,
            mstx_domain_include=["model_preparation"]
        )
        prof = ms.profiler.profile(activities=[ProfilerActivity.NPU],
                                   on_trace_ready=ms.profiler.tensorboard_trace_handler(dir_name=tmpdir),
                                   experimental_config=experimental_config)
        prof.start()
        for _ in range(6):
            train(net)
            prof.step()
        ckpt_path = os.path.join(tmpdir, "add.ckpt")
        ms.save_checkpoint(net, ckpt_path)
        prof.stop()
        trace_view_json_path = os.path.join(
            glob.glob(f"{tmpdir}/*_ascend_ms")[0],
            "ASCEND_PROFILER_OUTPUT",
            "trace_view.json"
        )
        FileChecker.check_timeline_values(trace_view_json_path, "name", {"save_checkpoint", })
