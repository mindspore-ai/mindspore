# Copyright 2024 Huawei Technologies Co., Ltd
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
The tests of mindspore, used to test mult stream and res limit.
"""
import os
import pandas as pd
import glob
import shutil
import numpy as np
from tests.mark_utils import arg_mark
from mindspore import Tensor, ops, Parameter, nn, Profiler
from mindspore.runtime import Stream
from mindspore.profiler import ProfilerLevel, ProfilerActivity, AicoreMetrics
import mindspore as ms
ms.set_context(mode=ms.context.GRAPH_MODE, jit_config={'jit_level': 'O0'})
device_id = int(os.getenv('DEVICE_ID', '0'))
a = Tensor(np.ones([3, 3]), ms.float32)
b = Tensor(np.ones([3, 3]), ms.float32)

output_path='./stream_event_with_morph_prof'


def quick_find_matmulv3(cube_num):
    """
    Check res limit from profling data
    """
    # Find all kernel_details.csv files
    csv_files = glob.glob(output_path + '/**/kernel_details.csv', recursive=True)
    if not csv_files:
        print("No kernel_details.csv files found")
        return
    print(f"Found {len(csv_files)} kernel_details.csv files")

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        # Simple search for MatMulV3
        matmulv3_mask = df['Type'].astype(str).str.upper().str.contains('MATMULV3', na=False)
        matmulv3_data = df[matmulv3_mask]
        if len(matmulv3_data) > 0:
            print(f"\nFile: {csv_file}")
            print(f"Found {len(matmulv3_data)} MatMulV3 records:")
            # Filter for Block Dim = cube_num
            block_dim_cube_num_data = matmulv3_data[matmulv3_data['Block Dim'] == cube_num]
            assert len(block_dim_cube_num_data) != 0
            assert len(block_dim_cube_num_data) == 12
            # Get unique Stream IDs with Block Dim=cube_num
            unique_stream_ids_cube_num = block_dim_cube_num_data['Stream ID'].unique()

            print("\nBlock Dim = cube_num analysis:")
            print(f"  Number of records with Block Dim=cube_num: {len(block_dim_cube_num_data)}")
            print(f"  Unique Stream IDs with Block Dim=cube_num: {unique_stream_ids_cube_num}")
            # Add assertion that all Stream IDs with Block Dim=cube_num are the same
            assert len(unique_stream_ids_cube_num) == 1
            stream_id_cube_num = unique_stream_ids_cube_num[0]

            # Filter for Block Dim != cube_num
            block_dim_not_cube_num_data = matmulv3_data[matmulv3_data['Block Dim'] != cube_num]
            assert len(block_dim_not_cube_num_data) != 0
            # Get unique Stream IDs with Block Dim != cube_num
            unique_stream_ids_not_cube_num = block_dim_not_cube_num_data['Stream ID'].unique()

            print("\nBlock Dim != cube_num analysis:")
            print(f"  Number of records with Block Dim != cube_num: {len(block_dim_not_cube_num_data)}")
            print(f"  Unique Stream IDs with Block Dim != cube_num: {unique_stream_ids_not_cube_num}")
            # Add assertion that no Stream IDs with Block Dim != cube_num are the same as the Block Dim=cube_num Stream ID
            assert stream_id_cube_num not in unique_stream_ids_not_cube_num

def test_with_stream_event_with_morph():
    """
    Feature: Support with stream event in morph.
    Description: Support with stream event in morph.
    Expectation: Run success.
    """
    s1 = Stream()
    s2 = Stream()
    s3 = Stream()
    s4 = Stream()
    s5 = Stream()

    def infer_dtype(args):
        return args


    def infer_shape(args):
        return args


    def mul_by(*args):
        def inner(input_data):
            event = ms.runtime.Event()
            event = ops.Depend()(event, input_data)
            event = event.record()
            output = []
            with ms.runtime.StreamCtx(s1):
                with ms.runtime.StreamLimitCtx(s1, 4, 8):
                    event_end_recv = event.wait()
                    x = ops.Depend()(input_data, event_end_recv)
                    x = ops.matmul(x, x)
                    output.append(x)
                    event1 = ms.runtime.Event()
                    event1 = ops.Depend()(event1, x)
                    event1 = event1.record()

            with ms.runtime.StreamCtx(s2):
                with ms.runtime.StreamLimitCtx(s2, 4, 8):
                    event_end_recv = event.wait()
                    x = ops.Depend()(input_data, event_end_recv)
                    x = ops.matmul(x, x+1)
                    output.append(x)
                    event2 = ms.runtime.Event()
                    event2 = ops.Depend()(event2, x)
                    event2 = event2.record()

            with ms.runtime.StreamCtx(s3):
                with ms.runtime.StreamLimitCtx(s3, 4, 8):
                    event_end_recv = event.wait()
                    x = ops.Depend()(input_data, event_end_recv)
                    x = ops.matmul(x, x+2)
                    output.append(x)
                    event3 = ms.runtime.Event()
                    event3 = ops.Depend()(event3, x)
                    event3 = event3.record()

            with ms.runtime.StreamCtx(s4):
                with ms.runtime.StreamLimitCtx(s4, 4, 8):
                    event_end_recv = event.wait()
                    x = ops.Depend()(input_data, event_end_recv)
                    x = ops.matmul(x, x+3)
                    output.append(x)
                    event4 = ms.runtime.Event()
                    event4 = ops.Depend()(event4, x)
                    event4 = event4.record()

            with ms.runtime.StreamCtx(s5):
                with ms.runtime.StreamLimitCtx(s5, 4, 10):
                    event_end_recv = event.wait()
                    x = ops.Depend()(input_data, event_end_recv)
                    x = ops.matmul(x, x+4)
                    output.append(x)
                    event5 = ms.runtime.Event()
                    event5 = ops.Depend()(event5, x)
                    event5 = event5.record()

            event_end_recv = event1.wait()
            output = ops.Depend()(output, event_end_recv)
            event_end_recv = event2.wait()
            output = ops.Depend()(output, event_end_recv)
            event_end_recv = event3.wait()
            output = ops.Depend()(output, event_end_recv)
            event_end_recv = event4.wait()
            output = ops.Depend()(output, event_end_recv)
            event_end_recv = event5.wait()
            output = ops.Depend()(output, event_end_recv)
            output = output[0] + output[1] - output[2]
            return output * 0.0000000001
        return inner


    class MorphNet(nn.Cell):
        def __init__(self):
            super(MorphNet, self).__init__() # pylint: disable=super-with-arguments
            self.weight0 = Parameter(ops.ones((8192, 8192), dtype=ms.float32), name="weight0")
            self.weight1 = Parameter(ops.ones((8192, 8192), dtype=ms.float32), name="weight1")
            self.mul_by_100 = ops.Morph(mul_by(100), infer_shape, infer_dtype)

        def construct(self, x):
            input_a = ops.matmul(x, self.weight0)
            input_a = input_a * 0.0
            input_a = ops.matmul(input_a, self.weight0)
            out = self.mul_by_100(input_a)
            out = out * self.weight1
            return out
    graphs_path="./test_with_stream_event_with_morph"
    ms.set_context(jit_config={"jit_level": "O0"}, save_graphs=3, save_graphs_path=graphs_path)
    input_x = ops.ones((8192, 8192), dtype=ms.float32)
    profiler = Profiler(profiler_level=ProfilerLevel.Level2,
                        activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],with_stack=True,
                        aic_metrics=AicoreMetrics.PipeUtilization)
    net = MorphNet()
    cell_ls = ms.nn.SequentialCell()
    for _ in range(6):
        cell_ls.append(net)
    print("net", cell_ls(input_x))
    profiler.analyse()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_with_stream_event_with_morph_prof_mem_track():
    """
    Feature: runtime with stream and res limit.
    Description: Test with stream and res limit api.
    Expectation: profile as expected.
    """
    env = "export MS_ALLOC_CONF='memory_tracker:True'"
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/test_with_stream_event_with_morph.py"
    assert os.path.exists(script)
    case_name = "test_with_stream_event_with_morph"
    output = real_path + "/test_with_stream_event_with_morph.log"

    cube_num=16
    vector_num=16
    ms.runtime.set_device_limit(device_id, cube_num, vector_num)

    cmd = f"{env}; pytest -sv {script}::{case_name} > {output} 2>&1"
    return_code = os.system(cmd)

    assert os.path.exists(output)
    with open(output, "r", encoding='utf-8') as f:
        output_log = f.read()
        print(output_log, flush=True)
    assert return_code == 0
    assert "RaceChecker: Read error" not in output_log
    assert "RaceChecker: Write error" not in output_log
    quick_find_matmulv3(cube_num)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
