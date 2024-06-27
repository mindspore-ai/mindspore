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
import argparse
import sys
import os
import glob
import pandas as pd
import numpy as np

workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(workspace, "mindformers"))
from mindspore import set_seed
from mindspore import Profiler
from mindformers.tools.register import MindFormerConfig
from mindformers import LlamaConfig, TransformerOpParallelConfig, LlamaForCausalLM, init_context

TOELERANCE = 5e-2


def str2bool(b):
    """String convert to Bool."""
    if b.lower() in ["false"]:
        output = False
    elif b.lower() in ["true"]:
        output = True
    else:
        raise Exception("Invalid Bool Value")
    return output


def generate_input_ids(batch_size, input_seq_lenth):
    return [[i+1 for i in range(input_seq_lenth)] for _ in range(batch_size)]


def get_total_time_from_profiler_file(file_path):
    if not file_path:
        raise FileExistsError(f"profiler file not exist")
    op_static = pd.read_csv(file_path)
    total_time = op_static['Total Time(us)'].sum()
    return total_time


def build_model(config_path, batch_size=1, model_parallel=1, use_bf16=False):
    set_seed(0)
    np.random.seed(0)
    # set model config
    config = MindFormerConfig(config_path)

    if model_parallel == 1:
        config.use_parallel = False
        device_id = int(os.getenv('DEVICE_ID', '0'))
        config.context.device_id = device_id
    else:
        # set parallel method
        config.use_parallel = True
        config.parallel_config.data_parallel = 1
        config.parallel_config.model_parallel = model_parallel
        config.parallel_config.pipeline_stage = 1
        print(config.parallel_config)

    if use_bf16:
        config.model.model_config.compute_dtype = "bfloat16"
        config.model.model_config.layernorm_compute_type = "bfloat16"
        config.model.model_config.softmax_compute_type = "bfloat16"
        config.model.model_config.rotary_dtype = "bfloat16"
        config.model.model_config.param_init_type = "bfloat16"

    # initialize env
    init_context(use_parallel=config.use_parallel,
                 context_config=config.context,
                 parallel_config=config.parallel)

    model_config = LlamaConfig(**config.model.model_config)

    # set model parameters
    model_config.parallel_config = TransformerOpParallelConfig(
        **config.parallel_config)
    model_config.batch_size = batch_size

    # build model from config
    model = LlamaForCausalLM(model_config)
    model.set_train(False)
    return model


def run_baichuan_1p_bs1(args):
    model = build_model(args.yaml_file, batch_size=1, model_parallel=1)

    inputs = ["I love Beijing, because"]
    inputs = inputs * 10

    inputs_ids = generate_input_ids(1, 10)
    outputs = model.generate(inputs_ids,
                             max_length=20,
                             do_sample=False)
    EXPECT_RES = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10834, 70913, 98066,
                           98066, 98066, 49616, 57632, 72154, 83812, 96142], dtype=np.int32)
    for output in outputs:
        assert (EXPECT_RES == output).all()

    inputs_ids = generate_input_ids(4, 12)
    outputs = model.generate(inputs_ids,
                             max_length=20,
                             do_sample=False)
    EXPECT_RES = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 106790,
                           98066, 59567, 83812, 59567, 70142, 31296, 73598], dtype=np.int32)
    for output in outputs:
        assert (EXPECT_RES == output).all()


def run_baichuan_1p_bs4(args):
    model = build_model(args.yaml_file, batch_size=4, model_parallel=1)

    inputs_ids = generate_input_ids(4, 10)
    outputs = model.generate(inputs_ids, max_length=20, do_sample=False)
    EXPECT_RES = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10834, 70913, 98066,
                           98066, 98066, 49616, 57632, 72154, 83812, 96142], dtype=np.int32)
    for output in outputs:
        assert (EXPECT_RES == output).all()

    inputs_ids = generate_input_ids(8, 12)
    outputs = model.generate(inputs_ids, max_length=20, do_sample=False)
    EXPECT_RES = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 106790,
                           98066, 59567, 83812, 59567, 70142, 31296, 73598], dtype=np.int32)
    for output in outputs:
        assert (EXPECT_RES == output).all()


def run_baichuan_4p_bs1(args):
    model = build_model(args.yaml_file, batch_size=1, model_parallel=4)

    inputs_ids = generate_input_ids(4, 10)
    outputs = model.generate(inputs_ids, max_length=20, do_sample=False)
    EXPECT_RES = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 87776, 103235, 43223,
                           37261, 77591, 22591, 117783, 109749, 121313, 33714], dtype=np.int32)
    for output in outputs:
        assert (EXPECT_RES == output).all()

    inputs_ids = generate_input_ids(8, 12)
    outputs = model.generate(inputs_ids, max_length=20, do_sample=False)
    EXPECT_RES = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 33069,
                           7065, 94742, 69391, 27691, 50605, 71209, 44256], dtype=np.int32)
    for output in outputs:
        assert (EXPECT_RES == output).all()


def run_baichuan_4p_bs4_bf16(args):
    model = build_model(args.yaml_file, batch_size=4,
                        model_parallel=4, use_bf16=True)

    inputs_ids = generate_input_ids(4, 10)
    model.generate(inputs_ids, max_length=20, do_sample=False)

    inputs_ids = generate_input_ids(8, 12)
    model.generate(inputs_ids, max_length=20, do_sample=False)


def run_baichuan_4p_bs4(args):
    model = build_model(args.yaml_file, batch_size=4, model_parallel=4)

    inputs_ids = generate_input_ids(4, 10)
    outputs = model.generate(inputs_ids,
                             max_length=20,
                             do_sample=False)
    EXPECT_RES = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 87776, 103235, 43223,
                           37261, 77591, 22591, 117783, 109749, 121313, 33714], dtype=np.int32)
    for output in outputs:
        assert (EXPECT_RES == output).all()

    profiler_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "predict_profiler")
    os.system(f"rm -rf {profiler_path}")
    profiler = Profiler(start_profile=False, output_path=profiler_path)
    profiler.start()
    inputs_ids = generate_input_ids(8, 12)
    outputs = model.generate(inputs_ids,
                             max_length=20,
                             do_sample=False)
    EXPECT_RES = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 33069,
                           7065, 94742, 69391, 27691, 50605, 71209, 44256], dtype=np.int32)
    profiler.stop()
    profiler.analyse()
    for output in outputs:
        assert (EXPECT_RES == output).all()

    profiler_file = glob.glob(os.path.join(os.path.join(
        profiler_path, "**"), "op_statistic_*.csv"), recursive=True)[0]
    expect_total_time = 23340
    total_time = get_total_time_from_profiler_file(profiler_file)
    print(f"total_time: {total_time}")
    assert total_time <= expect_total_time * (1 + TOELERANCE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_file', default="", type=str,
                        help='predict yaml path')
    parser.add_argument('--test_mode', default='', type=str,
                        help='run test mode.')
    args_ = parser.parse_args()
    test_mode = args_.test_mode
    if test_mode == "test_baichuan_1p_bs1":
        run_baichuan_1p_bs1(args_)
    if test_mode == "test_baichuan_1p_bs4":
        run_baichuan_1p_bs4(args_)
    if test_mode == "test_baichuan_4p_bs1":
        run_baichuan_4p_bs1(args_)
    if test_mode == "test_baichuan_4p_bs4":
        run_baichuan_4p_bs4(args_)
    if test_mode == "test_baichuan_4p_bs4_bf16":
        run_baichuan_4p_bs4_bf16(args_)
