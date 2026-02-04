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
Test module for testing the paralleled infer interface used for mindformers.
How to run this:
    pytest tests/st/networks/large_models/test_parallel_predict.py
"""
import argparse
import os
import sys
import numpy as np
from similarity import compare_distance

workspace = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, os.path.join(workspace, "networks/mindformers"))

from mindspore.nn.utils import no_init_parameters

from mindformers.models.llama.llama_tokenizer_fast import LlamaTokenizerFast
from mindformers import build_context, MindFormerConfig, build_parallel_config, LlamaConfig
from mindformers.tools.logger import logger
from research.qwen2_5.qwen2_5_tokenizer import Qwen2Tokenizer
from research.qwen2_5.infer.qwen2_5 import (
    ParallelQwenForCausalLM as ParallelQwenForCausalLM_MF,
)
from research.deepseek3.deepseek3_config import DeepseekV3Config
from research.deepseek3.deepseek3_model_infer import InferenceDeepseekV3ForCausalLM
from deepseekv3_weight_processor import DeepseekV3WeightProcessor
from qwen2_weight_processor import Qwen2WeightProcessor

def parallel_qwen2_0_5b_predict_mp2():
    """test qwen2 0.5B predict in model_parallel=2 with dynamic shape"""
    # Temporarily disable the test cases and add exception checks, it is necessary to upgrade MindFormer
    # to change the Python layer tensor CPU input tensor to pinned memory.
    # ms.runtime.set_kernel_launch_group()
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(cur_dir, "qwen/configs/ci_predict_qwen2_0_5b_instruct.yaml")

    vocab_file_path = "/home/workspace/mindspore_dataset/weight/Qwen2-0.5B-Instruct/vocab.json"
    merges_file_path = "/home/workspace/mindspore_dataset/weight/Qwen2-0.5B-Instruct/merges.txt"
    load_checkpoint = "/home/workspace/mindspore_dataset/weight/Qwen2-0.5B-Instruct/"

    seq_length = 128
    # init config with yaml
    config = MindFormerConfig(config_path)
    config.use_parallel = True
    config.parallel_config.model_parallel = 2
    config.parallel_config.data_parallel = 1
    config.parallel_config.pipeline_stage = 1
    config.load_checkpoint = load_checkpoint
    config.model.model_config.seq_length = seq_length
    config.model.model_config.qkv_concat = False
    config.processor.tokenizer.vocab_file = vocab_file_path
    config.processor.tokenizer.merges_file = merges_file_path
    config.parallel.parallel_mode = "STAND_ALONE"

    # init context
    build_context(config)
    build_parallel_config(config)

    config.model.model_config.parallel_config = config.parallel_config
    model_config = LlamaConfig(**config.model.model_config)
    model_config.checkpoint_name_or_path = None

    # build tokenizer
    tokenizer = Qwen2Tokenizer(**config.processor.tokenizer)

    # build model
    with no_init_parameters():
        network = ParallelQwenForCausalLM_MF(model_config)

    # load checkpoint
    if config.load_checkpoint:
        logger.info("----------------Transform and load checkpoint----------------")
        weight_processor = Qwen2WeightProcessor(config, network, False)
        weight_processor.load_safetensors_shard(config.load_checkpoint)

    # predict
    batch_datas = {1: {"prompt": "你好！",
                       "answer": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n你好！"
                                 "<|im_end|>\n<|im_start|>assistant\n你好！有什么可以帮助你的吗？<|im_end|>"},
                   4: {"prompt": "用python编写快速排序",
                       "answer": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n用"
                                 "python编写快速排序<|im_end|>\n<|im_start|>assistant\n以下是一个使用Python实现的快速排序"
                                 "算法：\n\n```python\ndef quick_sort(arr):\n    if len(arr) <= 1:\n        "
                                 "return arr\n    else:\n        pivot = arr[0]\n        left = [x for x in arr[1:] "
                                 "if x < pivot]\n        right = [x for x in arr[1:] if x >= pivot]\n        "
                                 "return quick_sort(left) + [pivot] + quick_sort(right)\n\n# 示例输入\narr = [3,6,8,1"},
                   8: {"prompt": "I believe the meaning of life is",
                       "answer": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nI "
                                 "believe the meaning of life is<|im_end|>\n<|im_start|>assistant\nThe meaning of "
                                 "life is a philosophical question that has been debated for centuries, and there "
                                 "is no one definitive answer to it. Some people believe that the meaning of life "
                                 "is to find happiness and fulfillment in their lives, while others believe that it "
                                 "is to achieve success or recognition.\n\nOthers may argue that the meaning of life "
                                 "is to make a positive impact on the world, to help others, and to contribute to "
                                 "society as a whole. Others may believe that the meaning of life is to pursue "
                                 "knowledge and understanding, to"},
                   }
    for batch_size, batch_data in batch_datas.items():
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": batch_data["prompt"]}
        ]
        input_ids = tokenizer.encode(tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        ))
        input_ids_list = []
        answer = batch_data["answer"]
        for i in range(0, batch_size):
            input_ids_list.append(input_ids)
        outputs = network.generate(input_ids_list,
                                   max_length=seq_length,
                                   do_sample=False,
                                   return_dict_in_generate=False)

        for i in range(0, len(outputs)):
            output_text = tokenizer.decode(outputs[i])
            print("parallel_qwen2_0.5b_predict_mp2, output_text:", output_text)
            print("parallel_qwen2_0.5b_predict_mp2, answer:", answer)
            compare_distance(output_text, answer, bench_sim=0.95)


def parallel_qwen2_0_5b_predict_dp2_mp2():
    """test qwen2 0.5B predict in data_parallel=2 and model_parallel=2 with dynamic shape"""
    # Temporarily disable the test cases and add exception checks, it is necessary to upgrade MindFormer
    # to change the Python layer tensor CPU input tensor to pinned memory.
    # ms.runtime.set_kernel_launch_group()
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(cur_dir, "qwen/configs/ci_predict_qwen2_0_5b_instruct.yaml")

    vocab_file_path = "/home/workspace/mindspore_dataset/weight/Qwen2-0.5B-Instruct/vocab.json"
    merges_file_path = "/home/workspace/mindspore_dataset/weight/Qwen2-0.5B-Instruct/merges.txt"
    load_checkpoint = "/home/workspace/mindspore_dataset/weight/Qwen2-0.5B-Instruct/"

    seq_length = 128
    # init config with yaml
    config = MindFormerConfig(config_path)
    config.use_parallel = True
    config.parallel_config.model_parallel = 2
    config.parallel_config.data_parallel = 2
    config.parallel_config.pipeline_stage = 1
    config.load_checkpoint = load_checkpoint
    config.model.model_config.seq_length = seq_length
    config.model.model_config.qkv_concat = False
    config.processor.tokenizer.vocab_file = vocab_file_path
    config.processor.tokenizer.merges_file = merges_file_path
    config.parallel.parallel_mode = "STAND_ALONE"

    # init context
    build_context(config)
    build_parallel_config(config)

    config.model.model_config.parallel_config = config.parallel_config
    model_config = LlamaConfig(**config.model.model_config)
    model_config.checkpoint_name_or_path = None

    # build tokenizer
    tokenizer = Qwen2Tokenizer(**config.processor.tokenizer)

    # build model
    with no_init_parameters():
        network = ParallelQwenForCausalLM_MF(model_config)

    # load checkpoint
    if config.load_checkpoint:
        logger.info("----------------Transform and load checkpoint----------------")
        weight_processor = Qwen2WeightProcessor(config, network, False)
        weight_processor.load_safetensors_shard(config.load_checkpoint)

    # predict
    batch_datas = {1: {"prompt": "你好！",
                       "answer": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n你好！"
                                 "<|im_end|>\n<|im_start|>assistant\n你好！有什么可以帮助你的吗？<|im_end|>"},
                   4: {"prompt": "用python编写快速排序",
                       "answer": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n用"
                                 "python编写快速排序<|im_end|>\n<|im_start|>assistant\n以下是一个使用Python实现的快速排序"
                                 "算法：\n\n```python\ndef quick_sort(arr):\n    if len(arr) <= 1:\n        "
                                 "return arr\n    else:\n        pivot = arr[0]\n        left = [x for x in arr[1:] "
                                 "if x < pivot]\n        right = [x for x in arr[1:] if x >= pivot]\n        "
                                 "return quick_sort(left) + [pivot] + quick_sort(right)\n\n# 示例输入\narr = [3,6,8,1"},
                   8: {"prompt": "I believe the meaning of life is",
                       "answer": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nI "
                                 "believe the meaning of life is<|im_end|>\n<|im_start|>assistant\nThe meaning of "
                                 "life is a philosophical question that has been debated for centuries, and there "
                                 "is no one definitive answer to it. Some people believe that the meaning of life "
                                 "is to find happiness and fulfillment in their lives, while others believe that it "
                                 "is to achieve success or recognition.\n\nOthers may argue that the meaning of life "
                                 "is to make a positive impact on the world, to help others, and to contribute to "
                                 "society as a whole. Others may believe that the meaning of life is to pursue "
                                 "knowledge and understanding, to"},
                   }
    for batch_size, batch_data in batch_datas.items():
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": batch_data["prompt"]}
        ]
        input_ids = tokenizer.encode(tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        ))
        input_ids_list = []
        answer = batch_data["answer"]
        for i in range(0, batch_size):
            input_ids_list.append(input_ids)
        outputs = network.generate(input_ids_list,
                                   max_length=seq_length,
                                   do_sample=False,
                                   return_dict_in_generate=False)

        for i in range(0, len(outputs)):
            output_text = tokenizer.decode(outputs[i])
            print("parallel_qwen2_0_5b_predict_dp2_mp2, output_text:", output_text)
            print("parallel_qwen2_0_5b_predict_dp2_mp2, answer:", answer)
            compare_distance(output_text, answer, bench_sim=0.95)


def parallel_deepseek_r1_bf16_predict_mp2():
    """test deepseek r1 bf16 predict in model_parallel=2 with dynamic shape"""
    # Temporarily disable the test cases and add exception checks, it is necessary to upgrade MindFormer
    # to change the Python layer tensor CPU input tensor to pinned memory.
    # ms.runtime.set_kernel_launch_group()
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(cur_dir, "deepseek/configs/ci_predict_deepseek3_671b.yaml")

    vocab_file_path = "/home/workspace/mindspore_dataset/weight/DeepSeek-R1-bf16/tokenizer.json"
    tokenizer_file_path = "/home/workspace/mindspore_dataset/weight/DeepSeek-R1-bf16/tokenizer.json"
    load_checkpoint = "/home/workspace/mindspore_dataset/weight/DeepSeek-R1-bf16/"

    seq_length = 128
    # init config with yaml
    config = MindFormerConfig(config_path)
    config.use_parallel = True
    config.parallel_config.model_parallel = 2
    config.parallel_config.data_parallel = 1
    config.parallel_config.pipeline_stage = 1
    config.load_checkpoint = load_checkpoint
    config.model.model_config.seq_length = seq_length
    config.processor.tokenizer.vocab_file = vocab_file_path
    config.processor.tokenizer.tokenizer_file = tokenizer_file_path

    # init context
    build_context(config)
    build_parallel_config(config)

    config.model.model_config.parallel_config = config.parallel_config
    config.model.model_config.moe_config = config.moe_config
    model_config = DeepseekV3Config(**config.model.model_config)
    model_config.checkpoint_name_or_path = None
    # build tokenizer
    tokenizer = LlamaTokenizerFast(config.processor.tokenizer.vocab_file,
                                   config.processor.tokenizer.tokenizer_file,
                                   unk_token=config.processor.tokenizer.unk_token,
                                   bos_token=config.processor.tokenizer.bos_token,
                                   eos_token=config.processor.tokenizer.eos_token,
                                   fast_tokenizer=True, trust_remote_code=True)

    # build model
    with no_init_parameters():
        network = InferenceDeepseekV3ForCausalLM(model_config)

    # load checkpoint
    if config.load_checkpoint:
        logger.info("----------------Transform and load checkpoint----------------")
        weight_processor = DeepseekV3WeightProcessor(config, network, False)
        weight_processor.load_safetensors_shard(config.load_checkpoint)

    # predict
    batch_datas = {
        4: {"prompt": "You are a helpful assistant.<｜User｜>将文本分类为中性、负面或正面。 \n文本：我认为这次假期还可以。 \n情感：<｜Assistant｜>\n",
            "answer": ["ugs611ాలు sic辨hara的开璞 SquaresInsp"]}
    }
    for batch_size, batch_data in batch_datas.items():
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": batch_data["prompt"]}
        ]
        input_ids = tokenizer.encode(tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        ))
        input_ids_list = []
        for _ in range(0, batch_size):
            input_ids_list.append(input_ids)
        outputs = network.generate(input_ids_list,
                                   max_length=seq_length,
                                   do_sample=False,
                                   return_dict_in_generate=False)
        assert np.array(outputs).shape == (4, 128)


TEST_MAP = {
    'parallel_qwen2_0_5b_predict_mp2': parallel_qwen2_0_5b_predict_mp2,
    'parallel_qwen2_0_5b_predict_dp2_mp2': parallel_qwen2_0_5b_predict_dp2_mp2,
    'parallel_deepseek_r1_bf16_predict_mp2': parallel_deepseek_r1_bf16_predict_mp2,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='test mode of llama2 model.')

    args = parser.parse_args()
    TEST_MAP[args.mode]()
