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
Test module for testing the stand alone Infer interface used for mindformers.
How to run this:
    pytest tests/st/networks/large_models/test_standalone_infer.py
"""
import os
import sys

workspace = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, os.path.join(workspace, "networks/mindformers"))

import mindspore as ms
from mindspore.nn.utils import no_init_parameters
from tests.mark_utils import arg_mark
from mindformers import build_context, MindFormerConfig, LlamaConfig, LlamaForCausalLM
from research.qwen2.qwen2_tokenizer import Qwen2Tokenizer
from similarity import compare_distance


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_qwen2_0_5b_predict_standalone():
    """
    Feature: Infer standalone st
    Description: Test infer interface for prediction with standalone.
    Expectation: AssertionError
    """
    # Temporarily disable the test cases and add exception checks, it is necessary to upgrade MindFormer
    # to change the Python layer tensor CPU input tensor to pinned memory.
    # ms.runtime.set_kernel_launch_group()
    ascend_home_path = os.getenv('ASCEND_HOME_PATH')
    if not ascend_home_path:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/cann"
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(cur_dir, "./qwen/configs/ci_predict_qwen2_0_5b_instruct.yaml")

    vocab_file_path = "/home/workspace/mindspore_dataset/weight/Qwen2.5-0.5B-Instruct-tokenizer/vocab.json"
    merges_file_path = "/home/workspace/mindspore_dataset/weight/Qwen2.5-0.5B-Instruct-tokenizer/merges.txt"
    load_safetensors = "/home/workspace/mindspore_dataset/weight/ms_safetensor_qwen2_0.5/model.safetensors"

    seq_length = 128
    # init config with yaml
    config = MindFormerConfig(config_path)
    config.use_parallel = False
    config.load_checkpoint = load_safetensors
    config.model.model_config.seq_length = seq_length
    config.processor.tokenizer.vocab_file = vocab_file_path
    config.processor.tokenizer.merges_file = merges_file_path
    config.context.device_id = int(os.environ.get("DEVICE_ID", "0"))

    # init context
    build_context(config)

    model_config = LlamaConfig(**config.model.model_config)
    model_config.checkpoint_name_or_path = None

    # build tokenizer
    tokenizer = Qwen2Tokenizer(**config.processor.tokenizer)

    # build model
    with no_init_parameters():
        network = LlamaForCausalLM(model_config)

    ms.load_checkpoint(
        ckpt_file_name=load_safetensors,
        net=network,
        format='safetensors'
    )

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
                                 "believe the meaning of life is<|im_end|>\n<|im_start|>assistant\nThe meaning of life "
                                 "is a philosophical question that has been debated for centuries, and there is no one "
                                 "definitive answer to it. Some people believe that the meaning of life is to find "
                                 "happiness and fulfillment in their lives, while others believe that it is to achieve "
                                 "success or recognition.\n\nOthers may argue that the meaning of life is to live a "
                                 "good life, to make a positive impact on the world, and to contribute to society in "
                                 "some way. Others may believe that the meaning of life is to seek knowledge and "
                                 "understanding"}
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
        for _ in range(0, batch_size):
            input_ids_list.append(input_ids)
        outputs = network.generate(input_ids_list,
                                   max_length=seq_length,
                                   do_sample=False,
                                   return_dict_in_generate=False)

        for output in outputs:
            output_text = tokenizer.decode(output)
            print("test_qwen2_0_5b_predict_standalone, output_text:", output_text)
            compare_distance(output_text, answer, bench_sim=0.95)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_qwen2_0_5b_predict_standalone_for_aclgraph():
    """
    Feature: Infer standalone st with aclgraph
    Description: Test infer interface for prediction with standalone.
    Expectation: AssertionError
    """
    ms.runtime.set_kernel_launch_capture(True)
    ascend_home_path = os.getenv('ASCEND_HOME_PATH')
    if not ascend_home_path:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(cur_dir, "./qwen/configs/ci_predict_qwen2_0_5b_instruct.yaml")

    vocab_file_path = "/home/workspace/mindspore_dataset/weight/Qwen2.5-0.5B-Instruct-tokenizer/vocab.json"
    merges_file_path = "/home/workspace/mindspore_dataset/weight/Qwen2.5-0.5B-Instruct-tokenizer/merges.txt"
    load_safetensors = "/home/workspace/mindspore_dataset/weight/ms_safetensor_qwen2_0.5/model.safetensors"

    seq_length = 128
    # init config with yaml
    config = MindFormerConfig(config_path)
    config.use_parallel = False
    config.load_checkpoint = load_safetensors
    config.model.model_config.seq_length = seq_length
    config.processor.tokenizer.vocab_file = vocab_file_path
    config.processor.tokenizer.merges_file = merges_file_path
    config.context.device_id = int(os.environ.get("DEVICE_ID", "0"))
    config.context.max_device_memory = "12GB"

    # init context
    build_context(config)

    model_config = LlamaConfig(**config.model.model_config)
    model_config.checkpoint_name_or_path = None

    # build tokenizer
    tokenizer = Qwen2Tokenizer(**config.processor.tokenizer)

    # build model
    with no_init_parameters():
        network = LlamaForCausalLM(model_config)

    ms.load_checkpoint(
        ckpt_file_name=load_safetensors,
        net=network,
        format='safetensors'
    )

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
                                 "believe the meaning of life is<|im_end|>\n<|im_start|>assistant\nThe meaning of life "
                                 "is a philosophical question that has been debated for centuries, and there is no one "
                                 "definitive answer to it. Some people believe that the meaning of life is to find "
                                 "happiness and fulfillment in their lives, while others believe that it is to achieve "
                                 "success or recognition.\n\nOthers may argue that the meaning of life is to live a "
                                 "good life, to make a positive impact on the world, and to contribute to society in "
                                 "some way. Others may believe that the meaning of life is to seek knowledge and "
                                 "understanding"}
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
        for _ in range(0, batch_size):
            input_ids_list.append(input_ids)
        outputs = network.generate(input_ids_list,
                                   max_length=seq_length,
                                   do_sample=False,
                                   return_dict_in_generate=False)

        for output in outputs:
            output_text = tokenizer.decode(output)
            print("test_qwen2_0_5b_predict_standalone_for_aclgraph, output_text:", output_text)
            compare_distance(output_text, answer, bench_sim=0.95)
