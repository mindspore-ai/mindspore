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
"""mcore qwen3-0.6b model ST of inference"""
import argparse
import os
import sys
import math
import jieba
import numpy as np

from transformers import AutoTokenizer

workspace = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    )))
sys.path.insert(0, os.path.join(workspace, "mindformers"))

from mindformers import AutoConfig
from mindformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from mindformers import build_context, MindFormerConfig
from mindformers.tools.logger import logger


def test_qwen3_0_6b_predict_mcore(device_num: int = 1):
    """
    Feature: Mcore Qwen3-0.6b predict task
    Description: Two-card tp parallel
    Expectation: Success or assert precision failed
    """
    max_decode_length = 64
    config_path = os.path.join(os.path.dirname(__file__), "qwen3_0_6b_infer.yaml")
    config = MindFormerConfig(config_path)
    config.use_parallel = device_num > 1
    config.parallel_config.model_parallel = device_num
    build_context(config)
    # Auto tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_dir)
    # Auto config
    model_config = AutoConfig.from_pretrained(config_path)
    model_config.parallel_config.model_parallel = device_num
    network = Qwen3ForCausalLM(model_config)
    # Load HF safetensors
    network.load_weights(config.pretrained_model_dir)
    # Build prompt and answer
    batch_datas = {1: {"prompt": "Give me a short introduction to large language model.",
                       "answer": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
                                 "Give me a short introduction to large language model.<|im_end|>\n"
                                 "<|im_start|>assistant\n<think>\nOkay, the user wants a short introduction to a large "
                                 "language model. Let me start by recalling what I know about them."
                                 "Large language models are AI systems designed to"},
                   4: {"prompt": "Please introduce some scenic spots in Beijing.",
                       "answer": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
                                 "Please introduce some scenic spots in Beijing.<|im_end|>\n<|im_start|>assistant\n"
                                 "<think>\nOkay, the user asked for some scenic spots in Beijing. "
                                 "Let me start by recalling the main attractions there. "
                                 "The Forbidden City is a top spot, so that's a good"},
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
                                   max_length=max_decode_length,
                                   do_sample=False,
                                   return_dict_in_generate=False)

        for output in outputs:
            output_text = tokenizer.decode(output)
            logger.info("test_qwen3_0_6b_predict, output_text:{}".format(str(output_text)))
            compare_distance(output_text, answer)


def _get_all_words(standard_cut_infer_ret_list, test_cut_infer_ret_list):
    all_words = []
    for s_cut in standard_cut_infer_ret_list:
        if s_cut not in all_words:
            all_words.append(s_cut)
    for t_cut in test_cut_infer_ret_list:
        if t_cut not in all_words:
            all_words.append(t_cut)
    return all_words


def _get_word_vector(standard_cut_infer_ret_list, test_cut_infer_ret_list, all_words):
    la_standard = []
    lb_test = []
    for word in all_words:
        la_standard.append(standard_cut_infer_ret_list.count(word))
        lb_test.append(test_cut_infer_ret_list.count(word))
    return la_standard, lb_test


def _get_calculate_cos(la_standard, lb_test):
    laa = np.array(la_standard)
    lbb = np.array(lb_test)
    cos = (np.dot(laa, lbb.T)) / ((math.sqrt(np.dot(laa, laa.T))) * (math.sqrt(np.dot(lbb, lbb.T))))
    return np.round(cos, 2)


def compare_distance(x1, x2, bench_sim=0.95):
    """compare distance"""
    y1 = list(jieba.cut(x1))
    y2 = list(jieba.cut(x2))
    all_words = _get_all_words(y1, y2)
    laa, lbb = _get_word_vector(y1, y2, all_words)
    sim = _get_calculate_cos(laa, lbb)
    logger.info("calculate sim is:{}".format(str(sim)))
    assert sim >= bench_sim


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Qwen3 ST")
    parser.add_argument("--device_num", type=int, default=2)

    args = parser.parse_args()
    test_qwen3_0_6b_predict_mcore(args.device_num)
