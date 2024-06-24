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
import os
import sys
import argparse
import numpy as np

workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(workspace, "mindformers"))
import mindspore as ms
from mindformers import LlamaConfig, TransformerOpParallelConfig, LlamaTokenizer, LlamaForCausalLM, init_context
from mindformers.tools.register import MindFormerConfig


def str2bool(b):
    """String convert to Bool."""
    if b.lower() in ["false"]:
        output = False
    elif b.lower() in ["true"]:
        output = True
    else:
        raise Exception("Invalid Bool Value")
    return output


def main(args):
    """main function."""
    # 多batch输入
    inputs = ["I love Beijing, because"]
    inputs = inputs * args.batch_size

    # set model config
    config = MindFormerConfig(args.yaml_file)

    config.use_parallel = args.use_parallel

    if config.use_parallel:
        # set parallel method
        config.parallel_config.data_parallel = args.data_parallel
        config.parallel_config.model_parallel = args.model_parallel
        config.parallel_config.pipeline_stage = args.pipeline_stage
        print(config.parallel_config)
    else:
        config.context.device_id = args.device_id

    if args.use_bf16:
        config.model.model_config.compute_dtype = "bfloat16"
        config.model.model_config.layernorm_compute_type = "bfloat16"
        config.model.model_config.softmax_compute_type = "bfloat16"
        config.model.model_config.rotary_dtype = "bfloat16"
        config.model.model_config.param_init_type = "bfloat16"

    # initialize env
    init_context(use_parallel=args.use_parallel,
                 context_config=config.context,
                 parallel_config=config.parallel)
    ms.set_context(jit_config={"jit_level": "O0"})

    model_config = LlamaConfig(**config.model.model_config)

    # set model parameters
    model_config.parallel_config = TransformerOpParallelConfig(**config.parallel_config)
    model_config.batch_size = args.batch_size
    model_config.use_past = args.use_past
    model_config.is_dynamic = args.is_dynamic
    model_config.seq_length = args.seq_length
    model_config.checkpoint_name_or_path = args.checkpoint_path
    model_config.max_decode_length = args.max_decode_length

    # build tokenizer
    tokenizer = LlamaTokenizer(vocab_file=args.tokenizer_path)
    # build model from config
    model = LlamaForCausalLM(model_config)
    model.set_train(False)

    inputs_ids = tokenizer(inputs, max_length=model_config.seq_length)["input_ids"]
    outputs = model.generate(inputs_ids,
                             max_length=model_config.max_decode_length,
                             do_sample=model_config.do_sample,
                             top_k=model_config.top_k,
                             top_p=model_config.top_p)

    EXPECT_RES = np.array([1, 306, 5360, 1522, 823, 292, 29892, 1363, 13142,
                           13142, 13142, 13142, 2665, 2665, 2665, 2665, 2665, 2665,
                           2665, 2665, 15408, 15408, 15408, 15408, 15408, 15408, 15408,
                           15408, 15408, 15408, 15408, 15408, 15408, 15408, 15408, 15408,
                           15408, 15408, 15408, 15408, 15408, 15408, 15408, 15408, 15408,
                           15408, 15408, 15408, 15408, 15408, 15408, 15408, 15408, 15408,
                           15408, 15408, 8975, 15408, 8975, 15408, 15408, 8975, 15408,
                           15408], dtype=np.int32)

    assert (EXPECT_RES == outputs).all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', default=0, type=int,
                        help='device_id')
    parser.add_argument('--checkpoint_path', default='', type=str,
                        help='set checkpoint path.')
    parser.add_argument('--tokenizer_path', default='', type=str,
                        help='set tokenizer model path.')
    parser.add_argument('--use_past', default=True, type=str2bool,
                        help='whether use past.')
    parser.add_argument('--is_dynamic', default=True, type=str2bool,
                        help='whether is dynamic shape.')
    parser.add_argument('--use_parallel', default=False, type=str2bool,
                        help='whether use past.')
    parser.add_argument('--yaml_file', default="", type=str,
                        help='predict yaml path')
    parser.add_argument('--seq_length', default=512, type=int,
                        help='predict max length')
    parser.add_argument('--max_decode_length', default=64, type=int,
                        help='predict max length')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batch_size')
    parser.add_argument('--data_parallel', default=1, type=int,
                        help='data_parallel')
    parser.add_argument('--model_parallel', default=1, type=int,
                        help='model_parallel')
    parser.add_argument('--pipeline_stage', default=1, type=int,
                        help='pipeline_stage')
    parser.add_argument("--use_bf16", default=False, type=str2bool,
                        help="whether use bfloat16 to train")
    args_ = parser.parse_args()
    main(args_)
