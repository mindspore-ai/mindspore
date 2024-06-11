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
import logging

import mindspore as ms
import numpy as np
from mindformers import LlamaConfig, LlamaForCausalLM, CausalLanguageModelDataset, LlamaTrainer, build_context
from mindformers.llama_utils import check_dataset_config, set_seed, str2bool
from mindformers.normal_config import MindFormerConfig
from mindformers.parallel_config import build_parallel_config

logger = logging.getLogger()


def main(args):
    """main function."""

    ms.set_context(jit_config={"jit_level": "O2"})

    # set model config
    config = MindFormerConfig(args.yaml_file)
    config.use_parallel = args.use_parallel

    # init context
    build_context(config)

    if config.seed and \
            ms.context.get_auto_parallel_context("parallel_mode") \
            not in ["semi_auto_parallel", "auto_parallel"]:
        set_seed(config.seed)
        np.random.seed(config.seed)

    # build context config
    logger.info(".........Build context config..........")
    build_parallel_config(config)

    # set model parameters
    if args.do_bf16:
        config.model.model_config.compute_dtype = "bfloat16"
        config.model.model_config.rotary_dtype = "float32"
        config.model.model_config.param_init_type = "bfloat16"
        config.runner_wrapper.scale_sense = 1

    model_config = LlamaConfig(**config.model.model_config)
    model_config.parallel_config = config.parallel_config
    model_config.batch_size = args.batch_size
    model_config.use_past = args.use_past
    model_config.seq_length = args.seq_length
    model_config.checkpoint_name_or_path = args.checkpoint_path
    model_config.use_flash_attention = args.use_flash_attention
    config.model.model_config = model_config

    network = LlamaForCausalLM(model_config)

    llama_trainer = LlamaTrainer(model_name="llama2_7b")
    llama_trainer.set_network(network, is_train=True)

    # set dataset path
    config.train_dataset.data_loader.dataset_dir = args.dataset_path
    config.train_dataset.data_loader.shuffle = False
    # set config
    config = llama_trainer.set_config(config, is_full_config=True)

    # dataset
    check_dataset_config(config)
    dataset = CausalLanguageModelDataset(config.train_dataset_task.dataset_config)
    llama_trainer.train(config, dataset=dataset, is_full_config=True)

    loss_collector = llama_trainer.callbacks[0].loss_collector
    throughput_list = llama_trainer.callbacks[0].throughput_list

    if args.rank_id == args.device_num - 1:
        print(f"Final loss is: {','.join(list(map(str, loss_collector)))}")

    actual_throughput = np.mean(throughput_list[1:])
    print(f"Actual Throughout is: {str(actual_throughput)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_file', default="", type=str,
                        help='predict yaml path')
    parser.add_argument('--device_id', default=0, type=int,
                        help='device_id')
    parser.add_argument('--rank_id', default=0, type=int,
                        help='rank_id')
    parser.add_argument('--device_num', default=0, type=int,
                        help='device_num')
    parser.add_argument('--checkpoint_path', default='', type=str,
                        help='set checkpoint path.')
    parser.add_argument('--dataset_path', default='', type=str,
                        help='set dataset path.')
    parser.add_argument('--use_past', default=False, type=str2bool,
                        help='whether use past.')
    parser.add_argument('--use_parallel', default=True, type=str2bool,
                        help='whether use past.')
    parser.add_argument('--seq_length', default=512, type=int,
                        help='predict max length')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batch_size')
    parser.add_argument('--data_parallel', default=2, type=int,
                        help='data_parallel')
    parser.add_argument('--model_parallel', default=1, type=int,
                        help='model_parallel')
    parser.add_argument('--pipeline_stage', default=1, type=int,
                        help='pipeline_stage')
    parser.add_argument("--do_bf16", default=True, type=str2bool,
                        help="whether use bfloat16 to train")
    parser.add_argument("--use_flash_attention", default=True, type=str2bool,
                        help="whether use flash attention")
    args_ = parser.parse_args()
    main(args_)
