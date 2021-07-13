# Copyright 2021 Huawei Technologies Co., Ltd
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
PanGu predict run
"""
import os
import numpy as np
from mindspore import context, Tensor
from mindspore.train.model import Model
import mindspore.communication.management as D
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_distributed_checkpoint
import mindspore.common.dtype as mstype
from mindspore.parallel._cost_model_context import _set_multi_subgraphs
from mindspore.parallel import set_algo_parameters
from src.pangu_alpha import PanguAlpha, EvalNet
from src.pangu_alpha_config import PANGUALPHAConfig, set_parse
from src.utils import get_args


def run_predict(args_opt):
    r"""
     The main function for running prediction
    """
    device_id = int(os.getenv("DEVICE_ID"))
    rank_id_str = os.getenv('RANK_ID', '0')
    rank_id = int(
        rank_id_str[rank_id_str.rfind('-') +
                    1:])
    print('rank_id:{}'.format(rank_id), "rank_id str:{}".format(rank_id_str))
    device_id = int(os.getenv('DEVICE_ID'))
    local_rank = rank_id
    print('local_rank:{}, device id:{} start to run...'.format(local_rank, device_id), flush=True)
    # Set execution mode
    context.set_context(save_graphs=False,
                        mode=context.GRAPH_MODE,
                        device_target="Ascend",
                        device_id=device_id)
    context.set_context(variable_memory_max_size="30GB")
    # Set parallel context
    if args_opt.distribute == "true":
        D.init()
        device_num = D.get_group_size()
        rank = D.get_rank()
        print("device_id is {}, rank_id is {}, device_num is {}".format(
            device_id, rank, device_num))
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
            gradients_mean=False,
            device_num=device_num,
            full_batch=True,
            loss_repeated_mean=True,
            enable_parallel_optimizer=False,
            strategy_ckpt_load_file=args_opt.strategy_load_ckpt_path,
            pipeline_stages=args_opt.stage_num)
        set_algo_parameters(elementwise_op_strategy_follow=True)
        _set_multi_subgraphs()

    else:
        rank = 0
        device_num = 1

    # Set model property
    model_parallel_num = args_opt.tensor_model_parallel_num
    data_parallel_num = int(device_num / model_parallel_num)
    per_batch_size = args_opt.per_batch_size
    batch_size = per_batch_size * data_parallel_num
    config = PANGUALPHAConfig(
        data_parallel_num=data_parallel_num,
        model_parallel_num=model_parallel_num,
        batch_size=batch_size,
        seq_length=args_opt.seq_length,
        vocab_size=args_opt.vocab_size,
        embedding_size=args_opt.embedding_size,
        num_layers=args_opt.num_layers,
        num_heads=args_opt.num_heads,
        expand_ratio=4,
        post_layernorm_residual=False,
        dropout_rate=0.0,
        compute_dtype=mstype.float16,
        use_past=False,
        self_layernorm=True,
        stage_num=args_opt.stage_num,
        micro_size=args_opt.micro_size,
        eod_reset=False,
        word_emb_dp=True,
        load_ckpt_path=args_opt.load_ckpt_path)
    print("===config is: ", config, flush=True)
    print("=====args_opt is: ", args_opt, flush=True)

    ckpt_name = args_opt.load_ckpt_name
    # Define network
    pangu_alpha = PanguAlpha(config)
    eval_net = EvalNet(pangu_alpha)
    eval_net.set_train(False)
    model_predict = Model(eval_net)

    # Compile network and obtain tensor layout for loading ckpt
    inputs_np = Tensor(np.ones(shape=(config.batch_size, config.seq_length)), mstype.int32)
    current_index = Tensor(np.array([0]), mstype.int32)
    predict_layout = model_predict.infer_predict_layout(inputs_np, current_index)
    print("======start load_distributed checkpoint", flush=True)
    # For 2.6B and 13B models, the number of ckpt files is 512.
    ckpt_name = 'filerted'
    ckpt_file_list = [os.path.join(args_opt.load_ckpt_path, f"{ckpt_name}_{ckpt_rank}.ckpt") for ckpt_rank in
                      range(0, 512)]
    print(f"Loading from path {ckpt_file_list[0]}", flush=True)
    # Load checkpoint files
    load_distributed_checkpoint(eval_net, ckpt_file_list, predict_layout)
    print("================load param ok=================", flush=True)

    from src.tokenization_jieba import JIEBATokenizer
    from src.generate import generate
    # Define tokenizer
    tokenizer = JIEBATokenizer(os.path.join(args_opt.tokenizer_path, 'vocab10.vocab'),
                               os.path.join(args_opt.tokenizer_path, 'vocab10.model'))

    # Tokenize input sentence to ids
    sample = "今天是一个好天气"
    tokenized_token = tokenizer.tokenize(sample)
    start_sentence = tokenizer.convert_tokens_to_ids(tokenized_token)
    input_ids = np.array(start_sentence).reshape(1, -1)
    # Call inference
    output_ids = generate(model_predict, input_ids, opt)
    # Decode output ids to sentence
    output_samples = tokenizer.convert_ids_to_tokens(output_ids.tolist())
    print('Output is:', output_samples, flush=True)

if __name__ == "__main__":
    opt = get_args(True)
    set_parse(opt)
    run_predict(opt)
