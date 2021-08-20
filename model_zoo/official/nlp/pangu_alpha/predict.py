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

import mindspore.common.dtype as mstype
import mindspore.communication.management as D
from mindspore import context, Tensor
from mindspore import export
from mindspore.context import ParallelMode
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._cost_model_context import _set_multi_subgraphs
from mindspore.train.model import Model
from mindspore.train.serialization import load_distributed_checkpoint
from mindspore.parallel.nn.transformer import TransformerOpParallelConfig
from src.pangu_alpha import EvalNet, PanguAlphaModel
from src.pangu_alpha_config import set_parse, PanguAlphaConfig
from src.utils import get_args


def load_model(args_opt):
    r"""
     The main function for load model
    """
    # Set execution mode
    context.set_context(save_graphs=False,
                        mode=context.GRAPH_MODE,
                        device_target=args_opt.device_target)
    context.set_context(variable_memory_max_size="30GB")
    # Set parallel context
    if args_opt.distribute == "true":
        D.init()
        device_num = D.get_group_size()
        rank = D.get_rank()
        print("rank_id is {}, device_num is {}".format(rank, device_num))
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
            gradients_mean=False,
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
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            strategy_ckpt_load_file=args_opt.strategy_load_ckpt_path)

    use_past = (args_opt.use_past == "true")
    print('local_rank:{}, start to run...'.format(rank), flush=True)
    if args_opt.export:
        use_past = True
    # Set model property
    model_parallel_num = args_opt.op_level_model_parallel_num
    data_parallel_num = int(device_num / model_parallel_num)

    parallel_config = TransformerOpParallelConfig(data_parallel=data_parallel_num,
                                                  model_parallel=model_parallel_num,
                                                  pipeline_stage=args_opt.stage_num,
                                                  micro_batch_num=args_opt.micro_size,
                                                  recompute=True)

    per_batch_size = args_opt.per_batch_size
    batch_size = per_batch_size * data_parallel_num
    # Now only support single batch_size for predict
    if args_opt.run_type == "predict":
        batch_size = 1
    config = PanguAlphaConfig(
        batch_size=batch_size,
        seq_length=args_opt.seq_length,
        vocab_size=args_opt.vocab_size,
        hidden_size=args_opt.embedding_size,
        num_layers=args_opt.num_layers,
        num_heads=args_opt.num_heads,
        post_layernorm_residual=False,
        dropout_rate=0.0,
        ffn_hidden_size=args_opt.embedding_size * 4,
        use_past=use_past,
        eod_reset=False,
        parallel_config=parallel_config,
        load_ckpt_path=args_opt.load_ckpt_path,
        param_init_type=mstype.float32 if args_opt.param_init_type == 'fp32' else mstype.float16)
    print("===config is: ", config, flush=True)
    print("=====args_opt is: ", args_opt, flush=True)

    ckpt_name = args_opt.load_ckpt_name
    # Define network
    pangu_alpha = PanguAlphaModel(config)
    eval_net = EvalNet(pangu_alpha)
    eval_net.set_train(False)
    model_predict = Model(eval_net)
    # Compile network and obtain tensor layout for loading ckpt
    inputs_np = Tensor(np.ones(shape=(config.batch_size, config.seq_length)), mstype.int32)
    current_index = Tensor(np.array([0]), mstype.int32)

    if args_opt.distribute == "false":
        predict_layout = None
    elif config.use_past:
        batch_valid_length = Tensor(np.array([0]), mstype.int32)
        init_true = Tensor([True], mstype.bool_)
        inputs_np_1 = Tensor(np.ones(shape=(config.batch_size, 1)), mstype.int32)
        model_predict.predict_network.add_flags_recursive(is_first_iteration=True)
        predict_layout = model_predict.infer_predict_layout(inputs_np, current_index, init_true, batch_valid_length)
        model_predict.predict_network.add_flags_recursive(is_first_iteration=False)
        _ = model_predict.infer_predict_layout(inputs_np_1, current_index, init_true, batch_valid_length)
    else:
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
    return model_predict, config


def export_mindir(model_predict, config):
    """Export mindir model"""
    inputs_np = Tensor(np.ones(shape=(config.batch_size, config.seq_length)), mstype.int32)
    current_index = Tensor(np.array([0]), mstype.int32)

    batch_valid_length = Tensor(np.array([0]), mstype.int32)
    init_true = Tensor([True], mstype.bool_)
    inputs_np_1 = Tensor(np.ones(shape=(config.batch_size, 1)), mstype.int32)

    model_predict.predict_network.add_flags_recursive(is_first_iteration=True)
    export(model_predict.predict_network, inputs_np, current_index,
           init_true, batch_valid_length, file_name='pangu_alpha_1024', file_format='MINDIR')
    model_predict.predict_network.add_flags_recursive(is_first_iteration=False)
    export(model_predict.predict_network, inputs_np_1, current_index,
           init_true, batch_valid_length, file_name='pangu_alpha_1', file_format='MINDIR')
    print("Export finished and now exit.")


def run_predict(model_predict, config, args_opt):
    """run predict"""
    from src.tokenization_jieba import JIEBATokenizer
    from src.generate import generate, generate_increment
    # Define tokenizer
    tokenizer = JIEBATokenizer(os.path.join(args_opt.tokenizer_path, 'vocab10.vocab'),
                               os.path.join(args_opt.tokenizer_path, 'vocab10.model'))

    # Tokenize input sentence to ids
    sample = "今天是一个好天气"
    tokenized_token = tokenizer.tokenize(sample)
    start_sentence = tokenizer.convert_tokens_to_ids(tokenized_token)
    input_ids = np.array(start_sentence).reshape(1, -1)
    # Call inference
    generate_func = generate_increment if config.use_past else generate
    output_ids = generate_func(model_predict, input_ids, args_opt)
    # Decode output ids to sentence
    output_samples = tokenizer.convert_ids_to_tokens(output_ids.tolist())
    print('Output is:', output_samples, flush=True)


def main():
    """Main process for predict or export model"""
    opt = get_args(True)
    set_parse(opt)
    model_predict, config = load_model(opt)
    if opt.export:
        export_mindir(model_predict, config)
    else:
        run_predict(model_predict, config, opt)


if __name__ == "__main__":
    main()
