import argparse
import time
import json
import numpy as np
import mindspore_lite as mslite


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', type=int, required=True)
    parser.add_argument('--rank_id', type=int, required=True)
    parser.add_argument('--prompt_model_path', type=str, required=True)
    parser.add_argument('--decoder_model_path', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--context', type=str, required=True)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--parallel_num', type=int, required=True)
    return parser.parse_args()


def create_context(args):
    context = mslite.Context()
    context.target = ["Ascend"]
    context.ascend.device_id = args.device_id
    context.ascend.rank_id = args.rank_id
    context.ascend.provider = "ge"
    context.cpu.thread_num = 1
    context.cpu.thread_affinity_mode = 2
    return context


def parse_configs(config_path):
    with open(config_path, 'r') as file:
        configs = json.load(file)
    return configs


def build_model(args):
    prompt_model_path = args.prompt_model_path
    decoder_model_path = args.decoder_model_path
    config_path = args.config_path
    context = args.context
    device = args.device
    model_type = args.model_type
    parallel_num = args.parallel_num
    print(
        f"Loading llama2, prompt model: {prompt_model_path}, decoder model: {decoder_model_path}")
    configs = parse_configs(config_path)
    config_ctx = configs[model_type + "_" + device + "_" +
                         context + "_" + str(parallel_num) + "p" + "_full"]
    config_inc = configs[model_type + "_" + device +
                         "_" + context + "_" + str(parallel_num) + "p" + "_inc"]
    start_time = time.time()
    context = create_context(args)
    prompt_model = mslite.Model()
    decoder_model = mslite.Model()
    model_group = mslite.ModelGroup(mslite.ModelGroupFlag.SHARE_WEIGHT)
    model_group.add_model([prompt_model, decoder_model])
    prompt_model.build_from_file(
        prompt_model_path, mslite.ModelType.MINDIR, context, "", config_ctx)
    prompt_model.predict(prompt_model.get_inputs())

    decoder_model.build_from_file(
        decoder_model_path, mslite.ModelType.MINDIR, context, "", config_inc)
    decoder_model.predict(decoder_model.get_inputs())
    print(f"Load model time: {time.time() - start_time:.3f} s.")
    return prompt_model, decoder_model


def infer_model(prompt_model: mslite.Model, decoder_model: mslite.Model, bs, seq_len, args):
    input_ids = np.ones((bs, seq_len), dtype=np.int32)
    input_ids_inc = np.ones((bs, 1), dtype=np.int32)
    input_position = np.array(([0] * bs), dtype=np.int32)
    init_reset = np.array([True], dtype=np.bool_)
    batch_valid_length = np.ones((bs), dtype=np.int32)
    input_ids_tensor = mslite.Tensor()
    input_ids_tensor.shape = [bs, seq_len]
    input_ids_tensor.dtype = mslite.DataType.INT32
    input_ids_tensor.set_data_from_numpy(input_ids)

    input_position_tensor = mslite.Tensor()
    input_position_tensor.shape = [bs]
    input_position_tensor.dtype = mslite.DataType.INT32
    input_position_tensor.set_data_from_numpy(input_position)

    init_reset_tensor = mslite.Tensor()
    init_reset_tensor.shape = [1]
    init_reset_tensor.dtype = mslite.DataType.BOOL
    init_reset_tensor.set_data_from_numpy(init_reset)

    batch_valid_length_tensor = mslite.Tensor()
    batch_valid_length_tensor.shape = [bs]
    batch_valid_length_tensor.dtype = mslite.DataType.INT32
    batch_valid_length_tensor.set_data_from_numpy(batch_valid_length)

    inputs_prompt = [input_ids_tensor, input_position_tensor,
                     init_reset_tensor, batch_valid_length_tensor]

    print("inputs of prompts model: ")
    for i in inputs_prompt:
        print(i.shape, i.dtype)

    input_ids_tensor_inc = mslite.Tensor()
    input_ids_tensor_inc.shape = [bs, 1]
    input_ids_tensor_inc.dtype = mslite.DataType.INT32
    input_ids_tensor_inc.set_data_from_numpy(input_ids_inc)

    inputs_decoder = [input_ids_tensor_inc, input_position_tensor,
                      init_reset_tensor, batch_valid_length_tensor]
    print("inputs of decoder model: ")
    for i in inputs_decoder:
        print(i.shape, i.dtype)

    print("checking accuracy: ")
    for i in range(5):
        prompt_output = prompt_model.predict(
            inputs_prompt)[0].get_data_to_numpy()
        decoder_output = decoder_model.predict(
            inputs_decoder)[0].get_data_to_numpy()
        print(f"{i}th round, prompt output: {prompt_output}", flush=True)
        print(f"{i}th round, decoder output: {decoder_output}", flush=True)

    prompt_time_list = []
    decoder_time_list = []
    decoder_loop = 299
    print("checking performance: ")
    for _ in range(10):
        prompt_start = time.time()
        prompt_model.predict(inputs_prompt)
        prompt_time_list.append(time.time() - prompt_start)

        decoder_start = time.time()
        for _ in range(decoder_loop):
            decoder_model.predict(inputs_decoder)
        decoder_time_list.append(time.time() - decoder_start)

    print(f"Rank({args.rank_id}) on device: {args.device_id}, Batch size {bs}, seq_length: {seq_len}, "
          f"1 prompt infer with {decoder_loop} decoder infer: ")
    print(
        f"The avg time cost of prompt: {np.mean(prompt_time_list) * 1000: .2f} ms.")
    print(
        f"The avg time cost of decoder: {np.mean(decoder_time_list) * 1000 / decoder_loop: .2f} ms.")


if __name__ == "__main__":
    args = get_args()
    prompt_model, decoder_model = build_model(args)
    infer_model(prompt_model, decoder_model, 1, 1024, args)
