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
"""Transformer evaluation script."""
import os
import argparse
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import context
from src.dataset import create_gru_dataset
from src.seq2seq import Seq2Seq
from src.gru_for_infer import GRUInferCell
from src.config import config

def run_gru_eval():
    """
    Transformer evaluation.
    """
    parser = argparse.ArgumentParser(description='GRU eval')
    parser.add_argument("--device_target", type=str, default="Ascend",
                        help="device where the code will be implemented, default is Ascend")
    parser.add_argument('--device_id', type=int, default=0, help='device id of GPU or Ascend, default is 0')
    parser.add_argument('--device_num', type=int, default=1, help='Use device nums, default is 1')
    parser.add_argument('--ckpt_file', type=str, default="", help='ckpt file path')
    parser.add_argument("--dataset_path", type=str, default="",
                        help="Dataset path, default: f`sns.")
    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, reserve_class_name_in_scope=False, \
        device_id=args.device_id, save_graphs=False)
    mindrecord_file = args.dataset_path
    if not os.path.exists(mindrecord_file):
        print("dataset file {} not exists, please check!".format(mindrecord_file))
        raise ValueError(mindrecord_file)
    dataset = create_gru_dataset(epoch_count=config.num_epochs, batch_size=config.eval_batch_size, \
        dataset_path=mindrecord_file, rank_size=args.device_num, rank_id=0, do_shuffle=False, is_training=False)
    dataset_size = dataset.get_dataset_size()
    print("dataset size is {}".format(dataset_size))
    network = Seq2Seq(config, is_training=False)
    network = GRUInferCell(network)
    network.set_train(False)
    if args.ckpt_file != "":
        parameter_dict = load_checkpoint(args.ckpt_file)
        load_param_into_net(network, parameter_dict)
    model = Model(network)

    predictions = []
    source_sents = []
    target_sents = []
    eval_text_len = 0
    for batch in dataset.create_dict_iterator(output_numpy=True, num_epochs=1):
        source_sents.append(batch["source_ids"])
        target_sents.append(batch["target_ids"])
        source_ids = Tensor(batch["source_ids"], mstype.int32)
        target_ids = Tensor(batch["target_ids"], mstype.int32)
        predicted_ids = model.predict(source_ids, target_ids)
        print("predicts is ", predicted_ids.asnumpy())
        print("target_ids is ", target_ids)
        predictions.append(predicted_ids.asnumpy())
        eval_text_len = eval_text_len + 1

    f_output = open(config.output_file, 'w')
    f_target = open(config.target_file, "w")
    for batch_out, true_sentence in zip(predictions, target_sents):
        for i in range(config.eval_batch_size):
            target_ids = [str(x) for x in true_sentence[i].tolist()]
            f_target.write(" ".join(target_ids) + "\n")
            token_ids = [str(x) for x in batch_out[i].tolist()]
            f_output.write(" ".join(token_ids) + "\n")
    f_output.close()
    f_target.close()

if __name__ == "__main__":
    run_gru_eval()
