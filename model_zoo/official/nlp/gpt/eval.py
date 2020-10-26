# Copyright 2020 Huawei Technologies Co., Ltd
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
GPT evaluation script.
"""

import math
import argparse
import numpy as np
from mindspore import context
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.inference import generate
from src.dataset import create_dataset
from src.gpt import GPT, EvalNet, GPTWithLoss, CrossEntropyLoss
from src.utils import GPTConfig

context.set_context(mode=context.GRAPH_MODE)

def ppl_score(probs, length, is_logsoftmax=True):
    """ calculate perplexity with prob or log_prob inputs """
    probs = probs[:length]
    if is_logsoftmax:
        prob = np.sum(probs) / length
        ppl = 1.0 / np.power(np.e, prob)
    else:
        prob = 1.0
        for p in probs:
            prob *= (1.0 / p)
        ppl = np.power(prob, 1.0/length)
    return ppl

def get_ppl(model, dataset):
    """ calculate perplexity for input dataset """
    PPL = []
    tokens = 0
    for data in dataset:
        data = data[0].asnumpy()
        input_ids = data

        logits = model(Tensor(input_ids, mstype.int32)).asnumpy()
        PPL.append(logits * len(data))
        tokens += len(data)

    val_loss = sum(PPL) / tokens
    ppl = math.exp(min(20, val_loss))
    return ppl

def get_acc(model, dataset):
    """ calculate accuracy for input dataset """
    total_num = 0
    acc_num = 0
    for data in dataset:
        data = data[0].asnumpy()
        input_mask = (data != 0).astype(np.int32)
        length = np.sum(input_mask, 1)
        label = np.zeros(length.shape)
        for i, idx in enumerate(length):
            label[i] = data[i][idx-1]
            input_mask[i][idx-1] = 0
            data[i][idx-1] = 0

        length = np.sum(data != 50256, 1)
        input_ids = data
        logits = model(Tensor(input_ids, mstype.int32)).asnumpy()
        logits = logits.reshape(len(length), -1)

        predicted_label = np.zeros(length.shape)
        for i, idx in enumerate(length):
            predicted_label[i] = logits[i][idx-2]

        total_num += len(label)
        acc_num += sum(label == predicted_label)

    acc = acc_num / total_num
    return acc


def run_eval():
    """ evaluate scripts """
    parser = argparse.ArgumentParser(description="GPT inferencing")
    parser.add_argument('--task_type', type=str, default="", help="Evaluation task.")
    parser.add_argument('--metrics', type=str, default="acc", choices=["ppl", "acc"], help="Evaluation metrics.")
    parser.add_argument('--ckpt_path', type=str, default="", help="path of checkpoint file.")
    parser.add_argument('--data_path', type=str, default="", help="path of MindRecord file.")

    args = parser.parse_args()
    task = args.task_type
    metrics = args.metrics
    ckpt_path = args.ckpt_path
    if task not in ["generate", "lambada", "wikitext"]:
        raise ValueError("{} is not supported now".format(task))

    if metrics not in ["acc", "ppl"]:
        raise ValueError("{} is not supported now".format(metrics))


    config = GPTConfig(batch_size=16,
                       seq_length=1024,
                       vocab_size=50257,
                       embedding_size=1024,
                       num_layers=24,
                       num_heads=16,
                       expand_ratio=4,
                       post_layernorm_residual=False,
                       dropout_rate=0.0,
                       compute_dtype=mstype.float16,
                       use_past=False)

    ckpt_dict = load_checkpoint(ckpt_path)

    gpt = GPT(config)
    if task == "generate":
        gpt_eval = EvalNet(gpt, generate=True)
    elif metrics == "acc":
        gpt_eval = EvalNet(gpt, generate=False)
    else:
        loss = CrossEntropyLoss(config)
        gpt_eval = GPTWithLoss(gpt, loss)

    gpt_eval.set_train(False)
    load_param_into_net(gpt_eval, ckpt_dict)

    if task == "generate":
        start_sentence = [6170, 318, 257]
        input_ids = np.array(start_sentence).reshape(1, -1)
        outputs = generate(gpt_eval, input_ids, config.seq_length)
        output_list = outputs.tolist()
        print("output id is ", output_list)
    else:
        data_path = args.data_path
        eval_dataset = create_dataset(config.batch_size, data_path=data_path, drop=False)
        if metrics == "acc":
            acc = get_acc(gpt_eval, eval_dataset)
            print("Accuracy is ", acc)
        elif metrics == "ppl":
            ppl = get_ppl(gpt_eval, eval_dataset)
            print("Perplexity is ", ppl)

if __name__ == "__main__":
    run_eval()
