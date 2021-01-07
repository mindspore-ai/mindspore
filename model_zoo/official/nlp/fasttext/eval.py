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
"""FastText for Evaluation"""
import argparse
import numpy as np

import mindspore.nn as nn
import mindspore.common.dtype as mstype
import mindspore.ops.operations as P
from mindspore.common.tensor import Tensor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as deC
from mindspore import context
from src.fasttext_model import FastText
parser = argparse.ArgumentParser(description='fasttext')
parser.add_argument('--data_path', type=str, help='infer dataset path..')
parser.add_argument('--data_name', type=str, required=True, default='ag',
                    help='dataset name. eg. ag, dbpedia')
parser.add_argument("--model_ckpt", type=str, required=True,
                    help="existed checkpoint address.")
args = parser.parse_args()
if args.data_name == "ag":
    from src.config import config_ag as config
    target_label1 = ['0', '1', '2', '3']
elif args.data_name == 'dbpedia':
    from src.config import config_db as config
    target_label1 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']
elif args.data_name == 'yelp_p':
    from  src.config import config_yelpp as config
    target_label1 = ['0', '1']
context.set_context(
    mode=context.GRAPH_MODE,
    save_graphs=False,
    device_target="Ascend")

class FastTextInferCell(nn.Cell):
    """
    Encapsulation class of FastText network infer.

    Args:
        network (nn.Cell): FastText model.

    Returns:
        Tuple[Tensor, Tensor], predicted_ids
    """
    def __init__(self, network):
        super(FastTextInferCell, self).__init__(auto_prefix=False)
        self.network = network
        self.argmax = P.ArgMaxWithValue(axis=1, keep_dims=True)
        self.log_softmax = nn.LogSoftmax(axis=1)

    def construct(self, src_tokens, src_tokens_lengths):
        """construct fasttext infer cell"""
        prediction = self.network(src_tokens, src_tokens_lengths)
        predicted_idx = self.log_softmax(prediction)
        predicted_idx, _ = self.argmax(predicted_idx)

        return predicted_idx

def load_infer_dataset(batch_size, datafile, bucket):
    """data loader for infer"""
    def batch_per_bucket(bucket_length, input_file):
        input_file = input_file + '/test_dataset_bs_' + str(bucket_length) + '.mindrecord'
        if not input_file:
            raise FileNotFoundError("input file parameter must not be empty.")

        data_set = ds.MindDataset(input_file,
                                  columns_list=['src_tokens', 'src_tokens_length', 'label_idx'])
        type_cast_op = deC.TypeCast(mstype.int32)
        data_set = data_set.map(operations=type_cast_op, input_columns="src_tokens")
        data_set = data_set.map(operations=type_cast_op, input_columns="src_tokens_length")
        data_set = data_set.map(operations=type_cast_op, input_columns="label_idx")

        data_set = data_set.batch(batch_size, drop_remainder=False)
        return data_set
    for i, _ in enumerate(bucket):
        bucket_len = bucket[i]
        ds_per = batch_per_bucket(bucket_len, datafile)
        if i == 0:
            data_set = ds_per
        else:
            data_set = data_set + ds_per

    return data_set

def run_fasttext_infer():
    """run infer with FastText"""
    dataset = load_infer_dataset(batch_size=config.batch_size, datafile=args.data_path, bucket=config.test_buckets)
    fasttext_model = FastText(config.vocab_size, config.embedding_dims, config.num_class)

    parameter_dict = load_checkpoint(args.model_ckpt)
    load_param_into_net(fasttext_model, parameter_dict=parameter_dict)

    ft_infer = FastTextInferCell(fasttext_model)

    model = Model(ft_infer)

    predictions = []
    target_sens = []

    for batch in dataset.create_dict_iterator(output_numpy=True, num_epochs=1):
        target_sens.append(batch['label_idx'])
        src_tokens = Tensor(batch['src_tokens'], mstype.int32)
        src_tokens_length = Tensor(batch['src_tokens_length'], mstype.int32)
        predicted_idx = model.predict(src_tokens, src_tokens_length)
        predictions.append(predicted_idx.asnumpy())

    from sklearn.metrics import accuracy_score, classification_report
    target_sens = np.array(target_sens).flatten()
    merge_target_sens = []
    for target_sen in target_sens:
        merge_target_sens.extend(target_sen)
    target_sens = merge_target_sens
    predictions = np.array(predictions).flatten()
    merge_predictions = []
    for prediction in predictions:
        merge_predictions.extend(prediction)
    predictions = merge_predictions
    acc = accuracy_score(target_sens, predictions)

    result_report = classification_report(target_sens, predictions, target_names=target_label1)
    print("********Accuracy: ", acc)
    print(result_report)

if __name__ == '__main__':
    run_fasttext_infer()
