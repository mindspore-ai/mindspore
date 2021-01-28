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
CRNN-Seq2Seq-OCR Evaluation.

"""

import os
import codecs
import argparse
import numpy as np

import mindspore.ops.operations as P
import mindspore.common.dtype as mstype

from mindspore.common import set_seed
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.config import config
from src.utils import initialize_vocabulary
from src.dataset import create_ocr_val_dataset
from src.attention_ocr import AttentionOCRInfer


set_seed(1)


def text_standardization(text_in):
    """
    replace some particular characters
    """
    stand_text = text_in.strip()
    stand_text = ' '.join(stand_text.split())
    stand_text = stand_text.replace(u'(', u'（')
    stand_text = stand_text.replace(u')', u'）')
    stand_text = stand_text.replace(u':', u'：')
    return stand_text


def LCS_length(str1, str2):
    """
    calculate longest common sub-sequence between str1 and str2
    """
    if str1 is None or str2 is None:
        return 0

    len1 = len(str1)
    len2 = len(str2)
    if len1 == 0 or len2 == 0:
        return 0

    lcs = [[0 for _ in range(len2 + 1)] for _ in range(2)]
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if str1[i - 1] == str2[j - 1]:
                lcs[i % 2][j] = lcs[(i - 1) % 2][j - 1] + 1
            else:
                if lcs[i % 2][j - 1] >= lcs[(i - 1) % 2][j]:
                    lcs[i % 2][j] = lcs[i % 2][j - 1]
                else:
                    lcs[i % 2][j] = lcs[(i - 1) % 2][j]

    return lcs[len1 % 2][-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CRNN-Seq2Seq-OCR Evaluation")
    parser.add_argument("--dataset_path", type=str, default="",
                        help="Test Dataset path")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Checkpoint of AttentionOCR (Default:None).")
    parser.add_argument("--device_target", type=str, default="Ascend",
                        help="device where the code will be implemented, default is Ascend")
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default: 0.")

    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)

    prefix = "fsns.mindrecord"
    mindrecord_dir = args.dataset_path
    mindrecord_file = os.path.join(mindrecord_dir, prefix + "0")
    print("mindrecord_file", mindrecord_file)
    dataset = create_ocr_val_dataset(mindrecord_file, config.eval_batch_size)
    data_loader = dataset.create_dict_iterator(num_epochs=1, output_numpy=True)
    print("Dataset creation Done!")

    #Network
    network = AttentionOCRInfer(config.eval_batch_size,
                                int(config.img_width / 4),
                                config.encoder_hidden_size,
                                config.decoder_hidden_size,
                                config.decoder_output_size,
                                config.max_length,
                                config.dropout_p)

    ckpt = load_checkpoint(args.checkpoint_path)
    load_param_into_net(network, ckpt)
    network.set_train(False)
    print("Checkpoint loading Done!")

    vocab, rev_vocab = initialize_vocabulary(config.vocab_path)
    eos_id = config.characters_dictionary.get("eos_id")
    sos_id = config.characters_dictionary.get("go_id")

    num_correct_char = 0
    num_total_char = 0
    num_correct_word = 0
    num_total_word = 0

    correct_file = 'result_correct.txt'
    incorrect_file = 'result_incorrect.txt'

    with codecs.open(correct_file, 'w', encoding='utf-8') as fp_output_correct, \
        codecs.open(incorrect_file, 'w', encoding='utf-8') as fp_output_incorrect:

        for data in data_loader:
            images = Tensor(data["image"])
            decoder_inputs = Tensor(data["decoder_input"])
            decoder_targets = Tensor(data["decoder_target"])

            decoder_hidden = Tensor(np.zeros((1, config.eval_batch_size, config.decoder_hidden_size),
                                             dtype=np.float16), mstype.float16)
            decoder_input = Tensor((np.ones((config.eval_batch_size, 1))*sos_id).astype(np.int32))
            encoder_outputs = network.encoder(images)
            batch_decoded_label = []

            for di in range(decoder_inputs.shape[1]):
                decoder_output, decoder_hidden, _ = network.decoder(decoder_input, decoder_hidden, encoder_outputs)
                topi = P.Argmax()(decoder_output)
                ni = P.ExpandDims()(topi, 1)
                decoder_input = ni
                topi_id = topi.asnumpy()
                batch_decoded_label.append(topi_id)

            for b in range(config.eval_batch_size):
                text = data["annotation"][b].decode("utf8")
                text = text_standardization(text)
                decoded_label = list(np.array(batch_decoded_label)[:, b])
                decoded_words = []
                for idx in decoded_label:
                    if idx == eos_id:
                        break
                    else:
                        decoded_words.append(rev_vocab[idx])
                predict = text_standardization("".join(decoded_words))

                if predict == text:
                    num_correct_word += 1
                    fp_output_correct.write('\t\t' + text + '\n')
                    fp_output_correct.write('\t\t' + predict + '\n\n')
                    print('correctly predicted : pred: {}, gt: {}'.format(predict, text))

                else:
                    fp_output_incorrect.write('\t\t' + text + '\n')
                    fp_output_incorrect.write('\t\t' + predict + '\n\n')
                    print('incorrectly predicted : pred: {}, gt: {}'.format(predict, text))

                num_total_word += 1
                num_correct_char += 2 * LCS_length(text, predict)
                num_total_char += len(text) + len(predict)

        print('\nnum of correct characters = %d' % (num_correct_char))
        print('\nnum of total characters = %d' % (num_total_char))
        print('\nnum of correct words = %d' % (num_correct_word))
        print('\nnum of total words = %d' % (num_total_word))
        print('\ncharacter precision = %f' % (float(num_correct_char) / num_total_char))
        print('\nAnnotation precision precision = %f' % (float(num_correct_word) / num_total_word))
