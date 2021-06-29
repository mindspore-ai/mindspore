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
import numpy as np

import mindspore.common.dtype as mstype
from mindspore.common import set_seed
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.model import Model

from src.utils import initialize_vocabulary
from src.dataset import create_ocr_val_dataset
from src.attention_ocr import AttentionOCRInfer

from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id

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

@moxing_wrapper()
def run_eval():
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=get_device_id())
    prefix = "fsns.mindrecord"
    if config.enable_modelarts:
        mindrecord_file = os.path.join(config.data_path, prefix + "0")
    else:
        mindrecord_file = os.path.join(config.test_data_dir, prefix + "0")
    print("mindrecord_file", mindrecord_file)
    dataset = create_ocr_val_dataset(mindrecord_file, config.eval_batch_size)
    data_loader = dataset.create_dict_iterator(num_epochs=1, output_numpy=True)
    print("Dataset creation Done!")

    # Network
    network = AttentionOCRInfer(config.eval_batch_size,
                                int(config.img_width / 4),
                                config.encoder_hidden_size,
                                config.decoder_hidden_size,
                                config.decoder_output_size,
                                config.max_length,
                                config.dropout_p)
    checkpoint_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), config.checkpoint_path)
    ckpt = load_checkpoint(checkpoint_path)
    load_param_into_net(network, ckpt)
    network.set_train(False)
    print("Checkpoint loading Done!")

    model = Model(network)

    vocab_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), config.vocab_path)
    _, rev_vocab = initialize_vocabulary(vocab_path)
    eos_id = config.characters_dictionary.eos_id
    sos_id = config.characters_dictionary.go_id

    num_correct_char = 0
    num_total_char = 0
    num_correct_word = 0
    num_total_word = 0

    correct_file = 'result_correct.txt'
    incorrect_file = 'result_incorrect.txt'

    with codecs.open(correct_file, 'w', encoding='utf-8') as fp_output_correct, \
            codecs.open(incorrect_file, 'w', encoding='utf-8') as fp_output_incorrect:

        for data in data_loader:
            images = Tensor(data["image"]).astype(np.float32)
            # decoder_targets = Tensor(data["decoder_target"])

            decoder_hidden = Tensor(np.zeros((1, config.eval_batch_size, config.decoder_hidden_size),
                                             dtype=np.float16), mstype.float16)
            decoder_input = Tensor((np.ones((config.eval_batch_size, 1)) * sos_id).astype(np.int32))

            result_batch_decoded_label = model.predict(images, decoder_input, decoder_hidden)

            batch_decoded_label = []
            for ele in result_batch_decoded_label:
                batch_decoded_label.append(ele.asnumpy())

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
if __name__ == '__main__':
    run_eval()
