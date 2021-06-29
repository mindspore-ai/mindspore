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
postprocess.

"""

import os
import codecs
import numpy as np

from src.utils import initialize_vocabulary
from src.model_utils.config import config
from eval import text_standardization, LCS_length


def get_acc():
    '''generate accuracy'''
    vocab_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), config.vocab_path)
    _, rev_vocab = initialize_vocabulary(vocab_path)
    eos_id = config.characters_dictionary.eos_id

    num_correct_char = 0
    num_total_char = 0
    num_correct_word = 0
    num_total_word = 0

    correct_file = 'result_correct.txt'
    incorrect_file = 'result_incorrect.txt'

    with codecs.open(correct_file, 'w', encoding='utf-8') as fp_output_correct, \
            codecs.open(incorrect_file, 'w', encoding='utf-8') as fp_output_incorrect:

        file_num = len(os.listdir(config.post_result_path)) // config.max_length
        for i in range(file_num):
            batch_decoded_label = []
            for j in range(config.max_length):
                f = "ocr_bs" + str(config.eval_batch_size) + "_" + str(i) + "_" + str(j) + ".bin"
                t = np.fromfile(os.path.join(config.post_result_path, f), np.int32)
                t = t.reshape(config.eval_batch_size,)
                batch_decoded_label.append(t)
            ann_f = os.path.join(config.pre_result_path, "annotation")
            annotation = np.load(os.path.join(ann_f, "ocr_bs" + str(config.eval_batch_size) + "_" + str(i) + ".npy"))

            for b in range(config.eval_batch_size):
                text = annotation[b].decode("utf8")
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
    get_acc()
