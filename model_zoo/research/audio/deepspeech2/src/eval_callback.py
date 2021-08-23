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
"""Eval CallBack of Deepspeech2"""

import json
import os
import logging
import numpy as np
from mindspore import save_checkpoint, load_checkpoint
from mindspore.train.callback import Callback

from src.config import eval_config
from src.dataset import create_dataset
from src.deepspeech2 import PredictWithSoftmax, DeepSpeechModel
from src.greedydecoder import MSGreedyDecoder


class SaveCallback(Callback):
    """
    EvalCallback body
    """

    def __init__(self, path):

        super(SaveCallback, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.init_logger()
        self.interval = 5
        self.eval_start_epoch = 30
        self.config = eval_config
        with open(self.config.DataConfig.labels_path) as label_file:
            self.labels = json.load(label_file)
        self.model = PredictWithSoftmax(DeepSpeechModel(batch_size=self.config.DataConfig.batch_size,
                                                        rnn_hidden_size=self.config.ModelConfig.hidden_size,
                                                        nb_layers=self.config.ModelConfig.hidden_layers,
                                                        labels=self.labels,
                                                        rnn_type=self.config.ModelConfig.rnn_type,
                                                        audio_conf=self.config.DataConfig.SpectConfig,
                                                        bidirectional=True))
        self.ds_eval = create_dataset(audio_conf=self.config.DataConfig.SpectConfig,
                                      manifest_filepath=self.config.DataConfig.test_manifest,
                                      labels=self.labels, normalize=True, train_mode=False,
                                      batch_size=self.config.DataConfig.batch_size, rank=0, group_size=1)
        self.wer = float('inf')
        self.cer = float('inf')
        if self.config.LMConfig.decoder_type == 'greedy':
            self.decoder = MSGreedyDecoder(
                labels=self.labels, blank_index=self.labels.index('_'))
        else:
            raise NotImplementedError("Only greedy decoder is supported now")
        self.target_decoder = MSGreedyDecoder(
            self.labels, blank_index=self.labels.index('_'))
        self.path = path

    def epoch_end(self, run_context):
        """
        select ckpt after some epoch
        """
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num

        if cur_epoch >= self.eval_start_epoch and (cur_epoch - self.eval_start_epoch) % self.interval == 0:
            message = '------------Epoch {} :start eval------------'.format(
                cur_epoch)
            self.logger.info(message)
            if not os.path.exists(self.path):
                os.makedirs(self.path)
            filename = os.path.join(
                self.path, 'Deepspeech2' + '_' + str(cur_epoch) + '.ckpt')
            save_checkpoint(save_obj=cb_params.train_network,
                            ckpt_file_name=filename)
            message = '------------Epoch {} :training ckpt saved------------'.format(
                cur_epoch)
            self.logger.info(message)
            load_checkpoint(ckpt_file_name=filename, net=self.model)
            message = '------------Epoch {} :training ckpt loaded------------'.format(
                cur_epoch)
            self.logger.info(message)
            total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
            output_data = []
            for data in self.ds_eval.create_dict_iterator():
                inputs, input_length, target_indices, targets = data['inputs'], data['input_length'], data[
                    'target_indices'], data['label_values']
                split_targets = []
                start, count, last_id = 0, 0, 0
                target_indices, targets = target_indices.asnumpy(), targets.asnumpy()
                for i in range(np.shape(targets)[0]):
                    if target_indices[i, 0] == last_id:
                        count += 1
                    else:
                        split_targets.append(list(targets[start:count]))
                        last_id += 1
                        start = count
                        count += 1
                split_targets.append(list(targets[start:]))
                out, output_sizes = self.model(inputs, input_length)
                decoded_output, _ = self.decoder.decode(out, output_sizes)
                target_strings = self.target_decoder.convert_to_strings(
                    split_targets)

                if self.config.save_output is not None:
                    output_data.append(
                        (out.asnumpy(), output_sizes.asnumpy(), target_strings))
                for doutput, toutput in zip(decoded_output, target_strings):
                    transcript, reference = doutput[0], toutput[0]
                    wer_inst = self.decoder.wer(transcript, reference)
                    cer_inst = self.decoder.cer(transcript, reference)
                    total_wer += wer_inst
                    total_cer += cer_inst
                    num_tokens += len(reference.split())
                    num_chars += len(reference.replace(' ', ''))
                    if self.config.verbose:
                        print("Ref:", reference.lower())
                        print("Hyp:", transcript.lower())
                        print("WER:", float(wer_inst) / len(reference.split()),
                              "CER:", float(cer_inst) / len(reference.replace(' ', '')), "\n")
            wer = float(total_wer) / num_tokens
            cer = float(total_cer) / num_chars
            message = "----------Epoch {} :wer is {}------------".format(
                cur_epoch, wer)
            self.logger.info(message)
            message = "----------Epoch {} :cer is {}------------".format(
                cur_epoch, cer)
            self.logger.info(message)
            if wer < self.wer and cer < self.cer:
                self.wer = wer
                self.cer = cer
                file_name = os.path.join(self.path,
                                         'Deepspeech2' + '_' + str(cur_epoch) + '_' + str(self.wer) + '_' + str(
                                             self.cer) + ".ckpt")
                save_checkpoint(save_obj=cb_params.train_network,
                                ckpt_file_name=file_name)
                message = "Save the minimum wer and cer checkpoint,now Epoch {} : and ,the wer is {}, the cer is \
                 {}".format(cur_epoch, self.wer, self.cer)
                self.logger.info(message)

    def init_logger(self):
        self.logger.setLevel(level=logging.INFO)
        handler = logging.FileHandler('eval_callback.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
