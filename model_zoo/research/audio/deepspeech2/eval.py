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
# ===========================================================================

"""
Eval DeepSpeech2
"""
import argparse
import json
import pickle
import numpy as np
from src.config import eval_config
from src.deepspeech2 import DeepSpeechModel, PredictWithSoftmax
from src.dataset import create_dataset
from src.greedydecoder import MSGreedyDecoder
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

parser = argparse.ArgumentParser(description='DeepSpeech evaluation')
parser.add_argument('--bidirectional', action="store_false", default=True, help='Use bidirectional RNN')
parser.add_argument('--pretrain_ckpt', type=str, default='', help='Pretrained checkpoint path')
parser.add_argument('--device_target', type=str, default="GPU", choices=("GPU", "CPU"),
                    help='Device target, support GPU and CPU, Default: GPU')
args = parser.parse_args()

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, save_graphs=False)
    config = eval_config
    with open(config.DataConfig.labels_path) as label_file:
        labels = json.load(label_file)

    model = PredictWithSoftmax(DeepSpeechModel(batch_size=config.DataConfig.batch_size,
                                               rnn_hidden_size=config.ModelConfig.hidden_size,
                                               nb_layers=config.ModelConfig.hidden_layers,
                                               labels=labels,
                                               rnn_type=config.ModelConfig.rnn_type,
                                               audio_conf=config.DataConfig.SpectConfig,
                                               bidirectional=args.bidirectional))

    ds_eval = create_dataset(audio_conf=config.DataConfig.SpectConfig,
                             manifest_filepath=config.DataConfig.test_manifest,
                             labels=labels, normalize=True, train_mode=False,
                             batch_size=config.DataConfig.batch_size, rank=0, group_size=1)

    param_dict = load_checkpoint(args.pretrain_ckpt)
    load_param_into_net(model, param_dict)
    print('Successfully loading the pre-trained model')


    if config.LMConfig.decoder_type == 'greedy':
        decoder = MSGreedyDecoder(labels=labels, blank_index=labels.index('_'))
    else:
        raise NotImplementedError("Only greedy decoder is supported now")
    target_decoder = MSGreedyDecoder(labels, blank_index=labels.index('_'))

    model.set_train(False)
    total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
    output_data = []
    for data in ds_eval.create_dict_iterator():
        inputs, input_length, target_indices, targets = data['inputs'], data['input_length'], data['target_indices'], \
                                                        data['label_values']

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
        out, output_sizes = model(inputs, input_length)
        decoded_output, _ = decoder.decode(out, output_sizes)
        target_strings = target_decoder.convert_to_strings(split_targets)

        if config.save_output is not None:
            output_data.append((out.asnumpy(), output_sizes.asnumpy(), target_strings))
        for doutput, toutput in zip(decoded_output, target_strings):
            transcript, reference = doutput[0], toutput[0]
            wer_inst = decoder.wer(transcript, reference)
            cer_inst = decoder.cer(transcript, reference)
            total_wer += wer_inst
            total_cer += cer_inst
            num_tokens += len(reference.split())
            num_chars += len(reference.replace(' ', ''))
            if config.verbose:
                print("Ref:", reference.lower())
                print("Hyp:", transcript.lower())
                print("WER:", float(wer_inst) / len(reference.split()),
                      "CER:", float(cer_inst) / len(reference.replace(' ', '')), "\n")
    wer = float(total_wer) / num_tokens
    cer = float(total_cer) / num_chars

    print('Test Summary \t'
          'Average WER {wer:.3f}\t'
          'Average CER {cer:.3f}\t'.format(wer=wer * 100, cer=cer * 100))

    if config.save_output is not None:
        with open(config.save_output + '.bin', 'wb') as output:
            pickle.dump(output_data, output)
