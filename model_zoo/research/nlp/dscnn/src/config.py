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
# ===========================================================================
"""Config setting, will be used in train.py and eval.py"""
from src.utils import prepare_words_list

def data_config(parser):
    '''config for data.'''

    parser.add_argument('--data_url', type=str,
                        default='http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz',
                        help='Location of speech training data archive on the web.')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Where to download the dataset.')
    parser.add_argument('--feat_dir', type=str, default='feat',
                        help='Where to save the feature of audios')
    parser.add_argument('--background_volume', type=float, default=0.1,
                        help='How loud the background noise should be, between 0 and 1.')
    parser.add_argument('--background_frequency', type=float, default=0.8,
                        help='How many of the training samples have background noise mixed in.')
    parser.add_argument('--silence_percentage', type=float, default=10.0,
                        help='How much of the training data should be silence.')
    parser.add_argument('--unknown_percentage', type=float, default=10.0,
                        help='How much of the training data should be unknown words.')
    parser.add_argument('--time_shift_ms', type=float, default=100.0,
                        help='Range to randomly shift the training audio by in time.')
    parser.add_argument('--testing_percentage', type=int, default=10,
                        help='What percentage of wavs to use as a test set.')
    parser.add_argument('--validation_percentage', type=int, default=10,
                        help='What percentage of wavs to use as a validation set.')
    parser.add_argument('--wanted_words', type=str, default='yes,no,up,down,left,right,on,off,stop,go',
                        help='Words to use (others will be added to an unknown label)')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Expected sample rate of the wavs')
    parser.add_argument('--clip_duration_ms', type=int, default=1000,
                        help='Expected duration in milliseconds of the wavs')
    parser.add_argument('--window_size_ms', type=float, default=40.0, help='How long each spectrogram timeslice is')
    parser.add_argument('--window_stride_ms', type=float, default=20.0, help='How long each spectrogram timeslice is')
    parser.add_argument('--dct_coefficient_count', type=int, default=20,
                        help='How many bins to use for the MFCC fingerprint')


def train_config(parser):
    '''config for train.'''
    data_config(parser)

    # network related
    parser.add_argument('--model_size_info', type=int, nargs="+",
                        default=[6, 276, 10, 4, 2, 1, 276, 3, 3, 2, 2, 276, 3, 3, 1, 1, 276, 3, 3, 1, 1, 276, 3, 3, 1,
                                 1, 276, 3, 3, 1, 1],
                        help='Model dimensions - different for various models')
    parser.add_argument('--drop', type=float, default=0.9, help='dropout')
    parser.add_argument('--pretrained', type=str, default='', help='model_path, local pretrained model to load')

    # training related
    parser.add_argument('--use_graph_mode', default=1, type=int, help='use graph mode or feed mode')
    parser.add_argument('--val_interval', type=int, default=1, help='validate interval')

    # dataset related
    parser.add_argument('--per_batch_size', default=100, type=int, help='batch size for per gpu')

    # optimizer and lr related
    parser.add_argument('--lr_scheduler', default='multistep', type=str,
                        help='lr-scheduler, option type: multistep, cosine_annealing')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate of the training')
    parser.add_argument('--lr_epochs', type=str, default='20,40,60,80', help='epoch of lr changing')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='decrease lr by a factor of exponential lr_scheduler')
    parser.add_argument('--eta_min', type=float, default=0., help='eta_min in cosine_annealing scheduler')
    parser.add_argument('--T_max', type=int, default=80, help='T-max in cosine_annealing scheduler')
    parser.add_argument('--max_epoch', type=int, default=80, help='max epoch num to train the model')
    parser.add_argument('--warmup_epochs', default=0, type=float, help='warmup epoch')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.98, help='momentum')

    # logging related
    parser.add_argument('--log_interval', type=int, default=100, help='logging interval')
    parser.add_argument('--ckpt_path', type=str, default='train_outputs/', help='checkpoint save location')
    parser.add_argument('--ckpt_interval', type=int, default=100, help='save ckpt_interval')

    flags, _ = parser.parse_known_args()
    flags.dataset_sink_mode = bool(flags.use_graph_mode)
    flags.lr_epochs = list(map(int, flags.lr_epochs.split(',')))

    model_settings = prepare_model_settings(
        len(prepare_words_list(flags.wanted_words.split(','))),
        flags.sample_rate, flags.clip_duration_ms, flags.window_size_ms,
        flags.window_stride_ms, flags.dct_coefficient_count)
    model_settings['dropout1'] = flags.drop
    return flags, model_settings


def eval_config(parser):
    '''config for eval.'''
    parser.add_argument('--feat_dir', type=str, default='feat',
                        help='Where to save the feature of audios')
    parser.add_argument('--model_dir', type=str,
                        default='outputs',
                        help='which folder the models are saved in or specific path of one model')
    parser.add_argument('--wanted_words', type=str, default='yes,no,up,down,left,right,on,off,stop,go',
                        help='Words to use (others will be added to an unknown label)')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Expected sample rate of the wavs')
    parser.add_argument('--clip_duration_ms', type=int, default=1000,
                        help='Expected duration in milliseconds of the wavs')
    parser.add_argument('--window_size_ms', type=float, default=40.0, help='How long each spectrogram timeslice is')
    parser.add_argument('--window_stride_ms', type=float, default=20.0, help='How long each spectrogram timeslice is')
    parser.add_argument('--dct_coefficient_count', type=int, default=20,
                        help='How many bins to use for the MFCC fingerprint')
    parser.add_argument('--model_size_info', type=int, nargs="+",
                        default=[6, 276, 10, 4, 2, 1, 276, 3, 3, 2, 2, 276, 3, 3, 1, 1, 276, 3, 3, 1, 1, 276, 3, 3, 1,
                                 1, 276, 3, 3, 1, 1],
                        help='Model dimensions - different for various models')

    parser.add_argument('--per_batch_size', default=100, type=int, help='batch size for per gpu')
    parser.add_argument('--drop', type=float, default=0.9, help='dropout')

    # logging related
    parser.add_argument('--log_path', type=str, default='eval_outputs/', help='path to save eval log')

    flags, _ = parser.parse_known_args()
    model_settings = prepare_model_settings(
        len(prepare_words_list(flags.wanted_words.split(','))),
        flags.sample_rate, flags.clip_duration_ms, flags.window_size_ms,
        flags.window_stride_ms, flags.dct_coefficient_count)
    model_settings['dropout1'] = flags.drop
    return flags, model_settings


def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms,
                           dct_coefficient_count):
    '''Prepare model setting.'''
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    length_minus_window = (desired_samples - window_size_samples)
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
    fingerprint_size = dct_coefficient_count * spectrogram_length
    return {
        'desired_samples': desired_samples,
        'window_size_samples': window_size_samples,
        'window_stride_samples': window_stride_samples,
        'spectrogram_length': spectrogram_length,
        'dct_coefficient_count': dct_coefficient_count,
        'fingerprint_size': fingerprint_size,
        'label_count': label_count,
        'sample_rate': sample_rate,
    }
