# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License Version 2.0(the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0#
#
# Unless required by applicable law or agreed to in writing software
# distributed under the License is distributed on an "AS IS" BASIS
# WITHOUT WARRANT IES OR CONITTONS OF ANY KIND， either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ====================================================================================

"""Parse arguments"""
import os
import ast
import argparse
from pprint import pprint, pformat
import yaml
from src.utils import prepare_words_list


_config_path = '../../default_config.yaml'


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


class Config:
    """
    Configuration namespace. Convert dictionary to members
    """
    def __init__(self, cfg_dict):
        for k, v in cfg_dict.items():
            if isinstance(v, (list, tuple)):
                setattr(self, k, [Config(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, Config(v) if isinstance(v, dict) else v)

    def __str__(self):
        return pformat(self.__dict__)

    def __repr__(self):
        return self.__str__()


def parse_cli_to_yaml(parser, cfg, helper=None, choices=None, cfg_path='default_config.yaml'):
    """
    Parse command line arguments to the configuration according to the default yaml

    Args:
        parser: Parent parser
        cfg: Base configuration
        helper: Helper description
        cfg_path: Path to the default yaml config
    """
    parser = argparse.ArgumentParser(description='[REPLACE THIS at config.py]',
                                     parents=[parser])
    helper = {} if helper is None else helper
    choices = {} if choices is None else choices
    for item in cfg:
        if not isinstance(cfg[item], list) and not isinstance(cfg[item], dict):
            help_description = helper[item] if item in helper else 'Please reference to {}'.format(cfg_path)
            choice = choices[item] if item in choices else None
            if isinstance(cfg[item], bool):
                parser.add_argument('--' + item, type=ast.literal_eval, default=cfg[item], choices=choice,
                                    help=help_description)
            else:
                parser.add_argument('--' + item, type=type(cfg[item]), default=cfg[item], choices=choice,
                                    help=help_description)
    args = parser.parse_args()
    return args


def parse_yaml(yaml_path):
    """
    Parse the yaml config file

    Args:
        yaml_path: Path to the yaml config
    """
    with open(yaml_path, 'r') as fin:
        try:
            cfgs = yaml.load_all(fin.read(), Loader=yaml.FullLoader)
            cfgs = [x for x in cfgs]
            if len(cfgs) == 1:
                cfg_helper = {}
                cfg = cfgs[0]
                cfg_choices = {}
            elif len(cfgs) == 2:
                cfg, cfg_helper = cfgs
                cfg_choices = {}
            elif len(cfgs) == 3:
                cfg, cfg_helper, cfg_choices = cfgs
            else:
                raise ValueError('At most 3 docs (config description for help, choices) are supported in config yaml')
            print(cfg_helper)
        except:
            raise ValueError('Failed to parse yaml')
    return cfg, cfg_helper, cfg_choices


def merge(args, cfg):
    """
    Merge the base config from yaml file and command line arguments

    Args:
        args: command line arguments
        cfg: Base configuration
    """
    args_var = vars(args)
    for item in args_var:
        cfg[item] = args_var[item]
    return cfg


def get_config():
    """
    Get Config according to the yaml file and cli arguments
    """
    parser = argparse.ArgumentParser(description='default name', add_help=False)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--config_path', type=str, default=os.path.join(current_dir, _config_path),
                        help='Config file path')
    path_args, _ = parser.parse_known_args()
    default, helper, choices = parse_yaml(path_args.config_path)
    pprint(default)
    args = parse_cli_to_yaml(parser=parser, cfg=default, helper=helper, choices=choices, cfg_path=path_args.config_path)
    final_config = merge(args, default)
    configs = Config(final_config)

    configs.dataset_sink_mode = bool(configs.use_graph_mode)
    configs.lr_epochs = list(map(int, configs.lr_epochs.split(',')))
    configs.model_setting_dropout1 = configs.drop

    configs.model_setting_desired_samples = int(configs.sample_rate * configs.clip_duration_ms / 1000)
    configs.model_setting_window_size_samples = int(configs.sample_rate * configs.window_size_ms / 1000)
    configs.model_setting_window_stride_samples = int(configs.sample_rate * configs.window_stride_ms / 1000)
    length_minus_window = (configs.model_setting_desired_samples - configs.model_setting_window_size_samples)

    if length_minus_window < 0:
        configs.model_setting_spectrogram_length = 0
    else:
        configs.model_setting_spectrogram_length = 1 + int(length_minus_window
                                                           / configs.model_setting_window_stride_samples)

    configs.model_setting_fingerprint_size = configs.dct_coefficient_count * configs.model_setting_spectrogram_length
    configs.model_setting_label_count = len(prepare_words_list(configs.wanted_words.split(',')))
    configs.model_setting_sample_rate = configs.sample_rate
    configs.model_setting_dct_coefficient_count = configs.dct_coefficient_count

    return configs


config = get_config()
