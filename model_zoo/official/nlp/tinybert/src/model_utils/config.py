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

"""Parse arguments"""

import os
import ast
import argparse
from pprint import pformat
import yaml
import mindspore.common.dtype as mstype
from src.tinybert_model import BertConfig


class Config:
    """
    Configuration namespace. Convert dictionary to members.
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


def parse_cli_to_yaml(parser, cfg, helper=None, choices=None, cfg_path="pretrain_base_config.yaml"):
    """
    Parse command line arguments to the configuration according to the default yaml.

    Args:
        parser: Parent parser.
        cfg: Base configuration.
        helper: Helper description.
        cfg_path: Path to the default yaml config.
    """
    parser = argparse.ArgumentParser(description="[REPLACE THIS at config.py]",
                                     parents=[parser])
    helper = {} if helper is None else helper
    choices = {} if choices is None else choices
    for item in cfg:
        if not isinstance(cfg[item], list) and not isinstance(cfg[item], dict):
            help_description = helper[item] if item in helper else "Please reference to {}".format(cfg_path)
            choice = choices[item] if item in choices else None
            if isinstance(cfg[item], bool):
                parser.add_argument("--" + item, type=ast.literal_eval, default=cfg[item], choices=choice,
                                    help=help_description)
            else:
                parser.add_argument("--" + item, type=type(cfg[item]), default=cfg[item], choices=choice,
                                    help=help_description)
    args = parser.parse_args()
    return args


def parse_yaml(yaml_path):
    """
    Parse the yaml config file.

    Args:
        yaml_path: Path to the yaml config.
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
                raise ValueError("At most 3 docs (config, description for help, choices) are supported in config yaml")
            # print(cfg_helper)
        except:
            raise ValueError("Failed to parse yaml")
    return cfg, cfg_helper, cfg_choices


def merge(args, cfg):
    """
    Merge the base config from yaml file and command line arguments.

    Args:
        args: Command line arguments.
        cfg: Base configuration.
    """
    args_var = vars(args)
    for item in args_var:
        cfg[item] = args_var[item]
    return cfg


def extra_operations(cfg):
    """
    Do extra work on config

    Args:
        config: Object after instantiation of class 'Config'.
    """
    def create_filter_fun(keywords):
        return lambda x: not (True in [key in x.name.lower() for key in keywords])

    if cfg.description == 'general_distill':
        cfg.common_cfg.loss_scale_value = 2 ** 16
        cfg.common_cfg.AdamWeightDecay.decay_filter = create_filter_fun(cfg.common_cfg.AdamWeightDecay.decay_filter)
        cfg.bert_teacher_net_cfg.dtype = mstype.float32
        cfg.bert_teacher_net_cfg.compute_type = mstype.float16
        cfg.bert_student_net_cfg.dtype = mstype.float32
        cfg.bert_student_net_cfg.compute_type = mstype.float16
        cfg.bert_teacher_net_cfg = BertConfig(**cfg.bert_teacher_net_cfg.__dict__)
        cfg.bert_student_net_cfg = BertConfig(**cfg.bert_student_net_cfg.__dict__)
    elif cfg.description == 'task_distill':
        cfg.phase1_cfg.loss_scale_value = 2 ** 8
        cfg.phase1_cfg.optimizer_cfg.AdamWeightDecay.decay_filter = create_filter_fun(
            cfg.phase1_cfg.optimizer_cfg.AdamWeightDecay.decay_filter)
        cfg.phase2_cfg.loss_scale_value = 2 ** 16
        cfg.phase2_cfg.optimizer_cfg.AdamWeightDecay.decay_filter = create_filter_fun(
            cfg.phase2_cfg.optimizer_cfg.AdamWeightDecay.decay_filter)
        cfg.td_teacher_net_cfg.dtype = mstype.float32
        cfg.td_teacher_net_cfg.compute_type = mstype.float16
        cfg.td_student_net_cfg.dtype = mstype.float32
        cfg.td_student_net_cfg.compute_type = mstype.float16
        cfg.td_teacher_net_cfg = BertConfig(**cfg.td_teacher_net_cfg.__dict__)
        cfg.td_student_net_cfg = BertConfig(**cfg.td_student_net_cfg.__dict__)
    else:
        pass


def get_config():
    """
    Get Config according to the yaml file and cli arguments.
    """
    def get_abs_path(path_input):
        if os.path.isabs(path_input):
            return path_input
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, path_input)
    parser = argparse.ArgumentParser(description="default name", add_help=False)
    parser.add_argument("--config_path", type=get_abs_path, default="../../gd_config.yaml",
                        help="Config file path")
    path_args, _ = parser.parse_known_args()
    default, helper, choices = parse_yaml(path_args.config_path)
    # pprint(default)
    args = parse_cli_to_yaml(parser=parser, cfg=default, helper=helper, choices=choices, cfg_path=path_args.config_path)
    final_config = merge(args, default)
    config_obj = Config(final_config)
    extra_operations(config_obj)
    return config_obj


config = get_config()
# td_teacher_net_cfg = config.td_teacher_net_cfg
# td_student_net_cfg = config.td_student_net_cfg
if config.description == 'general_distill':
    common_cfg = config.common_cfg
    bert_teacher_net_cfg = config.bert_teacher_net_cfg
    bert_student_net_cfg = config.bert_student_net_cfg
elif config.description == 'task_distill':
    phase1_cfg = config.phase1_cfg
    phase2_cfg = config.phase2_cfg
    eval_cfg = config.eval_cfg
    td_teacher_net_cfg = config.td_teacher_net_cfg
    td_student_net_cfg = config.td_student_net_cfg
else:
    pass
if __name__ == '__main__':
    print(config)
