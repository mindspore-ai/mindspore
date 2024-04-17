# Copyright 2024 Huawei Technologies Co., Ltd
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
import os
import ast
import yaml
import argparse
import collections
from copy import deepcopy

try:
    collectionsAbc = collections.abc
except AttributeError:
    collectionsAbc = collections


class Config(dict):
    """
    Configuration namespace. Convert dictionary to members.
    """

    def __init__(self, cfg_dict):
        super(Config, self).__init__()
        for k, v in cfg_dict.items():
            setattr(self, k, Config(v) if isinstance(v, dict) else v)

    def __setattr__(self, name, value):
        self[name] = value

    def __getattr__(self, name):
        if name in self:
            return self[name]

        raise AttributeError(name)

        # def __str__(self):
        #     return pformat(config_format_func(self))

        # def __repr__(self):
        #     return self.__str__()


def config_format_func(config, prefix=""):
    """
    Args:
        config: dict-like object
    Returns:
        formatted str
    """
    msg = ""
    if prefix:
        prefix += "."

    for k, v in config.__dict__.items():
        if isinstance(v, Config):
            msg += config_format_func(v, prefix=str(k))
        else:
            msg += format(prefix + str(k), "<40") + format(str(v), "<") + "\n"
    return msg


def load_config(file_path):
    BASE = "__BASE__"
    assert os.path.splitext(file_path)[-1] in [".yaml", ".yml"], f"[{file_path}] not yaml format."
    cfg_default, cfg_helper, cfg_choices = parse_yaml(file_path)

    # NOTE: cfgs outside have higher priority than cfgs in _BASE_
    if BASE in cfg_default:
        all_base_cfg_default = {}
        all_base_cfg_helper = {}
        all_base_cfg_choices = {}
        base_yamls = list(cfg_default[BASE])
        for base_yaml in base_yamls:
            if base_yaml.startswith("~"):
                base_yaml = os.path.expanduser(base_yaml)
            if not base_yaml.startswith("/"):
                base_yaml = os.path.join(os.path.dirname(file_path), base_yaml)

            base_cfg_default, base_cfg_helper, base_cfg_choices = load_config(base_yaml)
            all_base_cfg_default = merge_config(base_cfg_default, all_base_cfg_default)
            all_base_cfg_helper = merge_config(base_cfg_helper, all_base_cfg_helper)
            all_base_cfg_choices = merge_config(base_cfg_choices, all_base_cfg_choices)

        del cfg_default[BASE]
        return (
            merge_config(cfg_default, all_base_cfg_default),
            merge_config(cfg_helper, all_base_cfg_helper),
            merge_config(cfg_choices, all_base_cfg_choices),
        )

    return cfg_default, cfg_helper, cfg_choices


def parse_yaml(yaml_path):
    """
    Parse the yaml config file.

    Args:
        yaml_path: Path to the yaml config.
    """
    with open(yaml_path, "r") as fin:
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
        except:
            raise ValueError("Failed to parse yaml")
    return cfg, cfg_helper, cfg_choices


def merge_config(config, base):
    """Merge config"""
    new = deepcopy(base)
    for k, _ in config.items():
        if k in new and isinstance(new[k], dict) and isinstance(config[k], collectionsAbc.Mapping):
            new[k] = merge_config(config[k], new[k])
        else:
            new[k] = config[k]
    return new


def parse_cli_to_yaml(parents, args_main, cfg, helper=None, choices=None, cfg_path="default_config.yaml"):
    """
    Parse command line arguments to the configuration according to the default yaml.

    Args:
        parser: Parent parser.
        cfg: Base configuration.
        helper: Helper description.
        cfg_path: Path to the default yaml config.
    """
    parser = argparse.ArgumentParser(description="Main", parents=[parents])
    parser.set_defaults(config=cfg_path)
    helper = helper if helper is not None else {}
    choices = choices if choices is not None else {}
    for item in cfg:
        if not isinstance(cfg[item], list) and not isinstance(cfg[item], dict):
            help_description = helper[item] if item in helper else "Please reference to {}".format(cfg_path)
            choice = choices[item] if item in choices else None
            if isinstance(cfg[item], bool):
                parser.add_argument(
                    "--" + item, type=ast.literal_eval, default=cfg[item], choices=choice, help=help_description
                )
            else:
                parser.add_argument(
                    "--" + item, type=type(cfg[item]), default=cfg[item], choices=choice, help=help_description
                )
    args = parser.parse_args(args_main)
    return args


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
