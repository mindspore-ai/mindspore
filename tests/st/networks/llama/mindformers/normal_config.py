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
""" Transformer-Config dict parse module """

import os
import copy
from collections import OrderedDict
from typing import Optional, Union
import yaml

from mindformers.llama_utils import get_real_group_size


def ordered_yaml_load(stream, yaml_loader=yaml.SafeLoader,
                      object_pairs_hook=OrderedDict):
    """Load Yaml File in Orderedly."""
    class OrderedLoader(yaml_loader):
        pass

    def _construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        _construct_mapping)
    return yaml.load(stream, OrderedLoader)



BASE_CONFIG = 'base_config'

class DictConfig(dict):
    """config"""
    def __init__(self, **kwargs):
        super(DictConfig, self).__init__()
        self.update(kwargs)

    def __getattr__(self, key):
        if key not in self:
            return None
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]

    def __deepcopy__(self, memo=None):
        """Deep copy operation on arbitrary MindFormerConfig objects.

        Args:
            memo (dict) : Objects that already copied.
        Returns:
            MindFormerConfig : The deep copy of the given MindFormerConfig object.
        """
        config = self.__class__()
        for key in self.keys():
            config.__setattr__(copy.deepcopy(key, memo),
                               copy.deepcopy(self.__getattr__(key), memo))
        return config

    def to_dict(self):
        """
        for yaml dump,
        transform from Config to a strict dict class
        """
        return_dict = {}
        for key, val in self.items():
            if isinstance(val, self.__class__):
                val = val.to_dict()
            return_dict[key] = val
        return return_dict


class MindFormerConfig(DictConfig):
    """
    A Config class is inherit from dict.

    Config class can parse arguments from a config file of yaml or a dict.

    Args:
        args (list) : config filenames
        kwargs (dict) : config dictionary list

    Example:
        test.yaml:
            a:1
        >>> cfg = MindFormerConfig('./test.yaml')
        >>> cfg.a
        1

        >>> cfg = MindFormerConfig(**dict(a=1, b=dict(c=[0,1])))
        >>> cfg.b
        {'c': [0, 1]}
    """

    def __init__(self, *args, **kwargs):
        super(MindFormerConfig, self).__init__()
        cfg_dict = {}

        # load from file
        for arg in args:
            if isinstance(arg, str):
                if arg.endswith('yaml') or arg.endswith('yml'):
                    raw_dict = MindFormerConfig._file2dict(arg)
                    cfg_dict.update(raw_dict)

        # load dictionary configs
        if kwargs is not None:
            cfg_dict.update(kwargs)

        MindFormerConfig._dict2config(self, cfg_dict)

    def merge_from_dict(self, options):
        """Merge options into config file.

        Args:
            options (dict): dict of configs to merge from

        Examples:
            >>> options = {'model.arch': 'simmim'}
            >>> cfg = MindFormerConfig(**dict(model=dict(backbone=dict(type='vit'))))
            >>> cfg.merge_from_dict(options)
        """
        option_cfg_dict = {}
        for full_key, value in options.items():
            d = option_cfg_dict
            key_list = full_key.split('.')
            for sub_key in key_list[:-1]:
                d.setdefault(sub_key, MindFormerConfig())
                d = d[sub_key]
            sub_key = key_list[-1]
            d[sub_key] = value
        merge_dict = MindFormerConfig._merge_a_into_b(option_cfg_dict, self)
        MindFormerConfig._dict2config(self, merge_dict)

    @staticmethod
    def _merge_a_into_b(a, b):
        """Merge dict ``a`` into dict ``b``

        Values in ``a`` will overwrite ``b``

        Args:
            a (dict) : The source dict to be merged into b.
            b (dict) : The origin dict to be fetch keys from ``a``.
        Returns:
            dict: The modified dict of ``b`` using ``a``.
        """
        b = b.copy()
        for k, v in a.items():
            if isinstance(v, dict) and k in b:
                b[k] = MindFormerConfig._merge_a_into_b(v, b[k])
            else:
                b[k] = v
        return b

    @staticmethod
    def _file2dict(filename=None):
        """Convert config file to dictionary.

        Args:
            filename (str) : config file.
        """
        if filename is None:
            raise NameError('This {} cannot be empty.'.format(filename))

        filepath = os.path.realpath(filename)
        with open(filepath, encoding='utf-8') as fp:
            cfg_dict = ordered_yaml_load(fp, yaml_loader=yaml.FullLoader)

        # Load base config file.
        if BASE_CONFIG in cfg_dict:
            cfg_dir = os.path.dirname(filename)
            base_filenames = cfg_dict.pop(BASE_CONFIG)
            base_filenames = base_filenames if isinstance(
                base_filenames, list) else [base_filenames]

            cfg_dict_list = list()
            for base_filename in base_filenames:
                cfg_dict_item = MindFormerConfig._file2dict(
                    os.path.join(cfg_dir, base_filename))
                cfg_dict_list.append(cfg_dict_item)

            base_cfg_dict = dict()
            for cfg in cfg_dict_list:
                base_cfg_dict.update(cfg)

            # Merge config
            base_cfg_dict = MindFormerConfig._merge_a_into_b(cfg_dict, base_cfg_dict)
            cfg_dict = base_cfg_dict

        return cfg_dict

    @staticmethod
    def _dict2config(config, dic):
        """Convert dictionary to config.

        Args:
            config : Config object
            dic (dict) : dictionary
        Returns:

        Exceptions:

        """
        if isinstance(dic, dict):
            for key, value in dic.items():
                if isinstance(value, dict):
                    sub_config = MindFormerConfig()
                    dict.__setitem__(config, key, sub_config)
                    MindFormerConfig._dict2config(sub_config, value)
                else:
                    config[key] = dic[key]


class BaseArgsConfig:
    """Base Argument config."""
    _support_kwargs = []

    def __init__(self, **kwargs):
        if kwargs is not None:
            for key, value in kwargs.items():
                assert key in self._support_kwargs, \
                    f"The Config Class support input argument is {self._support_kwargs}, but get {key}"
                if value is None:
                    continue
                if isinstance(value, BaseArgsConfig):
                    value = value.__dict__
                self.__setattr__(key, value)


class ContextConfig(BaseArgsConfig):
    _support_kwargs = [
        'mode', 'precompile_only', 'device_target', 'device_id', 'save_graphs',
        'save_graphs_path', 'enable_dump', 'auto_tune_mode',
        'save_dump_path', 'enable_reduce_precision', 'variable_memory_max_size',
        'enable_profiling', 'profiling_options', 'enable_auto_mixed_precision',
        'enable_graph_kernel', 'reserve_class_name_in_scope', 'check_bprop',
        'max_device_memory', 'print_file_path', 'enable_sparse', 'max_call_depth',
        'env_config_path', 'graph_kernel_flags', 'save_compile_cache', 'runtime_num_threads',
        'load_compile_cache', 'grad_for_scalar', 'pynative_synchronize', 'mempool_block_size'
    ]

    def __init__(self,
                 mode: Optional[Union[int, str]] = 0,
                 device_target: str = "Ascend",
                 device_id: int = int(os.getenv('DEVICE_ID', '0')),
                 save_graphs: bool = False, save_graphs_path: str = ".", **kwargs):
        super(ContextConfig, self).__init__(mode=mode,
                                            device_id=device_id,
                                            device_target=device_target,
                                            save_graphs=save_graphs,
                                            save_graphs_path=save_graphs_path, **kwargs)


class ParallelContextConfig(BaseArgsConfig):
    _support_kwargs = [
        'device_num', 'global_rank', 'gradients_mean', 'gradient_fp32_sync', 'parallel_mode',
        'auto_parallel_search_mode', 'search_mode', 'parameter_broadcast', 'strategy_ckpt_load_file',
        'strategy_ckpt_save_file', 'full_batch', 'enable_parallel_optimizer', 'enable_alltoall',
        'all_reduce_fusion_config', 'pipeline_stages', 'grad_accumulation_step',
        'parallel_optimizer_config', 'comm_fusion'
    ]

    def __init__(self,
                 parallel_mode: str = 'STAND_ALONE',
                 device_num: int = get_real_group_size(),
                 gradients_mean: bool = False, **kwargs):
        super(ParallelContextConfig, self).__init__(parallel_mode=parallel_mode,
                                                    device_num=device_num,
                                                    gradients_mean=gradients_mean, **kwargs)
