# Copyright 2023 Huawei Technologies Co., Ltd
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
"""obfuscate network based on rewrite interfaces."""
import os
import re
from pathlib import Path
from string import Template
import numpy as np

import mindspore as ms
from mindspore import ops, nn
from mindspore import load_checkpoint, save_checkpoint, log
from mindspore.ops import functional as F
from mindspore.rewrite import SymbolTree, Node, NodeType, ScopedValue
from mindspore.rewrite.parsers import ClassDefParser
from mindspore.rewrite.parsers import ModuleParser

OBF_RATIOS_LENGTH = 1
MAX_OBF_RATIOS_NUM = 50
OBF_RATIOS_WIDTH = 0

_supported_ops = {
    'mul': ops.Mul,
    'matmul': ops.MatMul,
    'invert': ops.Inv
}

_supported_config_type = [
    'obf_metadata_config',
    'weight_obf_config',
    'network_obf_config'
]

_supported_metadata_type = [
    'random',
    'rearrange'
]

obf_medatadata_template = {
    'name': 'obf_metadata',
    'shape': [1,],
    'type': 'random',
    'save_metadata': True,
    'metadata_op': 'invert'
}

weight_obf_template = {
    'target': '',
    'weight_obf_ops': [{'name': 'mul', 'input_x': 'weight', 'input_y': 'obf_metadata'}]
}

network_obf_template = {
    'module': '',
    'target': '',
    'insert_new_input': [{'name': 'obf_metadata'}],
    'insert_ops': [{'name': 'mul', 'input_x': 'weight', 'input_y': 'obf_metadata'}]
}


def _transform_target_modules(target_modules):
    """transform target_modules to obf config"""
    obf_config = {}
    path = target_modules[0]
    target_list = target_modules[1].split('|')
    max_layers = 12
    layers = []
    obf_medatadata = obf_medatadata_template.copy()
    if len(target_modules) >= 3:
        obfuscate_layers = target_modules[2].split(':')
        if obfuscate_layers[1] != 'all':
            max_layers = int(obfuscate_layers[1])
        layers = list(range(0, max_layers))
        path_new = path.replace("blocks", "blocks/${layer}")
        network_obf_template['insert_ops'][0]['input_y'] = "obf_metadata_${layer}"
        weight_obf_template['weight_obf_ops'][0]['input_y'] = "obf_metadata_${layer}"
        weight_obf_template['name'] = "obf_metadata_${layer}"
        obf_medatadata['layers'] = layers
    else:
        path_new = path
    obf_config['obf_metadata_config'] = []
    obf_config['weight_obf_config'] = []
    obf_config['network_obf_config'] = []
    obf_config['obf_metadata_config'].append(obf_medatadata)

    for name in target_list:
        target_weight = '/'.join([path_new, name, 'weight'])
        target_bias = '/'.join([path_new, name, 'bias'])
        weight_obf = weight_obf_template.copy()
        weight_obf['target'] = target_weight
        bias_obf = weight_obf_template.copy()
        bias_obf['target'] = target_bias
        network_obf = network_obf_template.copy()
        network_obf['module'] = '/' + path_new
        network_obf['target'] = name
        if not layers:
            weight_obf['layers'] = layers
            bias_obf['layers'] = layers
            network_obf['layers'] = layers
        obf_config['weight_obf_config'].append(weight_obf)
        obf_config['weight_obf_config'].append(bias_obf)
        obf_config['network_obf_config'].append(network_obf)
    return obf_config


def _get_op(op_name):
    if op_name is None:
        return None
    if op_name not in _supported_ops:
        raise KeyError(f"'op name' must be in {list(_supported_ops.keys())}, but got {op_name}.")
    return _supported_ops[op_name]()


def obfuscate_ckpt(network, ckpt_files, target_modules=None, obf_config=None, saved_path='./', obfuscate_scale=100):
    """
    Obfuscate the plaintext checkpoint files according to the obfuscation config.

    Args:
        network (nn.Cell): The original network that need to be obfuscated.
        ckpt_files (str): The directory path of original ckpt files.
        target_modules (list[str], optional): The target ops that need to be obfuscated in the network. The first string
            represents the network path of the target ops in the original network, which should be in form of
            ``"A/B/C"``. The second string represents the names of multiple target ops in the same path, which
            should be in form of ``"D|E|F"``. For example, the target_modules of GPT2 can be ``['backbone/blocks
            /attention', 'dense1|dense2|dense3']``. If target_modules has the third value, it should be in the
            format of 'obfuscate_layers:all' or 'obfuscate_layers:int', which represents the number of layers
            need to be obfuscated of duplicate layers (such as transformer layers or resnet blocks).
            Default: ``None``.
        obf_config (dict, optional): The configuration of model obfuscation policies. Default: ``None``.
        saved_path (str, optional): The directory path for saving obfuscated ckpt files. Default: ``'./'``.
        obfuscate_scale (Union[float, int], optional): Obfuscate scale of weights.
            The generated random obf_ratios will be in
            range of (1 / obfuscate_scale, obfuscate_scale). Default: ``100``.

    Returns:
        dict[str], obf_metadata, which is the necessary data that needs to be loaded when running obfuscated network.

    Raises:
        TypeError: If `network` is not nn.Cell.
        TypeError: If `ckpt_files` is not string or `saved_path` is not string.
        TypeError: If `target_modules` is not list.
        TypeError: If target_modules's elements are not string.
        TypeError: If obf_config is not dict.
        ValueError: If `ckpt_files` does not exist or `saved_path` does not exist.
        ValueError: If the number of elements of `target_modules` is less than ``2``.
        ValueError: If the first string of `target_modules` contains characters other than uppercase and lowercase
            letters, numbers, ``'_'`` and ``'/'``.
        ValueError: If the second string of `target_modules` is empty or contains characters other than uppercase and
            lowercase letters, numbers, ``'_'`` and ``'/'``.
        ValueError: If the third string of `target_modules` is not in the format of 'obfuscate_layers:all' or
            'obfuscate_layers:int'.

    Examples:
        >>> from mindspore import obfuscate_ckpt, save_checkpoint
        >>> # Refer to https://atomgit.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>> save_checkpoint(net, './test_net.ckpt')
        >>> target_modules = ['', 'fc1|fc2']
        >>> obfuscate_ckpt(net, './', target_modules=target_modules, saved_path='./')
    """
    def _gen_obfuscate_tensor(tensor_shape, tensor_type='rearrange'):
        obf_tensor = None
        if tensor_type == 'rearrange':
            if len(tensor_shape) == 1:
                obf_tensor = ms.Tensor(np.random.permutation(tensor_shape[0]), dtype=ms.int32)
            if len(tensor_shape) == 2:
                tensor = ms.Tensor(np.identity(tensor_shape[0]), dtype=ms.int32)
                p = ms.Tensor(np.random.permutation(tensor_shape[1]), dtype=ms.int32)
                obf_tensor = tensor[:, p]
        if tensor_type == 'random':
            obf_tensor = ms.Tensor(np.random.randint(1, obfuscate_scale, size=tensor_shape), dtype=ms.float16)
        return obf_tensor

    def _gen_obf_metadata(config):
        name = config.get('name')
        if name is None:
            return
        save_metadata = config.get('save_metadata', False)
        metadata_op_name = config.get('metadata_op')
        layers = config.get('layers')
        if not layers:
            if not obf_metadata.get(name):
                obf_tensor = _gen_obfuscate_tensor(config.get('shape'), config.get('type'))
                obf_metadata[name] = obf_tensor
                if save_metadata:
                    saved_obf_tensor = obf_tensor
                    if metadata_op_name is not None:
                        metadata_op = _get_op(metadata_op_name)
                        saved_obf_tensor = metadata_op(saved_obf_tensor)
                    if saved_obf_tensor is not None:
                        saved_metadata[name] = saved_obf_tensor.asnumpy()
        else:
            for layer in layers:
                str_template = Template(name)
                obf_name = str_template.safe_substitute({"layer": str(layer)})
                obf_tensor = _gen_obfuscate_tensor(config.get('shape'), config.get('type'))
                obf_metadata[obf_name] = obf_tensor
                if save_metadata:
                    saved_obf_tensor = obf_tensor
                    if metadata_op_name is not None:
                        metadata_op = _get_op(metadata_op_name)
                        saved_obf_tensor = metadata_op(saved_obf_tensor)
                    if saved_obf_tensor is not None:
                        saved_metadata[obf_name] = saved_obf_tensor.asnumpy()

    if not isinstance(network, nn.Cell):
        raise TypeError("network must be nn.Cell, but got {}.".format(type(network)))
    _check_dir_path('ckpt_files', ckpt_files)
    _check_dir_path('saved_path', saved_path)

    if obf_config is None:
        if not _check_valid_target(network, target_modules):
            raise ValueError("{} is not exist, please check the input 'target_modules'.".format(target_modules))
        log.warning("'target_modules and obf_ratios' will be deprecated and "
                    "removed in a future version, use 'obf_config' instead.")
        obf_config = _transform_target_modules(target_modules)
    if not isinstance(obf_config, dict):
        raise TypeError("obf_config type should be dict, but got {}.".format(type(obf_config)))
    if not obf_config or not _check_valid_obf_config(obf_config, 'obf_metadata_config')\
        or not _check_valid_obf_config(obf_config, 'weight_obf_config'):
        raise ValueError("'obf_config' is empty or not valid, please check the input.")
    obf_metadata = {}
    obf_metadata_config = obf_config.get('obf_metadata_config', [])
    saved_metadata = {}
    for config in obf_metadata_config:
        _gen_obf_metadata(config)
    if (not isinstance(obfuscate_scale, (float, int))) or (obfuscate_scale <= 1):
        raise ValueError("obfuscate_scale must be float or int, and larger than 1, but got {}."
                         .format(obfuscate_scale))
    # start obfuscate ckpt
    ckpt_dir_files = os.listdir(ckpt_files)
    for ckpt_name in ckpt_dir_files:
        sub_path = os.path.realpath(ckpt_files) + '/' + ckpt_name
        if Path(sub_path).is_dir():
            sub_ckpt_file_list = os.listdir(sub_path)
            new_saved_path = os.path.realpath(saved_path) + '/' + ckpt_name
            if not os.path.exists(new_saved_path):
                try:
                    os.mkdir(new_saved_path, mode=0o700)
                except FileExistsError:
                    pass
            for sub_ckpt_name in sub_ckpt_file_list:
                if not sub_ckpt_name.endswith('.ckpt'):
                    continue
                _obfuscate_single_ckpt(os.path.realpath(sub_path) + '/' + sub_ckpt_name, obf_metadata,
                                       obf_config, new_saved_path)
        else:
            if not ckpt_name.endswith('.ckpt'):
                continue
            _obfuscate_single_ckpt(os.path.realpath(ckpt_files) + '/' + ckpt_name,
                                   obf_metadata, obf_config, saved_path)
    return saved_metadata


def _obfuscate_single_ckpt(ckpt_name, obf_metadata, obf_config, saved_path):
    """Obfuscate single ckpt file"""
    def _get_op_input_name(obf_op, name_key='input_x', layer=0):
        op_name = obf_op.get('name')
        input_name = obf_op.get(name_key)
        if input_name is None:
            log.error("can not find input: {} for op: {}.".format(name_key, op_name))
            return None
        str_template = Template(input_name)
        input_name = str_template.safe_substitute({"layer": str(layer)})
        return input_name

    def _get_op_input(input_name, obf_param):
        op_input = obf_metadata.get(input_name, None) if input_name.startswith('obf_metadata') else obf_param
        return op_input

    def _obfuscate_param(param, obf_metadata, obf_ops, layer=0):
        param_dtype = F.dtype(param)
        obf_param = param
        for obf_op in obf_ops:
            op_name = obf_op.get('name')
            if not isinstance(op_name, str):
                raise TypeError('{} should be str type, but got {}'.format(op_name, type(op_name)))
            if op_name == 'mul':
                input_x = obf_param
                input_y_name = _get_op_input_name(obf_op, 'input_y', layer)
                input_y = obf_metadata.get(input_y_name)
                if input_x is None or input_y is None:
                    log.error("input_x or input_y is None")
                    return None
                input_y = F.cast(input_y, param_dtype)
                obf_param = ops.mul(input_x, input_y)
            elif op_name == 'permuate':
                input_x_name = _get_op_input_name(obf_op, 'input_x', layer)
                p = obf_metadata.get(input_x_name, None)
                if p is None or obf_param is None:
                    log.error("input_x or param is None")
                    return None
                obf_param = obf_param[p]
            elif op_name == 'matmul':
                input_x_name = _get_op_input_name(obf_op, 'input_x', layer)
                input_y_name = _get_op_input_name(obf_op, 'input_y', layer)
                input_x = _get_op_input(input_x_name, obf_param)
                input_y = _get_op_input(input_y_name, obf_param)
                if input_x is None or input_y is None:
                    log.error("the input_x or input_y of op: {} is None.".format(op_name))
                    return None
                input_x = ops.transpose(input_x, (1, 0)) if obf_op.get('transpose_a', False) else input_x
                input_y = ops.transpose(input_y, (1, 0)) if obf_op.get('transpose_b', False) else input_y
                obf_param = ops.matmul(F.cast(input_x, param_dtype), F.cast(input_y, param_dtype))
            else:
                log.error("unsupported op, op must be matmul or permuate or mul, but got {}."
                          .format(op_name))
                return None
        return obf_param

    try:
        ckpt_param = load_checkpoint(ckpt_name)
    except (ValueError, TypeError, OSError):
        log.error("Load checkpoint failed for file {}.".format(ckpt_name))
        return False

    weight_obf_config = obf_config.get('weight_obf_config', [])
    for item, item_value in ckpt_param.items():
        item_split = item.split('.')
        param_path = '/'.join(item_split[:len(item_split)])
        for obf_target in weight_obf_config:
            if not isinstance(obf_target, dict):
                raise TypeError('{} should be dict type, but got {}'.format(obf_target, type(obf_target)))
            target = obf_target.get('target', None)
            layers = obf_target.get('layers', [])
            obf_ops = obf_target.get('weight_obf_ops', None)
            if not target or not obf_ops:
                raise KeyError("target or obf_ops is None.")
            if not layers:
                if target == param_path:
                    obf_param = _obfuscate_param(item_value.value(), obf_metadata, obf_ops)
                    if obf_param is None:
                        log.error("obfuscate weight {} failed.".format(item))
                        return False
                    item_value.set_data(obf_param)
            for layer in layers:
                str_template = Template(target)
                target_path = str_template.safe_substitute({"layer": str(layer)})
                if target_path == param_path:
                    obf_param = _obfuscate_param(item_value.value(), obf_metadata, obf_ops, layer)
                    if obf_param is None:
                        log.error("obfuscate weight {} failed.".format(item))
                        return False
                    item_value.set_data(obf_param)

    # save the obfuscated model to saved_path
    obf_param_list = []
    for item, item_value in ckpt_param.items():
        obf_param_list.append({'name': item, 'data': item_value})
    ckpt_file_name = ckpt_name.split('/')[-1]
    obf_ckpt_file_name = ckpt_file_name.split('.')[0] + '_obf' + '.ckpt'
    save_checkpoint(obf_param_list, os.path.realpath(saved_path) + '/' + obf_ckpt_file_name)
    return True


def load_obf_params_into_net(network, target_modules=None, obf_ratios=None, obf_config=None,
                             data_parallel_num=1, **kwargs):
    r"""
    Modify model structure according to obfuscation config and load obfuscated checkpoint into obfuscated network.

    Args:
        network (nn.Cell): The original network that needs to be obfuscated.
        target_modules (list[str], optional): The target ops that need to be obfuscated in the network.
            The first string
            represents the network path of the target ops in the original network, which should be in form of
            ``"A/B/C"``. The second string represents the names of multiple target ops in the same path, which
            should be in form of ``"D|E|F"``. For example, the target_modules of GPT2 can be ``['backbone
            /blocks/attention', 'dense1|dense2|dense3']``. If target_modules has the third value, it should be
            in the format of 'obfuscate_layers:all' or 'obfuscate_layers:int', which represents the number of
            layers that need to be obfuscated of duplicate layers (such as transformer layers or resnet
            blocks). Default: ``None``.
        obf_ratios (Tensor, optional): The obf ratios generated when executing :func:`mindspore.obfuscate_ckpt`.
            Default: ``None``.
        obf_config (dict, optional): The configuration of model obfuscation policies. Default: ``None``.
        data_parallel_num (int, optional): The data parallel number of parallel training. Default: ``1``.
        \*\*kwargs (dict): Configuration options dictionary.

            - ignored_func_decorators (list[str]): The name list of function decorators in network's python code.
            - ignored_class_decorators (list[str]): The name list of class decorators in network's python code.

    Returns:
        nn.Cell, new_net, which is the obfuscated network.

    Raises:
        TypeError: If `network` is not nn.Cell.
        TypeError: If `obf_ratios` is not Tensor.
        TypeError: If `target_modules` is not list.
        TypeError: If `obf_config` is not dict.
        TypeError: If target_modules's elements are not string.
        ValueError: If the number of elements of `target_modules` is less than ``2``.
        ValueError: If `obf_ratios` is empty Tensor.
        ValueError: If the first string of `target_modules` contains characters other than uppercase and lowercase
            letters, numbers, ``'_'`` and ``'/'``.
        ValueError: If the second string of `target_modules` is empty or contains characters other than uppercase and
            lowercase letters, numbers, ``'_'`` and ``'|'``.
        ValueError: If the third string of `target_modules` is not in the format of 'obfuscate_layers:all' or
            'obfuscate_layers:int'.
        TypeError: If `ignored_func_decorators` is not list[str] or `ignored_class_decorators` is not list[str].

    Examples:
        >>> from mindspore import obfuscate_ckpt, save_checkpoint, load_checkpoint, load_obf_params_into_net, Tensor
        >>> import mindspore.common.dtype as mstype
        >>> import numpy as np
        >>> # Refer to https://atomgit.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>> save_checkpoint(net, './test_net.ckpt')
        >>> target_modules = ['', 'fc1|fc2']
        >>> # obfuscate ckpt files
        >>> obfuscate_ckpt(net, './', target_modules=target_modules, saved_path='./')
        >>> # load obf ckpt into network
        >>> new_net = LeNet5()
        >>> load_checkpoint('./test_net_obf.ckpt', new_net)
        >>> obf_net = load_obf_params_into_net(new_net, target_modules)
    """
    if not isinstance(network, nn.Cell):
        raise TypeError("network must be nn.Cell, but got {}.".format(type(network)))
    if obf_config is None:
        if not _check_valid_target(network, target_modules):
            raise ValueError("{} is not exist, please check the input 'target_modules'.".format(target_modules))
        log.warning("'target_modules and obf_ratios' will be deprecated and "
                    "removed in a future version, use 'obf_config' instead.")
        obf_config = _transform_target_modules(target_modules)

    if not isinstance(obf_config, dict):
        raise TypeError('{} should be dict type, but got {}'.format(obf_config, type(obf_config)))

    if not obf_config or not _check_valid_obf_config(obf_config, 'network_obf_config'):
        raise ValueError("'obf_config' is empty or not valid, please check the input.")

    if (not isinstance(data_parallel_num, int)) or (data_parallel_num <= 0):
        raise ValueError("data_parallel_num must be positive number, but got {}.".format(data_parallel_num))

    network_obf_config = obf_config.get('network_obf_config', [])
    new_net = _obfuscate_network(network, network_obf_config, data_parallel_num=data_parallel_num, **kwargs)
    return new_net


def _check_dir_path(name, dir_path):
    """check directory path"""
    if not isinstance(dir_path, str):
        raise TypeError("{} must be string, but got {}.".format(name, type(dir_path)))
    if not os.path.exists(dir_path):
        raise ValueError("{} is not exist, please check the input {}.".format(dir_path, name))
    if not Path(dir_path).is_dir():
        raise TypeError("{} must be a directory path, but got {}.".format(name, dir_path))


def _check_valid_target(network, target_modules):
    """check whether the input 'target_modules' exists"""
    if not isinstance(target_modules, list):
        raise TypeError("target_modules type should be list, but got {}.".format(type(target_modules)))
    if len(target_modules) < 2:
        raise ValueError("target_modules should contain at least two string values, in the form of ['A/B/C', 'D1|D2'],"
                         "but got {}.".format(target_modules))
    if (not isinstance(target_modules[0], str)) or (not isinstance(target_modules[1], str)):
        raise TypeError("The values of target_modules should be string, but got {} and {}.".
                        format(type(target_modules[0]), type(target_modules[1])))

    if not target_modules[1]:
        raise ValueError("{} should be a non-empty string value, in the form of 'D1|D2'"
                         .format(target_modules[1]))
    if not re.fullmatch(pattern=r'([a-zA-Z]*[0-9]*\/*_*)*', string=target_modules[0]) \
            or not re.fullmatch(pattern=r'([a-zA-Z]*[0-9]*\|*_*)*', string=target_modules[1]):
        raise ValueError("please check the input 'target_modules'{},it should be in the form of ['A/B/C', 'D1|D2']."
                         "target_modules[0] can only contain uppercase and lowercase letters, numbers, '_' and '/',"
                         "target_modules[1] can only contain uppercase and lowercase letters, numbers, '_' and '|'"
                         .format(target_modules))
    # target_modules[0] is allowed to be '', it means the main network path
    path_list = target_modules[0].split('/')
    target_list = target_modules[1].split('|')
    net = network
    # DFS check whether path_list is valid
    stk = [net]
    i = 0
    global OBF_RATIOS_LENGTH
    OBF_RATIOS_LENGTH = 1
    while stk and i < len(path_list):
        net = stk.pop()
        if hasattr(net, path_list[i]):
            net = getattr(net, path_list[i])
            i += 1
            if isinstance(net, nn.CellList):
                OBF_RATIOS_LENGTH *= len(net)
                for n in net:
                    stk.append(n)
            elif isinstance(net, nn.Cell):
                stk.append(net)
            else:
                raise TypeError("Target_modules[0] should be a subgraph and it's type should be nn.Cell(nn.CellList),"
                                "but got type {}".format(type(net)))
    if target_modules[0] != '' and i != len(path_list):
        raise ValueError("the path {} does not exist.".format(target_modules[0]))
    # check whether target_list is valid
    global OBF_RATIOS_WIDTH
    OBF_RATIOS_WIDTH = 0
    for target in target_list:
        if not hasattr(net, target):
            log.warning("{} does not exist in the path {}".format(target, target_modules[0]))
        else:
            OBF_RATIOS_WIDTH += 1
    if OBF_RATIOS_WIDTH == 0:
        raise ValueError("all targets {} do not exist in the path {}.".format(target_list, target_modules[0]))
    _update_max_obf_ratios_num(target_modules)
    return True


def _check_ops_info(ops_info):
    """check ops info config"""
    for op in ops_info:
        op_name = op.get('name')
        if not isinstance(op_name, str):
            raise TypeError("op_name type should be str, but got {}.".format(type(op_name)))
        input_x_name = op.get('input_x')
        if not isinstance(input_x_name, str):
            raise TypeError("input_x_name type should be str, but got {}.".format(type(input_x_name)))
        input_y_name = op.get('input_y')
        if not isinstance(input_y_name, str):
            raise TypeError("input_y_name type should be str, but got {}.".format(type(input_y_name)))
        if not isinstance(op.get('transpose_a', False), bool):
            raise TypeError("transpose_a type should be bool, but got {}.".format(type(op.get('transpose_a'))))
        if not isinstance(op.get('transpose_b', False), bool):
            raise TypeError("transpose_b type should be bool, but got {}.".format(type(op.get('transpose_b'))))


def _check_new_input_info(insert_new_input):
    """check new input config"""
    if not isinstance(insert_new_input, list):
        raise TypeError("obf_config[][]['insert_new_input'] type should be list, but got {}."
                        .format(type(insert_new_input)))
    for new_input in insert_new_input:
        input_name = new_input.get('name')
        if not isinstance(input_name, str):
            raise TypeError("obf_config[][]['insert_new_input'][]['name'] type should be str, but got {}."
                            .format(type(input_name)))


def _check_obf_metadata_config(config):
    """check obf metadata config"""
    name = config.get('name')
    if not name or not isinstance(name, str):
        raise TypeError("obf_config[][]['name'] type should be str, but got {}.".format(type(name)))
    shape = config.get('shape')
    if not shape or not isinstance(shape, list):
        raise TypeError("obf_config[][]['shape'] type should be list, but got {}.".format(type(shape)))
    for item in shape:
        if not isinstance(item, int):
            raise TypeError("shape[] type should be int, but got {}.".format(type(item)))
    save_metadata = config.get('save_metadata', True)
    if not isinstance(save_metadata, bool):
        raise TypeError("obf_config[][]['save_metadata'] type should be bool, but got {}."
                        .format(type(save_metadata)))
    metadata_type = config.get('type')
    if metadata_type is not None:
        if not isinstance(metadata_type, str) or metadata_type not in _supported_metadata_type:
            raise TypeError("obf_config[][]['type'] should be str and must in {}, but got {}."
                            .format(str(_supported_metadata_type), type(metadata_type)))


def _check_weight_obf_config(config):
    """check weight obfuscation config"""
    target = config.get('target')
    if not target or not isinstance(target, str):
        raise TypeError("obf_config[][]['target'] type should be str, but got {}.".format(type(target)))
    weight_obf_ops = config.get('weight_obf_ops', [])
    if not isinstance(weight_obf_ops, list):
        raise TypeError("obf_config[][]['weight_obf_ops'] type should be list, but got {}."
                        .format(type(weight_obf_ops)))
    _check_ops_info(weight_obf_ops)


def _check_network_obf_config(config):
    """check network obfuscation config"""
    target = config.get('target')
    if not target or not isinstance(target, str):
        raise TypeError("obf_config[][]['target'] type should be str, but got {}.".format(type(target)))
    module = config.get('module')
    if not module or not isinstance(module, str):
        raise TypeError("obf_config[][]['module'] type should be str, but got {}.".format(type(module)))
    insert_new_input = config.get('insert_new_input', [])
    _check_new_input_info(insert_new_input)
    insert_ops = config.get('insert_ops', [])
    if not isinstance(insert_ops, list):
        raise TypeError("obf_config[][]['insert_ops'] type should be list, but got {}.".format(type(insert_ops)))
    _check_ops_info(insert_ops)


def _check_valid_obf_config(obf_config, config_type):
    """check obfuscation config"""
    if not isinstance(config_type, str) or config_type not in _supported_config_type:
        raise TypeError("config_type must be str, and in {}, but got {}."
                        .format(str(_supported_config_type), config_type))
    for config_type_item in obf_config.keys():
        if not isinstance(config_type_item, str) or config_type_item not in _supported_config_type:
            raise TypeError("config_type must be str, and in {}, but got {}."
                            .format(str(_supported_config_type), config_type_item))
    config_list = obf_config.get(config_type)
    if not isinstance(config_list, list):
        raise TypeError("obf_config[] type of should be list, but got {}.".format(type(config_list)))

    for config in config_list:
        if not isinstance(config, dict):
            raise TypeError("obf_config[][] type should be dict, but got {}.".format(type(config)))
        if config_type == 'obf_metadata_config':
            _check_obf_metadata_config(config)
        elif config_type == 'weight_obf_config':
            _check_weight_obf_config(config)
        elif config_type == 'network_obf_config':
            _check_network_obf_config(config)
        layers = config.get('layers')
        if layers is not None:
            if not isinstance(layers, list):
                raise TypeError("obf_config[][]['layers'] type should be list, but got {}.".format(type(layers)))
            for layer in layers:
                if not isinstance(layer, int):
                    raise TypeError("obf_config[][]['layers'][] type should be int, but got {}.".format(type(layer)))
    return True


def _update_max_obf_ratios_num(target_modules):
    """Update MAX_OBF_RATIOS_NUM"""
    if len(target_modules) >= 3:
        obfuscate_layers = target_modules[2].split(':')
        if len(obfuscate_layers) != 2 or obfuscate_layers[0] != 'obfuscate_layers':
            raise ValueError("The third value of target_modules should be in the format of 'obfuscate_layers:all' or"
                             "'obfuscate_layers:int'")
        global MAX_OBF_RATIOS_NUM
        if obfuscate_layers[1] == 'all':
            MAX_OBF_RATIOS_NUM = OBF_RATIOS_LENGTH * OBF_RATIOS_WIDTH
        else:
            if not obfuscate_layers[1].isdigit():
                raise ValueError(
                    "The third value of target_modules should be in the format of 'obfuscate_layers:all' or"
                    "'obfuscate_layers:int'")
            MAX_OBF_RATIOS_NUM = int(obfuscate_layers[1]) * OBF_RATIOS_WIDTH


def _remove_digit(item):
    """remove digit in the parameter path"""
    item_split = item.split('_')
    for tmp_str in item_split[:]:
        if tmp_str.isdigit():
            item_split.remove(tmp_str)
    return '_'.join(item_split)


def _remove_scope(item):
    """remove scope of name values"""
    item_split = item.split('.')
    for tmp_str in item_split[:]:
        if tmp_str == 'self':
            item_split.remove(tmp_str)
    return '.'.join(item_split)


def _obfuscate_network(model, obf_config=None, data_parallel_num=1, **kwargs):
    """obfuscate original network, including add deobfuscation ops and add inputs for passing obf_metadata."""

    def _insert_input(stree: SymbolTree, arg_name: str = 'obf_metadata'):
        """add inputs for passing obf_ratio"""
        last_input = None
        for node in stree.nodes():
            if node.get_node_type() == NodeType.Input:
                last_input = node
        position = stree.after(last_input)
        # the insert input node name would be 'input_obf_metadata'
        new_input_node = last_input.create_input(arg_name)
        stree.insert(position, new_input_node)

    def _update_subnet(substree: SymbolTree, subnode: Node):
        """update the network once the subnet is obfuscated"""
        input_y_node = substree.get_node("input_obf_metadata")
        if input_y_node is None:
            log.error("can not find input node: obf_metadata for net: {}.".format(subnode.get_name()))
            return False
        if hasattr(subnode, 'get_handler'):
            subnode.get_handler().append_kwarg({"obf_metadata": input_y_node.get_targets()[0]})
        else:
            subnode.append_kwarg({"obf_metadata": input_y_node.get_targets()[0]})
        return True

    def _insert_ops(stree: SymbolTree, node: Node, insert_ops: list):
        """add mul operation for original network"""
        current_node = node
        for insert_op in insert_ops:
            arg_list = current_node.get_targets().copy()
            obf_metadata = stree.get_node("input_obf_metadata")
            if obf_metadata is None:
                raise ValueError("can not find input node: obf_metadata for net: {}.".format(current_node.get_name()))
            v: str = obf_metadata.get_targets()[0].value
            index = insert_op['input_y']
            sv: ScopedValue = ScopedValue.create_naming_value(v + f'["{index}"]')
            arg_list.append(sv)
            target_list = current_node.get_targets().copy()
            name = insert_op['name']
            if data_parallel_num > 1:
                new_node = current_node.create_call_cell(cell=_get_op(name).shard(((data_parallel_num, 1), ())),
                                                         targets=target_list, args=arg_list, name=name)
            else:
                new_node = current_node.create_call_cell(cell=_get_op(name), targets=target_list, args=arg_list,
                                                         name=name)
            position = stree.after(current_node)
            stree.insert(position, new_node)
            current_node = new_node

    def _insert_ops_by_name(stree: SymbolTree, after_name_list: list, module: str):
        """add mul operation after the target nodes according the name of them"""
        if not after_name_list:
            return
        for node in stree.nodes():
            for after_name in after_name_list:
                if node.get_name() == after_name:
                    insert_ops = insert_ops_map[module+'/'+after_name]
                    _insert_ops(stree, node, insert_ops)

    def _process_controlflow_node(node: Node, stree: SymbolTree, full_path: str, path: str, targets: dict):
        ctrl = node.get_handler() if hasattr(node, 'get_handler') else node
        cell_loop_name = ''
        find_cell_loop = False
        if hasattr(ctrl, "loop_vars") and ctrl.loop_vars:
            cell_loop_name = ctrl.loop_vars[0]
            inputs = ctrl.get_inputs()
            for node_input in inputs:
                if node_input.get_node_type() == NodeType.CellContainer:
                    find_cell_loop = True
                    full_node_name = node_input.get_name()
                    node_name = _remove_digit(_remove_scope(full_node_name))
                    if not _process_cellcontainer_node(node_input, full_path+'/'+full_node_name,
                                                       path+'/'+node_name, targets):
                        log.error("_process_cellcontainer_node for node: {} failed.".format(node_name))
                        return False
        for c_node in ctrl.nodes():
            c_node_name = c_node.get_name()
            c_node_type = c_node.get_node_type()
            if c_node.get_node_type() == NodeType.ControlFlow:
                if not _process_controlflow_node(c_node, stree, full_path+'/'+c_node_name, path, targets):
                    return False
            elif c_node.get_node_type() == NodeType.Tree and _is_target_module(path + '/' + c_node_name, targets):
                sub_stree = SymbolTree(c_node.symbol_tree)
                _insert_input(sub_stree, arg_name='obf_metadata')
                _insert_ops_by_name(sub_stree, after_name_list=targets.get(path + '/' + c_node_name, None),
                                    module=path + '/' + c_node_name)
                if not _traverse(sub_stree, full_path+'/'+c_node_name, path+'/'+c_node_name, targets):
                    log.error("_traverse for node: {} failed.".format(c_node_name))
                    return False
                if not _update_subnet(sub_stree, c_node):
                    log.error("_update_subnet for node: {} failed.".format(c_node_name))
                    return False
            elif find_cell_loop and c_node_type == NodeType.CallFunction and c_node_name.startswith(cell_loop_name):
                input_y_node = stree.get_node("input_obf_metadata")
                if input_y_node is None:
                    log.error("input_y_node for node: {} is None.".format(c_node_name))
                    return False
                c_node.append_kwarg({"obf_metadata": input_y_node.get_targets()[0]})
        return True

    def _process_cellcontainer_node(node: Node, full_path: str, path: str, targets: dict):
        cellcontainer = node.get_handler() if hasattr(node, 'get_handler') else node
        for i in range(len(cellcontainer.nodes())):
            cell_node = cellcontainer.nodes()[i]
            # insert input for each sub_stree in cell_container
            if _is_target_module(path, targets) and cell_node.get_node_type() == NodeType.Tree:
                sub_stree = SymbolTree(cell_node.symbol_tree)
                _insert_input(sub_stree, arg_name='obf_metadata')
                _insert_ops_by_name(sub_stree, after_name_list=targets.get(path, None), module=path)
                if not _traverse(sub_stree, full_path + '/' + str(i), path + '/' + str(i), targets):
                    return False
        return True

    def _is_target_module(path, targets):
        for target_module in targets.keys():
            if target_module.startswith(path):
                return True
        return False

    def _traverse(stree: SymbolTree, full_path: str, path: str, targets: dict):
        for node in stree.nodes():
            node_name = node.get_name()
            if node.get_node_type() == NodeType.ControlFlow:
                if not _process_controlflow_node(node, stree, full_path + '/' + node_name, path, targets):
                    log.error("process controlflow node: {} failed.".format(node.get_name()))
                    return False
            elif node.get_node_type() == NodeType.Tree and _is_target_module(path + '/' + node_name, targets):
                sub_stree = node.get_sub_tree()
                _insert_input(sub_stree, arg_name='obf_metadata')
                _insert_ops_by_name(sub_stree, after_name_list=targets.get(path + '/' + node_name, None),
                                    module=path + '/' + node_name)
                if not _traverse(sub_stree, full_path + '/' + node_name, path + '/' + node_name, targets):
                    log.error("traverse sub_stree for node: {} failed.".format(node.get_name()))
                    return False
                if not _update_subnet(sub_stree, node):
                    log.error("update subnet for node: {} failed.".format(node.get_name()))
                    return False
        return True

    def _register_denied_func_decorators(fn):
        """set the function decorators which should be denied for parse"""
        name = "denied_function_decorator_list"
        setattr(ClassDefParser, name, fn)

    def _register_denied_class_decorators(fn):
        """set the class decorators which should be denied for parse"""
        name = "denied_class_decorator_list"
        setattr(ModuleParser, name, fn)

    if 'ignored_func_decorators' in kwargs:
        kw_func_dec = kwargs["ignored_func_decorators"]
        if not isinstance(kw_func_dec, list):
            raise TypeError('{} should be list, but got {}'.format(kw_func_dec, type(kw_func_dec)))
        if kw_func_dec and not isinstance(kw_func_dec[0], str):
            raise TypeError('elements of {} should be str, but got {}'.format(kw_func_dec, type(kw_func_dec[0])))
        _register_denied_func_decorators(kw_func_dec)
    else:
        _register_denied_func_decorators(["_args_type_validator_check", "_LogActionOnce", "cell_attr_register"])
    if 'ignored_class_decorators' in kwargs:
        kw_class_dec = kwargs["ignored_class_decorators"]
        _register_denied_class_decorators(kw_class_dec)
        if not isinstance(kw_class_dec, list):
            raise TypeError('{} should be list[str] type, but got {}'.format(kw_class_dec, type(kw_class_dec)))
        if kw_class_dec and not isinstance(kw_class_dec[0], str):
            raise TypeError('elements of {} should be str, but got {}'.format(kw_class_dec, type(kw_class_dec[0])))

    targets = {}
    insert_ops_map = {}
    for obf_item in obf_config:
        module = obf_item.get('module', None)
        target = obf_item.get('target', None)
        insert_ops_info = obf_item.get('insert_ops', None)
        layers = obf_item.get('layers', [])
        if not layers:
            real_insert_ops_info = []
            if not targets.get(module, None):
                targets[module] = []
            if target not in targets[module]:
                targets[module].append(target)
            target_path = module + '/' + target
            for op_info in insert_ops_info:
                real_op_info = op_info.copy()
                real_insert_ops_info.append(real_op_info)
            insert_ops_map[target_path] = real_insert_ops_info
        for layer in layers:
            real_insert_ops_info = []
            str_template = Template(module)
            real_module = str_template.safe_substitute({"layer": str(layer)})
            if not targets.get(real_module, None):
                targets[real_module] = []
            if target not in targets[real_module]:
                targets[real_module].append(target)
            target_path = real_module + '/' + target
            for op_info in insert_ops_info:
                real_op_info = op_info.copy()
                str_template = Template(real_op_info['input_x'])
                real_op_info['input_x'] = str_template.safe_substitute({"layer": str(layer)})
                str_template = Template(real_op_info['input_y'])
                real_op_info['input_y'] = str_template.safe_substitute({"layer": str(layer)})
                real_insert_ops_info.append(real_op_info)
            insert_ops_map[target_path] = real_insert_ops_info

    root_path = ""
    main_stree = SymbolTree.create(model)
    _insert_input(main_stree, arg_name='obf_metadata')
    _insert_ops_by_name(main_stree, after_name_list=targets.get(root_path, None), module=root_path)
    if not _traverse(main_stree, full_path=root_path, path=root_path, targets=targets):
        log.error("_traverse for root_path: {} failed.".format(root_path))
        return None
    new_net = main_stree.get_network()
    return new_net
