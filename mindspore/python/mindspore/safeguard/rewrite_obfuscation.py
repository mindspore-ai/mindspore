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
import secrets
from pathlib import Path

from mindspore import ops, nn
from mindspore.common.tensor import Tensor
from mindspore import log as logger
from mindspore import load_checkpoint, save_checkpoint
from mindspore.rewrite import SymbolTree, Node, NodeType, TreeNodeHelper, ScopedValue
from mindspore.rewrite.parsers.class_def_parser import ClassDefParser
from mindspore.rewrite.parsers.class_def_parser import ModuleParser

OBF_RATIOS_LENGTH = 1
MAX_OBF_RATIOS_NUM = 50
OBF_RATIOS_WIDTH = 0
OBF_RATIOS_INSERT_INDEX = 0


def obfuscate_ckpt(network, ckpt_files, target_modules=None, saved_path='./'):
    """
    obfuscate the plaintext checkpoint files. Usually used in conjunction with
    :func:`mindspore.load_obf_params_into_net`.
    interface.

    Args:
        network (nn.Cell): The original network that need to be obfuscated.
        ckpt_files (str): The directory path of original ckpt files.
        target_modules (list[str]): The target module of network that need to be obfuscated. The first string
            represents the network path of target module in original network, which should be in form of ``'A/B/C'``.
            The second string represents the obfuscation target module, which should be in form of ``'D|E|F'``. For
            example, thr target_modules of GPT2 can be ``['backbone/blocks/attention', 'dense1|dense2|dense3']``.
            If target_modules has the third value, it should be in the format of 'obfuscate_layers:all' or
            'obfuscate_layers:int', which represents the number of layers need to be obfuscated of duplicate layers
            (such as transformer layers or resnet blocks). If target_modules is ``None``, the function would search
            target modules by itself. If found, the searched target module would be used, otherwise suggested target
            modules would be given with warning log. Default: ``None``.
        saved_path (str): The directory path for saving obfuscated ckpt files. Default: ``'./'``.

    Raises:
        TypeError: If `network` is not nn.Cell.
        TypeError: If `ckpt_files` is not string or `saved_path` is not string.
        TypeError: If `target_modules` is not list.
        TypeError: If target_modules's elements are not string.
        ValueError: If `ckpt_files` is not exist or `saved_path` is not exist.
        ValueError: If the number of elements of `target_modules` is less than ``2``.
        ValueError: If the first string of `target_modules` contains characters other than uppercase and lowercase
            letters, numbers, ``'_'`` and ``'/'``.
        ValueError: If the second string of `target_modules` is empty or contains characters other than uppercase and
            lowercase letters, numbers, ``'_'`` and ``'|'``.
        ValueError: If the third string of `target_modules` is not in the format of 'obfuscate_layers:all' or
            'obfuscate_layers:int'.

    Returns:
        list[float], obf_ratios, which is the necessary data that needs to be load when running obfuscated network.

    Examples:
        >>> from mindspore import obfuscate_ckpt, save_checkpoint
        >>> # Refer to https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>> save_checkpoint(net, './test_net.ckpt')
        >>> target_modules = ['', 'fc1|fc2']
        >>> obfuscate_ckpt(net, target_modules, './', './')
    """
    if not isinstance(network, nn.Cell):
        raise TypeError("network must be nn.Cell, but got {}.".format(type(network)))
    _check_dir_path('ckpt_files', ckpt_files)
    _check_dir_path('saved_path', saved_path)
    # Try to find default target modules
    if target_modules is None:
        to_split_modules = _get_default_target_modules(ckpt_files)
    else:
        if len(target_modules) >= 1 and target_modules[0] == '/':
            target_modules[0] = ''
        to_split_modules = target_modules
    if not _check_valid_target(network, to_split_modules):
        raise ValueError("The obfuscate module path {} is not exist, please check the input 'target_modules'."
                         .format(to_split_modules))
    # generate and save obf_ratios to saved_path
    path_list = to_split_modules[0].split('/')
    target_list = to_split_modules[1].split('|')
    global OBF_RATIOS_LENGTH
    number_of_ratios = OBF_RATIOS_LENGTH * OBF_RATIOS_WIDTH
    if number_of_ratios > MAX_OBF_RATIOS_NUM:
        OBF_RATIOS_LENGTH = MAX_OBF_RATIOS_NUM // OBF_RATIOS_WIDTH
        number_of_ratios = OBF_RATIOS_LENGTH * OBF_RATIOS_WIDTH
    obf_ratios = []
    secrets_generator = secrets.SystemRandom()
    for _ in range(number_of_ratios):
        secure_float = secrets_generator.uniform(0.01, 100)
        obf_ratios.append(secure_float)
    # start obfuscate ckpt
    ckpt_dir_files = os.listdir(ckpt_files)
    for ckpt_name in ckpt_dir_files:
        if Path(ckpt_files + ckpt_name).is_dir():
            sub_path = os.path.abspath(ckpt_files) + '/' + ckpt_name
            sub_ckpt_file_list = os.listdir(sub_path)
            new_saved_path = os.path.abspath(saved_path) + '/' + ckpt_name
            if not os.path.exists(new_saved_path):
                try:
                    os.mkdir(new_saved_path, mode=0o700)
                except FileExistsError:
                    pass
            for sub_ckpt_name in sub_ckpt_file_list:
                if not sub_ckpt_name.endswith('.ckpt'):
                    continue
                _obfuscate_single_ckpt(os.path.abspath(sub_path) + '/' + sub_ckpt_name, obf_ratios, path_list,
                                       target_list, new_saved_path)
        else:
            if not ckpt_name.endswith('.ckpt'):
                continue
            _obfuscate_single_ckpt(os.path.abspath(ckpt_files) + '/' + ckpt_name, obf_ratios, path_list,
                                   target_list, saved_path)
    return obf_ratios


def _obfuscate_single_ckpt(ckpt_name, obf_ratios, path_list, target_list, saved_path):
    """Obfuscate single ckpt file"""
    module_has_been_obfuscated = set()
    try:
        ckpt_param = load_checkpoint(ckpt_name)
    except (ValueError, TypeError, OSError):
        logger.error("Load checkpoint failed for file {}.".format(ckpt_name))
        return None
    obf_ratios_index = -1
    for item in ckpt_param:
        module = _get_valid_module(item, path_list, target_list)
        if module:
            layer_index = _judge_layer_index(item)
            if layer_index >= OBF_RATIOS_LENGTH:
                break
            if module not in module_has_been_obfuscated:
                module_has_been_obfuscated.add(module)
                obf_ratios_index += 1
            ratio_total_index = layer_index * OBF_RATIOS_WIDTH + obf_ratios_index % OBF_RATIOS_WIDTH
            ckpt_param[item].set_data(ckpt_param[item].value() / obf_ratios[ratio_total_index])
    # save the obfuscated model to saved_path
    obf_param_list = []
    for item in ckpt_param:
        obf_param_list.append({'name': item, 'data': ckpt_param[item]})
    ckpt_file_name = ckpt_name.split('/')[-1]
    obf_ckpt_file_name = ckpt_file_name.split('.')[0] + '_obf' + '.ckpt'
    save_checkpoint(obf_param_list, os.path.abspath(saved_path) + '/' + obf_ckpt_file_name)
    return None


def load_obf_params_into_net(network, target_modules, obf_ratios, **kwargs):
    """
    load obfuscate ratios into obfuscated network. Usually used in conjunction with :func:`mindspore.obfuscate_ckpt`
    interface.

    Args:
        network (nn.Cell): The original network that need to be obfuscated.
        target_modules (list[str]): The target module of network that need to be obfuscated. The first string
            represents the network path of target module in original network, which should be in form of ``'A/B/C'``.
            The second string represents the obfuscation target module, which should be in form of ``'D|E|F'``. For
            example, thr target_modules of GPT2 can be ``['backbone/blocks/attention', 'dense1|dense2|dense3']``.
            If target_modules has the third value, it should be in the format of 'obfuscate_layers:all' or
            'obfuscate_layers:int', which represents the number of layers need to be obfuscated of duplicate layers
            (such as transformer layers or resnet blocks).
        obf_ratios (Tensor): The obf ratios generated when execute :func:`mindspore.obfuscate_ckpt`.
        kwargs (dict): Configuration options dictionary.

            - ignored_func_decorators (list[str]): The name list of function decorators in network's python code.
            - ignored_class_decorators (list[str]): The name list of class decorators in network's python code.

    Raises:
        TypeError: If `network` is not nn.Cell.
        TypeError: If `obf_ratios` is not Tensor.
        TypeError: If `target_modules` is not list.
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
        >>> from mindspore import obfuscate_ckpt, save_checkpoint, load_checkpoint, Tensor
        >>> import mindspore.common.dtype as mstype
        >>> import numpy as np
        >>> # Refer to https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>> save_checkpoint(net, './test_net.ckpt')
        >>> target_modules = ['', 'fc1|fc2']
        >>> # obfuscate ckpt files
        >>> obfuscate_ckpt(net, target_modules, './', './')
        >>> # load obf ckpt into network
        >>> new_net = LeNet5()
        >>> load_checkpoint('./test_net_obf.ckpt', new_net)
        >>> obf_ratios = Tensor(np.load('./obf_ratios.npy'), mstype.float16)
        >>> obf_net = load_obf_params_into_net(new_net, target_modules, obf_ratios)
    """
    if not isinstance(network, nn.Cell):
        raise TypeError("network must be nn.Cell, but got {}.".format(type(network)))
    if not isinstance(obf_ratios, Tensor):
        raise TypeError("obf_ratios must be MindSpore Tensor, but got {}.".format(type(obf_ratios)))
    if obf_ratios.size == 0:
        raise ValueError("obf_ratios can not be empty.")
    if not _check_valid_target(network, target_modules):
        raise ValueError("{} is not exist, please check the input 'target_modules'.".format(target_modules))
    if len(target_modules) >= 1 and target_modules[0] == '/':
        target_modules[0] = ''
    path_list = target_modules[0].split('/')
    path_len = len(path_list)
    target_list = []
    for _ in range(path_len):
        target_list.append([])
    target_list.append(target_modules[1].split('|'))
    global MAX_OBF_RATIOS_NUM, OBF_RATIOS_LENGTH
    number_of_ratios = OBF_RATIOS_LENGTH * OBF_RATIOS_WIDTH
    if number_of_ratios > MAX_OBF_RATIOS_NUM:
        OBF_RATIOS_LENGTH = MAX_OBF_RATIOS_NUM // OBF_RATIOS_WIDTH
        number_of_ratios = OBF_RATIOS_LENGTH * OBF_RATIOS_WIDTH
    MAX_OBF_RATIOS_NUM = number_of_ratios
    rewrite_network = _obfuscate_network(network, path_list, target_list, **kwargs)
    setattr(rewrite_network, 'obf_ratios', obf_ratios)
    return rewrite_network


def _check_dir_path(name, dir_path):
    """check directory path"""
    if not isinstance(dir_path, str):
        raise TypeError("{} must be string, but got {}.".format(name, type(dir_path)))
    if not os.path.exists(dir_path):
        raise ValueError("{} is not exist, please check the input {}.".format(dir_path, name))
    if not Path(dir_path).is_dir():
        raise TypeError("{} must be a directory path, but got {}.".format(name, dir_path))


def _judge_layer_index(layer_name):
    """Judge the layer index of target layers"""
    split_name = layer_name.split('.')
    for split_str in split_name[:]:
        if split_str.isdigit():
            return int(split_str)
    return 0


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
            logger.warning("{} does not exist in the path {}".format(target, target_modules[0]))
        else:
            OBF_RATIOS_WIDTH += 1
    if OBF_RATIOS_WIDTH == 0:
        raise ValueError("all targets {} do not exist in the path {}.".format(target_list, target_modules[0]))
    _update_max_obf_ratios_num(target_modules)
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


def _get_default_target_modules(ckpt_files):
    """Get the default or suggested target modules, if the target modules is None."""

    def _split_to_path_and_target(module, target):
        # split module into path list and target list
        target_index = module.index(target)
        path = module[:target_index - 1]
        target = module[target_index:].split('/')[0]
        return path, target

    def _find_default_obfuscate_modules(net_path):
        # find modules including the default paths
        default_module = {'attention'}
        for module in default_module:
            if module in net_path and module not in candidate_modules:
                candidate_modules.append(net_path)
        # find the default targets in the default module
        default_target = {'dense', 'query', 'key', 'value'}
        for target in default_target:
            for candidate in candidate_modules:
                if target in candidate:
                    path, target = _split_to_path_and_target(candidate, target)
                    if path not in paths:
                        paths.append(path)
                    if target not in targets:
                        targets.append(target)

    def _find_suggested_obfuscate_modules(net_path):
        default_target = {'dense', 'query', 'key', 'value'}
        for target in default_target:
            # find the suggest modules
            if target in net_path:
                path, target = _split_to_path_and_target(net_path, target)
                if [path, target] not in suggest_modules:
                    suggest_modules.append([path, target])

    # store the potential candidate_modules
    candidate_modules = []
    suggest_modules = []
    paths = []
    targets = []
    ckpt_dir_files = os.listdir(ckpt_files)
    for ckpt_name in ckpt_dir_files:
        if not ckpt_name.endswith('.ckpt'):
            continue
        try:
            ckpt_param = load_checkpoint(os.path.abspath(ckpt_files) + '/' + ckpt_name)
        except (ValueError, TypeError, OSError):
            logger.error("Load checkpoint failed for file {}.".format(os.path.abspath(ckpt_files) + '/' + ckpt_name))
            return None
        for item in ckpt_param:
            param_path = _remove_digit(item)
            param_path = '/'.join(param_path)
            # find candidate modules including the default paths and append candidate_modules
            _find_default_obfuscate_modules(param_path)
            # give the suggested modules and find the default targets in the default module
            _find_suggested_obfuscate_modules(param_path)
    if paths and targets:
        target_modules = [paths[0], '|'.join(targets)]
        logger.warning("The default obfuscate modules is obtained:{}".format(target_modules))
        return target_modules
    # logging the suggested target module
    logger.warning("The default obfuscate modules can not be obtained. The suggested possible paths are given below: {}"
                   .format(suggest_modules))
    raise ValueError("Can not get the default path, please specify the path in the form of ['A/B/C', 'D1|D2']")


def _get_valid_module(item, path_list, target_list):
    """get the valid module"""
    number_path = len(path_list)
    net_path = _remove_digit(item)
    net_path = '/'.join(net_path[:number_path])
    tar_path = '/'.join(path_list)
    # update the weights with obf_ratios in target module
    if net_path == tar_path:
        for target in target_list:
            if target in item.split('.'):
                target_index = item.split('.').index(target)
                module = ''.join(item.split('.')[:target_index + 1])
                return module
    return None


def _remove_digit(item):
    """remove digit in the parameter path"""
    param_path = item.split('.')
    for tmp_str in param_path[:]:
        if tmp_str.isdigit():
            param_path.remove(tmp_str)
    return param_path


def _obfuscate_network(model, path_list, target_list, **kwargs):
    """obfuscate original network, including add mul operation and add inputs for passing obf_ratio."""

    def _insert_input(stree: SymbolTree, arg_name: str = 'y_obf'):
        """add inputs for passing obf_ratio"""
        last_input = None
        for node in stree.nodes():
            if node.get_node_type() == NodeType.Input:
                last_input = node
        position = stree.after(last_input)
        # the insert input node name would be 'input_y_obf'
        new_input_node = last_input.create_input(arg_name)
        stree.insert(position, new_input_node)
        return new_input_node

    def _insert_mul(stree: SymbolTree, node: Node, index: int):
        """add mul operation for original network"""
        arg_list = node.get_targets().copy()
        input_y_node = stree.get_node("input_y_obf")
        v: str = input_y_node.get_targets()[0].value
        sv: ScopedValue = ScopedValue.create_naming_value(v + f'[{index}]')
        arg_list.append(sv)
        target_list = node.get_targets().copy()
        new_mul_node = node.create_call_cell(cell=ops.Mul(), targets=target_list, args=arg_list, name='mul')
        position = stree.after(node)
        stree.insert(position, new_mul_node)

    def _insert_mul_by_name(stree: SymbolTree, after_name_list: list):
        """add mul operation after the target nodes according the name of them"""
        if not after_name_list:
            return
        for node in stree.nodes():
            for after_name in after_name_list:
                if node.get_name() == after_name:
                    global OBF_RATIOS_INSERT_INDEX
                    if OBF_RATIOS_INSERT_INDEX < MAX_OBF_RATIOS_NUM:
                        _insert_mul(stree, node, OBF_RATIOS_INSERT_INDEX)
                        OBF_RATIOS_INSERT_INDEX += 1

    def _update_subnet(stree: SymbolTree, substree: SymbolTree, subnode: Node):
        """update the network once the subnet is obfuscated"""
        new_net = substree.get_network()
        input_y_node = substree.get_node("input_y_obf")
        if input_y_node is None:
            return
        arg_list = subnode.get_args().copy()
        kwargs_list = list(subnode.get_kwargs().values())
        arg_list.extend(kwargs_list)
        v: str = input_y_node.get_targets()[0].value
        arg_obf: ScopedValue = ScopedValue.create_naming_value("y_obf=" + v)
        arg_list.append(arg_obf)
        target_list = subnode.get_targets().copy()
        name = subnode.get_name()
        new_node = subnode.create_call_cell(cell=new_net, targets=target_list, args=arg_list, name=name)
        stree.replace(subnode, [new_node])

    def _traverse(stree, i=0):
        """traverse and obfuscate the original network"""
        if len(path_list) == i:
            return
        for node in stree.nodes():
            node_name = node.get_name()
            if node.get_node_type() == NodeType.Tree and node_name.startswith(path_list[i]):
                sub_stree = TreeNodeHelper.get_sub_tree(node)
                _traverse(sub_stree, i + 1)
                _insert_input(sub_stree, arg_name='y_obf')
                _insert_mul_by_name(sub_stree, after_name_list=target_list[i + 1])
                _update_subnet(stree, sub_stree, node)

    def _register_denied_func_decorators(fn):
        """set the function decorators which should be denied for parse"""
        name = "denied_function_decorator_list"
        setattr(ClassDefParser, name, fn)

    def _register_denied_class_decorators(fn):
        """set the class decorators which should be denied for parse"""
        name = "denied_class_decorator_list"
        setattr(ModuleParser, name, fn)

    if 'ignored_func_decorators' in kwargs.keys():
        kw_func_dec = kwargs["ignored_func_decorators"]
        if not isinstance(kw_func_dec, list):
            raise TypeError('{} should be list, but got {}'.format(kw_func_dec, type(kw_func_dec)))
        if kw_func_dec and not isinstance(kw_func_dec[0], str):
            raise TypeError('elements of {} should be str, but got {}'.format(kw_func_dec, type(kw_func_dec[0])))
        _register_denied_func_decorators(kw_func_dec)
    else:
        _register_denied_func_decorators(["_args_type_validator_check", "_LogActionOnce", "cell_attr_register"])
    if 'ignored_class_decorators' in kwargs.keys():
        kw_class_dec = kwargs["ignored_class_decorators"]
        _register_denied_class_decorators(kw_class_dec)
        if not isinstance(kw_class_dec, list):
            raise TypeError('{} should be list[str] type, but got {}'.format(kw_class_dec, type(kw_class_dec)))
        if kw_class_dec and not isinstance(kw_class_dec[0], str):
            raise TypeError('elements of {} should be str, but got {}'.format(kw_class_dec, type(kw_class_dec[0])))

    main_stree = SymbolTree.create(model)
    _traverse(main_stree, 0)
    _insert_input(main_stree, arg_name='y_obf')
    _insert_mul_by_name(main_stree, after_name_list=target_list[0])
    new_net = main_stree.get_network()
    return new_net
