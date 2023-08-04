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

"""split tool for prediction."""
import os
import sys
import stat
import json
import socket
import importlib
from typing import Dict, Any

__all__ = [
    'split_network',
    'split_ir'
]


def split_network(network, checkpoint_filenames, train_strategy_filename, strict_load,
                  *inputs, file_name, **kwargs):
    """
    Reshade for prediction, split by checkpoint and network, manually strategy

    Args:
        network (Cell): Network for distributed predication.
        checkpoint_filenames (list[str]): The name of Checkpoint files in order of rank id.
        train_strategy_filename (str): The filename of training strategy protocol buffer file.
                                       When train_strategy_filename is None, the training strategy file will be
                                       obtained from context.get_auto_parallel_context("strategy_ckpt_load_file").
                                       Therefore, the training strategy file needs to be specified
                                       in at least one of them. Default: None.
        strict_load (bool): Whether to strict load the parameter into net. If False, it will load parameter
                            into net when parameter name's suffix in checkpoint file is the same as the
                            parameter in the network. When the types are inconsistent perform type conversion
                            on the parameters of the same type, such as float32 to float16. Default: False.
        inputs (Union[Tensor, Dataset, List, Tuple, Number, Bool]): It represents the inputs
             of the `net`, if the network has multiple inputs, set them together. While its type is Dataset,
             it represents the preprocess behavior of the `net`, data preprocess operations will be serialized.
             In second situation, you should adjust batch size of dataset script manually which will impact on
             the batch size of 'net' input. Only supports parse "image" column from dataset currently.
        file_name (str): File name of the model to be exported.

            - AIR: Ascend Intermediate Representation. An intermediate representation format of Ascend model.
            - ONNX: Open Neural Network eXchange. An open format built to represent machine learning models.
            - MINDIR: MindSpore Native Intermediate Representation for Anf. An intermediate representation format
              for MindSpore models.

        kwargs (dict): Configuration options dictionary.

            - enc_key (byte): Byte-type key used for encryption. The valid length is 16, 24, or 32.
            - enc_mode (Union[str, function]): Specifies the encryption mode, to take effect when enc_key is set.

              - For 'AIR' and 'ONNX' models, only customized encryption is supported.
              - For 'MINDIR', all options are supported. Option: 'AES-GCM', 'AES-CBC', 'SM4-CBC'
                or Customized encryption.
                Default: 'AES-GCM'.
              - For details of using the customized encryption, please check the `tutorial
                <https://mindspore.cn/mindarmour/docs/en/master/model_encrypt_protection.html>`_.

            - dataset (Dataset): Specifies the preprocessing method of the dataset, which is used to import the
              preprocessing of the dataset into MindIR.

            - obf_config (dict): obfuscation config.

              - type (str): The type of obfuscation, only 'dynamic' is supported until now.
              - obf_ratio (float, str): The ratio of nodes in original model that would be obfuscated. `obf_ratio`
                should be in range of (0, 1] or in ["small", "medium", "large"].
              - customized_func (function): A python function used for customized function mode, which used for control
                the switch branch of obfuscation structure. The outputs of customized_func should be boolean. This
                function needs to ensure that its result is constant for any input. Users can refer to opaque
                predicates. If customized_func is set, then it should be passed to `load()` interface when loading
                obfuscated model.
              - obf_random_seed (int): The random seed used for determine the distribution of confusion branches and the
                weight confusion coefficient, which should be in (0, 9223372036854775807]. If `obf_random_seed` is set,
                then it should be passed to :class:`nn.GraphCell()` interface when loading obfuscated model. It should
                be noted that at least one of `customized_func` or `obf_random_seed` should be set, and the latter mode
                would be applied if both of them are set.


    Raises:
        TypeError: The type of inputs do not match the requirements.
        ValueError: Failed to load checkpoint into net.
    """
    _mindspore = None
    try:
        _mindspore = importlib.import_module('mindspore')
    except (ImportError, BaseException):
        raise ImportError("For 'LiteSplit', import mindspore fail.")

    model_predict = _mindspore.train.model.Model(network)

    predict_strategy = model_predict.infer_predict_layout(*inputs)

    _mindspore.load_distributed_checkpoint(network=network,
                                           checkpoint_filenames=checkpoint_filenames,
                                           predict_strategy=predict_strategy,
                                           train_strategy_filename=train_strategy_filename,
                                           strict_load=strict_load)

    _mindspore.export(model_predict.predict_network, *inputs, file_name=file_name, file_format="MINDIR", *kwargs)
    print("Export finished and now exit.")


def split_ir(file_name, device_num):
    """
    Auto Split MindIR.
    Reshade for prediction, split by mindir, automatically strategy

    The returned object can be executed by a `GraphCell`, see class :class:`mindspore.nn.GraphCell` for more details.

    Args:
        file_name (str): MindIR file name.
        device_num(str): The mindir will be automatically divided by device nums.

    Raises:
        ValueError: MindIR file does not exist or `file_name` is not a string.
        RuntimeError: Failed to split MindIR file.

    Examples:
        >>> split_ir("net.mindir", "[0,2)")


    """
    _mindspore = None
    try:
        _mindspore = importlib.import_module('mindspore')
    except (ImportError, BaseException):
        raise ImportError("For 'LiteSplit', import mindspore fail.")

    os.environ["RANK_TABLE_FILE"] = generate_rank_table(device_num)
    rank_size = int(device_num[3]) - int(device_num[1])
    assert rank_size in [2, 4, 8]
    os.environ["RANK_SIZE"] = str(rank_size)

    _mindspore.set_context(mode=_mindspore.GRAPH_MODE, device_target='Ascend', save_graphs=True)
    _mindspore.communication.init(backend_name="hccl")
    _mindspore.export_split_mindir(file_name)


def get_host_ip():
    """
    get host ip
    """
    ip = None
    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
    except EOFError:
        pass
    return ip


def generate_rank_table(device_num):
    """
    generate rank table
    """
    visible_devices = "0,1,2,3,4,5,6,7".split(',')
    print('visible_devices:{}'.format(visible_devices))

    server_id = get_host_ip()
    print('server_id:{}'.format(server_id))

    first_num = int(device_num[1])
    last_num = int(device_num[3])
    if first_num < 0 or last_num > 8:
        raise ValueError("device num {} must be in range [0,8] !".format(device_num))
    if first_num > last_num:
        raise ValueError("First num {} of device num {} must less than last num {} !".format(first_num, device_num,
                                                                                             last_num))
    if first_num < 4 < last_num:
        if first_num == 0 and last_num == 8:
            pass
        else:
            raise ValueError("device num {} must be in the same group of [0,4] or [4,8] !".format(device_num))

    device_num_list = list(range(first_num, last_num))
    print("device_num_list:", device_num_list)

    device_ips: Dict[Any, Any] = {}
    try:
        for device_id in device_num_list:
            ret = os.popen("hccn_tool -i %d -ip -g" % device_id).readlines()
            device_ips[str(device_id)] = ret[0].split(":")[1].replace('\n', '')
    except IndexError:
        print("Failed to call hccn_tool, try to read /etc/hccn.conf instead")
        try:
            with open('/etc/hccn.conf', 'r') as fin:
                for hccn_item in fin.readlines():
                    if hccn_item.strip().startswith('address_'):
                        device_id, device_ip = hccn_item.split('=')
                        device_id = device_id.split('_')[1]
                        device_ips[device_id] = device_ip.strip()
        except OSError:
            print("Failed to read /etc/hccn.conf")
            raise SystemError("Failed to find information for hccl")

    hccn_table = {'version': '1.0',
                  'server_count': '1',
                  'server_list': []}
    device_list = []
    rank_id = 0
    for instance_id in device_num_list:
        try:
            device_id = visible_devices[instance_id]
            device_ip = device_ips[device_id]
            device = {'device_id': device_id,
                      'device_ip': device_ip,
                      'rank_id': str(rank_id)}
            print('rank_id:{}, device_id:{}, device_ip:{}'.format(rank_id, device_id, device_ip))
        except KeyError:
            print("Failed to get device ids")
            raise KeyError("Failed to get device ids, key not in disk")

        rank_id += 1
        device_list.append(device)

    try:
        hccn_table['server_list'].append({
            'server_id': server_id,
            'device': device_list,
            'host_nic_ip': 'reserve'
        })
        hccn_table['status'] = 'completed'
    except KeyError:
        raise KeyError("Failed to create hccn_table, key not in disk")

    table_path = os.getcwd()
    table_fn = os.path.join(table_path,
                            'hccl_{}p_{}_{}.json'.format(len(device_num_list), "".join(map(str, device_num_list)),
                                                         server_id))
    flag = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    mode = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(table_fn, flag, mode), 'w') as table_fp:
        json.dump(hccn_table, table_fp, indent=4)
    sys.stdout.flush()
    print("Completed: hccl file was save in :", table_fn)
    return table_fn
