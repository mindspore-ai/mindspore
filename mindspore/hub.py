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
# ============================================================================
"""
hub for loading models:
Users can load pre-trained models using mindspore.hub.load() API.
"""
import os
import re
import shutil
import tarfile
import hashlib
from urllib.request import urlretrieve
import requests
from bs4 import BeautifulSoup

import mindspore
import mindspore.nn as nn
from mindspore import log as logger
from mindspore.train.serialization import load_checkpoint, load_param_into_net

DOWNLOAD_BASIC_URL = "http://download.mindspore.cn/model_zoo"
OFFICIAL_NAME = "official"
DEFAULT_CACHE_DIR = '.cache'
MODEL_TARGET_CV = ['alexnet', 'fasterrcnn', 'googlenet', 'lenet', 'resnet', 'resnet50', 'ssd', 'vgg', 'yolo']
MODEL_TARGET_NLP = ['bert', 'mass', 'transformer']


def _packing_targz(output_filename, savepath=DEFAULT_CACHE_DIR):
    """
    Packing the input filename to filename.tar.gz in source dir.
    """
    try:
        with tarfile.open(output_filename, "w:gz") as tar:
            tar.add(savepath, arcname=os.path.basename(savepath))
    except Exception as e:
        raise OSError("Cannot tar file {} for - {}".format(output_filename, e))


def _unpacking_targz(input_filename, savepath=DEFAULT_CACHE_DIR):
    """
    Unpacking the input filename to dirs.
    """
    try:
        t = tarfile.open(input_filename)
        t.extractall(path=savepath)
    except Exception as e:
        raise OSError("Cannot untar file {} for - {}".format(input_filename, e))


def _remove_path_if_exists(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)


def _create_path_if_not_exists(path):
    if not os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            os.mkdir(path)


def _get_weights_file(url, hash_md5=None, savepath=DEFAULT_CACHE_DIR):
    """
    get checkpoint weight from giving url.

    Args:
       url(string): checkpoint tar.gz url path.
       hash_md5(string): checkpoint file md5.
       savepath(string): checkpoint download save path.

    Returns:
       string.
    """

    def reporthook(a, b, c):
        percent = a * b * 100.0 / c
        show_str = ('[%%-%ds]' % 70) % (int(percent * 80) * '#')
        print("\rDownloading:", show_str, " %5.1f%%" % (percent), end="")

    def md5sum(file_name, hash_md5):
        fp = open(file_name, 'rb')
        content = fp.read()
        fp.close()
        m = hashlib.md5()
        m.update(content.encode('utf-8'))
        download_md5 = m.hexdigest()
        return download_md5 == hash_md5

    _remove_path_if_exists(os.path.realpath(savepath))
    _create_path_if_not_exists(os.path.realpath(savepath))
    ckpt_name = os.path.basename(url.split("/")[-1])
    # identify file exist or not
    file_path = os.path.join(savepath, ckpt_name)
    if os.path.isfile(file_path):
        if hash_md5 and md5sum(file_path, hash_md5):
            print('File already exists!')
            return file_path

    file_path_ = file_path[:-7] if ".tar.gz" in file_path else file_path
    _remove_path_if_exists(file_path_)

    # download the checkpoint file
    print('Downloading data from url {}'.format(url))
    try:
        urlretrieve(url, file_path, reporthook=reporthook)
    except HTTPError as e:
        raise Exception(e.code, e.msg, url)
    except URLError as e:
        raise Exception(e.errno, e.reason, url)
    print('\nDownload finished!')

    # untar file_path
    _unpacking_targz(file_path, os.path.realpath(savepath))

    filesize = os.path.getsize(file_path)
    # turn the file size to Mb format
    print('File size = %.2f Mb' % (filesize / 1024 / 1024))
    return file_path_


def _get_url_paths(url, ext='.tar.gz'):
    response = requests.get(url)
    if response.ok:
        response_text = response.text
    else:
        return response.raise_for_status()
    soup = BeautifulSoup(response_text, 'html.parser')
    parent = [url + node.get('href') for node in soup.find_all('a')
              if node.get('href').endswith(ext)]
    return parent


def _get_file_from_url(base_url, base_name):
    idx = 0
    urls = _get_url_paths(base_url + "/")
    files = [url.split('/')[-1] for url in urls]
    for i, name in enumerate(files):
        if re.match(base_name + '*', name) is not None:
            idx = i
            break
    return urls[idx]


def load_weights(network, network_name=None, force_reload=True, **kwargs):
    r"""
    Load a model from mindspore, with pretrained weights.

    Args:
        network (Cell): Cell network.
        network_name (string, optional): Cell network name get from network. Default: None.
        force_reload (bool, optional): Whether to force a fresh download unconditionally. Default: False.
        kwargs (dict, optional): The corresponding kwargs for download for model.

            - device_target (str, optional): Runtime device target. Default: 'ascend'.
            - dataset (str, optional): Dataset to train the network. Default: 'cifar10'.
            - version (str, optional): MindSpore version to save the checkpoint. Default: Latest version.

    Example:
        >>> hub.load(network, network_name='lenet',
                     **{'device_target': 'ascend', 'dataset':'mnist', 'version': '0.5.0'})
    """
    if not isinstance(network, nn.Cell):
        logger.error("Failed to combine the net and the parameters.")
        msg = ("Argument net should be a Cell, but got {}.".format(type(network)))
        raise TypeError(msg)

    if network_name is None:
        if hasattr(network, network_name):
            network_name = network.network_name
        else:
            msg = "Should input network name, but got None."
            raise TypeError(msg)

    device_target = kwargs['device_target'] if kwargs['device_target'] else 'ascend'
    dataset = kwargs['dataset'] if kwargs['dataset'] else 'imagenet'
    version = kwargs['version'] if kwargs['version'] else mindspore.version.__version__

    if network_name.split("_")[0] in MODEL_TARGET_CV:
        model_type = "cv"
    elif network_name.split("_")[0] in MODEL_TARGET_NLP:
        model_type = "nlp"
    else:
        raise ValueError("Unsupported network {} download checkpoint.".format(network_name.split("_")[0]))

    download_base_url = "/".join([DOWNLOAD_BASIC_URL,
                                  OFFICIAL_NAME, model_type, network_name])
    download_file_name = "_".join(
        [network_name, device_target, version, dataset, OFFICIAL_NAME])
    download_url = _get_file_from_url(download_base_url, download_file_name)

    if force_reload:
        ckpt_path = _get_weights_file(download_url, None, DEFAULT_CACHE_DIR)
    else:
        raise ValueError("Unsupported not force reload.")

    ckpt_file = os.path.join(ckpt_path, network_name + ".ckpt")
    param_dict = load_checkpoint(ckpt_file)
    load_param_into_net(network, param_dict)
