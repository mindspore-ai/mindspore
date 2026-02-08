# Copyright 2025 Huawei Technologies Co., Ltd
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
"""mindir export"""
import numpy as np
import os
import sys
from mindspore import context
from mindspore import log as logger
from mindspore import nn
from mindspore.train.serialization import export
from mindspore.train import Model
from mindspore.train.serialization import load

sys.path.append('../')

enc_path = os.path.split(os.path.abspath(__file__))[0] + "/data/mindir_enc/"
dec_path = os.path.split(os.path.abspath(__file__))[0] + "/data/net/"
if not os.path.exists(enc_path):
    os.makedirs(enc_path)
if not os.path.exists(dec_path):
    os.makedirs(dec_path)


def ms_function_export_mindir(func, input_data, name, enc_key=None, enc_mode='AES-GCM',
                              enc_flag=False):
    if enc_flag:
        export(func, *input_data, file_name=enc_path + name, file_format="MINDIR",
               enc_key=enc_key, enc_mode=enc_mode)
    else:
        export(func, *input_data, file_name=dec_path + name, file_format="MINDIR")


def ms_function_load_mindir(input_data, name, dec_key=None, dec_mode='AES-GCM', enc_flag=False):
    if enc_flag:
        data_path = enc_path
        graph = load(file_name=data_path + name + ".mindir", dec_key=dec_key, dec_mode=dec_mode)
    else:
        data_path = dec_path
        graph = load(file_name=data_path + name + ".mindir")
    graph_cell = nn.GraphCell(graph)
    model = Model(network=graph_cell)
    if None in input_data:
        output_me = model.predict(input_data[1])
    else:
        output_me = model.predict(*input_data)
    return output_me


def ms_function_save_inputs_outputs(inputs, outputs, name, enc_key=None, enc_mode="AES-GCM",
                                    enc_flag=False):
    if enc_flag:
        data_path = enc_path
    else:
        data_path = dec_path
    x = 1
    for i in inputs:
        if i is not None:
            i.asnumpy().tofile(data_path + "{}_input_x{}.bin".format(name, x))
            np.save(data_path + "{}_input_x{}.npy".format(name, x), i.asnumpy())
            x += 1
        else:
            logger.info("input data is None, can not save as npy file")
    if isinstance(outputs, tuple):
        o = 1
        for out in outputs:
            out.asnumpy().tofile(data_path + "{}_output_x{}.bin".format(name, o))
            np.save(data_path + "{}_output_x{}.npy".format(name, o), out.asnumpy())
            o += 1
    else:
        outputs.asnumpy().tofile(data_path + "{}_output_x1.bin".format(name))
        np.save(data_path + "{}_output_x1.npy".format(name), outputs.asnumpy())
    if enc_flag:
        string = "enc_key :" + enc_key + "\n" + "enc_mode :" + enc_mode
        with open("{}{}_enc.txt".format(data_path, name), "w", encoding='utf-8') as f:
            f.writelines(string)
        assert os.path.exists(os.path.join(data_path, "{}_enc.txt".format(name)))
    else:
        pass


def assert_file_exists(name, input_num, out_num, enc_flag=False):
    if enc_flag:
        data_path = enc_path
    else:
        data_path = dec_path
    assert os.path.exists(os.path.join(data_path, "{}.mindir".format(name)))
    if input_num >= 1:
        for i in range(input_num):
            assert os.path.exists(os.path.join(data_path, "{}_input_x{}.bin".format(name, i + 1)))
    else:
        pass
    if out_num >= 1:
        for i in range(out_num):
            assert os.path.exists(os.path.join(data_path, "{}_output_x{}.bin".format(name, i + 1)))
    else:
        pass


def excute_export_mindir(func, input_data, name, input_num, out_num, enc_key=None,
                         enc_mode="AES-GCM", dec_key=None, dec_mode='AES-GCM', enc_flag=False):
    mode = os.environ['CONTEXT_MODE']
    if enc_flag:
        ms_function_export_mindir(func, input_data, name, enc_key=enc_key,
                                  enc_mode=enc_mode, enc_flag=enc_flag)
        context.set_context(mode=context.GRAPH_MODE)
        output_me = ms_function_load_mindir(input_data, name, dec_key=dec_key,
                                            dec_mode=dec_mode, enc_flag=enc_flag)
        ms_function_save_inputs_outputs(input_data, output_me, name)
        assert_file_exists(name=name, input_num=input_num, out_num=out_num)
    else:
        ms_function_export_mindir(func, input_data, name)
        context.set_context(mode=context.GRAPH_MODE)
        output_me = ms_function_load_mindir(input_data, name)
        ms_function_save_inputs_outputs(input_data, output_me, name)
        assert_file_exists(name=name, input_num=input_num, out_num=out_num)
    if mode in ('GRAPH', 'GRAPH_MODE', 'CONTEXT.GRAPH_MODE'):
        context.set_context(mode=0)
    else:
        context.set_context(mode=1)
    return output_me
