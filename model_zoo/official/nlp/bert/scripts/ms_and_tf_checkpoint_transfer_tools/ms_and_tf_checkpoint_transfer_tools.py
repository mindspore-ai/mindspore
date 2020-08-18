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
    mindspore and tensorflow checkpoint transfer tools
"""

import argparse
import tensorflow as tf
from mindspore.common.tensor import Tensor
from mindspore.train.serialization import load_checkpoint, save_checkpoint
from ms2tf_config import param_name_dict as ms2tf_param_dict


def convert_ms_2_tf(tf_ckpt_path, ms_ckpt_path, new_ckpt_path):
    """
    convert ms checkpoint to tf checkpoint
    """
    # load MS checkpoint
    ms_param_dict = load_checkpoint(ms_ckpt_path)
    for name in ms_param_dict.keys():
        if isinstance(ms_param_dict[name].data, Tensor):
            ms_param_dict[name] = ms_param_dict[name].data.asnumpy()

    convert_count = 0
    with tf.Session() as sess:
        # convert ms shape to tf
        print("start convert parameter ...")
        new_var_list = []
        for var_name, shape in tf.contrib.framework.list_variables(tf_ckpt_path):
            if var_name in ms2tf_param_dict:
                ms_name = ms2tf_param_dict[var_name]

                new_tensor = tf.convert_to_tensor(ms_param_dict[ms_name])
                if len(shape) == 2:
                    if tuple(shape) != new_tensor.shape or new_tensor.shape[0] == new_tensor.shape[1]:
                        new_tensor = tf.transpose(new_tensor, (1, 0))
                        if new_tensor.shape != tuple(shape):
                            raise ValueError("shape is not matched after transpose!! {}, {}"
                                             .format(str(new_tensor.shape), str(tuple(shape))))

                if new_tensor.shape != tuple(shape):
                    raise ValueError("shape is not matched after transpose!! {}, {}"
                                     .format(str(new_tensor.shape), str(tuple(shape))))
                var = tf.Variable(new_tensor, name=var_name)
                convert_count = convert_count + 1
            else:
                var = tf.Variable(tf.contrib.framework.load_variable(tf_ckpt_path, var_name), name=var_name)
            new_var_list.append(var)
        print('convert value num: ', convert_count, " of ", len(ms2tf_param_dict))

        # saving tf checkpoint
        print("start saving ...")
        saver = tf.train.Saver(var_list=new_var_list)
        sess.run(tf.global_variables_initializer())
        saver.save(sess, new_ckpt_path)
        print("tf checkpoint was save in :", new_ckpt_path)

    return True


def convert_tf_2_ms(tf_ckpt_path, ms_ckpt_path, new_ckpt_path):
    """
    convert tf checkpoint to ms checkpoint
    """
    tf2ms_param_dict = dict(zip(ms2tf_param_dict.values(), ms2tf_param_dict.keys()))

    # load MS checkpoint
    ms_param_dict = load_checkpoint(ms_ckpt_path)

    new_params_list = []
    session = tf.compat.v1.Session()
    count = 0
    for ms_name in tf2ms_param_dict.keys():
        count += 1
        param_dict = {}

        tf_name = tf2ms_param_dict[ms_name]
        data = tf.train.load_variable(tf_ckpt_path, tf_name)
        ms_shape = ms_param_dict[ms_name].data.shape
        tf_shape = data.shape

        if len(ms_shape) == 2:
            if ms_shape != tf_shape or ms_shape[0] == ms_shape[1]:
                data = tf.transpose(data, (1, 0))
                data = data.eval(session=session)

        param_dict['name'] = ms_name
        param_dict['data'] = Tensor(data)

        new_params_list.append(param_dict)
    print("start saving checkpoint ...")
    save_checkpoint(new_params_list, new_ckpt_path)
    print("ms checkpoint was save in :", new_ckpt_path)

    return True


def main():
    """
    tf checkpoint transfer to ms or ms checkpoint transfer to tf
    """
    parser = argparse.ArgumentParser(description='checkpoint transfer.')
    parser.add_argument("--tf_ckpt_path", type=str, default='./tf-bert/bs64k_32k_ckpt_model.ckpt-28252',
                        help="TensorFlow checkpoint dir, default is: './tf-bert/bs64k_32k_ckpt_model.ckpt-28252'.")
    parser.add_argument("--ms_ckpt_path", type=str, default='./ms-bert/large_en.ckpt',
                        help="MindSpore checkpoint dir, default is: './ms-bert/large_en.ckpt'.")
    parser.add_argument("--new_ckpt_path", type=str, default='./new_ckpt/new_bert_large_en.ckpt',
                        help="New checkpoint dir, default is: './new_ckpt/new_bert_large_en.ckpt'.")
    parser.add_argument("--transfer_option", type=str, default='ms2tf',
                        help="option of transfer ms2tf or tf2ms, default is ms2tf.")

    args_opt = parser.parse_args()

    if args_opt.transfer_option == 'ms2tf':
        print("start ms2tf option ...")
        tf_ckpt_path = args_opt.tf_ckpt_path
        ms_ckpt_path = args_opt.ms_ckpt_path
        new_ckpt_path = args_opt.new_ckpt_path
        convert_ms_2_tf(tf_ckpt_path, ms_ckpt_path, new_ckpt_path)
    elif args_opt.transfer_option == 'tf2ms':
        print("start tf2ms option ...")
        tf_ckpt_path = args_opt.tf_ckpt_path
        ms_ckpt_path = args_opt.ms_ckpt_path
        new_ckpt_path = args_opt.new_ckpt_path
        convert_tf_2_ms(tf_ckpt_path, ms_ckpt_path, new_ckpt_path)
    else:
        print("ERROR: '--transfer_option' please select 0 or 1")


if __name__ == "__main__":
    main()
