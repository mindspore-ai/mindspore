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
"""
input argument parser functions
"""
import argparse


class ArgParser:
    """
    input argument parser functions
    """
    @classmethod
    def parse_arguments(cls):
        """parse input arguments for mslite bench"""
        parser = argparse.ArgumentParser(description='Easy Infer for model benchmark')
        cls.base_arg_parse(parser)
        cls.model_arg_parse(parser)
        cls.task_arg_parse(parser)
        cls.converter_arg_parse(parser)
        cls.auto_cmp_arg_parse(parser)
        args = parser.parse_args()
        return args

    @classmethod
    def task_arg_parse(cls, parser):
        """parse task related arguments"""
        # for task related
        parser.add_argument('--task_type',
                            type=str,
                            choices=["infer", "convert", "framework_cmp", "auto_cmp"],
                            default='auto_cmp',
                            help='benchmark task type:'
                                 'infer for framework accuracy compare,'
                                 'framework_cmp for multiple frameworks accuracy compare'
                                 'convert for mslite model converter,'
                                 'auto_cmp for mslite auto compare with third party framework,'
                                 'both single op and all ops accuracy compare are supported')
        # for inference related infos
        parser.add_argument('--model_file',
                            type=str,
                            default=None,
                            help='path to model file')
        parser.add_argument('--params_file',
                            type=str,
                            default="",
                            help='path to params file')
        parser.add_argument('--cmp_model_file',
                            type=str,
                            default=None,
                            help="the model path for model to be compared")
        parser.add_argument('--test_data',
                            type=str,
                            default=None,
                            help='path to data to do inference')
        parser.add_argument('--test_label',
                            type=str,
                            default=None,
                            help='path to test labels to calculate accuracy of model infer')

        # for benchmark and random accuracy test
        parser.add_argument('--input_tensor_shapes',
                            type=str,
                            default=None,
                            help="input tensor infos contain input tensor name and tensor shape"
                                 "format 'tensor_name: tensor_shape;")
        parser.add_argument('--input_data_file',
                            type=str,
                            default=None,
                            help="path to files contain input data, with key is input tensor name"
                                 "value is input numpy data")

        parser.add_argument('--save_file_type',
                            type=str,
                            default='not_save',
                            choices=['npy', 'bin', 'not_save'],
                            help="file type to save input output tensor info, "
                                 "default not save")

        parser.add_argument('--input_tensor_dtypes',
                            type=str,
                            default=None,
                            help="tensor dtype for each model input tensor, "
                                 "choices=[INT8, INT32, INT64"
                                 "FLOAT16, FLOAT, FLOAT64, UINT8]")

        parser.add_argument('--random_input_flag',
                            type=bool,
                            default=False,
                            help="flag indicate whether using random input to do inference")

        parser.add_argument('--loop_infer_times',
                            type=int,
                            default=1,
                            help="infer times for loop infer")

        parser.add_argument('--warmup_times',
                            type=int,
                            default=0,
                            help="warm times for model infer")

    @classmethod
    def model_arg_parse(cls, parser):
        """parse model and framework related arguments"""
        # for mslite config
        parser.add_argument('--thread_affinity_mode',
                            type=int,
                            default=2,
                            help='thread affinity number for mslite inference')

        parser.add_argument('--thread_num',
                            type=int,
                            default=1,
                            help='thread number for mslite inference')

        parser.add_argument('--mslite_model_type',
                            type=int,
                            default=0,
                            choices=[0, 4],
                            help='input model type for mslite inference, '
                                 '0 for MINDIR, 4 for MINDIR_LITE')

        parser.add_argument('--ascend_provider',
                            type=str,
                            default='',
                            choices=['', 'ge'],
                            help="Ascend infer method: '' for acl, 'ge' for GE")

        # for tensorrt infer
        parser.add_argument('--tensorrt_optim_input_shape',
                            type=str,
                            default=None,
                            help='optim input shape for tensorrt'
                                 'with key tensor name (str) '
                                 'and value shape info(List[int])')

        parser.add_argument('--tensorrt_min_input_shape',
                            type=str,
                            default=None,
                            help='optim input shape for tensorrt'
                                 'with key tensor name (str) '
                                 'and value shape info(List[int])')

        parser.add_argument('--tensorrt_max_input_shape',
                            type=str,
                            default=None,
                            help='optim input shape for tensorrt'
                                 'with key tensor name (str) '
                                 'and value shape info(List[int])')

        parser.add_argument('--gpu_memory_size',
                            type=int,
                            default=100,
                            help='gpu init memory size(M)')

        parser.add_argument('--is_enable_tensorrt',
                            type=bool,
                            default=False,
                            help="flag indicate whether use tensorrt engine")

        parser.add_argument('--is_fp16',
                            type=bool,
                            default=False,
                            help="flag indicate whether apply fp16 infer")

        parser.add_argument('--is_int8',
                            type=bool,
                            default=False,
                            help="flag indicate whether apply int8 infer")

    @staticmethod
    def converter_arg_parse(parser):
        """parse converter related arguments"""
        parser.add_argument('--converter_decrypt_key',
                            type=str,
                            default="",
                            help='decrypt key for mindir, '
                                 'only take effect when converter_save_type is MINDIR')

        parser.add_argument('--converter_decrypt_mode',
                            type=str,
                            choices=['AES-GCM', 'AES-CBC'],
                            default='AES-GCM',
                            help='decrypt model for mindir, '
                                 'only take effect when converter_decrypt_key is set')

        parser.add_argument('--converter_enable_encryption',
                            type=bool,
                            default=False,
                            help='whether enable encryption')

        parser.add_argument('--converter_input_shape',
                            type=str,
                            default=None,
                            help="input tensor shape contain input tensor name and tensor shape"
                                 "format 'tensor_name: tensor_shape;")

        parser.add_argument('--converter_weight_fp16',
                            type=bool,
                            default=False,
                            help='whether save model as fp16 model')

        parser.add_argument('--converter_encrypt_key',
                            type=str,
                            default="",
                            help='encrypt key for model encryption,'
                                 'only take effect when converter decrypt mode '
                                 'is AES-GCM')

        parser.add_argument('--converter_optimize',
                            type=str,
                            choices=['none', 'general', 'ascend_oriented', 'gpu_oriented'],
                            default='general',
                            help='optimize mode for converter')

        parser.add_argument('--converter_save_type',
                            type=str,
                            choices=['MINDIR', 'MINDIR_LITE'],
                            default='MINDIR',
                            help='model save type for mindspore lite')

        parser.add_argument('--converter_output_file',
                            type=str,
                            default="",
                            help='output path to save mindspore model')

        parser.add_argument('--converter_config_file',
                            type=str,
                            default="",
                            help='config file for converter')

        parser.add_argument('--quant_output_data_type',
                            type=str,
                            choices=['float32', 'int8', 'unknown', 'uint8'],
                            default='unknown',
                            help='data type for quant model, if unknown is set, data type is '
                                 'the same with model input')

        parser.add_argument('--quant_input_data_type',
                            type=str,
                            choices=['float32', 'int8', 'unknown', 'uint8'],
                            default='unknown',
                            help='data type for quant model, if unknown is set, data type is '
                                 'the same with model output')

        parser.add_argument('--converter_is_analysis',
                            type=bool,
                            default=False,
                            help='whether analysis converter status after model convert'
                                 'if True, will give summary about converter accuracy between'
                                 'mslite model and third party model')

        parser.add_argument('--converter_input_format',
                            type=str,
                            choices=['NCHW', 'NHWC'],
                            default='NCHW',
                            help='whether analysis converter status after model convert'
                                 'if True, will give summary about converter accuracy between'
                                 'mslite model and third party model')

    @staticmethod
    def auto_cmp_arg_parse(parser):
        """parse auto cmp related arguments"""
        parser.add_argument('--peak_node_names',
                            type=str,
                            default='all',
                            help='network node name to compare accuracy between'
                                 'third party framework with mslite framework,'
                                 'if all is set, every node in network would run accuracy compare')

        parser.add_argument('--cmp_result_file',
                            type=str,
                            default='',
                            help='path to save accuracy info, default in csv format')

    @staticmethod
    def base_arg_parse(parser):
        """parse base related arguments"""
        parser.add_argument('--device',
                            type=str,
                            default='ascend',
                            choices=['cpu', 'gpu', 'ascend'],
                            help='device type for model inference')
        parser.add_argument('--cmp_device',
                            type=str,
                            default='cpu',
                            choices=['cpu', 'gpu', 'ascend'],
                            help='device type for cmp model inference')
        parser.add_argument('--device_id',
                            type=int,
                            default=0,
                            help='device index for model inference')
        parser.add_argument('--log_level',
                            type=int,
                            choices=[0, 1, 2, 3],
                            default=1,
                            help='logging info for mslite bench'
                                 '0 for debug,'
                                 '1 for info,'
                                 '2 for warning'
                                 '3 for error')
        parser.add_argument('--frameworkType',
                            type=str,
                            default='MSLITE',
                            choices=['MSLITE', 'PB', 'ONNX', 'PADDLE'],
                            help='device type for model inference')
        parser.add_argument('--log_path',
                            type=str,
                            default=None,
                            help='path to save model inference log')
        parser.add_argument('--batch_size',
                            type=int,
                            default=1,
                            help='model inference batch size')
        parser.add_argument('--input_tensor_names',
                            nargs='+',
                            default=None,
                            help='model input tensor name list')
        parser.add_argument('--output_tensor_names',
                            nargs='+',
                            default=None,
                            help='model output tensor name list')
