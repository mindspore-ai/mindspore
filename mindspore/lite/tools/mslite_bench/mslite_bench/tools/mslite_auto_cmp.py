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
""" mindspore lite auto accuracy compare"""
import copy
import os


from mslite_bench.tools.converter import MsliteConverter
from mslite_bench.utils import InferLogger
from mslite_bench.graphs.graph_modifier_factory import create_graph_modifier


_logger = InferLogger().logger


class MsliteAutoCMP:
    """Auto compare between third party model and mslite model"""
    @classmethod
    def acc_infos_in_specific_node(cls,
                                   args,
                                   logger=None):
        """accuracy infos in specific node"""
        if args.input_tensor_shapes is None:
            logger.error('Shall input input_tensor_shapes for accuracy compare')
            raise ValueError('input_tensor_shapes is None')
        if args.input_tensor_dtypes is None:
            logger.error('Shall input input_tensor_dtypes for accuracy compare')
            raise ValueError('input_tensor_dtypes is None')
        if not args.model_file.endswith('onnx'):
            logger.error('Only onnx model accuracy compare is supported')
            raise ValueError('input model file shall be .onnx')
        graph_modifier = create_graph_modifier(args.model_file)
        sub_model_name = f'sub_{args.peak_node_names.replace("/", "_")}_{os.path.basename(args.model_file)}'
        sub_model_path = os.path.join(os.path.dirname(args.model_file), sub_model_name)
        if logger is None:
            logger = _logger

        peak_node_names = cls.parse_peak_node_names(args.peak_node_names)

        logger.info('Collect all node outputs to compare')
        graph_out_names = graph_modifier.extract_model(sub_model_path,
                                                       output_names=peak_node_names)
        logger.debug('Extract sub model successfully')

        args.converter_output_file = os.path.join(os.path.dirname(args.model_file),
                                                  f'sub_{args.peak_node_names.replace("/", "_")}')

        args_copy = copy.deepcopy(args)
        args_copy.model_file = sub_model_path
        args_copy.converter_is_analysis = True
        args_copy.output_tensor_names = graph_out_names
        args_copy.converter_input_shape = args.input_tensor_shapes

        logger.info('Start to convert model')
        MsliteConverter.convert(args_copy, logger, is_delete_ms_model=True)
        os.remove(sub_model_path)

    @staticmethod
    def parse_peak_node_names(peak_node_str):
        """parse peak node names"""
        return [item.strip() for item in peak_node_str.split(',')]
