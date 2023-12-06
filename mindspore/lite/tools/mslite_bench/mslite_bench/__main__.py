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
__main__ for mslite_bench
"""

import os

from mslite_bench.utils import ArgParser, InferLogger
from mslite_bench.common.model_info_enum import TaskType
from mslite_bench.common.task_common_func import CommonFunc


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_WARNINGS'] = '0'
os.environ['GLOG_v'] = '3'

if __name__ == '__main__':
    args = ArgParser.parse_arguments()
    mslite_logger = InferLogger(args.log_path)
    mslite_logger.set_level(CommonFunc.logging_level(args.log_level))
    logger = mslite_logger.logger
    logger.debug('Start model infer now!')

    if args.task_type.lower() == TaskType.FRAMEWORK_CMP.value:
        try:
            from mslite_bench.tools.cross_framework_accuracy import CrossFrameworkAccSummary
        except ImportError as e:
            logger.error('Failed to import CFA: %s', e)
            raise
        logger.debug('Start framework compare task!')
        CrossFrameworkAccSummary.accuracy_compare_func(args, logger)
    elif args.task_type.lower() in set(
            [TaskType.NPU_DYNAMIC_INFER.value, TaskType.MODEL_INFER.value]
    ):
        try:
            from mslite_bench.tools.easy_infer import EasyInfer
        except ImportError as e:
            logger.error('Failed to import easy infer: %s', e)
            raise
        if args.task_type.lower() == TaskType.NPU_DYNAMIC_INFER.value:
            logger.debug('Start mslite model dynamic infer task!')
            EasyInfer.ms_dynamic_input_infer(args, logger)
        else:
            logger.debug('Start model infer: %s', args.model_file)
            EasyInfer.easy_infer(args, logger)
    elif args.task_type.lower() == TaskType.CONVERTER.value:
        try:
            from mslite_bench.tools.converter import MsliteConverter
        except ImportError as e:
            logger.error('Failed to import MsliteConverter class')
            raise
        MsliteConverter.convert(args, logger)
    elif args.task_type.lower() == TaskType.AUTO_CMP.value:
        try:
            from mslite_bench.tools.mslite_auto_cmp import MsliteAutoCMP
        except ImportError as e:
            logger.error('Failed to import MsliteAutoCMP class')
            raise
        if args.input_tensor_shapes is None:
            logger.error('Shall input input_tensor_shapes for accuracy compare')
            raise ValueError('input_tensor_shapes is None')
        if args.input_tensor_dtypes is None:
            logger.error('Shall input input_tensor_dtypes for accuracy compare')
            raise ValueError('input_tensor_dtypes is None')
        MsliteAutoCMP.acc_infos_in_specific_node(args, logger)
    else:
        raise NotImplementedError(f'Task Type {args.task_type} ')
