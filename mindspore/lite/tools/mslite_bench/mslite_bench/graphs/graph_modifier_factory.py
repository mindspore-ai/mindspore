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
"""graph modifier factory"""
from mslite_bench.utils.infer_log import InferLogger
from mslite_bench.common.task_common_func import CommonFunc


_logger = InferLogger().logger


def create_graph_modifier(model_path):
    """create graph modifier"""
    if model_path.endswith('onnx'):
        try:
            infer_module = CommonFunc.import_module('mslite_bench.graphs.onnx_graph_modifier')
        except ImportError as e:
            _logger.info('import tf session failed: %s', e)
            raise
        return infer_module.OnnxModifier(model_path)
    if model_path.endswith('xx'):
        try:
            infer_module = CommonFunc.import_module('mslite_bench.graphs.tf_graph_modifier')
        except ImportError as e:
            _logger.info('import tf session failed: %s', e)
            raise
        return infer_module.TFModifier(model_path)
    raise NotImplementedError(f'model type of {model_path} is not supported ')
