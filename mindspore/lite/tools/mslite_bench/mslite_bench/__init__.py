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
mslite bench classes and functions
"""

from mslite_bench.common.config import (
    MsliteConfig, PaddleConfig, OnnxConfig, TFConfig
)
from mslite_bench.infer_base.infer_session_factory import InferSessionFactory
from mslite_bench.tools.cross_framework_accuracy import CrossFrameworkAccSummary

acc_info_between_features = CrossFrameworkAccSummary.acc_infos_between_features

__all__ = [
    'InferSessionFactory', 'MsliteConfig', 'PaddleConfig', 'OnnxConfig', 'TFConfig',
    'acc_info_between_features'
]
