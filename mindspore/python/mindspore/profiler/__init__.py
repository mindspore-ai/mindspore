# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
This module provides Python APIs to enable the profiling of MindSpore neural networks.
Users can import the mindspore.profiler.Profiler, initialize the Profiler object to start profiling,
and use Profiler.analyse() to stop profiling and analyse the results.
Users can visualize the results using the MindInsight tool.
Now, Profiler supports AICORE operator, AICPU operator, HostCPU operator, memory,
correspondence, cluster, etc data analysis.
"""
from mindspore.profiler.profiling import Profiler
from mindspore.profiler.envprofiling import EnvProfiler

__all__ = ["Profiler", "EnvProfiler"]
