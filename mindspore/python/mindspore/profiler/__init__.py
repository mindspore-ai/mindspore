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
Profiler Module Introduction.

This module provides Python APIs to enable the profiling of MindSpore neural networks.
Users can import the mindspore.profiler.Profiler, initialize the Profiler object to start profiling,
and use Profiler.analyse() to stop profiling and analyse the results.
To visualize the profiling results, users can open MindSpore Web, find the corresponding 'run'
button/option and click the profile link.
Now, Profiler supports the AICore operator analysis.
"""
from mindspore.profiler.profiling import Profiler
from mindspore.profiler.profiling import ProfileOption

__all__ = ["Profiler", "ProfileOption"]
