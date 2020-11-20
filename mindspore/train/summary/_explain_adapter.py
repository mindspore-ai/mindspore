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
"""Generate the explain event which conform to proto format."""
import time

from ..summary_pb2 import Event, Explain


def check_explain_proto(explain):
    """
    Package the explain event.

    Args:
        explain (Explain): The object of summary_pb2.Explain.
    """
    if not isinstance(explain, Explain):
        raise TypeError(f'Plugin explainer expects a {Explain.__name__} value.')

    if not explain.image_path and not explain.inference and not explain.metadata.label and not explain.benchmark:
        raise ValueError('One of metadata, image path, inference or benchmark has to be filled in.')


def package_explain_event(explain_str):
    """
    Package the explain event.

    Args:
        explain_str (string): The serialize string of summary_pb2.Explain.

    Returns:
        Event, event object.
    """
    event = Event()
    event.wall_time = time.time()
    event.explain.ParseFromString(explain_str)
    return event.SerializeToString()
