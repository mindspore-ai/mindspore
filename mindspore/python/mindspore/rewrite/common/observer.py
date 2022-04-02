# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Abstract class, observer of observer design pattern."""

import abc


class Observer(abc.ABC):
    """Abstract class, observer of observer design pattern."""

    def __init__(self):
        self._observing = False

    def start_observe(self):
        """
        Start observing so that current `Observer` can do response when any change occurred in `Observable`.
        """

        self._observing = True

    def stop_observe(self):
        """
        Stop observing so that current `Observer` will do nothing even when changes occurred in linked `Observable`.
        """

        self._observing = False

    def on_change(self):
        """
        Called back when any changes occurred in linked `Observable`.
        """

        if self._observing:
            self._on_change()

    @abc.abstractmethod
    def _on_change(self):
        """
        Abstract method for defining how to response when any changes occurred in linked `Observable`.
        """

        raise NotImplementedError
