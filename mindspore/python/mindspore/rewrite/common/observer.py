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
from .event import Event


class Observer(abc.ABC):
    """Abstract class, observer of observer design pattern."""

    def __init__(self):
        self._event_filter = set()
        self._event_filter.add(Event.CodeChangeEvent)

    def add_event(self, event: Event):
        """
        Add event filter. An observer only responses to event in its event_filter list.
        """
        self._event_filter.add(event)

    def remove_event(self, event: Event):
        """
        Remove event filter. An observer only responses to event in its event_filter list.
        """
        self._event_filter.remove(event)

    def on_change(self, event: Event):
        """
        Called back when any changes occurred in linked `Observable`.
        """

        if event in self._event_filter:
            self._on_change(event)

    @abc.abstractmethod
    def _on_change(self, event: Event):
        """
        Abstract method for defining how to response when any changes occurred in linked `Observable`.
        """

        raise NotImplementedError
