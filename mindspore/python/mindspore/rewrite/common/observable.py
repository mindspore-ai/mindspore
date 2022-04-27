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
"""Base class, observable of observer design pattern."""

from .observer import Observer
from .event import Event


class Observable:
    """Abstract class, observable of observer design pattern."""

    def __init__(self):
        self._observers: [Observer] = []

    def changed(self, event: Event):
        """
        Called when current observable is changed.
        `Observable` declares a change and all registered observers observe a change and do something for this change.
        """

        for observer in self._observers:
            observer.on_change(event)

    def reg_observer(self, observer: Observer):
        """
        Register an `observer` so that it can observe changes of current observable.

        Args:
             observer (Observer): An `Observer` to be registered into current observable.
        """

        self._observers.append(observer)
