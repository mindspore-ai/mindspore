# Copyright 2021 Huawei Technologies Co., Ltd
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
Fairmot for track
"""
import collections
import numpy as np


class TrackState:
    """TrackState"""
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3

    def __init__(self):
        pass


class BaseTrack:
    """
    Fairmot for BaseTrack
    """
    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = collections.OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    def __init__(self):
        pass

    @property
    def end_frame(self):
        """end frame"""
        return self.frame_id

    @staticmethod
    def next_id():
        """next id"""
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        """activate"""
        raise NotImplementedError

    def predict(self):
        """predict"""
        raise NotImplementedError

    def update(self, *args, **kwargs):
        """update"""
        raise NotImplementedError

    def mark_lost(self):
        """mark lost"""
        self.state = TrackState.Lost

    def mark_removed(self):
        """mark removed"""
        self.state = TrackState.Removed
