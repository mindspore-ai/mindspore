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
# ==============================================================================
"""Validators for TensorOps.
"""
from functools import wraps
from ...transforms.validators import check_uint32


def check_jieba_init(method):
    """Wrapper method to check the parameters of jieba add word."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        hmm_path, mp_path, model = (list(args) + 3 * [None])[:3]

        if "hmm_path" in kwargs:
            hmm_path = kwargs.get("hmm_path")
        if "mp_path" in kwargs:
            mp_path = kwargs.get("mp_path")
        if hmm_path is None:
            raise ValueError(
                "the dict of HMMSegment in cppjieba is not provided")
        kwargs["hmm_path"] = hmm_path
        if mp_path is None:
            raise ValueError(
                "the dict of MPSegment in cppjieba is not provided")
        kwargs["mp_path"] = mp_path
        if model is not None:
            kwargs["model"] = model
        return method(self, **kwargs)
    return new_method


def check_jieba_add_word(method):
    """Wrapper method to check the parameters of jieba add word."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        word, freq = (list(args) + 2 * [None])[:2]

        if "word" in kwargs:
            word = kwargs.get("word")
        if "freq" in kwargs:
            freq = kwargs.get("freq")
        if word is None:
            raise ValueError("word is not provided")
        kwargs["word"] = word
        if freq is not None:
            check_uint32(freq)
            kwargs["freq"] = freq
        return method(self, **kwargs)
    return new_method


def check_jieba_add_dict(method):
    """Wrapper method to check the parameters of add dict"""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        user_dict = (list(args) + [None])[0]
        if "user_dict" in kwargs:
            user_dict = kwargs.get("user_dict")
        if user_dict is None:
            raise ValueError("user_dict is not provided")
        kwargs["user_dict"] = user_dict
        return method(self, **kwargs)
    return new_method
