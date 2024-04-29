# Copyright 2024 Huawei Technologies Co., Ltd
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
"""utils for text generation."""

from threading import Thread
import numpy as np


def log_softmax(x, axis=None):
    """numpy implemented log softmax function.
    refers to https://github.com/scipy/scipy/blob/v1.11.1/scipy/special/_logsumexp.py"""
    x_max = np.amax(x, axis=axis, keepdims=True)

    if x_max.ndim > 0:
        x_max[~np.isfinite(x_max)] = 0
    elif not np.isfinite(x_max):
        x_max = 0

    tmp = x - x_max
    exp_tmp = np.exp(tmp)

    # suppress warnings about log of zero
    with np.errstate(divide="ignore"):
        s = np.sum(exp_tmp, axis=axis, keepdims=True)
        out = np.log(s)

    out = tmp - out
    return out


def softmax(x, axis=None):
    """numpy implemented softmax function.
    refers to https://github.com/scipy/scipy/blob/v1.11.1/scipy/special/_logsumexp.py"""
    x_max = np.amax(x, axis=axis, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)


def softmax_single(i, res, x):
    res[i] = softmax(x)


def softmax_with_threads(x, is_finished=None):
    """calculate softmax with threads"""
    res = np.ones_like(x)
    all_threads = []
    for i in range(0, res.shape[0]):
        if is_finished and is_finished[i]:
            continue
        thread = Thread(target=softmax_single,
                        args=(i, res, x[i]))
        all_threads.append(thread)
        thread.start()
    for thread in all_threads:
        thread.join()
    return res


def topk(x, top_k, axis=-1, largest=True, sort=True):
    """numpy implemented topk sample."""
    # safety check
    if x.shape[axis] < top_k:
        top_k = x.shape[axis] - 1
    if largest:
        topk_index = np.argpartition(-x, top_k, axis=axis)
    else:
        topk_index = np.argpartition(x, top_k, axis=axis)
    topk_index = np.take(topk_index, np.arange(top_k), axis=axis)
    topk_data = np.take_along_axis(x, topk_index, axis=axis)
    if sort:
        sort_index = (
            np.argsort(-topk_data, axis=axis)
            if largest
            else np.argsort(topk_data, axis=axis)
        )
        topk_data = np.take_along_axis(topk_data, sort_index, axis=axis)
        topk_index = np.take_along_axis(topk_index, sort_index, axis=axis)
    return topk_data, topk_index
