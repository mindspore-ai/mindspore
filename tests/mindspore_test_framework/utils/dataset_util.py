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

"""Dataset utils."""

import random

import numpy as np

from mindspore import Tensor


def generate_dataset_for_linear_regression(true_w, true_b, num_samples, batch_size):
    features = np.random.normal(scale=1, size=(num_samples, len(true_w)))
    labels = np.matmul(features, np.reshape(np.array(true_w), (-1, 1))) + true_b
    labels += np.random.normal(scale=0.01, size=labels.shape)
    indices = list(range(num_samples))
    random.shuffle(indices)

    for i in range(0, num_samples, batch_size):
        j = np.array(indices[i: min(i + batch_size, num_samples)])
        yield Tensor(features.take(j, 0).astype(np.float32)), Tensor(labels.take(j, 0).astype(np.float32))
