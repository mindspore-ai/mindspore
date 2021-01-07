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
# """test_auc"""

import math
import numpy as np
from mindspore import Tensor
from mindspore.nn.metrics import ROC, auc


def test_auc():
    """test_auc"""
    x = Tensor(np.array([[3, 0, 1], [1, 3, 0], [1, 0, 2]]))
    y = Tensor(np.array([[0, 2, 1], [1, 2, 1], [0, 0, 1]]))
    metric = ROC(pos_label=1)
    metric.clear()
    metric.update(x, y)
    fpr, tpr, thre = metric.eval()
    output = auc(fpr, tpr)

    assert math.isclose(output, 0.45, abs_tol=0.001)
    assert np.equal(thre, np.array([4, 3, 2, 1, 0])).all()
