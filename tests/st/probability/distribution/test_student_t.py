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
"""test cases for StudentT distribution"""
import numpy as np
from scipy import stats
import mindspore.context as context
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.nn.probability.distribution as msd
from mindspore import Tensor
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class LogProb(nn.Cell):
    """
    Test class: log probability of StudentT distribution.
    """

    def __init__(self, df, loc, scale):
        super(LogProb, self).__init__()
        self.n = msd.StudentT(df, loc, scale, dtype=mstype.float32)

    def construct(self, x_):
        return self.n.log_prob(x_)


class LogProb2(nn.Cell):
    """
    Test class: log probability of StudentT distribution.
    """

    def __init__(self):
        super(LogProb2, self).__init__()
        self.n = msd.StudentT(dtype=mstype.float32)

    def construct(self, x_, df, loc, scale):
        return self.n.log_prob(x_, df, loc, scale)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_log_likelihood():
    """
    Feature: StudentT distribution
    Description: test cases for log_prob() of StudentT distribution
    Expectation: the result match to stats
    """
    x = np.array([0.3, 4.0, np.pi, np.e, -2.0], dtype=np.float32)
    df = np.array([0.1, 0.3, 0.5, 0.7, 1.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.5, 0.7, 1.0], dtype=np.float32)
    scale = np.array([1.5, 1.0, 2.0, 3.0, 2.0], dtype=np.float32)

    # stats as benchmark
    expected = stats.t.logpdf(x, df=df, loc=loc, scale=scale).astype(np.float32)

    log_prob = LogProb(df, loc, scale)
    output = log_prob(Tensor(x, dtype=mstype.float32))

    log_prob2 = LogProb2()
    output2 = log_prob2(Tensor(x, dtype=mstype.float32), Tensor(df, dtype=mstype.float32),
                        Tensor(loc, dtype=mstype.float32), Tensor(scale, dtype=mstype.float32))

    tol = 1e-5

    output = output.asnumpy()
    assert (output[np.isinf(output)] == expected[np.isinf(expected)]).all()
    assert (np.abs(output[~np.isinf(output)] - expected[~np.isinf(expected)]) < tol).all()

    output2 = output2.asnumpy()
    assert (output2[np.isinf(output2)] == expected[np.isinf(expected)]).all()
    assert (np.abs(output2[~np.isinf(output2)] - expected[~np.isinf(expected)]) < tol).all()
