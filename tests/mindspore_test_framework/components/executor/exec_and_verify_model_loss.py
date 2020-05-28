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

"""Component that verify if the model can converge to expected loss."""

from ...components.icomponent import IExectorComponent
from ...utils import keyword
from ...utils.model_util import Model


class LossVerifierEC(IExectorComponent):
    """
    Verify if the model can converge to expected loss.

    Examples:
        'block': {
            'model': Linreg(2),
            'loss': SquaredLoss(),
            'opt': SGD(0.001, 20),
            'num_epochs': 1000,
            'loss_upper_bound': 0.03,
        }
    """

    def __call__(self):
        model = self.function[keyword.block][keyword.model]
        loss = self.function[keyword.block][keyword.loss]
        opt = self.function[keyword.block][keyword.opt]
        num_epochs = self.function[keyword.block][keyword.num_epochs]
        loss_upper_bound = self.function[keyword.block][keyword.loss_upper_bound]
        train_dataset = self.inputs[keyword.desc_inputs]
        model = Model(model, loss, opt)
        loss = model.train(num_epochs, train_dataset)
        assert loss.asnumpy().mean() <= loss_upper_bound
