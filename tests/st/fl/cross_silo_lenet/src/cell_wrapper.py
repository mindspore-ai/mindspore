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

from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.nn import TrainOneStepCell

class TrainOneStepCellForFLWorker(TrainOneStepCell):
    """
    Wraps the network with federated learning operators in worker.
    """
    def __init__(self, network, optimizer, sens=1.0, batch_size=32):
        super(TrainOneStepCellForFLWorker, self).__init__(network, optimizer, sens)
        self.batch_size = batch_size
        self.start_fl_job = P.StartFLJob(batch_size)
        self.update_model = P.UpdateModel()
        self.get_model = P.GetModel()
        self.depend = P.Depend()

    def construct(self, *inputs):
        start_fl_job = self.start_fl_job()
        inputs = self.depend(inputs, start_fl_job)

        loss = self.network(*inputs)
        sens = F.fill(loss.dtype, loss.shape, self.sens)
        grads = self.grad(self.network, self.weights)(*inputs, sens)
        grads = self.grad_reducer(grads)
        loss = self.depend(loss, self.optimizer(grads))

        self.update_model(self.weights)
        get_model = self.get_model(self.weights)

        loss = self.depend(loss, get_model)
        return loss
