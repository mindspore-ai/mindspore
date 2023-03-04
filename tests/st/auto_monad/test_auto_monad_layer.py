# Copyright 2021-2023 Huawei Technologies Co., Ltd
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
import pytest
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore.dataset import NumpySlicesDataset
from mindspore import context, Tensor

context.set_context(mode=context.GRAPH_MODE)

class AutoEncoderTrainNetwork(nn.Cell):
    def __init__(self):
        super(AutoEncoderTrainNetwork, self).__init__()
        self.loss_fun = nn.MSELoss()
        self.net = nn.CellList([nn.Dense(2, 32), nn.Dense(32, 2)])
        self.relu = nn.ReLU()

    def reconstruct_sample(self, x: Tensor):
        for _, layer in enumerate(self.net):
            x = layer(x)
            x = self.relu(x)
        return x

    def construct(self, x: Tensor):
        recon_x = self.reconstruct_sample(x)
        return self.loss_fun(recon_x, x)

    def sample_2d_data(self, n_normals=2000, n_outliers=400):
        z = np.random.randn(n_normals, 2)
        outliers = np.random.uniform(low=-6, high=6, size=(n_outliers, 2))
        centers = np.array([(2., 0), (-2., 0)])
        sigma = 0.3
        normal_points = sigma * z + centers[np.random.randint(len(centers), size=(n_normals,))]
        return np.vstack((normal_points, outliers))

    def create_synthetic_dataset(self):
        transformed_dataset = self.sample_2d_data()
        for dim in range(transformed_dataset.shape[1]):
            min_val = transformed_dataset[:, dim].min()
            max_val = transformed_dataset[:, dim].max()
            if min_val != max_val:
                transformed_dataset[:, dim] = (transformed_dataset[:, dim] - min_val) / (max_val - min_val)
            elif min_val != 1:
                transformed_dataset[:, dim] = transformed_dataset[:, dim] / min_val
        transformed_dataset = transformed_dataset.astype(np.float32)
        return transformed_dataset


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_auto_monad_layer():
    """
    Feature: Auto monad feature.
    Description: Verify auto monad feature.
    Expectation: No exception.
    """
    ae_with_loss = AutoEncoderTrainNetwork()
    transformed_dataset = ae_with_loss.create_synthetic_dataset()
    dataloader = NumpySlicesDataset(data=(transformed_dataset,), shuffle=True)
    dataloader = dataloader.batch(batch_size=16)
    optim = nn.RMSProp(params=ae_with_loss.trainable_params(), learning_rate=0.002,)
    train_net = nn.TrainOneStepCell(ae_with_loss, optim)
    train_net.set_train()
    gen_samples = dict()
    num_epoch = 21
    for epoch in range(num_epoch):
        loss = []
        for _, (batch,) in enumerate(dataloader):
            batch = Tensor(batch, dtype=ms.float32)
            loss_ = train_net(batch)
            loss.append(loss_.asnumpy())
        avg_loss = np.array(loss).mean()
        if epoch % 10 == 0:
            gen_samples[epoch] = ae_with_loss.reconstruct_sample(Tensor(transformed_dataset)).asnumpy()
        print(f"epoch: {epoch}/{num_epoch}, avg loss: {avg_loss}")
