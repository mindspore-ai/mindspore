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
"""Toolbox for anomaly detection by using VAE."""
import numpy as np

from mindspore._checkparam import Validator
from ..dpn import VAE
from ..infer import ELBO, SVI
from ...optim import Adam
from ...wrap.cell_wrapper import WithLossCell


class VAEAnomalyDetection:
    r"""
    Toolbox for anomaly detection by using VAE.

    Variational Auto-Encoder(VAE) can be used for Unsupervised Anomaly Detection. The anomaly score is the error
    between the X and the reconstruction of X. If the score is high, the X is mostly outlier.

    Args:
        encoder(Cell): The Deep Neural Network (DNN) model defined as encoder.
        decoder(Cell): The DNN model defined as decoder.
        hidden_size(int): The size of encoder's output tensor.
        latent_size(int): The size of the latent space.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """

    def __init__(self, encoder, decoder, hidden_size=400, latent_size=20):
        self.vae = VAE(encoder, decoder, hidden_size, latent_size)

    def train(self, train_dataset, epochs=5):
        """
        Train the VAE model.

        Args:
            train_dataset (Dataset): A dataset iterator to train model.
            epochs (int): Total number of iterations on the data. Default: 5.

        Returns:
            Cell, the trained model.
        """
        net_loss = ELBO()
        optimizer = Adam(params=self.vae.trainable_params(), learning_rate=0.001)
        net_with_loss = WithLossCell(self.vae, net_loss)
        vi = SVI(net_with_loss=net_with_loss, optimizer=optimizer)
        self.vae = vi.run(train_dataset, epochs)
        return self.vae

    def predict_outlier_score(self, sample_x):
        """
        Predict the outlier score.

        Args:
            sample_x (Tensor): The sample to be predicted, the shape is (N, C, H, W).

        Returns:
            numpy.dtype, the predicted outlier score of the sample.
        """
        reconstructed_sample = self.vae.reconstruct_sample(sample_x)
        return self._calculate_euclidean_distance(sample_x.asnumpy(), reconstructed_sample.asnumpy())

    def predict_outlier(self, sample_x, threshold=100.0):
        """
        Predict whether the sample is an outlier.

        Args:
            sample_x (Tensor): The sample to be predicted, the shape is (N, C, H, W).
            threshold (float): the threshold of the outlier. Default: 100.0.

        Returns:
            Bool, whether the sample is an outlier.
        """
        threshold = Validator.check_positive_float(threshold)
        score = self.predict_outlier_score(sample_x)
        return score >= threshold

    def _calculate_euclidean_distance(self, sample_x, reconstructed_sample):
        """
        Calculate the euclidean distance of the sample_x and reconstructed_sample.
        """
        return np.sqrt(np.sum(np.square(sample_x - reconstructed_sample)))
