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
"""Variational auto-encoder (VAE)"""
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore._checkparam import check_int_positive
from ....cell import Cell
from ....layer.basic import Dense


class VAE(Cell):
    r"""
    Variational Auto-Encoder (VAE).

    The VAE defines a generative model, `Z` is sampled from the prior, then used to reconstruct `X` by a decoder.
    see more details in `Auto-Encoding Variational Bayes <https://arxiv.org/abs/1312.6114>`_.

    Note:
        When define the encoder and decoder, the shape of the encoder's output tensor and decoder's input tensor
        should be :math:`(N, hidden\_size)`.
        The latent_size should be less than or equal to the hidden_size.

    Args:
        encoder(Cell): The DNN model defined as encoder.
        decoder(Cell): The DNN model defined as decoder.
        hidden_size(int): The size of encoder's output tensor.
        latent_size(int): The size of the latent space.

    Inputs:
        - **input** (Tensor) - the same shape as the input of encoder, the shape is :math:`(N, C, H, W)`.

    Outputs:
        - **output** (Tuple) - (recon_x(Tensor), x(Tensor), mu(Tensor), std(Tensor)).
    """

    def __init__(self, encoder, decoder, hidden_size, latent_size):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        if (not isinstance(encoder, Cell)) or (not isinstance(decoder, Cell)):
            raise TypeError('The encoder and decoder should be Cell type.')
        self.hidden_size = check_int_positive(hidden_size)
        self.latent_size = check_int_positive(latent_size)
        if hidden_size < latent_size:
            raise ValueError('The latent_size should be less than or equal to the hidden_size.')
        self.normal = C.normal
        self.exp = P.Exp()
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.to_tensor = P.ScalarToArray()
        self.dense1 = Dense(self.hidden_size, self.latent_size)
        self.dense2 = Dense(self.hidden_size, self.latent_size)
        self.dense3 = Dense(self.latent_size, self.hidden_size)

    def _encode(self, x):
        en_x = self.encoder(x)
        mu = self.dense1(en_x)
        log_var = self.dense2(en_x)
        return mu, log_var

    def _decode(self, z):
        z = self.dense3(z)
        recon_x = self.decoder(z)
        return recon_x

    def construct(self, x):
        mu, log_var = self._encode(x)
        std = self.exp(0.5 * log_var)
        z = self.normal(self.shape(mu), mu, std, seed=0)
        recon_x = self._decode(z)
        return recon_x, x, mu, std

    def generate_sample(self, generate_nums, shape):
        """
        Randomly sample from latent space to generate sample.

        Args:
            generate_nums (int): The number of samples to generate.
            shape(tuple): The shape of sample, it should be (generate_nums, C, H, W) or (-1, C, H, W).

        Returns:
            Tensor, the generated sample.
        """
        generate_nums = check_int_positive(generate_nums)
        if not isinstance(shape, tuple) or len(shape) != 4 or (shape[0] != -1 and shape[0] != generate_nums):
            raise ValueError('The shape should be (generate_nums, C, H, W) or (-1, C, H, W).')
        sample_z = self.normal((generate_nums, self.latent_size), self.to_tensor(0.0), self.to_tensor(1.0), seed=0)
        sample = self._decode(sample_z)
        sample = self.reshape(sample, shape)
        return sample

    def reconstruct_sample(self, x):
        """
        Reconstruct sample from original data.

        Args:
            x (Tensor): The input tensor to be reconstructed, the shape is (N, C, H, W).

        Returns:
            Tensor, the reconstructed sample.
        """
        mu, log_var = self._encode(x)
        std = self.exp(0.5 * log_var)
        z = self.normal(mu.shape, mu, std, seed=0)
        recon_x = self._decode(z)
        return recon_x
