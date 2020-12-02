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
"""Conditional Variational auto-encoder (CVAE)."""
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore._checkparam import Validator
from ....cell import Cell
from ....layer.basic import Dense, OneHot


class ConditionalVAE(Cell):
    r"""
    Conditional Variational Auto-Encoder (CVAE).

    The difference with VAE is that CVAE uses labels information.
    For more details, refer to `Learning Structured Output Representation using Deep Conditional Generative Models
    <http://papers.nips.cc/paper/5775-learning-structured-output-representation-using-deep-conditional-
    generative-models>`_.

    Note:
        When encoder and decoder ard defined, the shape of the encoder's output tensor and decoder's input tensor
        must be :math:`(N, hidden\_size)`.
        The latent_size must be less than or equal to the hidden_size.

    Args:
        encoder(Cell): The Deep Neural Network (DNN) model defined as encoder.
        decoder(Cell): The DNN model defined as decoder.
        hidden_size(int): The size of encoder's output tensor.
        latent_size(int): The size of the latent space.
        num_classes(int): The number of classes.

    Inputs:
        - **input_x** (Tensor) - The shape of input tensor is :math:`(N, C, H, W)`, which is the same as the input of
          encoder.

        - **input_y** (Tensor) - The tensor of the target data, the shape is :math:`(N,)`.

    Outputs:
        - **output** (tuple) - (recon_x(Tensor), x(Tensor), mu(Tensor), std(Tensor)).

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self, encoder, decoder, hidden_size, latent_size, num_classes):
        super(ConditionalVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        if (not isinstance(encoder, Cell)) or (not isinstance(decoder, Cell)):
            raise TypeError('The encoder and decoder should be Cell type.')
        self.hidden_size = Validator.check_positive_int(hidden_size)
        self.latent_size = Validator.check_positive_int(latent_size)
        if hidden_size < latent_size:
            raise ValueError('The latent_size should be less than or equal to the hidden_size.')
        self.num_classes = Validator.check_positive_int(num_classes)
        self.normal = C.normal
        self.exp = P.Exp()
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.concat = P.Concat(axis=1)
        self.to_tensor = P.ScalarToArray()
        self.one_hot = OneHot(depth=num_classes)
        self.dense1 = Dense(self.hidden_size, self.latent_size)
        self.dense2 = Dense(self.hidden_size, self.latent_size)
        self.dense3 = Dense(self.latent_size + self.num_classes, self.hidden_size)

    def _encode(self, x, y):
        en_x = self.encoder(x, y)
        mu = self.dense1(en_x)
        log_var = self.dense2(en_x)
        return mu, log_var

    def _decode(self, z):
        z = self.dense3(z)
        recon_x = self.decoder(z)
        return recon_x

    def construct(self, x, y):
        """
        The input are x and y, so the WithLossCell method needs to be rewritten when using cvae interface.
        """
        mu, log_var = self._encode(x, y)
        std = self.exp(0.5 * log_var)
        z = self.normal(self.shape(mu), mu, std, seed=0)
        y = self.one_hot(y)
        z_c = self.concat((z, y))
        recon_x = self._decode(z_c)
        return recon_x, x, mu, std

    def generate_sample(self, sample_y, generate_nums, shape):
        """
        Randomly sample from the latent space to generate samples.

        Args:
            sample_y (Tensor): Define the label of samples. Tensor of shape (generate_nums, ) and type mindspore.int32.
            generate_nums (int): The number of samples to generate.
            shape(tuple): The shape of sample, which must be the format of (generate_nums, C, H, W) or (-1, C, H, W).

        Returns:
            Tensor, the generated samples.
        """
        generate_nums = Validator.check_positive_int(generate_nums)
        if not isinstance(shape, tuple) or len(shape) != 4 or (shape[0] != -1 and shape[0] != generate_nums):
            raise ValueError('The shape should be (generate_nums, C, H, W) or (-1, C, H, W).')
        sample_z = self.normal((generate_nums, self.latent_size), self.to_tensor(0.0), self.to_tensor(1.0), seed=0)
        sample_y = self.one_hot(sample_y)
        sample_c = self.concat((sample_z, sample_y))
        sample = self._decode(sample_c)
        sample = self.reshape(sample, shape)
        return sample

    def reconstruct_sample(self, x, y):
        """
        Reconstruct samples from original data.

        Args:
            x (Tensor): The input tensor to be reconstructed, the shape is (N, C, H, W).
            y (Tensor): The label of the input tensor, the shape is (N,).

        Returns:
            Tensor, the reconstructed sample.
        """
        mu, log_var = self._encode(x, y)
        std = self.exp(0.5 * log_var)
        z = self.normal(mu.shape, mu, std, seed=0)
        y = self.one_hot(y)
        z_c = self.concat((z, y))
        recon_x = self._decode(z_c)
        return recon_x
