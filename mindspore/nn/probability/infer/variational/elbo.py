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
"""The Evidence Lower Bound (ELBO)."""
from mindspore.ops import operations as P
from ...distribution.normal import Normal
from ....cell import Cell
from ....loss.loss import MSELoss


class ELBO(Cell):
    r"""
    The Evidence Lower Bound (ELBO).

    Variational inference minimizes the Kullback-Leibler (KL) divergence from the variational distribution to
    the posterior distribution. It maximizes the ELBO, a lower bound on the logarithm of
    the marginal probability of the observations log p(x). The ELBO is equal to the negative KL divergence up to
    an additive constant.
    For more details, refer to `Variational Inference: A Review for Statisticians <https://arxiv.org/abs/1601.00670>`_.

    Args:
        latent_prior(str): The prior distribution of latent space. Default: Normal.
            - Normal: The prior distribution of latent space is Normal.
        output_prior(str): The distribution of output data. Default: Normal.
            - Normal: If the distribution of output data is Normal, the reconstruct loss is MSELoss.

    Inputs:
        - **input_data** (Tuple) - (recon_x(Tensor), x(Tensor), mu(Tensor), std(Tensor)).
        - **target_data** (Tensor) - the target tensor of shape :math:`(N,)`.

    Outputs:
        Tensor, loss float tensor.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self, latent_prior='Normal', output_prior='Normal'):
        super(ELBO, self).__init__()
        self.sum = P.ReduceSum()
        self.zeros = P.ZerosLike()
        if latent_prior == 'Normal':
            self.posterior = Normal()
        else:
            raise ValueError('The values of latent_prior now only support Normal')
        if output_prior == 'Normal':
            self.recon_loss = MSELoss(reduction='sum')
        else:
            raise ValueError('The values of output_dis now only support Normal')

    def construct(self, data, label):
        recon_x, x, mu, std = data
        reconstruct_loss = self.recon_loss(x, recon_x)
        kl_loss = self.posterior('kl_loss', 'Normal', self.zeros(mu), self.zeros(mu)+1, mu, std)
        elbo = reconstruct_loss + self.sum(kl_loss)
        return elbo
