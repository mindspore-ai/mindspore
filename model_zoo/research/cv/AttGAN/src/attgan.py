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
"""AttGAN Network Topology"""

import mindspore.ops.operations as P
from mindspore import nn

from src.block import LinearBlock, Conv2dBlock, ConvTranspose2dBlock

# Image size 128 x 128
MAX_DIM = 64 * 16

class Gen(nn.Cell):
    """Generator"""
    def __init__(self, enc_dim=64, enc_layers=5, enc_norm_fn="batchnorm", enc_acti_fn="lrelu",
                 dec_dim=64, dec_layers=5, dec_norm_fn="batchnorm", dec_acti_fn="relu",
                 n_attrs=13, shortcut_layers=1, inject_layers=1, img_size=128, mode="test"):
        super().__init__()
        self.shortcut_layers = min(shortcut_layers, dec_layers - 1)
        self.inject_layers = min(inject_layers, dec_layers - 1)
        self.f_size = img_size // 2 ** dec_layers  # f_size = 4 for 128x128

        layers = []
        n_in = 3
        for i in range(enc_layers):
            n_out = min(enc_dim * 2 ** i, MAX_DIM)
            layers += [Conv2dBlock(
                n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=enc_norm_fn, acti_fn=enc_acti_fn, mode=mode
            )]
            n_in = n_out
        self.enc_layers = nn.CellList(layers)

        layers = []
        n_in = n_in + n_attrs  # 1024 + 13
        for i in range(dec_layers):
            if i < dec_layers - 1:
                n_out = min(dec_dim * 2 ** (dec_layers - i - 1), MAX_DIM)
                layers += [ConvTranspose2dBlock(
                    n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=dec_norm_fn, acti_fn=dec_acti_fn, mode=mode
                )]
                n_in = n_out
                n_in = n_in + n_in // 2 if self.shortcut_layers > i else n_in
                n_in = n_in + n_attrs if self.inject_layers > i else n_in
            else:
                layers += [ConvTranspose2dBlock(
                    n_in, 3, (4, 4), stride=2, padding=1, norm_fn='none', acti_fn='tanh', mode=mode
                )]
        self.dec_layers = nn.CellList(layers)

        self.view = P.Reshape()
        self.repeat = P.Tile()
        self.cat = P.Concat(1)

    def encoder(self, x):
        """Encoder construct"""
        z = x
        zs = []
        for layer in self.enc_layers:
            z = layer(z)
            zs.append(z)
        return zs

    def decoder(self, zs, a):
        """Decoder construct"""
        a_tile = self.view(a, (a.shape[0], -1, 1, 1))
        multiples = (1, 1, self.f_size, self.f_size)
        a_tile = self.repeat(a_tile, multiples)

        z = self.cat((zs[-1], a_tile))
        i = 0
        for layer in self.dec_layers:
            z = layer(z)
            if self.shortcut_layers > i:
                z = self.cat((z, zs[len(self.dec_layers) - 2 - i]))
            if self.inject_layers > i:
                a_tile = self.view(a, (a.shape[0], -1, 1, 1))
                multiples = (1, 1, self.f_size * 2 ** (i + 1), self.f_size * 2 ** (i + 1))
                a_tile = self.repeat(a_tile, multiples)
                z = self.cat((z, a_tile))
            i = i + 1
        return z

    def construct(self, x, a=None, mode="enc-dec"):
        result = None
        if mode == "enc-dec":
            out = self.encoder(x)
            result = self.decoder(out, a)
        if mode == "enc":
            result = self.encoder(x)
        if mode == "dec":
            result = self.decoder(x, a)
        return result

class Dis(nn.Cell):
    """Discriminator"""
    def __init__(self, dim=64, norm_fn='none', acti_fn='lrelu',
                 fc_dim=1024, fc_norm_fn='none', fc_acti_fn='lrelu', n_layers=5, img_size=128, mode='test'):
        super().__init__()
        self.f_size = img_size // 2 ** n_layers

        layers = []
        n_in = 3
        for i in range(n_layers):
            n_out = min(dim * 2 ** i, MAX_DIM)
            layers += [Conv2dBlock(
                n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=norm_fn, acti_fn=acti_fn, mode=mode
            )]
            n_in = n_out
        self.conv = nn.SequentialCell(layers)
        self.fc_adv = nn.SequentialCell(
            [LinearBlock(1024 * self.f_size * self.f_size, fc_dim, fc_norm_fn, fc_acti_fn, mode),
             LinearBlock(fc_dim, 1, 'none', 'none', mode)])

        self.fc_cls = nn.SequentialCell(
            [LinearBlock(1024 * self.f_size * self.f_size, fc_dim, fc_norm_fn, fc_acti_fn, mode),
             LinearBlock(fc_dim, 13, 'none', 'none', mode)])

    def construct(self, x):
        """construct"""
        h = self.conv(x)
        view = P.Reshape()
        h = view(h, (h.shape[0], -1))
        return self.fc_adv(h), self.fc_cls(h)
