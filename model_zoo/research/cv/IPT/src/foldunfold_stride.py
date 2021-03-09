'''stride'''
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
import numpy as np
from mindspore import nn
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype


class _stride_unfold_(nn.Cell):
    '''stride'''

    def __init__(self,
                 kernel_size,
                 stride=-1):

        super(_stride_unfold_, self).__init__()
        if stride == -1:
            self.stride = kernel_size
        else:
            self.stride = stride
        self.kernel_size = kernel_size

        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.unfold = _unfold_(kernel_size)

    def construct(self, x):
        """stride"""
        N, C, H, W = x.shape
        leftup_idx_x = []
        leftup_idx_y = []
        nh = int(H / self.stride)
        nw = int(W / self.stride)
        for i in range(nh):
            leftup_idx_x.append(i * self.stride)
        for i in range(nw):
            leftup_idx_y.append(i * self.stride)
        NumBlock_x = len(leftup_idx_x)
        NumBlock_y = len(leftup_idx_y)
        zeroslike = P.ZerosLike()
        cc_2 = P.Concat(axis=2)
        cc_3 = P.Concat(axis=3)
        unf_x = P.Zeros()((N, C, NumBlock_x * self.kernel_size,
                           NumBlock_y * self.kernel_size), mstype.float32)
        N, C, H, W = unf_x.shape
        for i in range(NumBlock_x):
            for j in range(NumBlock_y):
                unf_i = i * self.kernel_size
                unf_j = j * self.kernel_size
                org_i = leftup_idx_x[i]
                org_j = leftup_idx_y[j]
                fills = x[:, :, org_i:org_i + self.kernel_size,
                          org_j:org_j + self.kernel_size]
                unf_x += cc_3((cc_3((zeroslike(unf_x[:, :, :, :unf_j]), cc_2((cc_2(
                    (zeroslike(unf_x[:, :, :unf_i, unf_j:unf_j + self.kernel_size]), fills)), zeroslike(
                        unf_x[:, :, unf_i + self.kernel_size:, unf_j:unf_j + self.kernel_size]))))),
                               zeroslike(unf_x[:, :, :, unf_j + self.kernel_size:])))
        y = self.unfold(unf_x)
        return y


class _stride_fold_(nn.Cell):
    '''stride'''

    def __init__(self,
                 kernel_size,
                 output_shape=(-1, -1),
                 stride=-1):

        super(_stride_fold_, self).__init__()
        if isinstance(kernel_size, (list, tuple)):
            self.kernel_size = kernel_size
        else:
            self.kernel_size = [kernel_size, kernel_size]

        if stride == -1:
            self.stride = kernel_size[0]
        else:
            self.stride = stride

        self.output_shape = output_shape

        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.fold = _fold_(kernel_size)

    def construct(self, x):
        '''stride'''
        if self.output_shape[0] == -1:
            large_x = self.fold(x)
            N, C, H, _ = large_x.shape
            leftup_idx = []
            for i in range(0, H, self.kernel_size[0]):
                leftup_idx.append(i)
            NumBlock = len(leftup_idx)
            fold_x = P.Zeros()((N, C, (NumBlock - 1) * self.stride + self.kernel_size[0],
                                (NumBlock - 1) * self.stride + self.kernel_size[0]), mstype.float32)

            for i in range(NumBlock):
                for j in range(NumBlock):
                    fold_i = i * self.stride
                    fold_j = j * self.stride
                    org_i = leftup_idx[i]
                    org_j = leftup_idx[j]
                    fills = x[:, :, org_i:org_i + self.kernel_size[0],
                              org_j:org_j + self.kernel_size[1]]
                    fold_x += cc_3((cc_3((zeroslike(fold_x[:, :, :, :fold_j]), cc_2((cc_2(
                        (zeroslike(fold_x[:, :, :fold_i, fold_j:fold_j + self.kernel_size[1]]), fills)), zeroslike(
                            fold_x[:, :, fold_i + self.kernel_size[0]:, fold_j:fold_j + self.kernel_size[1]]))))),
                                    zeroslike(fold_x[:, :, :, fold_j + self.kernel_size[1]:])))
            y = fold_x
        else:
            NumBlock_x = int(
                (self.output_shape[0] - self.kernel_size[0]) / self.stride + 1)
            NumBlock_y = int(
                (self.output_shape[1] - self.kernel_size[1]) / self.stride + 1)
            large_shape = [NumBlock_x * self.kernel_size[0],
                           NumBlock_y * self.kernel_size[1]]
            self.fold = _fold_(self.kernel_size, large_shape)
            large_x = self.fold(x)
            N, C, H, _ = large_x.shape
            leftup_idx_x = []
            leftup_idx_y = []
            for i in range(NumBlock_x):
                leftup_idx_x.append(i * self.kernel_size[0])
            for i in range(NumBlock_y):
                leftup_idx_y.append(i * self.kernel_size[1])
            fold_x = P.Zeros()((N, C, (NumBlock_x - 1) * self.stride + self.kernel_size[0],
                                (NumBlock_y - 1) * self.stride + self.kernel_size[1]), mstype.float32)
            for i in range(NumBlock_x):
                for j in range(NumBlock_y):
                    fold_i = i * self.stride
                    fold_j = j * self.stride
                    org_i = leftup_idx_x[i]
                    org_j = leftup_idx_y[j]
                    fills = x[:, :, org_i:org_i + self.kernel_size[0],
                              org_j:org_j + self.kernel_size[1]]
                    fold_x += cc_3((cc_3((zeroslike(fold_x[:, :, :, :fold_j]), cc_2((cc_2(
                        (zeroslike(fold_x[:, :, :fold_i, fold_j:fold_j + self.kernel_size[1]]), fills)), zeroslike(
                            fold_x[:, :, fold_i + self.kernel_size[0]:, fold_j:fold_j + self.kernel_size[1]]))))),
                                    zeroslike(fold_x[:, :, :, fold_j + self.kernel_size[1]:])))
            y = fold_x
        return y


class _unfold_(nn.Cell):
    '''stride'''

    def __init__(self,
                 kernel_size,
                 stride=-1):

        super(_unfold_, self).__init__()
        if stride == -1:
            self.stride = kernel_size
        self.kernel_size = kernel_size

        self.reshape = P.Reshape()
        self.transpose = P.Transpose()

    def construct(self, x):
        '''stride'''
        N, C, H, W = x.shape
        numH = int(H / self.kernel_size)
        numW = int(W / self.kernel_size)
        if numH * self.kernel_size != H or numW * self.kernel_size != W:
            x = x[:, :, :numH * self.kernel_size, :, numW * self.kernel_size]
        output_img = self.reshape(x, (N, C, numH, self.kernel_size, W))

        output_img = self.transpose(output_img, (0, 1, 2, 4, 3))

        output_img = self.reshape(output_img, (N, C, int(
            numH * numW), self.kernel_size, self.kernel_size))

        output_img = self.transpose(output_img, (0, 2, 1, 4, 3))

        output_img = self.reshape(output_img, (N, int(numH * numW), -1))
        return output_img


class _fold_(nn.Cell):
    '''stride'''

    def __init__(self,
                 kernel_size,
                 output_shape=(-1, -1),
                 stride=-1):

        super(_fold_, self).__init__()

        if isinstance(kernel_size, (list, tuple)):
            self.kernel_size = kernel_size
        else:
            self.kernel_size = [kernel_size, kernel_size]

        if stride == -1:
            self.stride = kernel_size[0]
        self.output_shape = output_shape

        self.reshape = P.Reshape()
        self.transpose = P.Transpose()

    def construct(self, x):
        '''stride'''
        N, C, L = x.shape
        org_C = int(L / self.kernel_size[0] / self.kernel_size[1])
        if self.output_shape[0] == -1:
            numH = int(np.sqrt(C))
            numW = int(np.sqrt(C))
            org_H = int(numH * self.kernel_size[0])
            org_W = org_H
        else:
            org_H = int(self.output_shape[0])
            org_W = int(self.output_shape[1])
            numH = int(org_H / self.kernel_size[0])
            numW = int(org_W / self.kernel_size[1])

        output_img = self.reshape(
            x, (N, C, org_C, self.kernel_size[0], self.kernel_size[1]))

        output_img = self.transpose(output_img, (0, 2, 1, 3, 4))
        output_img = self.reshape(
            output_img, (N, org_C, numH, numW, self.kernel_size[0], self.kernel_size[1]))

        output_img = self.transpose(output_img, (0, 1, 2, 4, 3, 5))

        output_img = self.reshape(output_img, (N, org_C, org_H, org_W))
        return output_img
