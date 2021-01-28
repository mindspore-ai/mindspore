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


import mindspore.ops.operations as P
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore.nn.loss.loss import _Loss

class DiceLoss(_Loss):
    def __init__(self, batch_size=4):
        super(DiceLoss, self).__init__()

        self.threshold0 = Tensor(0.5, mstype.float32)
        self.zero_float32 = Tensor(0.0, mstype.float32)
        self.k = int(640 * 640)
        self.negative_one_int32 = Tensor(-1, mstype.int32)
        self.batch_size = batch_size
        self.concat = P.Concat()
        self.less_equal = P.LessEqual()
        self.greater = P.Greater()
        self.reduce_sum = P.ReduceSum()
        self.reduce_sum_keep_dims = P.ReduceSum(keep_dims=True)
        self.reduce_mean = P.ReduceMean()
        self.reduce_min = P.ReduceMin()
        self.cast = P.Cast()
        self.minimum = P.Minimum()
        self.expand_dims = P.ExpandDims()
        self.select = P.Select()
        self.fill = P.Fill()
        self.topk = P.TopK(sorted=True)
        self.shape = P.Shape()
        self.sigmoid = P.Sigmoid()
        self.reshape = P.Reshape()
        self.slice = P.Slice()
        self.logical_and = P.LogicalAnd()
        self.logical_or = P.LogicalOr()
        self.equal = P.Equal()
        self.zeros_like = P.ZerosLike()
        self.add = P.Add()
        self.gather = P.Gather()

    def ohem_batch(self, scores, gt_texts, training_masks):
        '''

        :param scores: [N * H * W]
        :param gt_texts:  [N * H * W]
        :param training_masks: [N * H * W]
        :return: [N * H * W]
        '''
        selected_masks = ()
        for i in range(self.batch_size):
            score = self.slice(scores, (i, 0, 0), (1, 640, 640))
            score = self.reshape(score, (640, 640))

            gt_text = self.slice(gt_texts, (i, 0, 0), (1, 640, 640))
            gt_text = self.reshape(gt_text, (640, 640))

            training_mask = self.slice(training_masks, (i, 0, 0), (1, 640, 640))
            training_mask = self.reshape(training_mask, (640, 640))

            selected_mask = self.ohem_single(score, gt_text, training_mask)
            selected_masks = selected_masks + (selected_mask,)

        selected_masks = self.concat(selected_masks)
        return selected_masks

    def ohem_single(self, score, gt_text, training_mask):
        pos_num = self.logical_and(self.greater(gt_text, self.threshold0),
                                   self.greater(training_mask, self.threshold0))
        pos_num = self.reduce_sum(self.cast(pos_num, mstype.float32))

        neg_num = self.less_equal(gt_text, self.threshold0)
        neg_num = self.reduce_sum(self.cast(neg_num, mstype.float32))
        neg_num = self.minimum(3 * pos_num, neg_num)
        neg_num = self.cast(neg_num, mstype.int32)

        neg_num = self.add(neg_num, self.negative_one_int32)
        neg_mask = self.less_equal(gt_text, self.threshold0)
        ignore_score = self.fill(mstype.float32, (640, 640), -1e3)
        neg_score = self.select(neg_mask, score, ignore_score)
        neg_score = self.reshape(neg_score, (640 * 640,))

        topk_values, _ = self.topk(neg_score, self.k)
        threshold = self.gather(topk_values, neg_num, 0)

        selected_mask = self.logical_and(
            self.logical_or(self.greater(score, threshold),
                            self.greater(gt_text, self.threshold0)),
            self.greater(training_mask, self.threshold0))

        selected_mask = self.cast(selected_mask, mstype.float32)
        selected_mask = self.expand_dims(selected_mask, 0)

        return selected_mask

    def dice_loss(self, input_params, target, mask):
        '''

        :param input: [N, H, W]
        :param target: [N, H, W]
        :param mask: [N, H, W]
        :return:
        '''

        input_sigmoid = self.sigmoid(input_params)

        input_reshape = self.reshape(input_sigmoid, (self.batch_size, 640 * 640))
        target = self.reshape(target, (self.batch_size, 640 * 640))
        mask = self.reshape(mask, (self.batch_size, 640 * 640))

        input_mask = input_reshape * mask
        target = target * mask

        a = self.reduce_sum(input_mask * target, 1)
        b = self.reduce_sum(input_mask * input_mask, 1) + 0.001
        c = self.reduce_sum(target * target, 1) + 0.001
        d = (2 * a) / (b + c)
        dice_loss = self.reduce_mean(d)
        return 1 - dice_loss

    def avg_losses(self, loss_list):
        loss_kernel = loss_list[0]
        for i in range(1, len(loss_list)):
            loss_kernel += loss_list[i]
        loss_kernel = loss_kernel / len(loss_list)
        return loss_kernel

    def construct(self, model_predict, gt_texts, gt_kernels, training_masks):
        '''

        :param model_predict: [N * 7 * H * W]
        :param gt_texts: [N * H * W]
        :param gt_kernels:[N * 6 * H * W]
        :param training_masks:[N * H * W]
        :return:
        '''
        texts = self.slice(model_predict, (0, 0, 0, 0), (self.batch_size, 1, 640, 640))
        texts = self.reshape(texts, (self.batch_size, 640, 640))
        selected_masks_text = self.ohem_batch(texts, gt_texts, training_masks)
        loss_text = self.dice_loss(texts, gt_texts, selected_masks_text)

        kernels = []
        loss_kernels = []
        for i in range(1, 7):
            kernel = self.slice(model_predict, (0, i, 0, 0), (self.batch_size, 1, 640, 640))
            kernel = self.reshape(kernel, (self.batch_size, 640, 640))
            kernels.append(kernel)

        mask0 = self.sigmoid(texts)
        selected_masks_kernels = self.logical_and(self.greater(mask0, self.threshold0),
                                                  self.greater(training_masks, self.threshold0))
        selected_masks_kernels = self.cast(selected_masks_kernels, mstype.float32)

        for i in range(6):
            gt_kernel = self.slice(gt_kernels, (0, i, 0, 0), (self.batch_size, 1, 640, 640))
            gt_kernel = self.reshape(gt_kernel, (self.batch_size, 640, 640))
            loss_kernel_i = self.dice_loss(kernels[i], gt_kernel, selected_masks_kernels)
            loss_kernels.append(loss_kernel_i)
        loss_kernel = self.avg_losses(loss_kernels)

        loss = 0.7 * loss_text + 0.3 * loss_kernel
        return loss
