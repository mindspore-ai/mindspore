# Copyright 2020 Huawei Technologies Co., Ltd

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import time
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.nn.loss.loss import _Loss
from mindspore.train.callback import Callback
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.communication.management import get_group_size
from mindspore.context import ParallelMode
from mindspore import context

context.set_context(mode=context.GRAPH_MODE, save_graphs=True)
time_stamp_init = False
time_stamp_first = 0
grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()
GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0

@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * F.cast(reciprocal(scale), F.dtype(grad))

@grad_scale.register("Tensor", "RowTensor")
def tensor_grad_scale_row_tensor(scale, grad):
    return RowTensor(grad.indices,
                     grad.values * F.cast(reciprocal(scale), F.dtype(grad.values)),
                     grad.dense_shape)
clip_grad = C.MultitypeFuncGraph("clip_grad")

@clip_grad.register("Number", "Number", "Tensor")
class openpose_loss(_Loss):
    def __init__(self):
        super(openpose_loss, self).__init__()
        self.expand_dims = P.ExpandDims()
        self.tile = P.Tile()
        self.mul = P.Mul()
        self.l2_loss = P.L2Loss()
        self.square = P.Square()
        self.reduceMean = P.ReduceMean()
        self.reduceSum = P.ReduceSum()
        self.print = P.Print()
        self.shape = P.Shape()
        self.maxoftensor = P.ArgMaxWithValue(-1)

    def mean_square_error(self, map1, map2, mask=None):
        # print("mask", mask)
        # import pdb; pdb.set_trace()
        if mask is None:
            mse = self.reduceMean((map1 - map2) ** 2)
            return mse

        squareMap = self.square(map1 - map2)
        squareMap_mask = self.mul(squareMap, mask)
        mse = self.reduceMean(squareMap_mask)
        return mse

    def construct(self, logit_paf, logit_heatmap, gt_paf, gt_heatmap, ignore_mask):
        # Input
        # ignore_mask, make sure the ignore_mask the 0-1 array instead of the bool-false array
        heatmaps_loss = []
        pafs_loss = []
        total_loss = 0

        paf_masks = self.tile(self.expand_dims(ignore_mask, 1), (1, self.shape(gt_paf)[1], 1, 1))
        heatmap_masks = self.tile(self.expand_dims(ignore_mask, 1), (1, self.shape(gt_heatmap)[1], 1, 1))

        paf_masks = F.stop_gradient(paf_masks)
        heatmap_masks = F.stop_gradient(heatmap_masks)
        for logit_paf_t, logit_heatmap_t in zip(logit_paf, logit_heatmap):
            # TEST
            # tensor1 -- tuple
            # tensor1 = self.maxoftensor(logit_paf_t)[1]
            # tensor2 = self.maxoftensor(logit_heatmap_t)[1]
            # tensor3 = self.maxoftensor(tensor1)[1]
            # tensor4 = self.maxoftensor(tensor2)[1]
            # self.print("paf",tensor3)
            # self.print("heatmaps",tensor2)
            pafs_loss_t = self.mean_square_error(logit_paf_t, gt_paf, paf_masks)
            heatmaps_loss_t = self.mean_square_error(logit_heatmap_t, gt_heatmap, heatmap_masks)

            total_loss += pafs_loss_t + heatmaps_loss_t
            heatmaps_loss.append(heatmaps_loss_t)
            pafs_loss.append(pafs_loss_t)

        return total_loss, heatmaps_loss, pafs_loss

class Depend_network(nn.Cell):
    def __init__(self, network):
        super(Depend_network, self).__init__()
        self.network = network

    def construct(self, *args):
        loss, _, _ = self.network(*args) # loss, heatmaps_loss, pafs_loss
        return loss

class TrainingWrapper(nn.Cell):
    def __init__(self, network, optimizer, sens=1):
        super(TrainingWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.depend_network = Depend_network(network)
        # self.weights = ms.ParameterTuple(network.trainable_params())
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
        self.print = P.Print()
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            #if mean.get_device_num_is_set():
            # if mean:
                #degree = context.get_auto_parallel_context("device_num")
            # else:
            degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, *args):
        weights = self.weights
        loss, heatmaps_loss, pafs_loss = self.network(*args)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        #grads = self.grad(self.network, weights)(*args, sens)
        grads = self.grad(self.depend_network, weights)(*args, sens)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        #return F.depend(loss, self.optimizer(grads))
        # for grad in grads:
            # self.print(grad)
        loss = F.depend(loss, self.optimizer(grads))
        return loss, heatmaps_loss, pafs_loss

class BuildTrainNetwork(nn.Cell):
    def __init__(self, network, criterion):
        super(BuildTrainNetwork, self).__init__()
        self.network = network
        self.criterion = criterion

    def construct(self, input_data, gt_paf, gt_heatmap, mask):
        logit_pafs, logit_heatmap = self.network(input_data)
        loss, _, _ = self.criterion(logit_pafs, logit_heatmap, gt_paf, gt_heatmap, mask)
        return loss

class LossCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss is NAN or INF terminating training.
    Note:
        If per_print_times is 0 do not print loss.
    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """

    def __init__(self, per_print_times=1):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.count = 0
        self.loss_sum = 0

        global time_stamp_init, time_stamp_first
        if not time_stamp_init:
            time_stamp_first = time.time()
            time_stamp_init = True

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs.asnumpy()

        self.count += 1
        self.loss_sum += float(loss)

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if self.count >= 1:
            global time_stamp_first
            time_stamp_current = time.time()

            loss = self.loss_sum/self.count

            loss_file = open("./loss.log", "a+")
            loss_file.write("%lu epoch: %s step: %s ,loss: %.5f" %
                            (time_stamp_current - time_stamp_first, cb_params.cur_epoch_num, cur_step_in_epoch,
                             loss))
            loss_file.write("\n")
            loss_file.close()

            self.count = 0
            self.loss_sum = 0
