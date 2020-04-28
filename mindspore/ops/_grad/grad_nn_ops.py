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

"""Define the grad rules of neural network related operations."""
from mindspore.common import dtype as mstype
from .. import functional as F
from .. import operations as P
from ..operations import _grad_ops as G
from ..composite.multitype_ops.zeros_like_impl import zeros_like
from .grad_base import bprop_getters


@bprop_getters.register(P.BiasAdd)
def get_bprop_bias_add(self):
    """Grad definition for `BiasAdd` operation."""
    bias_grad = G.BiasAddGrad()

    def bprop(x, w, out, dout):
        return dout, bias_grad(dout)
    return bprop


@bprop_getters.register(P.Conv2D)
def get_bprop_conv2d(self):
    """Grad definition for `Conv2D` operation."""
    input_grad = P.Conv2DBackpropInput(
        self.out_channel, self.kernel_size, self.pad_mode, self.pad, self.pad_list, mode=self.mode,
        dilation=self.dilation, stride=self.stride, group=self.group
    )
    filter_grad = G.Conv2DBackpropFilter(
        self.out_channel, self.kernel_size, self.pad_mode, self.pad, self.pad_list, mode=self.mode,
        dilation=self.dilation, stride=self.stride, group=self.group
    )
    get_shape = P.Shape()

    def bprop(x, w, out, dout):
        dx = input_grad(dout, w, get_shape(x))
        dw = filter_grad(dout, x, get_shape(w))
        return dx, dw
    return bprop


@bprop_getters.register(P.ExtractImagePatches)
def get_bprop_extract_image_patches(self):
    """Grad definition for `ExtractImagePatches` operation."""
    get_shape = P.Shape()
    reshape = P.Reshape()
    extract_image_patches = P.ExtractImagePatches(ksizes=self.ksizes,
                                                  strides=self.strides,
                                                  rates=self.rates,
                                                  padding=self.padding)
    concat = P.Concat(axis=-1)
    expand_dims = P.ExpandDims()
    scatter_nd = P.ScatterNd()
    dtype = P.DType()
    fill = P.Fill()
    slice_op = P.Slice()
    transpose = P.Transpose()
    matmul = P.MatMul()
    cast = P.Cast()
    _, ksizes_row, ksizes_col, _ = self.ksizes

    def bprop(x, out, dout):
        x_shape = get_shape(x)
        x_batch, x_row, x_col, x_depth = x_shape
        x_indices_num = x_row * x_col + 1
        x_idx = F.tuple_to_array(range(1, x_indices_num))
        x_idx = reshape(x_idx, (1, x_row, x_col, 1))
        x_idx = cast(x_idx, mstype.float16)
        x_idx_patch = extract_image_patches(x_idx)
        x_idx_patch = transpose(x_idx_patch, (0, 3, 1, 2))
        x_idx_patch = cast(x_idx_patch, mstype.int32)

        out_shape = get_shape(out)
        _, out_row, out_col, _ = out_shape
        out_indices_num = out_row * out_col * ksizes_row * ksizes_col
        out_idx = F.tuple_to_array(range(out_indices_num))
        out_idx = reshape(out_idx, (1, ksizes_row * ksizes_col, out_row, out_col))

        idx_tensor = concat((expand_dims(x_idx_patch, -1), expand_dims(out_idx, -1)))
        idx_tensor = reshape(idx_tensor, (-1, 2))
        sp_shape = (x_indices_num, out_indices_num)
        sp_tensor = scatter_nd(idx_tensor, fill(dtype(dout), (out_indices_num,), 1), sp_shape)
        sp_tensor = slice_op(sp_tensor, (1, 0), (x_indices_num - 1, out_indices_num))

        grad = reshape(dout, (x_batch, out_row, out_col, ksizes_row, ksizes_col, x_depth))
        grad = transpose(grad, (1, 2, 3, 4, 0, 5))
        grad = reshape(grad, (-1, x_batch * x_depth))

        jac = matmul(sp_tensor, grad)
        dx = reshape(jac, (x_row, x_col, x_batch, x_depth))
        dx = transpose(dx, (2, 0, 1, 3))

        return (dx,)
    return bprop


@bprop_getters.register(P.DepthwiseConv2dNative)
def get_bprop_depthwise_conv2d_native(self):
    """Grad definition for `DepthwiseConv2dNative` operation."""
    input_grad = G.DepthwiseConv2dNativeBackpropInput(
        self.channel_multiplier, self.kernel_size, self.pad_mode, self.pad, self.pads, self.mode, self.stride,
        self.dilation, self.group
    )
    filter_grad = G.DepthwiseConv2dNativeBackpropFilter(
        self.channel_multiplier, self.kernel_size, self.pad_mode, self.pad, self.pads, self.mode, self.stride,
        self.dilation, self.group
    )
    get_shape = P.Shape()

    def bprop(x, w, out, dout):
        dx = input_grad(get_shape(x), w, dout)
        dw = filter_grad(x, get_shape(w), dout)
        return dx, dw
    return bprop


@bprop_getters.register(P.MaxPoolWithArgmax)
def get_bprop_max_pool_with_argmax(self):
    """Grad definition for `MaxPoolWithArgmax` operation."""
    maxpool_grad = G.MaxPoolGradWithArgmax(
        ksize=self.ksize,
        strides=self.strides,
        padding=self.padding,)

    def bprop(x, out, dout):
        dx = maxpool_grad(x, dout[0], out[1])
        return (dx,)
    return bprop


@bprop_getters.register(P.MaxPool)
def get_bprop_max_pool_grad(self):
    """Grad definition for `MaxPool` operation."""
    maxpool_grad = G.MaxPoolGrad(
        ksize=self.ksize,
        strides=self.strides,
        padding=self.padding)

    def bprop(x, out, dout):
        dx = maxpool_grad(x, out, dout)
        return (dx,)
    return bprop


@bprop_getters.register(P.AvgPool)
def get_bprop_avg_pool_grad(self):
    """Grad definition for `AvgPool` operation."""
    avgpool_grad = G.AvgPoolGrad(
        ksize=self.ksize,
        strides=self.strides,
        padding=self.padding)
    shape_op = P.Shape()

    avgpool_grad_gpu = G.AvgPoolGradGpu(
        ksize=self.ksize,
        strides=self.strides,
        padding=self.padding)

    def bprop(x, out, dout):
        dx = avgpool_grad(shape_op(x), dout)
        return (dx,)

    def bprop_gpu(x, out, dout):
        dx = avgpool_grad_gpu(x, out, dout)
        return (dx,)

    # the parameter of AvgPoolGrad in GPU and TBE/CPU is not same
    if self.target == "GPU":
        bprop_fn = bprop_gpu
    else:
        bprop_fn = bprop

    return bprop_fn


@bprop_getters.register(P.DropoutGenMask)
def get_bprop_dropout_gen_mask(self):
    """Grad definition for `DropoutGenMask` operation."""

    def bprop(shape, keep_prob, out, dout):
        return (zeros_like(shape), zeros_like(keep_prob))
    return bprop


@bprop_getters.register(P.DropoutDoMask)
def get_bprop_dropout_do_mask(self):
    """Grad definition for `DropoutDoMask` operation."""
    do_mask = P.DropoutDoMask()

    def bprop(x, y, keep_prob, out, dout):
        return (do_mask(dout, y, keep_prob), zeros_like(y), zeros_like(keep_prob))
    return bprop


@bprop_getters.register(P.ReLU)
def get_bprop_relu(self):
    """Grad definition for `ReLU` operation."""
    input_grad = G.ReluGrad()

    def bprop(x, out, dout):
        dx = input_grad(dout, out)
        return (dx,)
    return bprop


@bprop_getters.register(P.ReLU6)
def get_bprop_relu6(self):
    """Grad definition for `ReLU6` operation."""
    input_grad = G.ReLU6Grad()

    def bprop(x, out, dout):
        dx = input_grad(dout, x)
        return (dx,)
    return bprop


@bprop_getters.register(P.ReLUV2)
def get_bprop_relu_v2(self):
    """Grad definition for `ReLUV2` operation."""
    input_grad = G.ReluGradV2()

    def bprop(x, out, dout):
        mask = out[1]
        dx = input_grad(dout[0], mask)
        return (dx,)
    return bprop


@bprop_getters.register(P.HSwish)
def get_bprop_hswish(self):
    """Grad definition for `HSwish` operation."""
    input_grad = G.HSwishGrad()

    def bprop(x, out, dout):
        dx = input_grad(dout, x)
        return (dx,)
    return bprop


@bprop_getters.register(P.HSigmoid)
def get_bprop_hsigmoid(self):
    """Grad definition for `HSigmoid` operation."""
    input_grad = G.HSigmoidGrad()

    def bprop(x, out, dout):
        dx = input_grad(dout, x)
        return (dx,)
    return bprop


@bprop_getters.register(P.Elu)
def get_bprop_elu(self):
    """Grad definition for `Elu` operation."""
    input_grad = G.EluGrad()

    def bprop(x, out, dout):
        dx = input_grad(dout, x)
        return (dx,)
    return bprop


@bprop_getters.register(P.Sigmoid)
def get_bprop_sigmoid(self):
    """Grad definition for `Sigmoid` operation."""
    input_grad = G.SigmoidGrad()

    def bprop(x, out, dout):
        dx = input_grad(out, dout)
        return (dx,)
    return bprop


@bprop_getters.register(P.Softmax)
def get_bprop_softmax(self):
    """Grad definition for `Softmax` operation."""
    sum_func = P.ReduceSum(keep_dims=True)
    sub = P.Sub()
    mul = P.Mul()
    axis = self.axis

    def bprop(x, out, dout):
        dx = mul(sub(dout, sum_func(mul(dout, out), axis)), out)
        return (dx,)
    return bprop


@bprop_getters.register(P.LogSoftmax)
def get_bprop_log_softmax(self):
    """Grad definition for `LogSoftmax` operation."""
    logsoftmax_grad = G.LogSoftmaxGrad(self.axis)

    def bprop(x, out, dout):
        dx = logsoftmax_grad(out, dout)
        return (dx,)
    return bprop


@bprop_getters.register(P.Tanh)
def get_bprop_tanh(self):
    """Grad definition for `Tanh` operation."""
    logsoftmax_grad = G.TanhGrad()

    def bprop(x, out, dout):
        dx = logsoftmax_grad(out, dout)
        return (dx,)
    return bprop


@bprop_getters.register(P.Gelu)
def get_bprop_gelu(self):
    """Grad definition for `Gelu` operation."""
    input_grad = G.GeluGrad()

    def bprop(x, out, dout):
        dx = input_grad(dout, x, out)
        return (dx,)
    return bprop


@bprop_getters.register(P.FusedBatchNorm)
def get_bprop_fused_batch_norm(self):
    """Grad definition for `FusedBatchNorm` operation."""
    input_grad = G.FusedBatchNormGrad(self.epsilon, self.momentum)

    def bprop(x, scale, b, mean, variance, out, dout):
        saved_mean = out[3]
        saved_variance = out[4]
        out = input_grad(dout[0], x, scale, saved_mean, saved_variance)
        dx = out[0]
        dscale = out[1]
        dbias = out[2]
        return dx, dscale, dbias, zeros_like(mean), zeros_like(variance)
    return bprop


@bprop_getters.register(P.BatchNorm)
def get_bprop_batch_norm(self):
    """Grad definition for `BatchNorm` operation."""
    is_training = self.is_training
    input_grad = G.BatchNormGrad(is_training, self.epsilon)

    def bprop(x, scale, b, mean, variance, out, dout):
        if is_training:
            saved_reserve_1 = out[3]
            saved_reserve_2 = out[4]
            saved_reserve_3 = out[5]
        else:
            saved_reserve_1 = mean
            saved_reserve_2 = variance
            saved_reserve_3 = variance
        out = input_grad(dout[0], x, scale, saved_reserve_1, saved_reserve_2, saved_reserve_3)
        dx = out[0]
        dscale = out[1]
        dbias = out[2]
        return dx, dscale, dbias, zeros_like(mean), zeros_like(variance)
    return bprop


@bprop_getters.register(P.LayerNorm)
def get_bprop_layer_norm(self):
    """Grad definition for `LayerNorm` operation."""
    layer_norm_grad = G.LayerNormGrad(self.begin_norm_axis, self.begin_params_axis)

    def bprop(x, gamma, beta, out, dout):
        dx, d_gamma, d_beta = layer_norm_grad(x, dout[0], out[2], out[1], gamma)
        return dx, d_gamma, d_beta
    return bprop


@bprop_getters.register(P.L2Normalize)
def get_bprop_l2normalize(self):
    """Grad definition for `L2Normalize` operation."""
    input_grad = G.L2NormalizeGrad(self.axis, self.epsilon)

    def bprop(x, out, dout):
        dx = input_grad(x, out, dout)
        return (dx,)
    return bprop


@bprop_getters.register(P.SoftmaxCrossEntropyWithLogits)
def get_bprop_softmax_cross_entropy_with_logits(self):
    """Grad definition for `SoftmaxCrossEntropyWithLogits` operation."""
    expand = P.ExpandDims()

    def bprop(logits, labels, out, dout):
        grad = out[1]
        grad = grad * expand(dout[0], -1)
        return grad, zeros_like(labels)
    return bprop


@bprop_getters.register(P.SparseSoftmaxCrossEntropyWithLogits)
def get_bprop_sparse_softmax_cross_entropy_with_logits(self):
    """Grad definition for `SparseSoftmaxCrossEntropyWithLogits` operation."""
    is_grad = self.is_grad
    grad_op = P.SparseSoftmaxCrossEntropyWithLogits(is_grad=True)

    def bprop(logits, labels, out, dout):
        grad = out[0]
        if not is_grad:
            # if construct use loss
            grad = grad_op(logits, labels)
            grad = F.depend(grad, out)
            grad = grad * dout
        return grad, zeros_like(labels)
    return bprop


@bprop_getters.register(P.ResizeBilinear)
def get_bprop_resize_bilinear(self):
    """Grad definition for `ResizeBilinear` operation."""
    resize_grad = G.ResizeBilinearGrad(self.align_corners)

    def bprop(x, out, dout):
        dx = resize_grad(dout, x)
        return (dx,)
    return bprop


@bprop_getters.register(P.OneHot)
def get_bprop_onehot(self):
    """Grad definition for `OneHot` operation."""

    def bprop(indices, depth, on_value, off_value, out, dout):
        return zeros_like(indices), zeros_like(depth), zeros_like(on_value), zeros_like(off_value)
    return bprop


@bprop_getters.register(P.TopK)
def get_bprop_top_kv2(self):
    """Grad definition for `TopK` operation."""
    scatter = P.ScatterNd()
    expand_dims = P.ExpandDims()
    shape_op = P.Shape()

    def bprop(input_x, k, out, dout):
        indices = out[1]
        indices = expand_dims(indices, -1)
        updates = dout[0]
        shapes = shape_op(input_x)
        return scatter(indices, updates, shapes), zeros_like(k)
    return bprop


@bprop_getters.register(P.SmoothL1Loss)
def get_bprop_smooth_l1_loss(self):
    """Grad definition for `SmoothL1Loss` operation."""
    grad = G.SmoothL1LossGrad(self.sigma)

    def bprop(prediction, target, out, dout):
        dx = grad(prediction, target, dout)
        return dx, zeros_like(target)

    return bprop


@bprop_getters.register(P.L2Loss)
def get_bprop_l2_loss(self):
    """Grad definition for `L2Loss` operation."""

    def bprop(x, out, dout):
        dx = x * dout
        return (dx,)

    return bprop


@bprop_getters.register(P.PReLU)
def get_bprop_prelu(self):
    """Grad definition for `PReLU` operation."""
    grad = G.PReLUGrad()

    def bprop(x, w, out, dout):
        dx, dw = grad(dout, x, w)
        return dx, dw

    return bprop


@bprop_getters.register(P.LSTM)
def get_bprop_lstm(self):
    """Grad definition for `LSTM` operation."""
    lstm_grad_data = G.LSTMGradData(
        input_size=self.input_size,
        hidden_size=self.hidden_size,
        num_layers=self.num_layers,
        has_bias=self.has_bias,
        bidirectional=self.bidirectional,
        dropout=self.dropout
    )

    lstm_grad_weight = G.LSTMGradWeight(
        input_size=self.input_size,
        hidden_size=self.hidden_size,
        num_layers=self.num_layers,
        has_bias=self.has_bias,
        bidirectional=self.bidirectional,
        dropout=self.dropout
    )

    def bprop(x, hx, cx, w, out, dout):
        y, _, _, reserve, state = out
        dy, dhy, dcy, _, _ = dout
        dx, dhx, dcx = lstm_grad_data(y, dy, dhy, dcy, w, hx, cx, reserve, state)
        dw = lstm_grad_weight(F.depend(x, dx), hx, y, reserve, state)
        return dx, dhx, dcx, dw
    return bprop


@bprop_getters.register(P.SigmoidCrossEntropyWithLogits)
def get_bprop_sigmoid_crossentropy_with_logits(self):
    """Grad definition for `SigmoidCrossEntropyWithLogits` operation."""
    op = G.SigmoidCrossEntropyWithLogitsGrad()

    def bprop(x, y, out, dout):
        dx = op(x, y, dout)
        return (dx, zeros_like(y))
    return bprop


@bprop_getters.register(P.Pad)
def get_bprop_pad(self):
    """Grad definition for `Pad` operation."""
    shape_op = P.Shape()
    paddings = self.paddings

    def bprop(x, out, dout):
        begin = ()
        for item in paddings:
            begin += (item[0],)
        shp = shape_op(x)
        dx = P.Slice()(dout, begin, shp)
        return (dx,)
    return bprop


@bprop_getters.register(P.MirrorPad)
def get_bprop_mirror_pad(self):
    """Grad definition for `MirrorPad` operation."""
    mirror_pad_grad = G.MirrorPadGrad(self.mode)

    def bprop(x, paddings, out, dout):
        dx = mirror_pad_grad(dout, paddings, x)
        return (dx, zeros_like(paddings))
    return bprop


@bprop_getters.register(P.ROIAlign)
def get_bprop_roi_align(self):
    """Grad definition for `ROIAlign` operation."""
    shape_op = P.Shape()
    pooled_height = self.pooled_height
    pooled_width = self.pooled_width
    spatial_scale = self.spatial_scale
    sample_num = self.sample_num

    def bprop(inputs, rois, out, dout):
        rois_shape = shape_op(rois)
        inputs_shape = shape_op(inputs)
        dx = G.ROIAlignGrad(inputs_shape,
                            pooled_height,
                            pooled_width,
                            spatial_scale,
                            sample_num,
                            )(dout, rois)
        return dx, zeros_like(rois_shape)

    return bprop


@bprop_getters.register(P.Conv2DBackpropInput)
def get_bprop_conv2d_backprop_input(self):
    """Grad definition for `Conv2DBackpropInput` operation."""
    filter_grad = G.Conv2DBackpropFilter(
        self.out_channel, self.kernel_size, self.pad_mode, self.pad, self.pad_list, mode=self.mode,
        dilation=self.dilation, stride=self.stride, group=self.group
    )
    input_grad = P.Conv2D(
        self.out_channel, self.kernel_size, pad_mode=self.pad_mode.lower(), pad=self.pad,
        dilation=self.dilation, stride=self.stride, group=self.group
    )

    def bprop(x, w, f_sizes, out, dout):
        dx = input_grad(dout, w)
        dw = filter_grad(x, dout, F.shape(w))
        return dx, dw

    return bprop


@bprop_getters.register(P.BinaryCrossEntropy)
def get_bprop_binary_cross_entropy(self):
    """Grad definition for `BinaryCrossEntropy` operation."""
    grad = G.BinaryCrossEntropyGrad(self.reduction)

    def bprop(x, y, weight, out, dout):
        dx = grad(x, y, dout, weight)
        return dx, zeros_like(y), zeros_like(weight)

    return bprop
