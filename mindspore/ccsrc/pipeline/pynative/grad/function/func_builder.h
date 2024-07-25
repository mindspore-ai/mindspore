/**
 * Copyright 2024 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_FUNC_BUILDER_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_FUNC_BUILDER_H_

#include <utility>
#include <vector>
#include <string>
#include <memory>
#include "utils/hash_map.h"
#include "frontend/expander/bprop/bprop_irbuilder.h"
#include "pipeline/pynative/grad/function/func_pass.h"

namespace mindspore::pynative::autograd {
using NodePtr = expander::NodePtr;
using NodePtrList = expander::NodePtrList;
using BpropBuilder = expander::bprop::BpropBuilder;

class FuncBuilder : public BpropBuilder {
 public:
  FuncBuilder(const std::string &name, std::string device_target, const expander::ExpanderInferPtr &infer = nullptr);
  ~FuncBuilder() override = default;
  NodePtr EmitOp(const PrimitivePtr &prim, const NodePtrList &inputs) override;
  NodePtr EmitValue(const ValuePtr &value) override;
  NodePtrList ShapeCalc(const ShapeCalcBaseFunctorPtr &functor, const NodePtrList &inputs) override;
  NodePtrList ShapeCalc(const ShapeCalcBaseFunctorPtr &functor, const NodePtrList &inputs,
                        const std::vector<int64_t> &value_depend) override;
  // Override Stack to flatten tuple input.
  NodePtr Stack(const NodePtr &x, const ValuePtr &axis) override;
  NodePtr Stack(const NodePtrList &x, int64_t axis) override;
  // Override to optimize performance.
  NodePtr Cast(const NodePtr &node, const TypePtr &type) override;
  NodePtr Reshape(const NodePtr &node, const NodePtr &shape) override;
  NodePtr Transpose(const NodePtr &node, const NodePtr &perm) override;
  NodePtr MatMul(const NodePtr &a, const NodePtr &b, bool transpose_a, bool transpose_b) override;
  NodePtr MatMulExt(const NodePtr &a, const NodePtr &b) override;
  NodePtr Add(const NodePtr &lhs, const NodePtr &rhs) override;
  NodePtr Sub(const NodePtr &lhs, const NodePtr &rhs) override;
  NodePtr Mul(const NodePtr &lhs, const NodePtr &rhs) override;
  NodePtr Div(const NodePtr &lhs, const NodePtr &rhs) override;
  NodePtr Pow(const NodePtr &lhs, const NodePtr &rhs) override;
  NodePtr Equal(const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type) override;
  NodePtr NotEqual(const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type) override;
  NodePtr GreaterEqual(const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type) override;
  NodePtr Greater(const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type) override;
  NodePtr LessEqual(const NodePtr &input, const NodePtr &other, const TypePtr &dst_type) override;
  NodePtr Less(const NodePtr &input, const NodePtr &other, const TypePtr &dst_type) override;
  // to do
  NodePtr Log(const NodePtr &input);
  NodePtr Concat(const NodePtr &tensors, const NodePtr &axis) override;

  NodePtr Abs(const NodePtr &input) override;
  NodePtr AdamW(const NodePtr &var, const NodePtr &m, const NodePtr &v, const NodePtr &max_v, const NodePtr &gradient,
                const NodePtr &step, const NodePtr &lr, const NodePtr &beta1, const NodePtr &beta2,
                const NodePtr &decay, const NodePtr &eps, const NodePtr &amsgrad, const NodePtr &maximize) override;
  NodePtr AddExt(const NodePtr &input, const NodePtr &other, const NodePtr &alpha) override;
  NodePtr AddLayerNormV2(const NodePtr &x1, const NodePtr &x2, const NodePtr &gamma, const NodePtr &beta,
                         const NodePtr &epsilon, const NodePtr &additionalOut) override;
  NodePtr Addmm(const NodePtr &input, const NodePtr &mat1, const NodePtr &mat2, const NodePtr &beta,
                const NodePtr &alpha) override;
  NodePtr Arange(const NodePtr &start, const NodePtr &end, const NodePtr &step, const NodePtr &dtype) override;
  NodePtr ArgMaxExt(const NodePtr &input, const NodePtr &dim, const NodePtr &keepdim) override;
  NodePtr ArgMaxWithValue(const NodePtr &input, const NodePtr &axis, const NodePtr &keep_dims) override;
  NodePtr ArgMinWithValue(const NodePtr &input, const NodePtr &axis, const NodePtr &keep_dims) override;
  NodePtr Atan2Ext(const NodePtr &input, const NodePtr &other) override;
  NodePtr AvgPool2DGrad(const NodePtr &grad, const NodePtr &image, const NodePtr &kernel_size, const NodePtr &stride,
                        const NodePtr &padding, const NodePtr &ceil_mode, const NodePtr &count_include_pad,
                        const NodePtr &divisor_override) override;
  NodePtr AvgPool2D(const NodePtr &input, const NodePtr &kernel_size, const NodePtr &stride, const NodePtr &padding,
                    const NodePtr &ceil_mode, const NodePtr &count_include_pad,
                    const NodePtr &divisor_override) override;
  NodePtr BatchMatMul(const NodePtr &x, const NodePtr &y, const NodePtr &transpose_a,
                      const NodePtr &transpose_b) override;
  NodePtr BatchNormExt(const NodePtr &input, const NodePtr &weight, const NodePtr &bias, const NodePtr &running_mean,
                       const NodePtr &runnning_var, const NodePtr &training, const NodePtr &momentum,
                       const NodePtr &epsilon) override;
  NodePtr BatchNormGradExt(const NodePtr &dout, const NodePtr &input, const NodePtr &weight,
                           const NodePtr &running_mean, const NodePtr &running_var, const NodePtr &saved_mean,
                           const NodePtr &saved_rstd, const NodePtr &training, const NodePtr &eps) override;
  NodePtr BinaryCrossEntropyGrad(const NodePtr &input, const NodePtr &target, const NodePtr &grad_output,
                                 const NodePtr &weight, const NodePtr &reduction) override;
  NodePtr BinaryCrossEntropy(const NodePtr &input, const NodePtr &target, const NodePtr &weight,
                             const NodePtr &reduction) override;
  NodePtr BinaryCrossEntropyWithLogitsBackward(const NodePtr &grad_output, const NodePtr &input, const NodePtr &target,
                                               const NodePtr &weight, const NodePtr &posWeight,
                                               const NodePtr &reduction) override;
  NodePtr BCEWithLogitsLoss(const NodePtr &input, const NodePtr &target, const NodePtr &weight,
                            const NodePtr &posWeight, const NodePtr &reduction) override;
  NodePtr BatchMatMulExt(const NodePtr &input, const NodePtr &mat2) override;
  NodePtr BroadcastTo(const NodePtr &input, const NodePtr &shape) override;
  NodePtr Ceil(const NodePtr &input) override;
  NodePtr Chunk(const NodePtr &input, const NodePtr &chunks, const NodePtr &dim) override;
  NodePtr ClampScalar(const NodePtr &input, const NodePtr &min, const NodePtr &max) override;
  NodePtr ClampTensor(const NodePtr &input, const NodePtr &min, const NodePtr &max) override;
  NodePtr Col2ImExt(const NodePtr &input, const NodePtr &output_size, const NodePtr &kernel_size,
                    const NodePtr &dilation, const NodePtr &padding, const NodePtr &stride) override;
  NodePtr Col2ImGrad(const NodePtr &input, const NodePtr &kernel_size, const NodePtr &dilation, const NodePtr &padding,
                     const NodePtr &stride) override;
  NodePtr ConstantPadND(const NodePtr &input, const NodePtr &padding, const NodePtr &value) override;
  NodePtr Contiguous(const NodePtr &input) override;
  NodePtr ConvolutionGrad(const NodePtr &dout, const NodePtr &input, const NodePtr &weight, const NodePtr &bias,
                          const NodePtr &stride, const NodePtr &padding, const NodePtr &dilation,
                          const NodePtr &transposed, const NodePtr &output_padding, const NodePtr &groups,
                          const NodePtr &output_mask) override;
  NodePtr Convolution(const NodePtr &input, const NodePtr &weight, const NodePtr &bias, const NodePtr &stride,
                      const NodePtr &padding, const NodePtr &dilation, const NodePtr &transposed,
                      const NodePtr &output_padding, const NodePtr &groups) override;
  NodePtr Copy(const NodePtr &input) override;
  NodePtr Cos(const NodePtr &input) override;
  NodePtr CumsumExt(const NodePtr &input, const NodePtr &dim, const NodePtr &dtype) override;
  NodePtr Dense(const NodePtr &input, const NodePtr &weight, const NodePtr &bias) override;
  NodePtr DivMod(const NodePtr &x, const NodePtr &y, const NodePtr &rounding_mode) override;
  NodePtr Dot(const NodePtr &input, const NodePtr &other) override;
  NodePtr DropoutDoMaskExt(const NodePtr &input, const NodePtr &mask, const NodePtr &p) override;
  NodePtr DropoutExt(const NodePtr &input, const NodePtr &p, const NodePtr &seed, const NodePtr &offset) override;
  NodePtr DropoutGenMaskExt(const NodePtr &shape, const NodePtr &p, const NodePtr &seed, const NodePtr &offset,
                            const NodePtr &dtype) override;
  NodePtr DropoutGradExt(const NodePtr &input, const NodePtr &mask, const NodePtr &p) override;
  NodePtr EluExt(const NodePtr &input, const NodePtr &alpha) override;
  NodePtr EluGradExt(const NodePtr &dout, const NodePtr &x, const NodePtr &alpha) override;
  NodePtr EmbeddingDenseBackward(const NodePtr &grad, const NodePtr &indices, const NodePtr &num_weights,
                                 const NodePtr &padding_idx, const NodePtr &scale_grad_by_freq) override;
  NodePtr Embedding(const NodePtr &input, const NodePtr &weight, const NodePtr &padding_idx, const NodePtr &max_norm,
                    const NodePtr &norm_type, const NodePtr &scale_grad_by_freq) override;
  NodePtr Erf(const NodePtr &input) override;
  NodePtr Erfinv(const NodePtr &input) override;
  NodePtr Exp(const NodePtr &input) override;
  NodePtr Eye(const NodePtr &n, const NodePtr &m, const NodePtr &dtype) override;
  NodePtr FFNExt(const NodePtr &x, const NodePtr &weight1, const NodePtr &weight2, const NodePtr &expertTokens,
                 const NodePtr &bias1, const NodePtr &bias2, const NodePtr &scale, const NodePtr &offset,
                 const NodePtr &deqScale1, const NodePtr &deqScale2, const NodePtr &antiquant_scale1,
                 const NodePtr &antiquant_scale2, const NodePtr &antiquant_offset1, const NodePtr &antiquant_offset2,
                 const NodePtr &activation, const NodePtr &inner_precise) override;
  NodePtr FillScalar(const NodePtr &size, const NodePtr &fill_value, const NodePtr &dtype) override;
  NodePtr FillTensor(const NodePtr &size, const NodePtr &fill_value, const NodePtr &dtype) override;
  NodePtr FlashAttentionScoreGrad(const NodePtr &query, const NodePtr &key, const NodePtr &value, const NodePtr &dy,
                                  const NodePtr &pse_shift, const NodePtr &drop_mask, const NodePtr &padding_mask,
                                  const NodePtr &atten_mask, const NodePtr &softmax_max, const NodePtr &softmax_sum,
                                  const NodePtr &softmax_in, const NodePtr &attention_in, const NodePtr &prefix,
                                  const NodePtr &actual_seq_qlen, const NodePtr &actual_seq_kvlen,
                                  const NodePtr &head_num, const NodePtr &keep_prob, const NodePtr &scale_value,
                                  const NodePtr &pre_tokens, const NodePtr &next_tokens, const NodePtr &inner_precise,
                                  const NodePtr &input_layout, const NodePtr &sparse_mode) override;
  NodePtr FlashAttentionScore(const NodePtr &query, const NodePtr &key, const NodePtr &value, const NodePtr &real_shift,
                              const NodePtr &drop_mask, const NodePtr &padding_mask, const NodePtr &attn_mask,
                              const NodePtr &prefix, const NodePtr &actual_seq_qlen, const NodePtr &actual_seq_kvlen,
                              const NodePtr &head_num, const NodePtr &keep_prob, const NodePtr &scale_value,
                              const NodePtr &pre_tokens, const NodePtr &next_tokens, const NodePtr &inner_precise,
                              const NodePtr &input_layout, const NodePtr &sparse_mode) override;
  NodePtr FlattenExt(const NodePtr &input, const NodePtr &start_dim, const NodePtr &end_dim) override;
  NodePtr Floor(const NodePtr &input) override;
  NodePtr GatherDGradV2(const NodePtr &x, const NodePtr &dim, const NodePtr &index, const NodePtr &dout) override;
  NodePtr GatherD(const NodePtr &x, const NodePtr &dim, const NodePtr &index) override;
  NodePtr GeLUGrad(const NodePtr &dy, const NodePtr &x, const NodePtr &y) override;
  NodePtr GeLU(const NodePtr &input) override;
  NodePtr Generator(const NodePtr &cmd, const NodePtr &inputs) override;
  NodePtr GridSampler2DGrad(const NodePtr &grad, const NodePtr &input_x, const NodePtr &grid,
                            const NodePtr &interpolation_mode, const NodePtr &padding_mode,
                            const NodePtr &align_corners) override;
  NodePtr GridSampler2D(const NodePtr &input_x, const NodePtr &grid, const NodePtr &interpolation_mode,
                        const NodePtr &padding_mode, const NodePtr &align_corners) override;
  NodePtr GridSampler3DGrad(const NodePtr &grad, const NodePtr &input_x, const NodePtr &grid,
                            const NodePtr &interpolation_mode, const NodePtr &padding_mode,
                            const NodePtr &align_corners) override;
  NodePtr GridSampler3D(const NodePtr &input_x, const NodePtr &grid, const NodePtr &interpolation_mode,
                        const NodePtr &padding_mode, const NodePtr &align_corners) override;
  NodePtr GroupNormGrad(const NodePtr &dy, const NodePtr &x, const NodePtr &mean, const NodePtr &rstd,
                        const NodePtr &gamma_opt, const NodePtr &num_groups, const NodePtr &dx_is_require,
                        const NodePtr &dgamma_is_require, const NodePtr &dbeta_is_require) override;
  NodePtr GroupNorm(const NodePtr &input, const NodePtr &num_groups, const NodePtr &weight, const NodePtr &bias,
                    const NodePtr &eps) override;
  NodePtr Im2ColExt(const NodePtr &input, const NodePtr &kernel_size, const NodePtr &dilation, const NodePtr &padding,
                    const NodePtr &stride) override;
  NodePtr IndexAddExt(const NodePtr &input, const NodePtr &index, const NodePtr &source, const NodePtr &axis,
                      const NodePtr &alpha) override;
  NodePtr IndexSelect(const NodePtr &input, const NodePtr &dim, const NodePtr &index) override;
  NodePtr IsClose(const NodePtr &input, const NodePtr &other, const NodePtr &rtol, const NodePtr &atol,
                  const NodePtr &equal_nan) override;
  NodePtr IsFinite(const NodePtr &x) override;
  NodePtr LayerNormExt(const NodePtr &input, const NodePtr &normalized_shape, const NodePtr &weight,
                       const NodePtr &bias, const NodePtr &eps) override;
  NodePtr LayerNormGradExt(const NodePtr &dy, const NodePtr &x, const NodePtr &normalized_shape, const NodePtr &mean,
                           const NodePtr &variance, const NodePtr &gamma, const NodePtr &beta) override;
  NodePtr LeakyReLUExt(const NodePtr &input, const NodePtr &negative_slope) override;
  NodePtr LeakyReLUGradExt(const NodePtr &dy, const NodePtr &input, const NodePtr &negative_slope,
                           const NodePtr &is_result) override;
  NodePtr LinSpaceExt(const NodePtr &start, const NodePtr &end, const NodePtr &steps, const NodePtr &dtype) override;
  NodePtr LogicalAnd(const NodePtr &x, const NodePtr &y) override;
  NodePtr LogicalNot(const NodePtr &input) override;
  NodePtr LogicalOr(const NodePtr &x, const NodePtr &y) override;
  NodePtr MaskedFill(const NodePtr &input_x, const NodePtr &mask, const NodePtr &value) override;
  NodePtr MatrixInverseExt(const NodePtr &input) override;
  NodePtr Max(const NodePtr &input) override;
  NodePtr MaxPoolGradWithIndices(const NodePtr &x, const NodePtr &grad, const NodePtr &argmax,
                                 const NodePtr &kernel_size, const NodePtr &strides, const NodePtr &pads,
                                 const NodePtr &dilation, const NodePtr &ceil_mode,
                                 const NodePtr &argmax_type) override;
  NodePtr MaxPoolGradWithMask(const NodePtr &x, const NodePtr &grad, const NodePtr &mask, const NodePtr &kernel_size,
                              const NodePtr &strides, const NodePtr &pads, const NodePtr &dilation,
                              const NodePtr &ceil_mode, const NodePtr &argmax_type) override;
  NodePtr MaxPoolWithIndices(const NodePtr &x, const NodePtr &kernel_size, const NodePtr &strides, const NodePtr &pads,
                             const NodePtr &dilation, const NodePtr &ceil_mode, const NodePtr &argmax_type) override;
  NodePtr MaxPoolWithMask(const NodePtr &x, const NodePtr &kernel_size, const NodePtr &strides, const NodePtr &pads,
                          const NodePtr &dilation, const NodePtr &ceil_mode, const NodePtr &argmax_type) override;
  NodePtr Maximum(const NodePtr &input, const NodePtr &other) override;
  NodePtr MeanExt(const NodePtr &input, const NodePtr &axis, const NodePtr &keep_dims, const NodePtr &dtype) override;
  NodePtr Min(const NodePtr &input) override;
  NodePtr Minimum(const NodePtr &input, const NodePtr &other) override;
  NodePtr Mv(const NodePtr &input, const NodePtr &vec) override;
  NodePtr Neg(const NodePtr &input) override;
  NodePtr NonZeroExt(const NodePtr &input) override;
  NodePtr NonZero(const NodePtr &input) override;
  NodePtr Norm(const NodePtr &input_x, const NodePtr &ord, const NodePtr &dim, const NodePtr &keepdim,
               const NodePtr &dtype) override;
  NodePtr NormalFloatFloat(const NodePtr &mean, const NodePtr &std, const NodePtr &size, const NodePtr &seed,
                           const NodePtr &offset) override;
  NodePtr NormalFloatTensor(const NodePtr &mean, const NodePtr &std, const NodePtr &seed,
                            const NodePtr &offset) override;
  NodePtr NormalTensorFloat(const NodePtr &mean, const NodePtr &std, const NodePtr &seed,
                            const NodePtr &offset) override;
  NodePtr NormalTensorTensor(const NodePtr &mean, const NodePtr &std, const NodePtr &seed,
                             const NodePtr &offset) override;
  NodePtr OneHotExt(const NodePtr &tensor, const NodePtr &num_classes, const NodePtr &on_value,
                    const NodePtr &off_value, const NodePtr &axis) override;
  NodePtr OnesLikeExt(const NodePtr &input, const NodePtr &dtype) override;
  NodePtr Ones(const NodePtr &shape, const NodePtr &dtype) override;
  NodePtr ProdExt(const NodePtr &input, const NodePtr &axis, const NodePtr &keep_dims, const NodePtr &dtype) override;
  NodePtr RandExt(const NodePtr &shape, const NodePtr &seed, const NodePtr &offset, const NodePtr &dtype) override;
  NodePtr RandLikeExt(const NodePtr &tensor, const NodePtr &seed, const NodePtr &offset, const NodePtr &dtype) override;
  NodePtr Reciprocal(const NodePtr &x) override;
  NodePtr ReduceAll(const NodePtr &input, const NodePtr &axis, const NodePtr &keep_dims) override;
  NodePtr ReduceAny(const NodePtr &x, const NodePtr &axis, const NodePtr &keep_dims) override;
  NodePtr ReflectionPad1DGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &padding) override;
  NodePtr ReflectionPad1D(const NodePtr &input, const NodePtr &padding) override;
  NodePtr ReflectionPad2DGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &padding) override;
  NodePtr ReflectionPad2D(const NodePtr &input, const NodePtr &padding) override;
  NodePtr ReflectionPad3DGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &padding) override;
  NodePtr ReflectionPad3D(const NodePtr &input, const NodePtr &padding) override;
  NodePtr ReluGrad(const NodePtr &y_backprop, const NodePtr &x) override;
  NodePtr ReLU(const NodePtr &input) override;
  NodePtr RepeatInterleaveGrad(const NodePtr &input, const NodePtr &repeats, const NodePtr &dim) override;
  NodePtr RepeatInterleaveInt(const NodePtr &input, const NodePtr &repeats, const NodePtr &dim,
                              const NodePtr &output_size) override;
  NodePtr RepeatInterleaveTensor(const NodePtr &input, const NodePtr &repeats, const NodePtr &dim,
                                 const NodePtr &output_size) override;
  NodePtr ReplicationPad1DGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &padding) override;
  NodePtr ReplicationPad1D(const NodePtr &input, const NodePtr &padding) override;
  NodePtr ReplicationPad2DGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &padding) override;
  NodePtr ReplicationPad2D(const NodePtr &input, const NodePtr &padding) override;
  NodePtr ReplicationPad3DGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &padding) override;
  NodePtr ReplicationPad3D(const NodePtr &input, const NodePtr &padding) override;
  NodePtr ReverseV2(const NodePtr &input, const NodePtr &axis) override;
  NodePtr RmsNormGrad(const NodePtr &dy, const NodePtr &x, const NodePtr &rstd, const NodePtr &gamma) override;
  NodePtr RmsNorm(const NodePtr &x, const NodePtr &gamma, const NodePtr &epsilon) override;
  NodePtr Rsqrt(const NodePtr &input) override;
  NodePtr ScatterAddExt(const NodePtr &input, const NodePtr &dim, const NodePtr &index, const NodePtr &src) override;
  NodePtr Scatter(const NodePtr &input, const NodePtr &dim, const NodePtr &index, const NodePtr &src,
                  const NodePtr &reduce) override;
  NodePtr SearchSorted(const NodePtr &sorted_sequence, const NodePtr &values, const NodePtr &sorter,
                       const NodePtr &dtype, const NodePtr &right) override;
  NodePtr Select(const NodePtr &condition, const NodePtr &input, const NodePtr &other) override;
  NodePtr SigmoidGrad(const NodePtr &y, const NodePtr &dy) override;
  NodePtr Sigmoid(const NodePtr &input) override;
  NodePtr Sign(const NodePtr &input) override;
  NodePtr SiLUGrad(const NodePtr &dout, const NodePtr &x) override;
  NodePtr SiLU(const NodePtr &input) override;
  NodePtr Sin(const NodePtr &input) override;
  NodePtr SliceExt(const NodePtr &input, const NodePtr &dim, const NodePtr &start, const NodePtr &end,
                   const NodePtr &step) override;
  NodePtr SoftmaxBackward(const NodePtr &dout, const NodePtr &out, const NodePtr &dim) override;
  NodePtr Softmax(const NodePtr &input, const NodePtr &axis) override;
  NodePtr SoftplusExt(const NodePtr &input, const NodePtr &beta, const NodePtr &threshold) override;
  NodePtr SoftplusGradExt(const NodePtr &dout, const NodePtr &x, const NodePtr &beta,
                          const NodePtr &threshold) override;
  NodePtr SortExt(const NodePtr &input, const NodePtr &dim, const NodePtr &descending, const NodePtr &stable) override;
  NodePtr SplitTensor(const NodePtr &input_x, const NodePtr &split_int, const NodePtr &axis) override;
  NodePtr SplitWithSize(const NodePtr &input_x, const NodePtr &split_sections, const NodePtr &axis) override;
  NodePtr Sqrt(const NodePtr &x) override;
  NodePtr Square(const NodePtr &input) override;
  NodePtr StackExt(const NodePtr &tensors, const NodePtr &dim) override;
  NodePtr SubExt(const NodePtr &input, const NodePtr &other, const NodePtr &alpha) override;
  NodePtr SumExt(const NodePtr &input, const NodePtr &dim, const NodePtr &keepdim, const NodePtr &dtype) override;
  NodePtr TanhGrad(const NodePtr &y, const NodePtr &dy) override;
  NodePtr Tanh(const NodePtr &input) override;
  NodePtr Tile(const NodePtr &input, const NodePtr &dims) override;
  NodePtr TopkExt(const NodePtr &input, const NodePtr &k, const NodePtr &dim, const NodePtr &largest,
                  const NodePtr &sorted) override;
  NodePtr Triu(const NodePtr &input, const NodePtr &diagonal) override;
  NodePtr UniformExt(const NodePtr &tensor, const NodePtr &a, const NodePtr &b, const NodePtr &seed,
                     const NodePtr &offset) override;
  NodePtr Unique2(const NodePtr &input, const NodePtr &sorted, const NodePtr &return_inverse,
                  const NodePtr &return_counts) override;
  NodePtr UniqueDim(const NodePtr &input, const NodePtr &sorted, const NodePtr &return_inverse,
                    const NodePtr &dim) override;
  NodePtr UpsampleBilinear2DGrad(const NodePtr &dy, const NodePtr &input_size, const NodePtr &output_size,
                                 const NodePtr &scales, const NodePtr &align_corners) override;
  NodePtr UpsampleBilinear2D(const NodePtr &x, const NodePtr &output_size, const NodePtr &scales,
                             const NodePtr &align_corners) override;
  NodePtr UpsampleLinear1DGrad(const NodePtr &dy, const NodePtr &input_size, const NodePtr &output_size,
                               const NodePtr &scales, const NodePtr &align_corners) override;
  NodePtr UpsampleLinear1D(const NodePtr &x, const NodePtr &output_size, const NodePtr &scales,
                           const NodePtr &align_corners) override;
  NodePtr UpsampleNearest1DGrad(const NodePtr &dy, const NodePtr &input_size, const NodePtr &output_size,
                                const NodePtr &scales) override;
  NodePtr UpsampleNearest1D(const NodePtr &x, const NodePtr &output_size, const NodePtr &scales) override;
  NodePtr UpsampleNearest2DGrad(const NodePtr &dy, const NodePtr &input_size, const NodePtr &output_size,
                                const NodePtr &scales) override;
  NodePtr UpsampleNearest2D(const NodePtr &x, const NodePtr &output_size, const NodePtr &scales) override;
  NodePtr UpsampleNearest3DGrad(const NodePtr &dy, const NodePtr &input_size, const NodePtr &output_size,
                                const NodePtr &scales) override;
  NodePtr UpsampleNearest3D(const NodePtr &x, const NodePtr &output_size, const NodePtr &scales) override;
  NodePtr UpsampleTrilinear3DGrad(const NodePtr &dy, const NodePtr &input_size, const NodePtr &output_size,
                                  const NodePtr &scales, const NodePtr &align_corners) override;
  NodePtr UpsampleTrilinear3D(const NodePtr &x, const NodePtr &output_size, const NodePtr &scales,
                              const NodePtr &align_corners) override;
  NodePtr ZerosLikeExt(const NodePtr &input, const NodePtr &dtype) override;
  NodePtr Zeros(const NodePtr &size, const NodePtr &dtype) override;
  NodePtr DynamicQuantExt(const NodePtr &x, const NodePtr &smooth_scales) override;
  NodePtr GroupedMatmul(const NodePtr &x, const NodePtr &weight, const NodePtr &bias, const NodePtr &scale,
                        const NodePtr &offset, const NodePtr &antiquant_scale, const NodePtr &antiquant_offset,
                        const NodePtr &group_list, const NodePtr &split_item, const NodePtr &group_type) override;
  NodePtr MoeFinalizeRouting(const NodePtr &expanded_x, const NodePtr &x1, const NodePtr &x2, const NodePtr &bias,
                             const NodePtr &scales, const NodePtr &expanded_row_idx,
                             const NodePtr &expanded_expert_idx) override;
  NodePtr QuantBatchMatmul(const NodePtr &x1, const NodePtr &x2, const NodePtr &scale, const NodePtr &offset,
                           const NodePtr &bias, const NodePtr &transpose_x1, const NodePtr &transpose_x2,
                           const NodePtr &dtype) override;
  NodePtr QuantV2(const NodePtr &x, const NodePtr &scale, const NodePtr &offset, const NodePtr &sqrt_mode,
                  const NodePtr &rounding_mode, const NodePtr &dst_type) override;
  NodePtr WeightQuantBatchMatmul(const NodePtr &x, const NodePtr &weight, const NodePtr &antiquant_scale,
                                 const NodePtr &antiquant_offset, const NodePtr &quant_scale,
                                 const NodePtr &quant_offset, const NodePtr &bias, const NodePtr &transpose_x,
                                 const NodePtr &transpose_weight, const NodePtr &antiquant_group_size) override;

  // paas
  NodePtr BatchNormGrad(const NodePtrList &inputs, bool is_scale_or_bias_grad) override;
  NodePtr SparseSoftmaxCrossEntropyWithLogits(const NodePtrList &inputs, const expander::DAttr &attrs,
                                              const NodePtr &out, const NodePtr &dout, bool is_graph_mode) override;
  NodePtr Depend(const NodePtr &value, const NodePtr &expr) override;
  NodePtr TupleGetItem(const NodePtr &input, size_t i) override;
  NodePtr TupleGetItem(const NodePtr &input, const NodePtr &index) override;
  NodePtr MakeTuple(const NodePtrList &inputs) override;
  NodePtr MakeList(const NodePtrList &inputs) override;
  NodePtr Conditional(const NodePtr &cond, const BlockFunc &true_case, const BlockFunc &false_case) override;
  NodePtr ScalarEq(const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type) override;
  NodePtr OutZeros(const NodePtr &node) override;
  ValuePtr Ones(const ValuePtr &value);
  ValuePtr Zeros(const ValuePtr &value);
  ValuePtr Add(const ValuePtr &input, const ValuePtr &other);
  void SetInputs(std::string instance_name, const std::vector<NodePtr> *inputs,
                 mindspore::HashMap<std::string, ValuePtr> *attrs_ptr);
  ValuePtr FillZeros(const ValuePtr &value, const abstract::AbstractBasePtr &abs);

 private:
  NodePtrList FlattenNode(const NodePtr &input);
  std::string device_target_;
  bprop_pass::FuncPassForwardPtr pass_forward_;
};
using FuncBuilderPtr = std::shared_ptr<FuncBuilder>;
}  // namespace mindspore::pynative::autograd

#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_FUNC_BUILDER_H_
