/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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
#include "pynative/backward/op_grad/func_pass.h"
#include "include/frontend/expander/bprop_interface.h"

namespace mindspore::pynative::autograd {
using NodePtr = expander::NodePtr;
using NodePtrList = expander::NodePtrList;
using BpropBuilder = expander::bprop::BpropBuilder;

class FuncBuilder : public BpropBuilder {
 public:
  FuncBuilder(const std::string &name, device::DeviceType device_target,
              const expander::ExpanderInferPtr &infer = nullptr);
  ~FuncBuilder() override = default;
  NodePtr EmitOp(const PrimitivePtr &prim, const NodePtrList &inputs) override;
  NodePtr EmitValue(const ValuePtr &value) override;
  NodePtr Shape(const NodePtr &node, bool tensor = false) override;
  void MarkSharedGradTensor(const NodePtr &lhs, const NodePtr &rhs) override;
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
  // The second element y is a target tensor, not target shape!
  NodePtr BroadcastTo(const NodePtr &x, const NodePtr &y) override;
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
  NodePtr Concat(const NodePtr &tensors, const NodePtr &axis) override;
  NodePtr InplaceCopy(const NodePtr &variable, const NodePtr &value, bool non_blocking = false) override;
  NodePtr AsStrided(const NodePtr &input, const NodePtr &size, const NodePtr &stride,
                    const NodePtr &storage_offset) override;
  NodePtr FloorDiv(const NodePtr &input, const NodePtr &other) override;

  // Here is auto generate.
  NodePtr Ones(const NodePtr &shape, const NodePtr &dtype) override;
  NodePtr LerpScalar(const NodePtr &input, const NodePtr &end, const NodePtr &weight) override;
  NodePtr Atanh(const NodePtr &input) override;
  NodePtr ClampScalar(const NodePtr &input, const NodePtr &min, const NodePtr &max) override;
  NodePtr InplaceRandom(const NodePtr &input, const NodePtr &from_, const NodePtr &to, const NodePtr &seed,
                        const NodePtr &offset) override;
  NodePtr ClampTensor(const NodePtr &input, const NodePtr &min, const NodePtr &max) override;
  NodePtr Kthvalue(const NodePtr &input, const NodePtr &k, const NodePtr &dim, const NodePtr &keepdim) override;
  NodePtr CumsumExt(const NodePtr &input, const NodePtr &dim, const NodePtr &dtype) override;
  NodePtr SplitTensor(const NodePtr &input, const NodePtr &split_size, const NodePtr &dim) override;
  NodePtr InplaceUniform(const NodePtr &input, const NodePtr &from_, const NodePtr &to, const NodePtr &seed,
                         const NodePtr &offset) override;
  NodePtr RotaryPositionEmbeddingGrad(const NodePtr &dy, const NodePtr &cos, const NodePtr &sin, const NodePtr &dx,
                                      const NodePtr &mode) override;
  NodePtr KLDiv(const NodePtr &input, const NodePtr &target, const NodePtr &reduction,
                const NodePtr &log_target) override;
  NodePtr OnesLikeExt(const NodePtr &input, const NodePtr &dtype) override;
  NodePtr Embedding(const NodePtr &input, const NodePtr &weight, const NodePtr &padding_idx, const NodePtr &max_norm,
                    const NodePtr &norm_type, const NodePtr &scale_grad_by_freq) override;
  NodePtr SoftplusExt(const NodePtr &input, const NodePtr &beta, const NodePtr &threshold) override;
  NodePtr ViewAs(const NodePtr &input, const NodePtr &other) override;
  NodePtr Cosh(const NodePtr &input) override;
  NodePtr GroupNorm(const NodePtr &input, const NodePtr &num_groups, const NodePtr &weight, const NodePtr &bias,
                    const NodePtr &eps) override;
  NodePtr InnerIndex(const NodePtr &input, const NodePtr &indices) override;
  NodePtr InplaceIndexPut(const NodePtr &input, const NodePtr &indices, const NodePtr &values,
                          const NodePtr &accumulate) override;
  NodePtr AddRmsNorm(const NodePtr &x1, const NodePtr &x2, const NodePtr &gamma, const NodePtr &epsilon) override;
  NodePtr ReplicationPad3DGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &padding) override;
  NodePtr FlashAttentionScoreGrad(const NodePtr &query, const NodePtr &key, const NodePtr &value, const NodePtr &dy,
                                  const NodePtr &pse_shift, const NodePtr &drop_mask, const NodePtr &padding_mask,
                                  const NodePtr &atten_mask, const NodePtr &softmax_max, const NodePtr &softmax_sum,
                                  const NodePtr &softmax_in, const NodePtr &attention_in, const NodePtr &prefix,
                                  const NodePtr &actual_seq_qlen, const NodePtr &actual_seq_kvlen,
                                  const NodePtr &head_num, const NodePtr &keep_prob, const NodePtr &scale_value,
                                  const NodePtr &pre_tokens, const NodePtr &next_tokens, const NodePtr &inner_precise,
                                  const NodePtr &input_layout, const NodePtr &sparse_mode) override;
  NodePtr BitwiseNot(const NodePtr &input) override;
  NodePtr ConvolutionStr(const NodePtr &input, const NodePtr &weight, const NodePtr &bias, const NodePtr &stride,
                         const NodePtr &padding, const NodePtr &dilation, const NodePtr &transposed,
                         const NodePtr &output_padding, const NodePtr &groups) override;
  NodePtr LogSoftmax(const NodePtr &logits, const NodePtr &axis) override;
  NodePtr RemainderScalarTensor(const NodePtr &input, const NodePtr &other) override;
  NodePtr Addmm(const NodePtr &input, const NodePtr &mat1, const NodePtr &mat2, const NodePtr &beta,
                const NodePtr &alpha) override;
  NodePtr GridSampler3DGrad(const NodePtr &grad, const NodePtr &input_x, const NodePtr &grid,
                            const NodePtr &interpolation_mode, const NodePtr &padding_mode,
                            const NodePtr &align_corners, const NodePtr &output_mask) override;
  NodePtr MoeDistributeDispatch(const NodePtr &x, const NodePtr &expert_ids, const NodePtr &ep_world_size,
                                const NodePtr &ep_rank_id, const NodePtr &moe_expert_num, const NodePtr &expert_scales,
                                const NodePtr &scales, const NodePtr &x_active_mask, const NodePtr &group_ep,
                                const NodePtr &group_tp, const NodePtr &tp_world_size, const NodePtr &tp_rank_id,
                                const NodePtr &expert_shard_type, const NodePtr &shared_expert_num,
                                const NodePtr &shared_expert_rank_num, const NodePtr &quant_mode,
                                const NodePtr &global_bs, const NodePtr &expert_token_nums_type) override;
  NodePtr Polar(const NodePtr &abs, const NodePtr &angle) override;
  NodePtr Sqrt(const NodePtr &x) override;
  NodePtr TraceExt(const NodePtr &input) override;
  NodePtr Unique2(const NodePtr &input, const NodePtr &sorted, const NodePtr &return_inverse,
                  const NodePtr &return_counts) override;
  NodePtr LogSigmoidGrad(const NodePtr &dy, const NodePtr &input, const NodePtr &buffer) override;
  NodePtr BatchMatMulExt(const NodePtr &input, const NodePtr &mat2) override;
  NodePtr RepeatInterleaveGrad(const NodePtr &input, const NodePtr &repeats, const NodePtr &dim) override;
  NodePtr MoeTokenUnpermuteGrad(const NodePtr &permuted_tokens, const NodePtr &unpermuted_tokens_grad,
                                const NodePtr &sorted_indices, const NodePtr &probs, const NodePtr &padded_mode,
                                const NodePtr &restore_shape) override;
  NodePtr FillTensor(const NodePtr &size, const NodePtr &fill_value, const NodePtr &dtype) override;
  NodePtr AvgPool2DGrad(const NodePtr &grad, const NodePtr &image, const NodePtr &kernel_size, const NodePtr &stride,
                        const NodePtr &padding, const NodePtr &ceil_mode, const NodePtr &count_include_pad,
                        const NodePtr &divisor_override) override;
  NodePtr BitwiseXorTensor(const NodePtr &input, const NodePtr &other) override;
  NodePtr ReplicationPad2D(const NodePtr &input, const NodePtr &padding) override;
  NodePtr SigmoidGrad(const NodePtr &y, const NodePtr &dy) override;
  NodePtr AvgPool3DGradExt(const NodePtr &grad, const NodePtr &input, const NodePtr &kernel_size, const NodePtr &stride,
                           const NodePtr &padding, const NodePtr &ceil_mode, const NodePtr &count_include_pad,
                           const NodePtr &divisor_override) override;
  NodePtr RandExt(const NodePtr &shape, const NodePtr &seed, const NodePtr &offset, const NodePtr &dtype,
                  const NodePtr &device) override;
  NodePtr GreaterEqualScalar(const NodePtr &input, const NodePtr &other) override;
  NodePtr HSigmoidGrad(const NodePtr &grads, const NodePtr &input_x) override;
  NodePtr Swiglu(const NodePtr &input, const NodePtr &dim) override;
  NodePtr SplitWithSizeView(const NodePtr &input, const NodePtr &split_size, const NodePtr &dim) override;
  NodePtr Squeeze(const NodePtr &input, const NodePtr &axis) override;
  NodePtr UpsampleNearest2D(const NodePtr &x, const NodePtr &output_size, const NodePtr &scales) override;
  NodePtr Sin(const NodePtr &input) override;
  NodePtr TopkExt(const NodePtr &input, const NodePtr &k, const NodePtr &dim, const NodePtr &largest,
                  const NodePtr &sorted) override;
  NodePtr BinaryCrossEntropyGrad(const NodePtr &input, const NodePtr &target, const NodePtr &grad_output,
                                 const NodePtr &weight, const NodePtr &reduction) override;
  NodePtr SwigluGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &dim) override;
  NodePtr InplaceScatterValue(const NodePtr &input, const NodePtr &dim, const NodePtr &index,
                              const NodePtr &value) override;
  NodePtr InplaceReLU(const NodePtr &input) override;
  NodePtr SiLU(const NodePtr &input) override;
  NodePtr AddLayerNormGrad(const NodePtr &dy, const NodePtr &x1, const NodePtr &x2, const NodePtr &rstd,
                           const NodePtr &mean, const NodePtr &gamma, const NodePtr &dsumOptional) override;
  NodePtr HShrink(const NodePtr &input, const NodePtr &lambd) override;
  NodePtr Take(const NodePtr &input, const NodePtr &index) override;
  NodePtr Std(const NodePtr &input, const NodePtr &dim, const NodePtr &correction, const NodePtr &keepdim) override;
  NodePtr InplaceErfinv(const NodePtr &input) override;
  NodePtr ToDevice(const NodePtr &input, const NodePtr &device, const NodePtr &dtype, const NodePtr &non_blocking,
                   const NodePtr &copy) override;
  NodePtr FmodTensor(const NodePtr &input, const NodePtr &other) override;
  NodePtr MaskedFill(const NodePtr &input_x, const NodePtr &mask, const NodePtr &value) override;
  NodePtr InplaceTanh(const NodePtr &input) override;
  NodePtr Expm1(const NodePtr &input) override;
  NodePtr InplaceMaskedScatter(const NodePtr &input, const NodePtr &mask, const NodePtr &source) override;
  NodePtr Neg(const NodePtr &input) override;
  NodePtr Tile(const NodePtr &input, const NodePtr &dims) override;
  NodePtr InplaceBernoulliTensor(const NodePtr &input, const NodePtr &p, const NodePtr &seed,
                                 const NodePtr &offset) override;
  NodePtr DiagonalView(const NodePtr &input, const NodePtr &offset, const NodePtr &dim1, const NodePtr &dim2) override;
  NodePtr FillScalar(const NodePtr &size, const NodePtr &fill_value, const NodePtr &dtype) override;
  NodePtr AdaptiveMaxPool1D(const NodePtr &input, const NodePtr &output_size) override;
  NodePtr LinalgQr(const NodePtr &A, const NodePtr &mode) override;
  NodePtr ArgMinWithValue(const NodePtr &input, const NodePtr &axis, const NodePtr &keep_dims) override;
  NodePtr L1LossBackwardExt(const NodePtr &grad_output, const NodePtr &input, const NodePtr &target,
                            const NodePtr &reduction) override;
  NodePtr ReflectionPad2D(const NodePtr &input, const NodePtr &padding) override;
  NodePtr LogicalXor(const NodePtr &input, const NodePtr &other) override;
  NodePtr Cummax(const NodePtr &input, const NodePtr &axis) override;
  NodePtr Minimum(const NodePtr &input, const NodePtr &other) override;
  NodePtr AdaptiveAvgPool2DExt(const NodePtr &input, const NodePtr &output_size) override;
  NodePtr GatherDGradV2(const NodePtr &x, const NodePtr &dim, const NodePtr &index, const NodePtr &dout) override;
  NodePtr SmoothL1Loss(const NodePtr &prediction, const NodePtr &target, const NodePtr &beta,
                       const NodePtr &reduction) override;
  NodePtr CumminExt(const NodePtr &input, const NodePtr &dim) override;
  NodePtr BCEWithLogitsLoss(const NodePtr &input, const NodePtr &target, const NodePtr &weight,
                            const NodePtr &posWeight, const NodePtr &reduction) override;
  NodePtr BroadcastToView(const NodePtr &input, const NodePtr &shape) override;
  NodePtr RandLikeExt(const NodePtr &tensor, const NodePtr &seed, const NodePtr &offset, const NodePtr &dtype,
                      const NodePtr &device) override;
  NodePtr InplaceExp(const NodePtr &input) override;
  NodePtr BitwiseAndTensor(const NodePtr &input, const NodePtr &other) override;
  NodePtr UpsampleNearest3DGrad(const NodePtr &dy, const NodePtr &input_size, const NodePtr &output_size,
                                const NodePtr &scales) override;
  NodePtr MultiScaleDeformableAttn(const NodePtr &value, const NodePtr &shape, const NodePtr &offset,
                                   const NodePtr &locations, const NodePtr &weight) override;
  NodePtr LogicalOr(const NodePtr &x, const NodePtr &y) override;
  NodePtr MaxPoolWithMask(const NodePtr &x, const NodePtr &kernel_size, const NodePtr &strides, const NodePtr &pads,
                          const NodePtr &dilation, const NodePtr &ceil_mode, const NodePtr &argmax_type) override;
  NodePtr InplaceFloorDivides(const NodePtr &input, const NodePtr &other) override;
  NodePtr ScatterAddExt(const NodePtr &input, const NodePtr &dim, const NodePtr &index, const NodePtr &src) override;
  NodePtr ReflectionPad3D(const NodePtr &input, const NodePtr &padding) override;
  NodePtr HSwishGrad(const NodePtr &y_grad, const NodePtr &x) override;
  NodePtr FlattenExt(const NodePtr &input, const NodePtr &start_dim, const NodePtr &end_dim) override;
  NodePtr Square(const NodePtr &input) override;
  NodePtr Addbmm(const NodePtr &input, const NodePtr &batch1, const NodePtr &batch2, const NodePtr &beta,
                 const NodePtr &alpha) override;
  NodePtr Arange(const NodePtr &start, const NodePtr &end, const NodePtr &step, const NodePtr &dtype) override;
  NodePtr InplaceIndexFillTensor(const NodePtr &input, const NodePtr &dim, const NodePtr &index,
                                 const NodePtr &value) override;
  NodePtr Round(const NodePtr &input, const NodePtr &decimals) override;
  NodePtr SliceExtView(const NodePtr &input, const NodePtr &dim, const NodePtr &start, const NodePtr &end,
                       const NodePtr &step) override;
  NodePtr ArgMinExt(const NodePtr &input, const NodePtr &dim, const NodePtr &keepdim) override;
  NodePtr ReplicationPad1DGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &padding) override;
  NodePtr MaskedSelectGrad(const NodePtr &input, const NodePtr &mask, const NodePtr &grad) override;
  NodePtr SubExt(const NodePtr &input, const NodePtr &other, const NodePtr &alpha) override;
  NodePtr InnerMoeTokenUnpermute(const NodePtr &permuted_tokens, const NodePtr &sorted_indices, const NodePtr &probs,
                                 const NodePtr &padded_mode, const NodePtr &restore_shape) override;
  NodePtr SelectExtView(const NodePtr &input, const NodePtr &dim, const NodePtr &index) override;
  NodePtr InplaceMaskedFillTensor(const NodePtr &input, const NodePtr &mask, const NodePtr &value) override;
  NodePtr InplaceDivMod(const NodePtr &input, const NodePtr &other, const NodePtr &rounding_mode) override;
  NodePtr NormalFloatTensor(const NodePtr &mean, const NodePtr &std, const NodePtr &seed,
                            const NodePtr &offset) override;
  NodePtr SplitTensorView(const NodePtr &input, const NodePtr &split_size, const NodePtr &dim) override;
  NodePtr ReflectionPad2DGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &padding) override;
  NodePtr Sign(const NodePtr &input) override;
  NodePtr Narrow(const NodePtr &input, const NodePtr &dim, const NodePtr &start, const NodePtr &length) override;
  NodePtr GridSampler3D(const NodePtr &input_x, const NodePtr &grid, const NodePtr &interpolation_mode,
                        const NodePtr &padding_mode, const NodePtr &align_corners) override;
  NodePtr AddLayerNormV2(const NodePtr &x1, const NodePtr &x2, const NodePtr &gamma, const NodePtr &beta,
                         const NodePtr &epsilon, const NodePtr &additionalOut) override;
  NodePtr IsInf(const NodePtr &input) override;
  NodePtr InplaceIndexAddExt(const NodePtr &input, const NodePtr &dim, const NodePtr &index, const NodePtr &source,
                             const NodePtr &alpha) override;
  NodePtr BatchNormGradExt(const NodePtr &dout, const NodePtr &input, const NodePtr &weight,
                           const NodePtr &running_mean, const NodePtr &running_var, const NodePtr &saved_mean,
                           const NodePtr &saved_rstd, const NodePtr &training, const NodePtr &eps,
                           const NodePtr &output_mask) override;
  NodePtr DivMod(const NodePtr &input, const NodePtr &other, const NodePtr &rounding_mode) override;
  NodePtr Slice(const NodePtr &input, const NodePtr &begin, const NodePtr &size) override;
  NodePtr RandIntLike(const NodePtr &input, const NodePtr &low, const NodePtr &high, const NodePtr &seed,
                      const NodePtr &offset, const NodePtr &dtype, const NodePtr &device) override;
  NodePtr MaxPoolGradWithIndices(const NodePtr &x, const NodePtr &grad, const NodePtr &argmax,
                                 const NodePtr &kernel_size, const NodePtr &strides, const NodePtr &pads,
                                 const NodePtr &dilation, const NodePtr &ceil_mode,
                                 const NodePtr &argmax_type) override;
  NodePtr RemainderTensorScalar(const NodePtr &input, const NodePtr &other) override;
  NodePtr FmodScalar(const NodePtr &input, const NodePtr &other) override;
  NodePtr Randn(const NodePtr &shape, const NodePtr &seed, const NodePtr &offset, const NodePtr &dtype,
                const NodePtr &device) override;
  NodePtr BitwiseXorScalar(const NodePtr &input, const NodePtr &other) override;
  NodePtr UpsampleTrilinear3D(const NodePtr &x, const NodePtr &output_size, const NodePtr &scales,
                              const NodePtr &align_corners) override;
  NodePtr ArgMaxWithValue(const NodePtr &input, const NodePtr &axis, const NodePtr &keep_dims) override;
  NodePtr InplaceFloor(const NodePtr &input) override;
  NodePtr UnstackExtView(const NodePtr &input, const NodePtr &dim) override;
  NodePtr InplaceFloorDivide(const NodePtr &input, const NodePtr &other) override;
  NodePtr InplaceSubExt(const NodePtr &input, const NodePtr &other, const NodePtr &alpha) override;
  NodePtr GeLU(const NodePtr &input) override;
  NodePtr ReplicationPad1D(const NodePtr &input, const NodePtr &padding) override;
  NodePtr InplaceCopy(const NodePtr &input, const NodePtr &src, const NodePtr &non_blocking) override;
  NodePtr Baddbmm(const NodePtr &input, const NodePtr &batch1, const NodePtr &batch2, const NodePtr &beta,
                  const NodePtr &alpha) override;
  NodePtr ExpandDims(const NodePtr &input_x, const NodePtr &axis) override;
  NodePtr LeakyReLUExt(const NodePtr &input, const NodePtr &negative_slope) override;
  NodePtr UniqueDim(const NodePtr &input, const NodePtr &sorted, const NodePtr &return_inverse,
                    const NodePtr &dim) override;
  NodePtr HistcExt(const NodePtr &input, const NodePtr &bins, const NodePtr &min, const NodePtr &max) override;
  NodePtr IncreFlashAttention(const NodePtr &query, const NodePtr &key, const NodePtr &value, const NodePtr &attn_mask,
                              const NodePtr &actual_seq_lengths, const NodePtr &pse_shift,
                              const NodePtr &dequant_scale1, const NodePtr &quant_scale1, const NodePtr &dequant_scale2,
                              const NodePtr &quant_scale2, const NodePtr &quant_offset2, const NodePtr &antiquant_scale,
                              const NodePtr &antiquant_offset, const NodePtr &block_table,
                              const NodePtr &kv_padding_size, const NodePtr &num_heads, const NodePtr &input_layout,
                              const NodePtr &scale_value, const NodePtr &num_key_value_heads, const NodePtr &block_size,
                              const NodePtr &inner_precise) override;
  NodePtr Log10(const NodePtr &input) override;
  NodePtr EmbeddingDenseBackward(const NodePtr &grad, const NodePtr &indices, const NodePtr &num_weights,
                                 const NodePtr &padding_idx, const NodePtr &scale_grad_by_freq) override;
  NodePtr Mm(const NodePtr &input, const NodePtr &mat2) override;
  NodePtr Col2ImExt(const NodePtr &input, const NodePtr &output_size, const NodePtr &kernel_size,
                    const NodePtr &dilation, const NodePtr &padding, const NodePtr &stride) override;
  NodePtr GeLUGrad(const NodePtr &dy, const NodePtr &x, const NodePtr &y) override;
  NodePtr OneHotExt(const NodePtr &tensor, const NodePtr &num_classes, const NodePtr &on_value,
                    const NodePtr &off_value, const NodePtr &axis) override;
  NodePtr SiLUGrad(const NodePtr &dout, const NodePtr &x) override;
  NodePtr ConvolutionStrGrad(const NodePtr &dout, const NodePtr &input, const NodePtr &weight, const NodePtr &bias,
                             const NodePtr &stride, const NodePtr &padding, const NodePtr &dilation,
                             const NodePtr &transposed, const NodePtr &output_padding, const NodePtr &groups,
                             const NodePtr &output_mask) override;
  NodePtr InplaceDivMods(const NodePtr &input, const NodePtr &other, const NodePtr &rounding_mode) override;
  NodePtr SortExt(const NodePtr &input, const NodePtr &dim, const NodePtr &descending, const NodePtr &stable) override;
  NodePtr Generator(const NodePtr &cmd, const NodePtr &inputs) override;
  NodePtr LinSpaceExt(const NodePtr &start, const NodePtr &end, const NodePtr &steps, const NodePtr &dtype) override;
  NodePtr InnerUnique(const NodePtr &input, const NodePtr &sorted, const NodePtr &return_inverse) override;
  NodePtr AddcdivExt(const NodePtr &input, const NodePtr &tensor1, const NodePtr &tensor2,
                     const NodePtr &value) override;
  NodePtr LogAddExp2(const NodePtr &input, const NodePtr &other) override;
  NodePtr ThresholdGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &threshold) override;
  NodePtr LogSoftmaxExt(const NodePtr &input, const NodePtr &dim, const NodePtr &dtype) override;
  NodePtr PowScalarTensor(const NodePtr &input, const NodePtr &exponent) override;
  NodePtr AvgPool3DExt(const NodePtr &input, const NodePtr &kernel_size, const NodePtr &stride, const NodePtr &padding,
                       const NodePtr &ceil_mode, const NodePtr &count_include_pad,
                       const NodePtr &divisor_override) override;
  NodePtr InplaceFillDiagonal(const NodePtr &input, const NodePtr &fill_value, const NodePtr &wrap) override;
  NodePtr Col2ImGrad(const NodePtr &input, const NodePtr &kernel_size, const NodePtr &dilation, const NodePtr &padding,
                     const NodePtr &stride) override;
  NodePtr AllGatherMatmul(const NodePtr &input, const NodePtr &x2, const NodePtr &group, const NodePtr &world_size,
                          const NodePtr &bias, const NodePtr &gather_index, const NodePtr &gather_output,
                          const NodePtr &comm_turn, const NodePtr &trans_input, const NodePtr &trans_x2) override;
  NodePtr MaxUnpool2DExt(const NodePtr &input, const NodePtr &indices, const NodePtr &kernel_size,
                         const NodePtr &stride, const NodePtr &padding, const NodePtr &output_size) override;
  NodePtr InplaceGroupedMatmulAdd(const NodePtr &x, const NodePtr &weight, const NodePtr &group_list,
                                  const NodePtr &out) override;
  NodePtr MaxPoolWithIndices(const NodePtr &x, const NodePtr &kernel_size, const NodePtr &strides, const NodePtr &pads,
                             const NodePtr &dilation, const NodePtr &ceil_mode, const NodePtr &argmax_type) override;
  NodePtr SoftmaxBackward(const NodePtr &dout, const NodePtr &out, const NodePtr &dim) override;
  NodePtr MatrixInverseExt(const NodePtr &input) override;
  NodePtr Tanh(const NodePtr &input) override;
  NodePtr DropoutGradExt(const NodePtr &input, const NodePtr &mask, const NodePtr &p) override;
  NodePtr InnerNonZero(const NodePtr &input) override;
  NodePtr AllFinite(const NodePtr &tensors) override;
  NodePtr ReshapeAndCache(const NodePtr &key, const NodePtr &value, const NodePtr &key_cache,
                          const NodePtr &value_cache, const NodePtr &slot_mapping) override;
  NodePtr InplaceClampScalar(const NodePtr &input, const NodePtr &min, const NodePtr &max) override;
  NodePtr NewOnes(const NodePtr &input, const NodePtr &size, const NodePtr &dtype) override;
  NodePtr Dot(const NodePtr &input, const NodePtr &other) override;
  NodePtr InplaceAddExt(const NodePtr &input, const NodePtr &other, const NodePtr &alpha) override;
  NodePtr XLogYScalarOther(const NodePtr &input, const NodePtr &other) override;
  NodePtr AvgPool1D(const NodePtr &input, const NodePtr &kernel_size, const NodePtr &stride, const NodePtr &padding,
                    const NodePtr &ceil_mode, const NodePtr &count_include_pad) override;
  NodePtr RotaryPositionEmbedding(const NodePtr &x, const NodePtr &cos, const NodePtr &sin,
                                  const NodePtr &mode) override;
  NodePtr RmsNorm(const NodePtr &x, const NodePtr &gamma, const NodePtr &epsilon) override;
  NodePtr InplaceZero(const NodePtr &input) override;
  NodePtr ExpandDimsView(const NodePtr &input, const NodePtr &dim) override;
  NodePtr Outer(const NodePtr &input, const NodePtr &vec2) override;
  NodePtr InplaceLog(const NodePtr &input) override;
  NodePtr ToOther(const NodePtr &input, const NodePtr &other, const NodePtr &non_blocking,
                  const NodePtr &copy) override;
  NodePtr InplaceAddmm(const NodePtr &input, const NodePtr &mat1, const NodePtr &mat2, const NodePtr &beta,
                       const NodePtr &alpha) override;
  NodePtr InplaceThreshold(const NodePtr &input, const NodePtr &threshold, const NodePtr &value) override;
  NodePtr IsClose(const NodePtr &input, const NodePtr &other, const NodePtr &rtol, const NodePtr &atol,
                  const NodePtr &equal_nan) override;
  NodePtr GridSampler2DGrad(const NodePtr &grad, const NodePtr &input_x, const NodePtr &grid,
                            const NodePtr &interpolation_mode, const NodePtr &padding_mode,
                            const NodePtr &align_corners, const NodePtr &output_mask) override;
  NodePtr ReflectionPad1D(const NodePtr &input, const NodePtr &padding) override;
  NodePtr InplaceIndexCopy(const NodePtr &input, const NodePtr &dim, const NodePtr &index,
                           const NodePtr &tensor) override;
  NodePtr InplaceStopGradient(const NodePtr &input) override;
  NodePtr BernoulliExt(const NodePtr &input, const NodePtr &seed, const NodePtr &offset) override;
  NodePtr InplaceDiv(const NodePtr &input, const NodePtr &other) override;
  NodePtr Log1p(const NodePtr &input) override;
  NodePtr SubScalar(const NodePtr &input, const NodePtr &other, const NodePtr &alpha) override;
  NodePtr Addmv(const NodePtr &input, const NodePtr &mat, const NodePtr &vec, const NodePtr &beta,
                const NodePtr &alpha) override;
  NodePtr SearchSorted(const NodePtr &sorted_sequence, const NodePtr &values, const NodePtr &sorter,
                       const NodePtr &dtype, const NodePtr &right) override;
  NodePtr UpsampleBicubic2D(const NodePtr &x, const NodePtr &output_size, const NodePtr &scales,
                            const NodePtr &align_corners) override;
  NodePtr GatherD(const NodePtr &x, const NodePtr &dim, const NodePtr &index) override;
  NodePtr Scatter(const NodePtr &input, const NodePtr &dim, const NodePtr &index, const NodePtr &src,
                  const NodePtr &reduce) override;
  NodePtr AcoshExt(const NodePtr &input) override;
  NodePtr Convolution(const NodePtr &input, const NodePtr &weight, const NodePtr &bias, const NodePtr &stride,
                      const NodePtr &padding, const NodePtr &dilation, const NodePtr &transposed,
                      const NodePtr &output_padding, const NodePtr &groups) override;
  NodePtr Chunk(const NodePtr &input, const NodePtr &chunks, const NodePtr &dim) override;
  NodePtr Clone(const NodePtr &input) override;
  NodePtr ReLU(const NodePtr &input) override;
  NodePtr VarMean(const NodePtr &input, const NodePtr &dim, const NodePtr &correction, const NodePtr &keepdim) override;
  NodePtr InplaceFillScalar(const NodePtr &input, const NodePtr &value) override;
  NodePtr MultinomialExt(const NodePtr &input, const NodePtr &num_samples, const NodePtr &replacement,
                         const NodePtr &seed, const NodePtr &offset) override;
  NodePtr MishGradExt(const NodePtr &dout, const NodePtr &x) override;
  NodePtr ReduceMax(const NodePtr &x, const NodePtr &axis, const NodePtr &keep_dims) override;
  NodePtr ArgSort(const NodePtr &input, const NodePtr &dim, const NodePtr &descending, const NodePtr &stable) override;
  NodePtr GeluGradExt(const NodePtr &grad, const NodePtr &input, const NodePtr &approximate) override;
  NodePtr BinaryCrossEntropyWithLogitsBackward(const NodePtr &grad_output, const NodePtr &input, const NodePtr &target,
                                               const NodePtr &weight, const NodePtr &posWeight,
                                               const NodePtr &reduction) override;
  NodePtr LinalgVectorNorm(const NodePtr &x, const NodePtr &ord, const NodePtr &dim, const NodePtr &keepdim,
                           const NodePtr &dtype) override;
  NodePtr Norm(const NodePtr &input, const NodePtr &p, const NodePtr &dim, const NodePtr &keepdim,
               const NodePtr &dtype) override;
  NodePtr BatchNormElemtGrad(const NodePtr &dout, const NodePtr &input, const NodePtr &mean, const NodePtr &invstd,
                             const NodePtr &weight, const NodePtr &sumd_dy, const NodePtr &sum_dy_xmu,
                             const NodePtr &count) override;
  NodePtr RepeatInterleaveTensor(const NodePtr &input, const NodePtr &repeats, const NodePtr &dim,
                                 const NodePtr &output_size) override;
  NodePtr TrilExt(const NodePtr &input, const NodePtr &diagonal) override;
  NodePtr PReLUGrad(const NodePtr &dy, const NodePtr &x, const NodePtr &weight) override;
  NodePtr InplaceScatterSrcReduce(const NodePtr &input, const NodePtr &dim, const NodePtr &index, const NodePtr &src,
                                  const NodePtr &reduce) override;
  NodePtr AdaptiveAvgPool3DGradExt(const NodePtr &input_grad, const NodePtr &input) override;
  NodePtr BitwiseOrScalar(const NodePtr &input, const NodePtr &other) override;
  NodePtr InplaceNormal(const NodePtr &input, const NodePtr &mean, const NodePtr &std, const NodePtr &seed,
                        const NodePtr &offset) override;
  NodePtr CountNonZero(const NodePtr &input, const NodePtr &dim) override;
  NodePtr EqualExt(const NodePtr &input, const NodePtr &other) override;
  NodePtr StdMean(const NodePtr &input, const NodePtr &dim, const NodePtr &correction, const NodePtr &keepdim) override;
  NodePtr BatchNormReduceGrad(const NodePtr &dout, const NodePtr &input, const NodePtr &mean, const NodePtr &invstd,
                              const NodePtr &weight, const NodePtr &input_g, const NodePtr &weight_g,
                              const NodePtr &bias_g) override;
  NodePtr GroupNormGrad(const NodePtr &dy, const NodePtr &x, const NodePtr &mean, const NodePtr &rstd,
                        const NodePtr &gamma_opt, const NodePtr &num_groups, const NodePtr &dx_is_require,
                        const NodePtr &dgamma_is_require, const NodePtr &dbeta_is_require) override;
  NodePtr TanhGrad(const NodePtr &y, const NodePtr &dy) override;
  NodePtr MaskedScatter(const NodePtr &input, const NodePtr &mask, const NodePtr &source) override;
  NodePtr Exp(const NodePtr &input) override;
  NodePtr BitwiseOrTensor(const NodePtr &input, const NodePtr &other) override;
  NodePtr NLLLoss2d(const NodePtr &input, const NodePtr &target, const NodePtr &weight, const NodePtr &reduction,
                    const NodePtr &ignore_index) override;
  NodePtr BatchNormElemt(const NodePtr &input, const NodePtr &weight, const NodePtr &bias, const NodePtr &mean,
                         const NodePtr &invstd, const NodePtr &eps) override;
  NodePtr Hardtanh(const NodePtr &input, const NodePtr &min_val, const NodePtr &max_val) override;
  NodePtr Exp2(const NodePtr &input) override;
  NodePtr Cos(const NodePtr &input) override;
  NodePtr SmoothL1LossGrad(const NodePtr &prediction, const NodePtr &target, const NodePtr &dout, const NodePtr &beta,
                           const NodePtr &reduction) override;
  NodePtr MishExt(const NodePtr &input) override;
  NodePtr Select(const NodePtr &condition, const NodePtr &input, const NodePtr &other) override;
  NodePtr TransposeView(const NodePtr &input, const NodePtr &input_perm) override;
  NodePtr AddExt(const NodePtr &input, const NodePtr &other, const NodePtr &alpha) override;
  NodePtr TransposeExtView(const NodePtr &input, const NodePtr &dim0, const NodePtr &dim1) override;
  NodePtr ZerosLikeExt(const NodePtr &input, const NodePtr &dtype) override;
  NodePtr NewZeros(const NodePtr &input, const NodePtr &size, const NodePtr &dtype) override;
  NodePtr Roll(const NodePtr &input, const NodePtr &shifts, const NodePtr &dims) override;
  NodePtr InplaceClampTensor(const NodePtr &input, const NodePtr &min, const NodePtr &max) override;
  NodePtr ExpandAs(const NodePtr &input, const NodePtr &other) override;
  NodePtr Conv1DExt(const NodePtr &input, const NodePtr &weight, const NodePtr &bias, const NodePtr &stride,
                    const NodePtr &padding, const NodePtr &dilation, const NodePtr &groups) override;
  NodePtr ReflectionPad3DGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &padding) override;
  NodePtr AvgPool2D(const NodePtr &input, const NodePtr &kernel_size, const NodePtr &stride, const NodePtr &padding,
                    const NodePtr &ceil_mode, const NodePtr &count_include_pad,
                    const NodePtr &divisor_override) override;
  NodePtr FlashAttentionScore(const NodePtr &query, const NodePtr &key, const NodePtr &value, const NodePtr &real_shift,
                              const NodePtr &drop_mask, const NodePtr &padding_mask, const NodePtr &attn_mask,
                              const NodePtr &prefix, const NodePtr &actual_seq_qlen, const NodePtr &actual_seq_kvlen,
                              const NodePtr &head_num, const NodePtr &keep_prob, const NodePtr &scale_value,
                              const NodePtr &pre_tokens, const NodePtr &next_tokens, const NodePtr &inner_precise,
                              const NodePtr &input_layout, const NodePtr &sparse_mode) override;
  NodePtr BatchNormGatherStatsWithCounts(const NodePtr &input, const NodePtr &mean, const NodePtr &invstd,
                                         const NodePtr &running_mean, const NodePtr &running_var,
                                         const NodePtr &momentum, const NodePtr &eps, const NodePtr &counts) override;
  NodePtr AtanExt(const NodePtr &input) override;
  NodePtr Log2(const NodePtr &input) override;
  NodePtr RandpermExt(const NodePtr &n, const NodePtr &seed, const NodePtr &offset, const NodePtr &dtype) override;
  NodePtr LogAddExp(const NodePtr &input, const NodePtr &other) override;
  NodePtr LogSigmoid(const NodePtr &input) override;
  NodePtr XLogYScalarSelf(const NodePtr &input, const NodePtr &other) override;
  NodePtr TriangularSolve(const NodePtr &b, const NodePtr &A, const NodePtr &upper, const NodePtr &transpose,
                          const NodePtr &unitriangular) override;
  NodePtr SpeedFusionAttention(const NodePtr &query, const NodePtr &key, const NodePtr &value, const NodePtr &head_num,
                               const NodePtr &input_layout, const NodePtr &seed, const NodePtr &offset,
                               const NodePtr &pse, const NodePtr &padding_mask, const NodePtr &atten_mask,
                               const NodePtr &scale, const NodePtr &keep_prob, const NodePtr &pre_tokens,
                               const NodePtr &next_tokens, const NodePtr &inner_precise, const NodePtr &prefix,
                               const NodePtr &actual_seq_qlen, const NodePtr &actual_seq_kvlen,
                               const NodePtr &sparse_mode, const NodePtr &gen_mask_parallel, const NodePtr &sync,
                               const NodePtr &pse_type, const NodePtr &q_start_idx,
                               const NodePtr &kv_start_idx) override;
  NodePtr GluGrad(const NodePtr &grads, const NodePtr &x, const NodePtr &axis) override;
  NodePtr IsNegInf(const NodePtr &input) override;
  NodePtr DropoutGenMaskExt(const NodePtr &shape, const NodePtr &p, const NodePtr &seed, const NodePtr &offset,
                            const NodePtr &dtype) override;
  NodePtr HShrinkGrad(const NodePtr &gradients, const NodePtr &features, const NodePtr &lambd) override;
  NodePtr EmptyLike(const NodePtr &input, const NodePtr &dtype, const NodePtr &device,
                    const NodePtr &pin_memory) override;
  NodePtr MeanExt(const NodePtr &input, const NodePtr &dim, const NodePtr &keepdim, const NodePtr &dtype) override;
  NodePtr InplaceScatterAdd(const NodePtr &input, const NodePtr &dim, const NodePtr &index,
                            const NodePtr &src) override;
  NodePtr InplaceMul(const NodePtr &input, const NodePtr &other) override;
  NodePtr LayerNormExt(const NodePtr &input, const NodePtr &normalized_shape, const NodePtr &weight,
                       const NodePtr &bias, const NodePtr &eps) override;
  NodePtr LogicalAnd(const NodePtr &x, const NodePtr &y) override;
  NodePtr Divs(const NodePtr &input, const NodePtr &other) override;
  NodePtr InnerInplaceIndexPut(const NodePtr &input, const NodePtr &indices, const NodePtr &values,
                               const NodePtr &accumulate) override;
  NodePtr InplaceIndexFillScalar(const NodePtr &input, const NodePtr &dim, const NodePtr &index,
                                 const NodePtr &value) override;
  NodePtr NormalTensorFloat(const NodePtr &mean, const NodePtr &std, const NodePtr &seed,
                            const NodePtr &offset) override;
  NodePtr AdaptiveAvgPool2DGradExt(const NodePtr &grad_output, const NodePtr &x) override;
  NodePtr ProdExt(const NodePtr &input, const NodePtr &dim, const NodePtr &keepdim, const NodePtr &dtype) override;
  NodePtr Softmax(const NodePtr &input, const NodePtr &axis) override;
  NodePtr InplaceElu(const NodePtr &input, const NodePtr &alpha) override;
  NodePtr NeScalar(const NodePtr &input, const NodePtr &other) override;
  NodePtr Conv2DExt(const NodePtr &input, const NodePtr &weight, const NodePtr &bias, const NodePtr &stride,
                    const NodePtr &padding, const NodePtr &dilation, const NodePtr &groups) override;
  NodePtr RandnLike(const NodePtr &input, const NodePtr &seed, const NodePtr &offset, const NodePtr &dtype,
                    const NodePtr &device) override;
  NodePtr Conv3DPadding(const NodePtr &input, const NodePtr &weight, const NodePtr &bias, const NodePtr &stride,
                        const NodePtr &padding, const NodePtr &dilation, const NodePtr &groups) override;
  NodePtr Ceil(const NodePtr &input) override;
  NodePtr EluGradExt(const NodePtr &dout, const NodePtr &x_or_out, const NodePtr &alpha,
                     const NodePtr &is_result) override;
  NodePtr TypeAs(const NodePtr &input, const NodePtr &other) override;
  NodePtr BatchNormStats(const NodePtr &input, const NodePtr &eps) override;
  NodePtr MaxDim(const NodePtr &input, const NodePtr &dim, const NodePtr &keepdim) override;
  NodePtr FFNExt(const NodePtr &x, const NodePtr &weight1, const NodePtr &weight2, const NodePtr &expertTokens,
                 const NodePtr &bias1, const NodePtr &bias2, const NodePtr &scale, const NodePtr &offset,
                 const NodePtr &deqScale1, const NodePtr &deqScale2, const NodePtr &antiquant_scale1,
                 const NodePtr &antiquant_scale2, const NodePtr &antiquant_offset1, const NodePtr &antiquant_offset2,
                 const NodePtr &activation, const NodePtr &inner_precise) override;
  NodePtr ConvolutionGrad(const NodePtr &dout, const NodePtr &input, const NodePtr &weight, const NodePtr &bias,
                          const NodePtr &stride, const NodePtr &padding, const NodePtr &dilation,
                          const NodePtr &transposed, const NodePtr &output_padding, const NodePtr &groups,
                          const NodePtr &output_mask) override;
  NodePtr MSELossExt(const NodePtr &input, const NodePtr &target, const NodePtr &reduction) override;
  NodePtr NLLLoss2dGrad(const NodePtr &loss_grad, const NodePtr &input, const NodePtr &target, const NodePtr &weight,
                        const NodePtr &reduction, const NodePtr &ignore_index, const NodePtr &total_weight) override;
  NodePtr ReflectionPad1DGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &padding) override;
  NodePtr AdaptiveAvgPool3DExt(const NodePtr &input, const NodePtr &output_size) override;
  NodePtr PromptFlashAttention(const NodePtr &query, const NodePtr &key, const NodePtr &value, const NodePtr &attn_mask,
                               const NodePtr &actual_seq_lengths, const NodePtr &actual_seq_lengths_kv,
                               const NodePtr &pse_shift, const NodePtr &deq_scale1, const NodePtr &quant_scale1,
                               const NodePtr &deq_scale2, const NodePtr &quant_scale2, const NodePtr &quant_offset2,
                               const NodePtr &num_heads, const NodePtr &scale_value, const NodePtr &pre_tokens,
                               const NodePtr &next_tokens, const NodePtr &input_layout,
                               const NodePtr &num_key_value_heads, const NodePtr &sparse_mode,
                               const NodePtr &inner_precise) override;
  NodePtr MinDim(const NodePtr &input, const NodePtr &dim, const NodePtr &keepdim) override;
  NodePtr MatmulReduceScatter(const NodePtr &input, const NodePtr &x2, const NodePtr &group, const NodePtr &world_size,
                              const NodePtr &reduce_op, const NodePtr &bias, const NodePtr &comm_turn,
                              const NodePtr &trans_input, const NodePtr &trans_x2) override;
  NodePtr SpeedFusionAttentionGrad(const NodePtr &query, const NodePtr &key, const NodePtr &value, const NodePtr &dy,
                                   const NodePtr &head_num, const NodePtr &input_layout, const NodePtr &pse,
                                   const NodePtr &padding_mask, const NodePtr &atten_mask, const NodePtr &softmax_max,
                                   const NodePtr &softmax_sum, const NodePtr &softmax_in, const NodePtr &attention_in,
                                   const NodePtr &scale_value, const NodePtr &keep_prob, const NodePtr &pre_tokens,
                                   const NodePtr &next_tokens, const NodePtr &inner_precise, const NodePtr &seed,
                                   const NodePtr &offset, const NodePtr &numels, const NodePtr &prefix,
                                   const NodePtr &actual_seq_qlen, const NodePtr &actual_seq_kvlen,
                                   const NodePtr &sparse_mode, const NodePtr &gen_mask_parallel, const NodePtr &sync,
                                   const NodePtr &pse_type, const NodePtr &q_start_idx,
                                   const NodePtr &kv_start_idx) override;
  NodePtr MaskedFillScalar(const NodePtr &input, const NodePtr &mask, const NodePtr &value) override;
  NodePtr Atan2Ext(const NodePtr &input, const NodePtr &other) override;
  NodePtr DequantSwigluQuant(const NodePtr &x, const NodePtr &weight_scale, const NodePtr &activation_scale,
                             const NodePtr &bias, const NodePtr &quant_scale, const NodePtr &quant_offset,
                             const NodePtr &group_index, const NodePtr &activate_left,
                             const NodePtr &quant_mode) override;
  NodePtr InplaceSiLU(const NodePtr &input) override;
  NodePtr Var(const NodePtr &input, const NodePtr &dim, const NodePtr &correction, const NodePtr &keepdim) override;
  NodePtr Mv(const NodePtr &input, const NodePtr &vec) override;
  NodePtr AdamW(const NodePtr &var, const NodePtr &m, const NodePtr &v, const NodePtr &max_v, const NodePtr &gradient,
                const NodePtr &step, const NodePtr &lr, const NodePtr &beta1, const NodePtr &beta2,
                const NodePtr &decay, const NodePtr &eps, const NodePtr &amsgrad, const NodePtr &maximize) override;
  NodePtr InplaceMatmulAdd(const NodePtr &x, const NodePtr &weight, const NodePtr &C) override;
  NodePtr BincountExt(const NodePtr &input, const NodePtr &weights, const NodePtr &minlength) override;
  NodePtr SeluGrad(const NodePtr &gradient, const NodePtr &result) override;
  NodePtr StackExt(const NodePtr &tensors, const NodePtr &dim) override;
  NodePtr NormalTensorTensor(const NodePtr &mean, const NodePtr &std, const NodePtr &seed,
                             const NodePtr &offset) override;
  NodePtr ReduceAll(const NodePtr &input, const NodePtr &axis, const NodePtr &keep_dims) override;
  NodePtr DropoutDoMaskExt(const NodePtr &input, const NodePtr &mask, const NodePtr &p) override;
  NodePtr UpsampleBicubic2DGrad(const NodePtr &dy, const NodePtr &input_size, const NodePtr &output_size,
                                const NodePtr &scales, const NodePtr &align_corners) override;
  NodePtr AddcmulExt(const NodePtr &input, const NodePtr &tensor1, const NodePtr &tensor2,
                     const NodePtr &value) override;
  NodePtr InplaceScatterValueReduce(const NodePtr &input, const NodePtr &dim, const NodePtr &index,
                                    const NodePtr &value, const NodePtr &reduce) override;
  NodePtr Gcd(const NodePtr &input, const NodePtr &other) override;
  NodePtr Eye(const NodePtr &n, const NodePtr &m, const NodePtr &dtype) override;
  NodePtr NanToNum(const NodePtr &input, const NodePtr &nan, const NodePtr &posinf, const NodePtr &neginf) override;
  NodePtr GeluExt(const NodePtr &input, const NodePtr &approximate) override;
  NodePtr Repeat(const NodePtr &input, const NodePtr &repeats) override;
  NodePtr Conv2DPadding(const NodePtr &input, const NodePtr &weight, const NodePtr &bias, const NodePtr &stride,
                        const NodePtr &padding, const NodePtr &dilation, const NodePtr &groups) override;
  NodePtr InplaceSubScalar(const NodePtr &input, const NodePtr &other, const NodePtr &alpha) override;
  NodePtr Copy(const NodePtr &input) override;
  NodePtr Zeros(const NodePtr &size, const NodePtr &dtype) override;
  NodePtr Muls(const NodePtr &input, const NodePtr &other) override;
  NodePtr NLLLossGrad(const NodePtr &logits, const NodePtr &loss_grad, const NodePtr &labels, const NodePtr &weight,
                      const NodePtr &total_weight, const NodePtr &reduction, const NodePtr &ignore_index) override;
  NodePtr AdaptiveAvgPool1D(const NodePtr &input, const NodePtr &output_size) override;
  NodePtr Index(const NodePtr &input, const NodePtr &indices) override;
  NodePtr HardtanhGrad(const NodePtr &dout, const NodePtr &input, const NodePtr &min_val,
                       const NodePtr &max_val) override;
  NodePtr RepeatInterleaveInt(const NodePtr &input, const NodePtr &repeats, const NodePtr &dim,
                              const NodePtr &output_size) override;
  NodePtr Conv3DExt(const NodePtr &input, const NodePtr &weight, const NodePtr &bias, const NodePtr &stride,
                    const NodePtr &padding, const NodePtr &dilation, const NodePtr &groups) override;
  NodePtr Sigmoid(const NodePtr &input) override;
  NodePtr Threshold(const NodePtr &input, const NodePtr &threshold, const NodePtr &value) override;
  NodePtr NormalFloatFloat(const NodePtr &mean, const NodePtr &std, const NodePtr &size, const NodePtr &seed,
                           const NodePtr &offset) override;
  NodePtr BatchNormExt(const NodePtr &input, const NodePtr &weight, const NodePtr &bias, const NodePtr &running_mean,
                       const NodePtr &runnning_var, const NodePtr &training, const NodePtr &momentum,
                       const NodePtr &epsilon) override;
  NodePtr AsinExt(const NodePtr &input) override;
  NodePtr Cast(const NodePtr &input, const NodePtr &dtype) override;
  NodePtr LayerNormGradExt(const NodePtr &dy, const NodePtr &x, const NodePtr &normalized_shape, const NodePtr &mean,
                           const NodePtr &variance, const NodePtr &gamma, const NodePtr &beta,
                           const NodePtr &output_mask) override;
  NodePtr KLDivGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &target, const NodePtr &reduction,
                    const NodePtr &log_target) override;
  NodePtr ApplyRotaryPosEmb(const NodePtr &query, const NodePtr &key, const NodePtr &cos, const NodePtr &sin,
                            const NodePtr &position_ids, const NodePtr &cos_format) override;
  NodePtr BatchMatMul(const NodePtr &x, const NodePtr &y, const NodePtr &transpose_a,
                      const NodePtr &transpose_b) override;
  NodePtr HSigmoid(const NodePtr &input) override;
  NodePtr NonZero(const NodePtr &input) override;
  NodePtr Meshgrid(const NodePtr &inputs, const NodePtr &indexing) override;
  NodePtr Erfinv(const NodePtr &input) override;
  NodePtr MaxPoolGradWithMask(const NodePtr &x, const NodePtr &grad, const NodePtr &mask, const NodePtr &kernel_size,
                              const NodePtr &strides, const NodePtr &pads, const NodePtr &dilation,
                              const NodePtr &ceil_mode, const NodePtr &argmax_type) override;
  NodePtr UniformExt(const NodePtr &tensor, const NodePtr &a, const NodePtr &b, const NodePtr &seed,
                     const NodePtr &offset) override;
  NodePtr GridSampler2D(const NodePtr &input_x, const NodePtr &grid, const NodePtr &interpolation_mode,
                        const NodePtr &padding_mode, const NodePtr &align_corners) override;
  NodePtr RemainderTensorTensor(const NodePtr &input, const NodePtr &other) override;
  NodePtr Dense(const NodePtr &input, const NodePtr &weight, const NodePtr &bias) override;
  NodePtr SeLUExt(const NodePtr &input) override;
  NodePtr AsinhExt(const NodePtr &input) override;
  NodePtr AcosExt(const NodePtr &input) override;
  NodePtr SoftMarginLoss(const NodePtr &input, const NodePtr &target, const NodePtr &reduction) override;
  NodePtr ChunkView(const NodePtr &input, const NodePtr &chunks, const NodePtr &dim) override;
  NodePtr InplaceMuls(const NodePtr &input, const NodePtr &other) override;
  NodePtr HSwish(const NodePtr &input) override;
  NodePtr TExt(const NodePtr &input) override;
  NodePtr UpsampleBilinear2D(const NodePtr &x, const NodePtr &output_size, const NodePtr &scales,
                             const NodePtr &align_corners) override;
  NodePtr Cross(const NodePtr &input, const NodePtr &other, const NodePtr &dim) override;
  NodePtr SumExt(const NodePtr &input, const NodePtr &dim, const NodePtr &keepdim, const NodePtr &dtype) override;
  NodePtr InplacePut(const NodePtr &input, const NodePtr &index, const NodePtr &source,
                     const NodePtr &accumulate) override;
  NodePtr SliceExt(const NodePtr &input, const NodePtr &dim, const NodePtr &start, const NodePtr &end,
                   const NodePtr &step) override;
  NodePtr ScatterValue(const NodePtr &input, const NodePtr &dim, const NodePtr &index, const NodePtr &src,
                       const NodePtr &reduce) override;
  NodePtr ReverseV2(const NodePtr &input, const NodePtr &axis) override;
  NodePtr UpsampleNearest2DGrad(const NodePtr &dy, const NodePtr &input_size, const NodePtr &output_size,
                                const NodePtr &scales) override;
  NodePtr Nansum(const NodePtr &input, const NodePtr &dim, const NodePtr &keepdim, const NodePtr &dtype) override;
  NodePtr UpsampleNearest1DGrad(const NodePtr &dy, const NodePtr &input_size, const NodePtr &output_size,
                                const NodePtr &scales) override;
  NodePtr Maximum(const NodePtr &input, const NodePtr &other) override;
  NodePtr MoeTokenPermuteGrad(const NodePtr &permuted_tokens_grad, const NodePtr &sorted_indices,
                              const NodePtr &num_topk, const NodePtr &padded_mode) override;
  NodePtr DivMods(const NodePtr &input, const NodePtr &other, const NodePtr &rounding_mode) override;
  NodePtr Trunc(const NodePtr &input) override;
  NodePtr MedianDim(const NodePtr &input, const NodePtr &dim, const NodePtr &keepdim) override;
  NodePtr Max(const NodePtr &input) override;
  NodePtr MedianExt(const NodePtr &input) override;
  NodePtr Erfc(const NodePtr &input) override;
  NodePtr GLU(const NodePtr &x, const NodePtr &axis) override;
  NodePtr Reciprocal(const NodePtr &input) override;
  NodePtr SoftShrink(const NodePtr &input, const NodePtr &lambd) override;
  NodePtr InplaceRemainderTensorScalar(const NodePtr &input, const NodePtr &other) override;
  NodePtr Contiguous(const NodePtr &input) override;
  NodePtr ToDtype(const NodePtr &input, const NodePtr &dtype, const NodePtr &non_blocking,
                  const NodePtr &copy) override;
  NodePtr SplitWithSize(const NodePtr &input, const NodePtr &split_size, const NodePtr &dim) override;
  NodePtr MoeTokenPermute(const NodePtr &tokens, const NodePtr &indices, const NodePtr &num_out_tokens,
                          const NodePtr &padded_mode) override;
  NodePtr AdaptiveMaxPool2D(const NodePtr &input, const NodePtr &output_size) override;
  NodePtr ReplicationPad3D(const NodePtr &input, const NodePtr &padding) override;
  NodePtr SilentCheckV3(const NodePtr &val, const NodePtr &max, const NodePtr &avg, const NodePtr &input_grad,
                        const NodePtr &step, const NodePtr &c_thresh_l1, const NodePtr &c_thresh_l2,
                        const NodePtr &beta1, const NodePtr &npu_asd_detect) override;
  NodePtr BinaryCrossEntropy(const NodePtr &input, const NodePtr &target, const NodePtr &weight,
                             const NodePtr &reduction) override;
  NodePtr L1LossExt(const NodePtr &input, const NodePtr &target, const NodePtr &reduction) override;
  NodePtr Min(const NodePtr &input) override;
  NodePtr InplaceBernoulliScalar(const NodePtr &input, const NodePtr &p, const NodePtr &seed,
                                 const NodePtr &offset) override;
  NodePtr FloorDivScalar(const NodePtr &input, const NodePtr &other) override;
  NodePtr FullLike(const NodePtr &input, const NodePtr &fill_value, const NodePtr &dtype) override;
  NodePtr Empty(const NodePtr &size, const NodePtr &dtype, const NodePtr &device, const NodePtr &pin_memory) override;
  NodePtr MultiScaleDeformableAttnGrad(const NodePtr &value, const NodePtr &shape, const NodePtr &offset,
                                       const NodePtr &locations_trans, const NodePtr &weight,
                                       const NodePtr &grad_output) override;
  NodePtr LogSoftmaxGrad(const NodePtr &logits, const NodePtr &grad, const NodePtr &axis) override;
  NodePtr RandInt(const NodePtr &low, const NodePtr &high, const NodePtr &shape, const NodePtr &seed,
                  const NodePtr &offset, const NodePtr &dtype, const NodePtr &device) override;
  NodePtr Frac(const NodePtr &input) override;
  NodePtr ArgMaxExt(const NodePtr &input, const NodePtr &dim, const NodePtr &keepdim) override;
  NodePtr UniqueConsecutive(const NodePtr &input, const NodePtr &return_inverse, const NodePtr &return_counts,
                            const NodePtr &dim) override;
  NodePtr ReduceAny(const NodePtr &x, const NodePtr &axis, const NodePtr &keep_dims) override;
  NodePtr UpsampleLinear1DGrad(const NodePtr &dy, const NodePtr &input_size, const NodePtr &output_size,
                               const NodePtr &scales, const NodePtr &align_corners) override;
  NodePtr InplaceHardtanh(const NodePtr &input, const NodePtr &min_val, const NodePtr &max_val) override;
  NodePtr IndexFillScalar(const NodePtr &input, const NodePtr &dim, const NodePtr &index,
                          const NodePtr &value) override;
  NodePtr PagedAttention(const NodePtr &query, const NodePtr &key_cache, const NodePtr &value_cache,
                         const NodePtr &block_tables, const NodePtr &context_lens, const NodePtr &antiquant_scale,
                         const NodePtr &antiquant_offset, const NodePtr &attn_mask, const NodePtr &q_seq_lens,
                         const NodePtr &alibi_mask, const NodePtr &head_num, const NodePtr &scale_value,
                         const NodePtr &kv_head_num, const NodePtr &kv_cache_quant_mode, const NodePtr &mask_mode,
                         const NodePtr &mla_v_dim) override;
  NodePtr PowTensorScalar(const NodePtr &input, const NodePtr &exponent) override;
  NodePtr NonZeroExt(const NodePtr &input) override;
  NodePtr SoftMarginLossGrad(const NodePtr &predict, const NodePtr &label, const NodePtr &dout,
                             const NodePtr &reduction) override;
  NodePtr SelectV2(const NodePtr &condition, const NodePtr &input, const NodePtr &other) override;
  NodePtr ReluGrad(const NodePtr &y_backprop, const NodePtr &x) override;
  NodePtr EluExt(const NodePtr &input, const NodePtr &alpha) override;
  NodePtr IndexSelect(const NodePtr &input, const NodePtr &dim, const NodePtr &index) override;
  NodePtr Split(const NodePtr &input_x, const NodePtr &axis, const NodePtr &output_num) override;
  NodePtr IndexAddExt(const NodePtr &input, const NodePtr &dim, const NodePtr &index, const NodePtr &source,
                      const NodePtr &alpha) override;
  NodePtr DropoutExt(const NodePtr &input, const NodePtr &p, const NodePtr &seed, const NodePtr &offset) override;
  NodePtr SoftplusGradExt(const NodePtr &dout, const NodePtr &x, const NodePtr &beta,
                          const NodePtr &threshold) override;
  NodePtr IsFinite(const NodePtr &input) override;
  NodePtr Abs(const NodePtr &input) override;
  NodePtr NLLLoss(const NodePtr &logits, const NodePtr &labels, const NodePtr &weight, const NodePtr &reduction,
                  const NodePtr &ignore_index) override;
  NodePtr UpsampleTrilinear3DGrad(const NodePtr &dy, const NodePtr &input_size, const NodePtr &output_size,
                                  const NodePtr &scales, const NodePtr &align_corners) override;
  NodePtr RmsNormGrad(const NodePtr &dy, const NodePtr &x, const NodePtr &rstd, const NodePtr &gamma) override;
  NodePtr LeakyReLUGradExt(const NodePtr &dy, const NodePtr &input, const NodePtr &negative_slope,
                           const NodePtr &is_result) override;
  NodePtr LogSumExp(const NodePtr &input, const NodePtr &dim, const NodePtr &keepdim) override;
  NodePtr Erf(const NodePtr &input) override;
  NodePtr SilentCheckV2(const NodePtr &val, const NodePtr &input_grad, const NodePtr &sfda, const NodePtr &step,
                        const NodePtr &c_min_steps, const NodePtr &c_thresh_l1, const NodePtr &c_coeff_l1,
                        const NodePtr &c_thresh_l2, const NodePtr &c_coeff_l2, const NodePtr &npu_asd_detect) override;
  NodePtr InplaceScatterSrc(const NodePtr &input, const NodePtr &dim, const NodePtr &index,
                            const NodePtr &src) override;
  NodePtr BitwiseAndScalar(const NodePtr &input, const NodePtr &other) override;
  NodePtr MSELossGradExt(const NodePtr &dout, const NodePtr &x, const NodePtr &target,
                         const NodePtr &reduction) override;
  NodePtr UpsampleLinear1D(const NodePtr &x, const NodePtr &output_size, const NodePtr &scales,
                           const NodePtr &align_corners) override;
  NodePtr ReduceMin(const NodePtr &x, const NodePtr &axis, const NodePtr &keep_dims) override;
  NodePtr LogicalNot(const NodePtr &input) override;
  NodePtr SoftShrinkGrad(const NodePtr &input_grad, const NodePtr &input_x, const NodePtr &lambd) override;
  NodePtr CrossEntropyLossGrad(const NodePtr &grad_loss, const NodePtr &log_prob, const NodePtr &target,
                               const NodePtr &weight, const NodePtr &grad_zloss, const NodePtr &lse_for_zloss,
                               const NodePtr &reduction, const NodePtr &ignore_index, const NodePtr &label_smoothing,
                               const NodePtr &lse_square_scale_for_zloss) override;
  NodePtr MatMul(const NodePtr &input, const NodePtr &mat2, const NodePtr &transpose_a,
                 const NodePtr &transpose_b) override;
  NodePtr Triu(const NodePtr &input, const NodePtr &diagonal) override;
  NodePtr Lerp(const NodePtr &input, const NodePtr &end, const NodePtr &weight) override;
  NodePtr ReplicationPad2DGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &padding) override;
  NodePtr InplaceDivs(const NodePtr &input, const NodePtr &other) override;
  NodePtr Im2ColExt(const NodePtr &input, const NodePtr &kernel_size, const NodePtr &dilation, const NodePtr &padding,
                    const NodePtr &stride) override;
  NodePtr DiagExt(const NodePtr &input, const NodePtr &diagonal) override;
  NodePtr InplaceFillTensor(const NodePtr &input, const NodePtr &value) override;
  NodePtr NewFull(const NodePtr &input, const NodePtr &size, const NodePtr &fill_value, const NodePtr &dtype) override;
  NodePtr PReLU(const NodePtr &input, const NodePtr &weight) override;
  NodePtr IndexFillTensor(const NodePtr &input, const NodePtr &dim, const NodePtr &index,
                          const NodePtr &value) override;
  NodePtr ConvTranspose2D(const NodePtr &input, const NodePtr &weight, const NodePtr &bias, const NodePtr &stride,
                          const NodePtr &padding, const NodePtr &output_padding, const NodePtr &groups,
                          const NodePtr &dilation) override;
  NodePtr InplaceRemainderTensorTensor(const NodePtr &input, const NodePtr &other) override;
  NodePtr Sinc(const NodePtr &input) override;
  NodePtr InplaceAddsExt(const NodePtr &input, const NodePtr &other, const NodePtr &alpha) override;
  NodePtr Tan(const NodePtr &input) override;
  NodePtr UpsampleNearest1D(const NodePtr &x, const NodePtr &output_size, const NodePtr &scales) override;
  NodePtr MoeDistributeCombine(const NodePtr &expand_x, const NodePtr &expert_ids, const NodePtr &expand_idx,
                               const NodePtr &ep_send_counts, const NodePtr &expert_scales,
                               const NodePtr &ep_world_size, const NodePtr &ep_rank_id, const NodePtr &moe_expert_num,
                               const NodePtr &tp_send_counts, const NodePtr &x_active_mask,
                               const NodePtr &activate_scale, const NodePtr &weight_scale, const NodePtr &group_list,
                               const NodePtr &expand_scales, const NodePtr &group_ep, const NodePtr &group_tp,
                               const NodePtr &tp_world_size, const NodePtr &tp_rank_id,
                               const NodePtr &expert_shard_type, const NodePtr &shared_expert_num,
                               const NodePtr &shared_export_rank_num, const NodePtr &global_bs,
                               const NodePtr &out_dtype, const NodePtr &common_quant_mode,
                               const NodePtr &group_list_type) override;
  NodePtr ConstantPadND(const NodePtr &input, const NodePtr &padding, const NodePtr &value) override;
  NodePtr UpsampleNearest3D(const NodePtr &x, const NodePtr &output_size, const NodePtr &scales) override;
  NodePtr Rsqrt(const NodePtr &input) override;
  NodePtr RingAttentionUpdate(const NodePtr &prev_attn_out, const NodePtr &prev_softmax_max,
                              const NodePtr &prev_softmax_sum, const NodePtr &cur_attn_out,
                              const NodePtr &cur_softmax_max, const NodePtr &cur_softmax_sum,
                              const NodePtr &actual_seq_qlen, const NodePtr &layout) override;
  NodePtr InplaceMaskedFillScalar(const NodePtr &input, const NodePtr &mask, const NodePtr &value) override;
  NodePtr NewEmpty(const NodePtr &input, const NodePtr &size, const NodePtr &dtype, const NodePtr &device) override;
  NodePtr CrossEntropyLoss(const NodePtr &input, const NodePtr &target, const NodePtr &weight, const NodePtr &reduction,
                           const NodePtr &ignore_index, const NodePtr &label_smoothing,
                           const NodePtr &lse_square_scale_for_zloss, const NodePtr &return_zloss) override;
  NodePtr AddScalar(const NodePtr &input, const NodePtr &other, const NodePtr &alpha) override;
  NodePtr UpsampleBilinear2DGrad(const NodePtr &dy, const NodePtr &input_size, const NodePtr &output_size,
                                 const NodePtr &scales, const NodePtr &align_corners) override;
  NodePtr Floor(const NodePtr &input) override;
  NodePtr Mla(const NodePtr &query, const NodePtr &q_rope, const NodePtr &kv_cache, const NodePtr &k_rope,
              const NodePtr &block_tables, const NodePtr &attn_mask, const NodePtr &deq_scale_qk,
              const NodePtr &deq_scale_pv, const NodePtr &q_seq_lens, const NodePtr &context_lens,
              const NodePtr &head_num, const NodePtr &scale_value, const NodePtr &kv_head_num, const NodePtr &mask_mode,
              const NodePtr &is_ring) override;
  NodePtr MaskedSelect(const NodePtr &input, const NodePtr &mask) override;
  NodePtr NarrowView(const NodePtr &input, const NodePtr &dim, const NodePtr &start, const NodePtr &length) override;
  NodePtr Sinh(const NodePtr &input) override;
  NodePtr Conv1DPadding(const NodePtr &input, const NodePtr &weight, const NodePtr &bias, const NodePtr &stride,
                        const NodePtr &padding, const NodePtr &dilation, const NodePtr &groups) override;
  NodePtr QuantMatmul(const NodePtr &x1, const NodePtr &x2, const NodePtr &scale, const NodePtr &offset,
                      const NodePtr &pertoken_scale, const NodePtr &bias, const NodePtr &output_dtype,
                      const NodePtr &x1_dtype, const NodePtr &x2_dtype, const NodePtr &pertoken_scale_dtype,
                      const NodePtr &scale_dtype, const NodePtr &group_sizes) override;
  NodePtr AddRmsNormQuantV2(const NodePtr &x1, const NodePtr &x2, const NodePtr &gamma, const NodePtr &scale,
                            const NodePtr &offset, const NodePtr &epsilon) override;
  NodePtr MoeInitRouting(const NodePtr &x, const NodePtr &row_idx, const NodePtr &expert_idx,
                         const NodePtr &active_num) override;
  NodePtr FusedInferAttentionScore(
    const NodePtr &query, const NodePtr &key, const NodePtr &value, const NodePtr &pse_shift, const NodePtr &attn_mask,
    const NodePtr &actual_seq_lengths, const NodePtr &actual_seq_lengths_kv, const NodePtr &dequant_scale1,
    const NodePtr &quant_scale1, const NodePtr &dequant_scale2, const NodePtr &quant_scale2,
    const NodePtr &quant_offset2, const NodePtr &antiquant_scale, const NodePtr &antiquant_offset,
    const NodePtr &block_table, const NodePtr &query_padding_size, const NodePtr &kv_padding_size,
    const NodePtr &key_antiquant_scale, const NodePtr &key_antiquant_offset, const NodePtr &value_antiquant_scale,
    const NodePtr &value_antiquant_offset, const NodePtr &key_shared_prefix, const NodePtr &value_shared_prefix,
    const NodePtr &actual_shared_prefix_len, const NodePtr &num_heads, const NodePtr &scale_value,
    const NodePtr &pre_tokens, const NodePtr &next_tokens, const NodePtr &input_layout,
    const NodePtr &num_key_value_heads, const NodePtr &sparse_mode, const NodePtr &inner_precise,
    const NodePtr &block_size, const NodePtr &antiquant_mode, const NodePtr &softmax_lse_flag,
    const NodePtr &key_antiquant_mode, const NodePtr &value_antiquant_mode) override;
  NodePtr GroupedMatmulV4(const NodePtr &x, const NodePtr &weight, const NodePtr &bias, const NodePtr &scale,
                          const NodePtr &offset, const NodePtr &antiquant_scale, const NodePtr &antiquant_offset,
                          const NodePtr &pre_token_scale, const NodePtr &group_list, const NodePtr &activation_input,
                          const NodePtr &activation_quant_scale, const NodePtr &activation_quant_offset,
                          const NodePtr &split_item, const NodePtr &group_type, const NodePtr &group_list_type,
                          const NodePtr &act_type, const NodePtr &output_dtype) override;
  NodePtr QuantBatchMatmul(const NodePtr &x1, const NodePtr &x2, const NodePtr &scale, const NodePtr &offset,
                           const NodePtr &bias, const NodePtr &pertokenScaleOptional, const NodePtr &transpose_x1,
                           const NodePtr &transpose_x2, const NodePtr &dtype) override;
  NodePtr MoeComputeExpertTokens(const NodePtr &sorted_experts, const NodePtr &num_expert) override;
  NodePtr GroupedMatmul(const NodePtr &x, const NodePtr &weight, const NodePtr &bias, const NodePtr &scale,
                        const NodePtr &offset, const NodePtr &antiquant_scale, const NodePtr &antiquant_offset,
                        const NodePtr &group_list, const NodePtr &split_item, const NodePtr &group_type,
                        const NodePtr &transpose_a, const NodePtr &transpose_b) override;
  NodePtr WeightQuantBatchMatmul(const NodePtr &x, const NodePtr &weight, const NodePtr &antiquant_scale,
                                 const NodePtr &antiquant_offset, const NodePtr &quant_scale,
                                 const NodePtr &quant_offset, const NodePtr &bias, const NodePtr &transpose_x,
                                 const NodePtr &transpose_weight, const NodePtr &antiquant_group_size) override;
  NodePtr MatmulAllReduceAddRmsNorm(const NodePtr &x1, const NodePtr &x2, const NodePtr &bias, const NodePtr &residual,
                                    const NodePtr &gamma, const NodePtr &epsilon, const NodePtr &group,
                                    const NodePtr &reduce_op, const NodePtr &comm_turn,
                                    const NodePtr &stream_mode) override;
  NodePtr GroupedMatmulV2(const NodePtr &x, const NodePtr &weight, const NodePtr &bias, const NodePtr &scale,
                          const NodePtr &offset, const NodePtr &antiquant_scale, const NodePtr &antiquant_offset,
                          const NodePtr &group_list, const NodePtr &split_item, const NodePtr &group_type) override;
  NodePtr QuantV2(const NodePtr &x, const NodePtr &scale, const NodePtr &offset, const NodePtr &sqrt_mode,
                  const NodePtr &rounding_mode, const NodePtr &dst_type) override;
  NodePtr MoeInitRoutingV2(const NodePtr &x, const NodePtr &expert_idx, const NodePtr &active_num,
                           const NodePtr &expert_capacity, const NodePtr &expert_num, const NodePtr &drop_pad_mode,
                           const NodePtr &expert_tokens_count_or_cumsum_flag,
                           const NodePtr &expert_tokens_before_capacity_flag) override;
  NodePtr MoeGatingTopKSoftmax(const NodePtr &x, const NodePtr &finished, const NodePtr &k) override;
  NodePtr MoeFinalizeRouting(const NodePtr &expanded_x, const NodePtr &x1, const NodePtr &x2, const NodePtr &bias,
                             const NodePtr &scales, const NodePtr &expanded_row_idx,
                             const NodePtr &expanded_expert_idx) override;
  NodePtr MoeInitRoutingQuantV2(const NodePtr &x, const NodePtr &expert_idx, const NodePtr &active_num,
                                const NodePtr &expert_capacity, const NodePtr &expert_num, const NodePtr &drop_pad_mode,
                                const NodePtr &expert_tokens_count_or_cumsum_flag,
                                const NodePtr &expert_tokens_before_capacity_flag, const NodePtr &quant_mode,
                                const NodePtr &scale, const NodePtr &offset) override;
  NodePtr DynamicQuantExt(const NodePtr &x, const NodePtr &smooth_scales) override;
  NodePtr KVCacheScatterUpdate(const NodePtr &var, const NodePtr &indices, const NodePtr &updates, const NodePtr &axis,
                               const NodePtr &reduce) override;
  NodePtr FuncDropoutExt(const NodePtr &input, const NodePtr &p, const NodePtr &training, const NodePtr &inplace,
                         const NodePtr &seed, const NodePtr &offset) override;
  NodePtr GmmBackward(const NodePtr &grad, const NodePtr &x, const NodePtr &weight, const NodePtr &group_list,
                      const NodePtr &group_list_type) override;
  NodePtr PixelShuffle(const NodePtr &input, const NodePtr &upscale_factor) override;
  NodePtr GmmV2BackwardFusion(const NodePtr &grad, const NodePtr &weight, const NodePtr &group_list,
                              const NodePtr &group_list_type) override;
  NodePtr AnyExt(const NodePtr &input, const NodePtr &dim, const NodePtr &keepdim) override;
  NodePtr GmmV2Backward(const NodePtr &grad, const NodePtr &x, const NodePtr &weight, const NodePtr &group_list,
                        const NodePtr &group_list_type) override;
  NodePtr FuncMaxPool2D(const NodePtr &input, const NodePtr &kernel_size, const NodePtr &stride, const NodePtr &padding,
                        const NodePtr &dilation, const NodePtr &ceil_mode, const NodePtr &return_indices) override;
  NodePtr MoeTokenUnpermute(const NodePtr &permuted_tokens, const NodePtr &sorted_indices, const NodePtr &probs,
                            const NodePtr &padded_mode, const NodePtr &restore_shape) override;
  NodePtr Any(const NodePtr &input) override;
  NodePtr InplaceExponential(const NodePtr &input, const NodePtr &lambd, const NodePtr &seed,
                             const NodePtr &offset) override;
  NodePtr Dropout2dExt(const NodePtr &input, const NodePtr &p, const NodePtr &training, const NodePtr &inplace,
                       const NodePtr &seed, const NodePtr &offset) override;
  NodePtr GmmBackwardFusion(const NodePtr &grad, const NodePtr &weight, const NodePtr &group_list,
                            const NodePtr &group_list_type) override;
  NodePtr Gmm(const NodePtr &x, const NodePtr &weight, const NodePtr &bias, const NodePtr &group_list,
              const NodePtr &group_type, const NodePtr &group_list_type) override;
  NodePtr EinsumExt(const NodePtr &equation, const NodePtr &operands) override;
  NodePtr GmmV2(const NodePtr &x, const NodePtr &weight, const NodePtr &bias, const NodePtr &group_list,
                const NodePtr &group_type, const NodePtr &group_list_type) override;

  NodePtr InnerCommAllReduce(const NodePtr &grad, const NodePtr &op, const NodePtr &group) override;
  NodePtr InnerCommAllGather(const NodePtr &grad, const NodePtr &rank_size, const NodePtr &group) override;
  NodePtr InnerCommReduceScatter(const NodePtr &grad, const NodePtr &rank_size, const NodePtr &type,
                                 const NodePtr &group) override;
  NodePtr InnerCommIsend(const NodePtr &grad, const NodePtr &rank_size, const NodePtr &group,
                         const NodePtr &tag) override;
  NodePtr InnerCommIrecv(const NodePtr &tag, const NodePtr &rank_size, const NodePtr &shape, const NodePtr &group,
                         const NodePtr &type) override;
  NodePtr InnerCommAllToAllV(const NodePtr &grad, const NodePtr &group, const NodePtr &send_numel_list,
                             const NodePtr &recv_numel_list, const NodePtr &rank_size,
                             const NodePtr &split_sizes_empty) override;
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
  ValuePtr Ones(const tensor::TensorPtr &tensor);
  ValuePtr Zeros(const tensor::TensorPtr &tensor);
  ValuePtr Add(const ValuePtr &input, const ValuePtr &other);
  void SetInputs(std::string instance_name, const std::vector<NodePtr> *inputs,
                 mindspore::HashMap<std::string, ValuePtr> *attrs_ptr);
  void ResetInputs();
  ValuePtr FillZeros(const ValuePtr &value, const abstract::AbstractBasePtr &abs);

 private:
  NodePtrList FlattenNode(const NodePtr &input);
  device::DeviceType device_target_;
  bprop_pass::FuncPassForwardPtr pass_forward_;
};
using FuncBuilderPtr = std::shared_ptr<FuncBuilder>;
}  // namespace mindspore::pynative::autograd

#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_FUNC_BUILDER_H_
