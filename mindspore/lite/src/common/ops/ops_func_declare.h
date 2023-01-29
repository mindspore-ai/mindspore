/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_COMMON_OPS_OPS_FUNC_DECLARE_H_
#define MINDSPORE_LITE_SRC_COMMON_OPS_OPS_FUNC_DECLARE_H_

#ifdef PRIMITIVE_WRITEABLE
#include <memory>
#include "schema/inner/model_generated.h"
#include "ops/abs.h"
#include "ops/adam.h"
#include "ops/add.h"
#include "ops/adder.h"
#include "ops/addn.h"
#include "ops/all.h"
#include "ops/apply_momentum.h"
#include "ops/arg_max.h"
#include "ops/arg_min.h"
#include "ops/asin.h"
#include "ops/assert.h"
#include "ops/assign.h"
#include "ops/assign_add.h"
#include "ops/attention.h"
#include "ops/atan.h"
#include "ops/audio_spectrogram.h"
#include "ops/avg_pool.h"
#include "ops/batch_norm.h"
#include "ops/batch_to_space.h"
#include "ops/batch_to_space_nd.h"
#include "ops/bias_add.h"
#include "ops/binary_cross_entropy.h"
#include "ops/broadcast_to.h"
#include "ops/broadcast.h"
#include "ops/cast.h"
#include "ops/ceil.h"
#include "ops/clip.h"
#include "ops/custom.h"
#include "ops/custom_normalize.h"
#include "ops/custom_predict.h"
#include "ops/custom_extract_features.h"
#include "ops/concat.h"
#include "ops/constant_of_shape.h"
#include "ops/cos.h"
#include "ops/crop.h"
#include "ops/depth_to_space.h"
#include "ops/depend.h"
#include "ops/detection_post_process.h"
#include "ops/div.h"
#include "ops/dropout.h"
#include "ops/eltwise.h"
#include "ops/elu.h"
#include "ops/embedding_lookup.h"
#include "ops/equal.h"
#include "ops/expand_dims.h"
#include "ops/exp.h"
#include "ops/fake_quant_with_min_max_vars.h"
#include "ops/fake_quant_with_min_max_vars_per_channel.h"
#include "ops/fft_imag.h"
#include "ops/fft_real.h"
#include "ops/fill.h"
#include "ops/flatten.h"
#include "ops/floor.h"
#include "ops/floor_div.h"
#include "ops/floor_mod.h"
#include "ops/fused_batch_norm.h"
#include "ops/gather.h"
#include "ops/gather_nd.h"
#include "ops/greater_equal.h"
#include "ops/greater.h"
#include "ops/hashtable_lookup.h"
#include "ops/instance_norm.h"
#include "ops/l2_normalize.h"
#include "ops/layer_norm.h"
#include "ops/leaky_relu.h"
#include "ops/less.h"
#include "ops/less_equal.h"
#include "ops/log.h"
#include "ops/logical_and.h"
#include "ops/logical_not.h"
#include "ops/logical_or.h"
#include "ops/logical_xor.h"
#include "ops/log1p.h"
#include "ops/lp_normalization.h"
#include "ops/lrn.h"
#include "ops/lsh_projection.h"
#include "ops/lstm.h"
#include "ops/fusion/mat_mul_fusion.h"
#include "ops/max_pool.h"
#include "ops/maximum.h"
#include "ops/switch_layer.h"
#include "ops/mfcc.h"
#include "ops/minimum.h"
#include "ops/mod.h"
#include "ops/mul.h"
#include "ops/neg.h"
#include "ops/non_max_suppression.h"
#include "ops/not_equal.h"
#include "ops/one_hot.h"
#include "ops/ones_like.h"
#include "ops/pad.h"
#include "ops/prelu.h"
#include "ops/prior_box.h"
#include "ops/proposal.h"
#include "ops/quant_dtype_cast.h"
#include "ops/ragged_range.h"
#include "ops/range.h"
#include "ops/rank.h"
#include "ops/real_div.h"
#include "ops/reciprocal.h"
#include "ops/reduce.h"
#include "ops/relu6.h"
#include "ops/reshape.h"
#include "ops/resize.h"
#include "ops/reverse_sequence.h"
#include "ops/reverse_v2.h"
#include "ops/rfft.h"
#include "ops/roi_pooling.h"
#include "ops/round.h"
#include "ops/rsqrt.h"
#include "ops/scale.h"
#include "ops/scatter_nd.h"
#include "ops/scatter_nd_update.h"
#include "ops/select.h"
#include "ops/sgd.h"
#include "ops/shape.h"
#include "ops/sigmoid.h"
#include "ops/sigmoid_cross_entropy_with_logits.h"
#include "ops/sin.h"
#include "ops/skip_gram.h"
#include "ops/smooth_l1_loss.h"
#include "ops/softmax.h"
#include "ops/softmax_cross_entropy_with_logits.h"
#include "ops/space_to_batch.h"
#include "ops/space_to_batch_nd.h"
#include "ops/space_to_depth.h"
#include "ops/sparse_softmax_cross_entropy_with_logits.h"
#include "ops/sparse_to_dense.h"
#include "ops/sparse_fill_empty_rows.h"
#include "ops/sparse_reshape.h"
#include "ops/sparse_segment_sum.h"
#include "ops/split.h"
#include "ops/square.h"
#include "ops/squeeze.h"
#include "ops/sqrt.h"
#include "ops/squared_difference.h"
#include "ops/stack.h"
#include "ops/strided_slice.h"
#include "ops/sub.h"
#include "ops/switch.h"
#include "ops/tan.h"
#include "ops/tanh.h"
#include "ops/tensor_list_from_tensor.h"
#include "ops/tensor_list_get_item.h"
#include "ops/tensor_list_reserve.h"
#include "ops/tensor_list_set_item.h"
#include "ops/tensor_list_stack.h"
#include "ops/tile.h"
#include "ops/transpose.h"
#include "ops/unique.h"
#include "ops/unstack.h"
#include "ops/unsqueeze.h"
#include "ops/unsorted_segment_sum.h"
#include "ops/where.h"
#include "ops/zeros_like.h"
#include "ops/grad/activation_grad.h"
#include "ops/grad/add_grad.h"
#include "ops/grad/avg_pool_grad.h"
#include "ops/grad/bias_add_grad.h"
#include "ops/grad/batch_norm_grad.h"
#include "ops/grad/binary_cross_entropy_grad.h"
#include "ops/grad/de_conv2d_grad_filter.h"
#include "ops/grad/div_grad.h"
#include "ops/grad/dropout_grad.h"
#include "ops/grad/flatten_grad.h"
#include "ops/grad/group_conv2d_grad_input.h"
#include "ops/grad/layer_norm_grad.h"
#include "ops/grad/log_grad.h"
#include "ops/grad/lstm_grad.h"
#include "ops/grad/lstm_grad_data.h"
#include "ops/grad/lstm_grad_weight.h"
#include "ops/grad/max_pool_grad.h"
#include "ops/grad/maximum_grad.h"
#include "ops/grad/minimum_grad.h"
#include "ops/grad/mul_grad.h"
#include "ops/grad/neg_grad.h"
#include "ops/grad/pooling_grad.h"
#include "ops/grad/power_grad.h"
#include "ops/grad/resize_grad.h"
#include "ops/grad/rsqrt_grad.h"
#include "ops/grad/sigmoid_cross_entropy_with_logits_grad.h"
#include "ops/grad/smooth_l1_loss_grad.h"
#include "ops/grad/sqrt_grad.h"
#include "ops/grad/sub_grad.h"
#include "ops/fusion/activation.h"
#include "ops/fusion/add_fusion.h"
#include "ops/fusion/adder_fusion.h"
#include "ops/fusion/arg_max_fusion.h"
#include "ops/fusion/arg_min_fusion.h"
#include "ops/fusion/avg_pool_fusion.h"
#include "ops/fusion/conv2d_backprop_filter_fusion.h"
#include "ops/fusion/conv2d_backprop_input_fusion.h"
#include "ops/fusion/conv2d_fusion.h"
#include "ops/fusion/conv2d_transpose_fusion.h"
#include "ops/fusion/div_fusion.h"
#include "ops/fusion/embedding_lookup_fusion.h"
#include "ops/fusion/exp_fusion.h"
#include "ops/fusion/full_connection.h"
#include "ops/fusion/l2_normalize_fusion.h"
#include "ops/fusion/layer_norm_fusion.h"
#include "ops/fusion/max_pool_fusion.h"
#include "ops/fusion/mul_fusion.h"
#include "ops/fusion/pad_fusion.h"
#include "ops/fusion/partial_fusion.h"
#include "ops/fusion/pow_fusion.h"
#include "ops/fusion/prelu_fusion.h"
#include "ops/fusion/reduce_fusion.h"
#include "ops/fusion/scale_fusion.h"
#include "ops/fusion/slice_fusion.h"
#include "ops/fusion/sub_fusion.h"
#include "ops/fusion/tile_fusion.h"
#include "ops/fusion/topk_fusion.h"
#include "ops/fusion/groupnorm_fusion.h"
#include "ops/gru.h"
#include "ops/non_zero.h"
#include "ops/invert_permutation.h"
#include "ops/size.h"
#include "ops/random_standard_normal.h"
#include "ops/crop_and_resize.h"
#include "ops/erf.h"
#include "ops/grad/strided_slice_grad.h"
#include "ops/is_finite.h"
#include "ops/lin_space.h"
#include "ops/uniform_real.h"
#include "ops/grad/abs_grad.h"
#include "ops/splice.h"
#include "ops/log_softmax.h"
#include "ops/call.h"
#include "ops/cumsum.h"
#include "ops/split_with_overlap.h"
#include "ops/glu.h"
#include "ops/tensor_array.h"
#include "ops/tensor_array_read.h"
#include "ops/tensor_array_write.h"
#include "ops/affine.h"
#include "ops/all_gather.h"
#include "ops/reduce_scatter.h"
#include "ops/dynamic_quant.h"
#include "ops/random_normal.h"
#include "ops/nllloss.h"
#include "ops/grad/nllloss_grad.h"
#include "ops/format_transpose.h"
#include "ops/gather_d.h"
#include "ops/tensor_scatter_add.h"
#include "ops/scatter_elements.h"
#include "ops/triu.h"
#include "ops/tril.h"

namespace mindspore::lite::ops {
#define FUNC_MSOP2SCHEMAOP_DECLARE(OP) std::unique_ptr<schema::PrimitiveT> MSOp2SchemaOp(const mindspore::ops::OP *op);

#ifdef PRIMITIVE_WRITEABLE
FUNC_MSOP2SCHEMAOP_DECLARE(Abs)
FUNC_MSOP2SCHEMAOP_DECLARE(Activation)
FUNC_MSOP2SCHEMAOP_DECLARE(ActivationGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(Adam)
FUNC_MSOP2SCHEMAOP_DECLARE(AddFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(AdderFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(AddGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(AddN)
FUNC_MSOP2SCHEMAOP_DECLARE(All)
FUNC_MSOP2SCHEMAOP_DECLARE(ApplyMomentum)
FUNC_MSOP2SCHEMAOP_DECLARE(Argmax)
FUNC_MSOP2SCHEMAOP_DECLARE(ArgMaxFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(ArgMin)
FUNC_MSOP2SCHEMAOP_DECLARE(ArgMinFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(Asin)
FUNC_MSOP2SCHEMAOP_DECLARE(Assert)
FUNC_MSOP2SCHEMAOP_DECLARE(Assign)
FUNC_MSOP2SCHEMAOP_DECLARE(AssignAdd)
FUNC_MSOP2SCHEMAOP_DECLARE(Atan)
FUNC_MSOP2SCHEMAOP_DECLARE(AudioSpectrogram)
FUNC_MSOP2SCHEMAOP_DECLARE(AvgPoolFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(AvgPoolGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(BatchNorm)
FUNC_MSOP2SCHEMAOP_DECLARE(BatchNormGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(BatchToSpace)
FUNC_MSOP2SCHEMAOP_DECLARE(BatchToSpaceND)
FUNC_MSOP2SCHEMAOP_DECLARE(BiasAdd)
FUNC_MSOP2SCHEMAOP_DECLARE(BiasAddGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(BinaryCrossEntropy)
FUNC_MSOP2SCHEMAOP_DECLARE(BinaryCrossEntropyGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(BroadcastTo)
FUNC_MSOP2SCHEMAOP_DECLARE(Cast)
FUNC_MSOP2SCHEMAOP_DECLARE(Ceil)
FUNC_MSOP2SCHEMAOP_DECLARE(Clip)
FUNC_MSOP2SCHEMAOP_DECLARE(Concat)
FUNC_MSOP2SCHEMAOP_DECLARE(Attention)
FUNC_MSOP2SCHEMAOP_DECLARE(ConstantOfShape)
FUNC_MSOP2SCHEMAOP_DECLARE(Conv2DBackpropFilterFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(Conv2DBackpropInputFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(Conv2DFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(Conv2dTransposeFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(Cos)
FUNC_MSOP2SCHEMAOP_DECLARE(Crop)
FUNC_MSOP2SCHEMAOP_DECLARE(CustomExtractFeatures)
FUNC_MSOP2SCHEMAOP_DECLARE(CustomNormalize)
FUNC_MSOP2SCHEMAOP_DECLARE(CustomPredict)
FUNC_MSOP2SCHEMAOP_DECLARE(DeConv2DGradFilter)
FUNC_MSOP2SCHEMAOP_DECLARE(Depend)
FUNC_MSOP2SCHEMAOP_DECLARE(DepthToSpace)
FUNC_MSOP2SCHEMAOP_DECLARE(DetectionPostProcess)
FUNC_MSOP2SCHEMAOP_DECLARE(DivFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(DivGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(Dropout)
FUNC_MSOP2SCHEMAOP_DECLARE(DropoutGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(Eltwise)
FUNC_MSOP2SCHEMAOP_DECLARE(Elu)
FUNC_MSOP2SCHEMAOP_DECLARE(EmbeddingLookupFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(Equal)
FUNC_MSOP2SCHEMAOP_DECLARE(ExpFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(ExpandDims)
FUNC_MSOP2SCHEMAOP_DECLARE(FakeQuantWithMinMaxVars)
FUNC_MSOP2SCHEMAOP_DECLARE(FakeQuantWithMinMaxVarsPerChannel)
FUNC_MSOP2SCHEMAOP_DECLARE(FftImag)
FUNC_MSOP2SCHEMAOP_DECLARE(FftReal)
FUNC_MSOP2SCHEMAOP_DECLARE(Fill)
FUNC_MSOP2SCHEMAOP_DECLARE(Flatten)
FUNC_MSOP2SCHEMAOP_DECLARE(FlattenGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(Floor)
FUNC_MSOP2SCHEMAOP_DECLARE(FloorDiv)
FUNC_MSOP2SCHEMAOP_DECLARE(FloorMod)
FUNC_MSOP2SCHEMAOP_DECLARE(FullConnection)
FUNC_MSOP2SCHEMAOP_DECLARE(FusedBatchNorm)
FUNC_MSOP2SCHEMAOP_DECLARE(Gather)
FUNC_MSOP2SCHEMAOP_DECLARE(GatherNd)
FUNC_MSOP2SCHEMAOP_DECLARE(Greater)
FUNC_MSOP2SCHEMAOP_DECLARE(GreaterEqual)
FUNC_MSOP2SCHEMAOP_DECLARE(GroupConv2DGradInput)
FUNC_MSOP2SCHEMAOP_DECLARE(HashtableLookup)
FUNC_MSOP2SCHEMAOP_DECLARE(InstanceNorm)
FUNC_MSOP2SCHEMAOP_DECLARE(LayerNormFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(LeakyRelu)
FUNC_MSOP2SCHEMAOP_DECLARE(Less)
FUNC_MSOP2SCHEMAOP_DECLARE(LessEqual)
FUNC_MSOP2SCHEMAOP_DECLARE(Log)
FUNC_MSOP2SCHEMAOP_DECLARE(LogGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(LogicalAnd)
FUNC_MSOP2SCHEMAOP_DECLARE(LogicalNot)
FUNC_MSOP2SCHEMAOP_DECLARE(LogicalOr)
FUNC_MSOP2SCHEMAOP_DECLARE(LogicalXor)
FUNC_MSOP2SCHEMAOP_DECLARE(LpNormalization)
FUNC_MSOP2SCHEMAOP_DECLARE(LRN)
FUNC_MSOP2SCHEMAOP_DECLARE(LshProjection)
FUNC_MSOP2SCHEMAOP_DECLARE(LSTM)
FUNC_MSOP2SCHEMAOP_DECLARE(LSTMGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(LSTMGradData)
FUNC_MSOP2SCHEMAOP_DECLARE(LSTMGradWeight)
FUNC_MSOP2SCHEMAOP_DECLARE(L2NormalizeFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(MatMulFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(Maximum)
FUNC_MSOP2SCHEMAOP_DECLARE(MaximumGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(MaxPoolFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(MaxPoolGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(SwitchLayer)
FUNC_MSOP2SCHEMAOP_DECLARE(Mfcc)
FUNC_MSOP2SCHEMAOP_DECLARE(Minimum)
FUNC_MSOP2SCHEMAOP_DECLARE(MinimumGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(Mod)
FUNC_MSOP2SCHEMAOP_DECLARE(MulFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(MulGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(Neg)
FUNC_MSOP2SCHEMAOP_DECLARE(NegGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(NotEqual)
FUNC_MSOP2SCHEMAOP_DECLARE(NonMaxSuppression)
FUNC_MSOP2SCHEMAOP_DECLARE(OneHot)
FUNC_MSOP2SCHEMAOP_DECLARE(OnesLike)
FUNC_MSOP2SCHEMAOP_DECLARE(PadFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(PartialFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(PowFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(PowerGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(PReLUFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(PriorBox)
FUNC_MSOP2SCHEMAOP_DECLARE(Proposal)
FUNC_MSOP2SCHEMAOP_DECLARE(RaggedRange)
FUNC_MSOP2SCHEMAOP_DECLARE(Rank)
FUNC_MSOP2SCHEMAOP_DECLARE(Range)
FUNC_MSOP2SCHEMAOP_DECLARE(Rank)
FUNC_MSOP2SCHEMAOP_DECLARE(RealDiv)
FUNC_MSOP2SCHEMAOP_DECLARE(Reciprocal)
FUNC_MSOP2SCHEMAOP_DECLARE(ReduceFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(Reshape)
FUNC_MSOP2SCHEMAOP_DECLARE(Resize)
FUNC_MSOP2SCHEMAOP_DECLARE(ReverseSequence)
FUNC_MSOP2SCHEMAOP_DECLARE(ReverseV2)
FUNC_MSOP2SCHEMAOP_DECLARE(Rfft)
FUNC_MSOP2SCHEMAOP_DECLARE(ROIPooling)
FUNC_MSOP2SCHEMAOP_DECLARE(Round)
FUNC_MSOP2SCHEMAOP_DECLARE(Rsqrt)
FUNC_MSOP2SCHEMAOP_DECLARE(QuantDTypeCast)
FUNC_MSOP2SCHEMAOP_DECLARE(Scale)
FUNC_MSOP2SCHEMAOP_DECLARE(ScaleFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(ScatterNd)
FUNC_MSOP2SCHEMAOP_DECLARE(Select)
FUNC_MSOP2SCHEMAOP_DECLARE(SGD)
FUNC_MSOP2SCHEMAOP_DECLARE(Shape)
FUNC_MSOP2SCHEMAOP_DECLARE(SigmoidCrossEntropyWithLogits)
FUNC_MSOP2SCHEMAOP_DECLARE(SigmoidCrossEntropyWithLogitsGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(Sin)
FUNC_MSOP2SCHEMAOP_DECLARE(SkipGram)
FUNC_MSOP2SCHEMAOP_DECLARE(SliceFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(SmoothL1Loss)
FUNC_MSOP2SCHEMAOP_DECLARE(SmoothL1LossGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(Softmax)
FUNC_MSOP2SCHEMAOP_DECLARE(SoftmaxCrossEntropyWithLogits)
FUNC_MSOP2SCHEMAOP_DECLARE(SpaceToBatch)
FUNC_MSOP2SCHEMAOP_DECLARE(SpaceToBatchND)
FUNC_MSOP2SCHEMAOP_DECLARE(SpaceToDepth)
FUNC_MSOP2SCHEMAOP_DECLARE(SparseSoftmaxCrossEntropyWithLogits)
FUNC_MSOP2SCHEMAOP_DECLARE(SparseToDense)
FUNC_MSOP2SCHEMAOP_DECLARE(Split)
FUNC_MSOP2SCHEMAOP_DECLARE(Sqrt)
FUNC_MSOP2SCHEMAOP_DECLARE(Square)
FUNC_MSOP2SCHEMAOP_DECLARE(SquaredDifference)
FUNC_MSOP2SCHEMAOP_DECLARE(Squeeze)
FUNC_MSOP2SCHEMAOP_DECLARE(Stack)
FUNC_MSOP2SCHEMAOP_DECLARE(StridedSlice)
FUNC_MSOP2SCHEMAOP_DECLARE(Sub)
FUNC_MSOP2SCHEMAOP_DECLARE(SubFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(SubGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(Switch)
FUNC_MSOP2SCHEMAOP_DECLARE(Tan)
FUNC_MSOP2SCHEMAOP_DECLARE(TensorListFromTensor)
FUNC_MSOP2SCHEMAOP_DECLARE(TensorListGetItem)
FUNC_MSOP2SCHEMAOP_DECLARE(TensorListReserve)
FUNC_MSOP2SCHEMAOP_DECLARE(TensorListSetItem)
FUNC_MSOP2SCHEMAOP_DECLARE(TensorListStack)
FUNC_MSOP2SCHEMAOP_DECLARE(TileFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(TopKFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(Transpose)
FUNC_MSOP2SCHEMAOP_DECLARE(Unique)
FUNC_MSOP2SCHEMAOP_DECLARE(UnsortedSegmentSum)
FUNC_MSOP2SCHEMAOP_DECLARE(Unsqueeze)
FUNC_MSOP2SCHEMAOP_DECLARE(Unstack)
FUNC_MSOP2SCHEMAOP_DECLARE(Where)
FUNC_MSOP2SCHEMAOP_DECLARE(ZerosLike)
FUNC_MSOP2SCHEMAOP_DECLARE(Select)
FUNC_MSOP2SCHEMAOP_DECLARE(GRU)
FUNC_MSOP2SCHEMAOP_DECLARE(NonZero)
FUNC_MSOP2SCHEMAOP_DECLARE(InvertPermutation)
FUNC_MSOP2SCHEMAOP_DECLARE(Size)
FUNC_MSOP2SCHEMAOP_DECLARE(RandomStandardNormal)
FUNC_MSOP2SCHEMAOP_DECLARE(CropAndResize)
FUNC_MSOP2SCHEMAOP_DECLARE(Erf)
FUNC_MSOP2SCHEMAOP_DECLARE(StridedSliceGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(IsFinite)
FUNC_MSOP2SCHEMAOP_DECLARE(LinSpace)
FUNC_MSOP2SCHEMAOP_DECLARE(UniformReal)
FUNC_MSOP2SCHEMAOP_DECLARE(AbsGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(RsqrtGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(SqrtGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(LayerNormGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(ResizeGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(Splice)
FUNC_MSOP2SCHEMAOP_DECLARE(LogSoftmax)
FUNC_MSOP2SCHEMAOP_DECLARE(Call)
FUNC_MSOP2SCHEMAOP_DECLARE(CumSum)
FUNC_MSOP2SCHEMAOP_DECLARE(SplitWithOverlap)
FUNC_MSOP2SCHEMAOP_DECLARE(GLU)
FUNC_MSOP2SCHEMAOP_DECLARE(TensorArray)
FUNC_MSOP2SCHEMAOP_DECLARE(TensorArrayRead)
FUNC_MSOP2SCHEMAOP_DECLARE(TensorArrayWrite)
FUNC_MSOP2SCHEMAOP_DECLARE(Affine)
FUNC_MSOP2SCHEMAOP_DECLARE(ScatterNdUpdate)
FUNC_MSOP2SCHEMAOP_DECLARE(AllGather)
FUNC_MSOP2SCHEMAOP_DECLARE(ReduceScatter)
FUNC_MSOP2SCHEMAOP_DECLARE(DynamicQuant)
FUNC_MSOP2SCHEMAOP_DECLARE(RandomNormal)
FUNC_MSOP2SCHEMAOP_DECLARE(NLLLoss)
FUNC_MSOP2SCHEMAOP_DECLARE(NLLLossGrad)
FUNC_MSOP2SCHEMAOP_DECLARE(FormatTranspose)
FUNC_MSOP2SCHEMAOP_DECLARE(GatherD)
FUNC_MSOP2SCHEMAOP_DECLARE(GroupNormFusion)
FUNC_MSOP2SCHEMAOP_DECLARE(Log1p)
FUNC_MSOP2SCHEMAOP_DECLARE(TensorScatterAdd)
FUNC_MSOP2SCHEMAOP_DECLARE(ScatterElements)
FUNC_MSOP2SCHEMAOP_DECLARE(Triu)
FUNC_MSOP2SCHEMAOP_DECLARE(Tril)
FUNC_MSOP2SCHEMAOP_DECLARE(SparseFillEmptyRows)
FUNC_MSOP2SCHEMAOP_DECLARE(SparseReshape)
FUNC_MSOP2SCHEMAOP_DECLARE(SparseSegmentSum)
#endif
}  // namespace mindspore::lite::ops
#else
#define FUNC_MSOP2SCHEMAOP_DECLARE(OP)
#endif
#endif  // MINDSPORE_LITE_SRC_COMMON_OPS_OPS_FUNC_DECLARE_H_
