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
#include "nnacl/infer/infer_register.h"

#ifdef _MSC_VER
#include "nnacl/infer/activation_grad_infer.h"
#include "nnacl/infer/adam_infer.h"
#include "nnacl/infer/add_sub_grad_infer.h"
#include "nnacl/infer/addn_infer.h"
#include "nnacl/infer/affine_infer.h"
#include "nnacl/infer/all_gather_infer.h"
#include "nnacl/infer/apply_momentum_infer.h"
#include "nnacl/infer/argmin_max_infer.h"
#include "nnacl/infer/arithmetic_compare_infer.h"
#include "nnacl/infer/arithmetic_grad_infer.h"
#include "nnacl/infer/arithmetic_infer.h"
#include "nnacl/infer/assert_op_infer.h"
#include "nnacl/infer/assign_add_infer.h"
#include "nnacl/infer/assign_infer.h"
#include "nnacl/infer/attention_infer.h"
#include "nnacl/infer/encoder_layer_infer.h"
#include "nnacl/infer/audio_spectrogram_infer.h"
#include "nnacl/infer/batch_to_space_infer.h"
#include "nnacl/infer/bias_grad_infer.h"
#include "nnacl/infer/binary_cross_entropy_infer.h"
#include "nnacl/infer/bn_grad_infer.h"
#include "nnacl/infer/broadcast_to_infer.h"
#include "nnacl/infer/cast_infer.h"
#include "nnacl/infer/common_infer.h"
#include "nnacl/infer/concat_infer.h"
#include "nnacl/infer/constant_of_shape_infer.h"
#ifdef MSLITE_ENABLE_CONTROLFLOW
#include "nnacl/infer/control/tensor_array_infer.h"
#include "nnacl/infer/control/tensor_array_read_infer.h"
#include "nnacl/infer/control/tensor_array_write_infer.h"
#include "nnacl/infer/control/tensorlist_fromtensor_infer.h"
#include "nnacl/infer/control/tensorlist_getitem_infer.h"
#include "nnacl/infer/control/tensorlist_reserve_infer.h"
#include "nnacl/infer/control/tensorlist_setitem_infer.h"
#include "nnacl/infer/control/tensorlist_stack_infer.h"
#endif
#include "nnacl/infer/conv2d_grad_filter_infer.h"
#include "nnacl/infer/conv2d_grad_input_infer.h"
#include "nnacl/infer/conv2d_infer.h"
#include "nnacl/infer/crop_and_resize_infer.h"
#include "nnacl/infer/crop_infer.h"
#include "nnacl/infer/cumsum_infer.h"
#include "nnacl/infer/deconv2d_infer.h"
#include "nnacl/infer/depth_to_space_infer.h"
#include "nnacl/infer/depthwise_conv2d_infer.h"
#include "nnacl/infer/detection_post_process_infer.h"
#include "nnacl/infer/dropout_grad_infer.h"
#include "nnacl/infer/dropout_infer.h"
#include "nnacl/infer/dynamic_quant_infer.h"
#include "nnacl/infer/embedding_lookup_infer.h"
#include "nnacl/infer/expand_dims_infer.h"
#include "nnacl/infer/fft_imag_infer.h"
#include "nnacl/infer/fft_real_infer.h"
#include "nnacl/infer/fill_infer.h"
#include "nnacl/infer/flatten_grad_infer.h"
#include "nnacl/infer/flatten_infer.h"
#include "nnacl/infer/full_connection_infer.h"
#include "nnacl/infer/fused_batchnorm_infer.h"
#include "nnacl/infer/gather_infer.h"
#include "nnacl/infer/gather_nd_infer.h"
#include "nnacl/infer/glu_infer.h"
#include "nnacl/infer/group_conv2d_grad_input_infer.h"
#include "nnacl/infer/gru_infer.h"
#include "nnacl/infer/instance_norm_infer.h"
#include "nnacl/infer/invert_permutation_infer.h"
#include "nnacl/infer/layer_norm_grad_infer.h"
#include "nnacl/infer/layer_norm_infer.h"
#include "nnacl/infer/lin_space_infer.h"
#include "nnacl/infer/log_softmax_infer.h"
#include "nnacl/infer/lstm_grad_data_infer.h"
#include "nnacl/infer/lstm_grad_infer.h"
#include "nnacl/infer/lstm_grad_weight_infer.h"
#include "nnacl/infer/lstm_infer.h"
#include "nnacl/infer/matmul_infer.h"
#include "nnacl/infer/max_min_grad_infer.h"
#include "nnacl/infer/mean_infer.h"
#include "nnacl/infer/mfcc_infer.h"
#include "nnacl/infer/nllloss_grad_infer.h"
#include "nnacl/infer/nllloss_infer.h"
#include "nnacl/infer/non_max_suppression_infer.h"
#include "nnacl/infer/one_hot_infer.h"
#include "nnacl/infer/pad_infer.h"
#include "nnacl/infer/partial_infer.h"
#include "nnacl/infer/pooling_grad_infer.h"
#include "nnacl/infer/pooling_infer.h"
#include "nnacl/infer/power_infer.h"
#include "nnacl/infer/prior_box_infer.h"
#include "nnacl/infer/quant_dtype_cast_infer.h"
#include "nnacl/infer/ragged_range_infer.h"
#include "nnacl/infer/random_normal_infer.h"
#include "nnacl/infer/random_standard_normal_infer.h"
#include "nnacl/infer/range_infer.h"
#include "nnacl/infer/rank_infer.h"
#include "nnacl/infer/reduce_infer.h"
#include "nnacl/infer/reduce_scatter_infer.h"
#include "nnacl/infer/reshape_infer.h"
#include "nnacl/infer/resize_grad_infer.h"
#include "nnacl/infer/resize_infer.h"
#include "nnacl/infer/rfft_infer.h"
#include "nnacl/infer/roi_pooling_infer.h"
#include "nnacl/infer/scatter_nd_infer.h"
#include "nnacl/infer/scatter_nd_update_infer.h"
#include "nnacl/infer/select_infer.h"
#include "nnacl/infer/sgd_infer.h"
#ifndef RUNTIME_PASS_CLIP
#include "nnacl/infer/shape_fusion_infer.h"
#endif
#include "nnacl/infer/shape_infer.h"
#include "nnacl/infer/size_infer.h"
#include "nnacl/infer/slice_infer.h"
#include "nnacl/infer/softmax_cross_entropy_infer.h"
#include "nnacl/infer/softmax_infer.h"
#include "nnacl/infer/space_to_batch_infer.h"
#include "nnacl/infer/space_to_batch_nd_infer.h"
#include "nnacl/infer/space_to_depth_infer.h"
#include "nnacl/infer/sparse_softmax_cross_entropy_with_logits_infer.h"
#include "nnacl/infer/sparse_to_dense_infer.h"
#include "nnacl/infer/splice_infer.h"
#include "nnacl/infer/split_infer.h"
#include "nnacl/infer/split_with_over_lap_infer.h"
#include "nnacl/infer/squeeze_infer.h"
#include "nnacl/infer/stack_infer.h"
#include "nnacl/infer/strided_slice_grad_infer.h"
#include "nnacl/infer/strided_slice_infer.h"
#ifdef MSLITE_ENABLE_STRING_KERNEL
#include "nnacl/infer/string/custom_extract_features_infer.h"
#include "nnacl/infer/string/custom_normalize_infer.h"
#include "nnacl/infer/string/custom_predict_infer.h"
#include "nnacl/infer/string/hashtable_lookup_infer.h"
#include "nnacl/infer/string/lsh_projection_infer.h"
#include "nnacl/infer/string/skip_gram_infer.h"
#endif
#include "nnacl/infer/tile_infer.h"
#include "nnacl/infer/topk_infer.h"
#include "nnacl/infer/transpose_infer.h"
#include "nnacl/infer/uniform_real_infer.h"
#include "nnacl/infer/unique_infer.h"
#include "nnacl/infer/unsorted_segment_sum_infer.h"
#include "nnacl/infer/unsqueeze_infer.h"
#include "nnacl/infer/unstack_infer.h"
#include "nnacl/infer/where_infer.h"
#include "nnacl/infer/isfinite_infer.h"

InferShape g_infer_func[PrimType_MAX] = {0};
InferShape g_inner_op_infer_func[PrimType_InnerOpMax - PrimType_InnerOpMin] = {0};
void RegAllInferFunc1() {
  g_infer_func[PrimType_NONE] = NULL;
  g_infer_func[PrimType_Abs] = CommonInferShape;
  g_infer_func[PrimType_AbsGrad] = CommonGradInferShape;
  g_infer_func[PrimType_Activation] = CommonInferShape;
  g_infer_func[PrimType_ActivationGrad] = ActivationGradInferShape;
  g_infer_func[PrimType_Adam] = AdamInferShape;
  g_infer_func[PrimType_AdderFusion] = Conv2dInferShape;
  g_infer_func[PrimType_AddFusion] = ArithmeticInferShape;
  g_infer_func[PrimType_AddGrad] = AddSubGradInferShape;
  g_infer_func[PrimType_AddN] = AddnInferShape;
  g_infer_func[PrimType_Affine] = AffineInferShape;
  g_infer_func[PrimType_All] = NULL;
  g_infer_func[PrimType_AllGather] = AllGatherInferShape;
  g_infer_func[PrimType_ApplyMomentum] = ApplyMomentumInferShape;
  g_infer_func[PrimType_ArgMaxFusion] = ArgMinMaxInferShape;
  g_infer_func[PrimType_ArgMinFusion] = ArgMinMaxInferShape;
  g_infer_func[PrimType_Assert] = AssertOpInferShape;
  g_infer_func[PrimType_Assign] = AssignInferShape;
  g_infer_func[PrimType_AssignAdd] = AssignAddInferShape;
  g_infer_func[PrimType_Attention] = AttentionInferShape;
  g_infer_func[PrimType_AudioSpectrogram] = AudioSpectrogramInferShape;
  g_infer_func[PrimType_AvgPoolFusion] = PoolingInferShape;
  g_infer_func[PrimType_AvgPoolGrad] = PoolingGradInferShape;
  g_infer_func[PrimType_BatchNorm] = CommonInferShape;
  g_infer_func[PrimType_BatchNormGrad] = BnGradInferShape;
  g_infer_func[PrimType_BatchToSpace] = BatchToSpaceInferShape;
  g_infer_func[PrimType_BatchToSpaceND] = NULL;
  g_infer_func[PrimType_BiasAdd] = ArithmeticInferShape;
  g_infer_func[PrimType_BiasAddGrad] = BiasGradInferShape;
  g_infer_func[PrimType_BinaryCrossEntropy] = BinaryCrossEntropyInferShape;
  g_infer_func[PrimType_BinaryCrossEntropyGrad] = CommonInferShape;
  g_infer_func[PrimType_BroadcastTo] = BroadcastToInferShape;
  g_infer_func[PrimType_Call] = NULL;
  g_infer_func[PrimType_Cast] = CastInferShape;
  g_infer_func[PrimType_Ceil] = CommonInferShape;
  g_infer_func[PrimType_Clip] = CommonInferShape;
  g_infer_func[PrimType_Concat] = ConcatInferShape;
  g_infer_func[PrimType_ConstantOfShape] = ConstantOfShapeInferShape;
  g_infer_func[PrimType_Conv2DBackpropFilterFusion] = Conv2dGradFilterInferShape;
  g_infer_func[PrimType_Conv2DBackpropInputFusion] = Conv2dGradInputInferShape;
  g_infer_func[PrimType_Conv2DFusion] = Conv2dInferShape;
  g_infer_func[PrimType_Conv2dTransposeFusion] = Deconv2dInferShape;
  g_infer_func[PrimType_Cos] = CommonInferShape;
  g_infer_func[PrimType_Crop] = CropInferShape;
  g_infer_func[PrimType_CropAndResize] = CropAndResizeInferShape;
  g_infer_func[PrimType_CumSum] = CumsumInferShape;
  g_infer_func[PrimType_Custom] = NULL;
#ifdef MSLITE_ENABLE_STRING_KERNEL
  g_infer_func[PrimType_CustomExtractFeatures] = CustomExtractFeaturesInferShape;
#endif
}

void RegAllInferFunc2() {
#ifdef MSLITE_ENABLE_STRING_KERNEL
  g_infer_func[PrimType_CustomNormalize] = CustomNormalizeInferShape;
  g_infer_func[PrimType_CustomPredict] = CustomPredictInferShape;
#endif
  g_infer_func[PrimType_DeConv2DGradFilter] = NULL;
  g_infer_func[PrimType_Depend] = CommonInferShape;
  g_infer_func[PrimType_DepthToSpace] = DepthToSpaceInferShape;
  g_infer_func[PrimType_DetectionPostProcess] = DetectionPostProcessInferShape;
  g_infer_func[PrimType_DivFusion] = ArithmeticInferShape;
  g_infer_func[PrimType_DivGrad] = ArithmeticGradInferShape;
  g_infer_func[PrimType_Dropout] = DropoutInferShape;
  g_infer_func[PrimType_DropoutGrad] = DropoutGradInferShape;
  g_infer_func[PrimType_DynamicQuant] = DynamicQuantInferShape;
  g_infer_func[PrimType_Eltwise] = ArithmeticInferShape;
  g_infer_func[PrimType_Elu] = CommonInferShape;
  g_infer_func[PrimType_EmbeddingLookupFusion] = EmbeddingLookupInferShape;
  g_infer_func[PrimType_Equal] = ArithmeticCompareInferShape;
  g_infer_func[PrimType_Erf] = CommonInferShape;
  g_infer_func[PrimType_ExpandDims] = ExpandDimsInferShape;
  g_infer_func[PrimType_ExpFusion] = CommonInferShape;
  g_infer_func[PrimType_FakeQuantWithMinMaxVars] = CommonInferShape;
  g_infer_func[PrimType_FakeQuantWithMinMaxVarsPerChannel] = NULL;
  g_infer_func[PrimType_FftImag] = FftImagInferShape;
  g_infer_func[PrimType_FftReal] = FftRealInferShape;
  g_infer_func[PrimType_Fill] = FillInferShape;
  g_infer_func[PrimType_Flatten] = FlattenInferShape;
  g_infer_func[PrimType_FlattenGrad] = FlattenGradInferShape;
  g_infer_func[PrimType_Floor] = CommonInferShapeWithOneInput;
  g_infer_func[PrimType_FloorDiv] = ArithmeticInferShape;
  g_infer_func[PrimType_FloorMod] = ArithmeticInferShape;
  g_infer_func[PrimType_FullConnection] = FullConnectionInferShape;
  g_infer_func[PrimType_FusedBatchNorm] = FusedBatchNormInferShape;
  g_infer_func[PrimType_Gather] = GatherInferShape;
  g_infer_func[PrimType_GatherNd] = GatherNdInferShape;
  g_infer_func[PrimType_GenOP] = NULL;
  g_infer_func[PrimType_GLU] = GluInferShape;
  g_infer_func[PrimType_Greater] = ArithmeticCompareInferShape;
  g_infer_func[PrimType_GreaterEqual] = ArithmeticCompareInferShape;
  g_infer_func[PrimType_GRU] = GruInferShape;
#ifdef MSLITE_ENABLE_STRING_KERNEL
  g_infer_func[PrimType_HashtableLookup] = HashtableLoopupInferShape;
#endif
  g_infer_func[PrimType_InstanceNorm] = InstanceNormInferShape;
  g_infer_func[PrimType_InvertPermutation] = InvertPermutationInferShape;
  g_infer_func[PrimType_IsFinite] = IsFiniteInferShape;
  g_infer_func[PrimType_L2NormalizeFusion] = CommonInferShape;
  g_infer_func[PrimType_LayerNormFusion] = LayerNormInferShape;
  g_infer_func[PrimType_LayerNormGrad] = LayerNormGradInferShape;
  g_infer_func[PrimType_LeakyRelu] = CommonInferShape;
  g_infer_func[PrimType_Less] = ArithmeticCompareInferShape;
  g_infer_func[PrimType_LessEqual] = ArithmeticCompareInferShape;
  g_infer_func[PrimType_LinSpace] = LinSpaceInferShape;
}

void RegAllInferFunc3() {
  g_infer_func[PrimType_Log] = CommonInferShape;
  g_infer_func[PrimType_LogGrad] = CommonGradInferShape;
  g_infer_func[PrimType_LogicalAnd] = ArithmeticInferShape;
  g_infer_func[PrimType_LogicalNot] = CommonInferShape;
  g_infer_func[PrimType_LogicalOr] = ArithmeticInferShape;
  g_infer_func[PrimType_LogSoftmax] = LogSoftmaxInferShape;
  g_infer_func[PrimType_LpNormalization] = NULL;
  g_infer_func[PrimType_LRN] = CommonInferShapeWithNHWC;
#ifdef MSLITE_ENABLE_STRING_KERNEL
  g_infer_func[PrimType_LshProjection] = LshProjectionInferShape;
#endif
  g_infer_func[PrimType_LSTM] = LstmInferShape;
  g_infer_func[PrimType_LSTMGrad] = LstmGradInferShape;
  g_infer_func[PrimType_LSTMGradData] = LstmGradDataInferShape;
  g_infer_func[PrimType_LSTMGradWeight] = LstmGradWeightInferShape;
  g_infer_func[PrimType_MatMulFusion] = MatmulInferShape;
  g_infer_func[PrimType_Maximum] = ArithmeticInferShape;
  g_infer_func[PrimType_MaximumGrad] = MaxMinGradInferShape;
  g_infer_func[PrimType_MaxPoolFusion] = PoolingInferShape;
  g_infer_func[PrimType_MaxPoolGrad] = PoolingGradInferShape;
  g_infer_func[PrimType_Merge] = NULL;
  g_infer_func[PrimType_Mfcc] = MfccInferShape;
  g_infer_func[PrimType_MIN] = NULL;
  g_infer_func[PrimType_Minimum] = ArithmeticInferShape;
  g_infer_func[PrimType_MinimumGrad] = MaxMinGradInferShape;
  g_infer_func[PrimType_Mod] = ArithmeticInferShape;
  g_infer_func[PrimType_MulFusion] = ArithmeticInferShape;
  g_infer_func[PrimType_MulGrad] = ArithmeticGradInferShape;
  g_infer_func[PrimType_Neg] = CommonInferShape;
  g_infer_func[PrimType_NegGrad] = CommonGradInferShape;
  g_infer_func[PrimType_NLLLoss] = NLLLossInferShape;
  g_infer_func[PrimType_NLLLossGrad] = NLLLossGradInferShape;
  g_infer_func[PrimType_NonMaxSuppression] = NonMaxSuppressionInferShape;
  g_infer_func[PrimType_NonZero] = NULL;
  g_infer_func[PrimType_NotEqual] = ArithmeticCompareInferShape;
  g_infer_func[PrimType_OneHot] = OneHotInferShape;
  g_infer_func[PrimType_OnesLike] = NULL;
  g_infer_func[PrimType_PadFusion] = PadInferShape;
  g_infer_func[PrimType_PartialFusion] = PartialInferShape;
  g_infer_func[PrimType_PowerGrad] = CommonGradInferShape;
  g_infer_func[PrimType_PowFusion] = PowerInferShape;
  g_infer_func[PrimType_PReLUFusion] = CommonInferShape;
  g_infer_func[PrimType_PriorBox] = PriorBoxInferShape;
  g_infer_func[PrimType_QuantDTypeCast] = QuantDtypeCastInferShape;
  g_infer_func[PrimType_RaggedRange] = RaggedRangeInferShape;
  g_infer_func[PrimType_RandomNormal] = RandomNormalInferShape;
  g_infer_func[PrimType_RandomStandardNormal] = RandomStandardNormalInferShape;
  g_infer_func[PrimType_Range] = RangeInferShape;
  g_infer_func[PrimType_Rank] = RankInferShape;
}

void RegAllInferFunc4() {
  g_infer_func[PrimType_RealDiv] = ArithmeticInferShape;
  g_infer_func[PrimType_Reciprocal] = CommonInferShape;
  g_infer_func[PrimType_ReduceFusion] = ReduceInferShape;
  g_infer_func[PrimType_ReduceScatter] = ReduceScatterInferShape;
  g_infer_func[PrimType_Reshape] = ReshapeInferShape;
  g_infer_func[PrimType_Resize] = ResizeInferShape;
  g_infer_func[PrimType_ResizeGrad] = ResizeGradInferShape;
  g_infer_func[PrimType_ReverseSequence] = CommonInferShape;
  g_infer_func[PrimType_ReverseV2] = CommonInferShape;
  g_infer_func[PrimType_Rfft] = RfftInferShape;
  g_infer_func[PrimType_ROIPooling] = ROIPoolingInferShape;
  g_infer_func[PrimType_Round] = CommonInferShape;
  g_infer_func[PrimType_Rsqrt] = CommonInferShape;
  g_infer_func[PrimType_RsqrtGrad] = NULL;
  g_infer_func[PrimType_ScaleFusion] = CommonInferShape;
  g_infer_func[PrimType_ScatterNd] = ScatterNdInferShape;
  g_infer_func[PrimType_ScatterNdUpdate] = ScatterNdUpdateInferShape;
  g_infer_func[PrimType_TensorScatterAdd] = ScatterNdUpdateInferShape;
  g_infer_func[PrimType_Select] = SelectInferShape;
  g_infer_func[PrimType_SGD] = SgdInferShape;
  g_infer_func[PrimType_Shape] = ShapeInferShape;
  g_infer_func[PrimType_SigmoidCrossEntropyWithLogits] = CommonInferShape;
  g_infer_func[PrimType_SigmoidCrossEntropyWithLogitsGrad] = CommonInferShape;
  g_infer_func[PrimType_Sin] = CommonInferShape;
  g_infer_func[PrimType_Size] = SizeInferShape;
#ifdef MSLITE_ENABLE_STRING_KERNEL
  g_infer_func[PrimType_SkipGram] = SkipGramInferShape;
#endif
  g_infer_func[PrimType_SliceFusion] = SliceInferShape;
  g_infer_func[PrimType_SmoothL1Loss] = CommonInferShape;
  g_infer_func[PrimType_SmoothL1LossGrad] = CommonInferShape;
  g_infer_func[PrimType_Softmax] = SoftMaxInferShape;
  g_infer_func[PrimType_SoftmaxCrossEntropyWithLogits] = SoftmaxCrossEntropyInferShape;
  g_infer_func[PrimType_SpaceToBatch] = SpaceToBatchInferShape;
  g_infer_func[PrimType_SpaceToBatchND] = SpaceToBatchNdInferShape;
  g_infer_func[PrimType_SpaceToDepth] = SpaceToDepthInferShape;
  g_infer_func[PrimType_SparseSoftmaxCrossEntropyWithLogits] = SparseSoftmaxCrossEntropyWithLogitsInferShape;
  g_infer_func[PrimType_SparseToDense] = SparseToDenseInferShape;
  g_infer_func[PrimType_Splice] = SpliceInferShape;
  g_infer_func[PrimType_Split] = SplitInferShape;
  g_infer_func[PrimType_SplitWithOverlap] = SplitWithOverlapInferShape;
  g_infer_func[PrimType_Sqrt] = CommonInferShape;
  g_infer_func[PrimType_SqrtGrad] = NULL;
  g_infer_func[PrimType_Square] = CommonInferShape;
  g_infer_func[PrimType_SquaredDifference] = ArithmeticInferShape;
  g_infer_func[PrimType_Squeeze] = SqueezeInferShape;
  g_infer_func[PrimType_Stack] = StackInferShape;
  g_infer_func[PrimType_StridedSlice] = StridedSliceInferShape;
  g_infer_func[PrimType_StridedSliceGrad] = StridedSliceGradInferShape;
  g_infer_func[PrimType_SubFusion] = ArithmeticInferShape;
  g_infer_func[PrimType_SubGrad] = AddSubGradInferShape;
}

void RegAllInferFunc5() {
  g_infer_func[PrimType_Switch] = NULL;
#ifdef MSLITE_ENABLE_CONTROLFLOW
  g_infer_func[PrimType_TensorArray] = TensorArrayInferShape;
  g_infer_func[PrimType_TensorArrayRead] = TensorArrayReadInferShape;
  g_infer_func[PrimType_TensorArrayWrite] = TensorArrayWriteInferShape;
  g_infer_func[PrimType_TensorListFromTensor] = TensorListFromTensorInferShape;
  g_infer_func[PrimType_TensorListGetItem] = TensorListGetItemInferShape;
  g_infer_func[PrimType_TensorListReserve] = TensorListReserveInferShape;
  g_infer_func[PrimType_TensorListSetItem] = TensorListSetItemInferShape;
  g_infer_func[PrimType_TensorListStack] = TensorListStackInferShape;
#endif
  g_infer_func[PrimType_TileFusion] = TileInferShape;
  g_infer_func[PrimType_TopKFusion] = TopKInferShape;
  g_infer_func[PrimType_Transpose] = TransposeInferShape;
  g_infer_func[PrimType_UniformReal] = UniformRealInferShape;
  g_infer_func[PrimType_Unique] = UniqueInferShape;
  g_infer_func[PrimType_UnsortedSegmentSum] = UnsortedSegmentSumInferShape;
  g_infer_func[PrimType_Unsqueeze] = UnsqueezeInferShape;
  g_infer_func[PrimType_Unstack] = UnstackInferShape;
  g_infer_func[PrimType_Where] = WhereInferShape;
  g_infer_func[PrimType_ZerosLike] = CommonInferShape;

  // fused operators.
  g_inner_op_infer_func[PrimType_Inner_GltextureToOpencl - PrimType_InnerOpMin] = NULL;
  g_inner_op_infer_func[PrimType_Inner_Identity - PrimType_InnerOpMin] = NULL;
#ifndef RUNTIME_PASS_CLIP
  g_inner_op_infer_func[PrimType_Inner_ShapeFusion - PrimType_InnerOpMin] = ShapeFusionInferShape;
  g_inner_op_infer_func[PrimType_Inner_EncoderLayer - PrimType_InnerOpMin] = EncoderLayerInferShape;

#endif
  g_inner_op_infer_func[PrimType_Inner_ToFormat - PrimType_InnerOpMin] = NULL;
}

#else
__attribute__((init_priority(101))) InferShape g_infer_func[PrimType_MAX] = {0};
__attribute__((init_priority(101))) InferShape g_inner_op_infer_func[PrimType_InnerOpMax - PrimType_InnerOpMin] = {0};
#endif  // _MSC_VER

InferShape GetInferFunc(int prim_type) {
#ifdef _MSC_VER
  if (g_infer_func[PrimType_Abs] == NULL) {
    RegAllInferFunc1();
    RegAllInferFunc2();
    RegAllInferFunc3();
    RegAllInferFunc4();
    RegAllInferFunc5();
  }
#endif
  if (prim_type > PrimType_MIN && prim_type < PrimType_MAX) {
    return g_infer_func[prim_type];
  } else if (prim_type >= PrimType_InnerOpMin && prim_type < PrimType_InnerOpMax) {
    return g_inner_op_infer_func[prim_type - PrimType_InnerOpMin];
  }
  return NULL;
}

void RegInfer(int prim_type, InferShape func) {
  if (prim_type > PrimType_MIN && prim_type < PrimType_MAX) {
    g_infer_func[prim_type] = func;
  } else if (prim_type >= PrimType_InnerOpMin && prim_type < PrimType_InnerOpMax) {
    g_inner_op_infer_func[prim_type - PrimType_InnerOpMin] = func;
  }
}
