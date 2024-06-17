/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_BASE_NN_OP_NAME_H_
#define MINDSPORE_CORE_BASE_NN_OP_NAME_H_

namespace mindspore {
// Loss
constexpr auto kCTCLossOpName = "CTCLoss";
constexpr auto kNLLLossOpName = "NLLLoss";
constexpr auto kNLLLossGradOpName = "NLLLossGrad";
constexpr auto kMultiMarginLossOpName = "MultiMarginLoss";
constexpr auto kMultiMarginLossGradOpName = "MultiMarginLossGrad";
constexpr auto kMultilabelMarginLossOpName = "MultilabelMarginLoss";
constexpr auto kMultilabelMarginLossGradOpName = "MultilabelMarginLossGrad";
constexpr auto kTripletMarginLossOpName = "TripletMarginLoss";

constexpr auto kLayerNormOpName = "LayerNorm";
constexpr auto kLayerNormGradOpName = "LayerNormGrad";
constexpr auto kLayerNormV3OpName = "LayerNormV3";
constexpr auto kLayerNormGradV3OpName = "LayerNormGradV3";
constexpr auto kPadV3OpName = "PadV3";
constexpr auto kPadV3GradOpName = "PadV3Grad";
constexpr auto kMirrorPadGradOpName = "MirrorPadGrad";
constexpr auto kDataFormatVecPermuteOpName = "DataFormatVecPermute";
constexpr auto kDropoutGenMaskOpName = "DropoutGenMask";
constexpr auto kDropoutGenMaskV3OpName = "DropoutGenMaskV3";
constexpr auto kStatelessDropOutGenMaskOpName = "StatelessDropOutGenMask";
constexpr auto kDropoutDoMaskOpName = "DropoutDoMask";
constexpr auto kDropoutDoMaskV3OpName = "DropoutDoMaskV3";
constexpr auto kDropoutOpName = "Dropout";
constexpr auto kDropoutGradOpName = "DropoutGrad";
constexpr auto kDropout2DOpName = "Dropout2D";
constexpr auto kDropout3DOpName = "Dropout3D";
constexpr auto kMishOpName = "Mish";
constexpr auto kLRNOpName = "LRN";
constexpr auto kGridSampler2DOpName = "GridSampler2D";
constexpr auto kGridSampler2DGradOpName = "GridSampler2DGrad";
constexpr auto kGridSampler3DOpName = "GridSampler3D";
constexpr auto kGridSampler3DGradOpName = "GridSampler3DGrad";
constexpr auto kHSwishOpName = "HSwish";
constexpr auto kHSwishGradOpName = "HSwishGrad";
constexpr auto kNuclearNormOpName = "NuclearNorm";
constexpr auto kIFMROpName = "IFMR";
constexpr auto kRenormOpName = "Renorm";
constexpr auto kChannelShuffleOpName = "ChannelShuffle";
constexpr auto kBiasAddOpName = "BiasAdd";
constexpr auto kBiasAddGradOpName = "BiasAddGrad";
constexpr auto kBatchNormOpName = "BatchNorm";
constexpr auto kBatchNormGradOpName = "BatchNormGrad";
constexpr auto kBatchNormGradGradOpName = "BatchNormGradGrad";
constexpr auto kBatchNormGradWithActivationOpName = "BatchNormGradWithActivation";
constexpr auto kBatchNormGradWithAddAndActivationOpName = "BatchNormGradWithAddAndActivation";
constexpr auto kBatchNormWithActivationOpName = "BatchNormWithActivation";
constexpr auto kBatchNormWithAddAndActivationOpName = "BatchNormWithAddAndActivation";
constexpr auto kBCEWithLogitsLossOpName = "BCEWithLogitsLoss";
constexpr auto kBNInferGradOpName = "BNInferGrad";
constexpr auto kBNInferOpName = "BNInfer";
constexpr auto kBNTrainingReduceGradOpName = "BNTrainingReduceGrad";
constexpr auto kBNTrainingReduceOpName = "BNTrainingReduce";
constexpr auto kBNTrainingUpdateGradOpName = "BNTrainingUpdateGrad";
constexpr auto kBNTrainingUpdateOpName = "BNTrainingUpdate";
constexpr auto kBpropCutOpName = "bprop_cut";
constexpr auto kClipByNormNoDivSumOpName = "ClipByNormNoDivSum";
constexpr auto kDeformableOffsetsOpName = "DeformableOffsets";
constexpr auto kDeformableOffsetsGradOpName = "DeformableOffsetsGrad";
constexpr auto kDeformableConv2dOpName = "DeformableConv2d";
constexpr auto kCTCGreedyDecoderOpName = "CTCGreedyDecoder";
constexpr auto kDataFormatDimMapOpName = "DataFormatDimMap";
constexpr auto kDenseOpName = "Dense";
constexpr auto kDenseGradOpName = "DenseGrad";
constexpr auto kDepthwiseConv2DOpName = "DepthwiseConv2D";
constexpr auto kDropOutDoMaskOpName = "DropOutDoMask";
constexpr auto kDropOutDoMaskV3OpName = "DropOutDoMaskV3";
constexpr auto kDropOutDoMaskV3DOpName = "DropOutDoMaskV3D";
constexpr auto kDynamicStitchOpName = "DynamicStitch";
constexpr auto kEmbeddingLookupCommGradOpName = "EmbeddingLookupCommGrad";
constexpr auto kEmbeddingLookupOpName = "EmbeddingLookup";
constexpr auto kFlattenOpName = "Flatten";
constexpr auto kFlattenGradOpName = "FlattenGrad";
constexpr auto kFusedMulAddOpName = "FusedMulAdd";
constexpr auto kHShrinkOpName = "HShrink";
constexpr auto kHShrinkGradOpName = "HShrinkGrad";
constexpr auto kHardSwishOpName = "HardSwish";
constexpr auto kHardSwishGradOpName = "HardSwishGrad";
constexpr auto kInstanceNormOpName = "InstanceNorm";
constexpr auto kInstanceNormGradOpName = "InstanceNormGrad";
constexpr auto kInstanceNormV2OpName = "InstanceNormV2";
constexpr auto kInstanceNormV2GradOpName = "InstanceNormV2Grad";
constexpr auto kROIAlignOpName = "ROIAlign";
constexpr auto kL2NormalizeOpName = "L2Normalize";
constexpr auto kL2NormalizeGradOpName = "L2NormalizeGrad";
constexpr auto kLARSUpdateOpName = "LARSUpdate";
constexpr auto kLarsV2UpdateOpName = "LarsV2Update";
constexpr auto kLayerNormBetaGammaBackpropOpName = "LayerNormBetaGammaBackprop";
constexpr auto kLayerNormBetaGammaBackpropV2OpName = "LayerNormBetaGammaBackpropV2";
constexpr auto kLayerNormGradGradOpName = "LayerNormGradGrad";
constexpr auto kLayerNormXBackpropOpName = "LayerNormXBackprop";
constexpr auto kLayerNormXBackpropV2OpName = "LayerNormXBackpropV2";
constexpr auto kLog1pOpName = "Log1p";
constexpr auto kLogSoftmaxOpName = "LogSoftmax";
constexpr auto kLogSoftmaxV2OpName = "LogSoftmaxV2";
constexpr auto kLogSoftmaxGradOpName = "LogSoftmaxGrad";
constexpr auto kLSTMGradOpName = "LSTMGrad";
constexpr auto kLSTMOpName = "LSTM";
constexpr auto kMatrixExpOpName = "MatrixExp";
constexpr auto kNthElementOpName = "NthElement";
constexpr auto kOneHotOpName = "OneHot";
constexpr auto kOneHotDOpName = "OneHotD";
constexpr auto kPdistGradOpName = "PdistGrad";
constexpr auto kQuantileOpName = "Quantile";
constexpr auto kROIAlignGradOpName = "ROIAlignGrad";
constexpr auto kSigmoidCrossEntropyWithLogitsV2OpName = "SigmoidCrossEntropyWithLogitsV2";
constexpr auto kSmoothL1LossOpName = "SmoothL1Loss";
constexpr auto kSmoothL1LossV2OpName = "SmoothL1LossV2";
constexpr auto kSmoothL1LossGradOpName = "SmoothL1LossGrad";
constexpr auto kSmoothL1LossGradV2OpName = "SmoothL1LossGradV2";
constexpr auto kSoftmaxOpName = "Softmax";
constexpr auto kSoftmaxV2OpName = "SoftmaxV2";
constexpr auto kSoftmaxCrossEntropyWithLogitsOpName = "SoftmaxCrossEntropyWithLogits";
constexpr auto kSoftmaxGradExtOpName = "SoftmaxGradExt";
constexpr auto kSoftmaxV2WithDropoutDoMaskV3OpName = "SoftmaxV2WithDropoutDoMaskV3";
constexpr auto kSparseSoftmaxCrossEntropyWithLogitsOpName = "SparseSoftmaxCrossEntropyWithLogits";
constexpr auto kSparseSoftmaxCrossEntropyWithLogitsV2OpName = "SparseSoftmaxCrossEntropyWithLogitsV2";
constexpr auto kSoftMarginLossOpName = "SoftMarginLoss";
constexpr auto kSoftplusOpName = "Softplus";
constexpr auto kSoftsignOpName = "Softsign";
constexpr auto kApplyCamePart1OpName = "ApplyCamePart1";
constexpr auto kApplyCamePart2OpName = "ApplyCamePart2";
constexpr auto kApplyCamePart3OpName = "ApplyCamePart3";
constexpr auto kApplyCamePart4OpName = "ApplyCamePart4";
constexpr auto kFlashPromptFlashAttentionOpName = "PromptFlashAttention";
constexpr auto kFlashIncreFlashAttentionOpName = "IncreFlashAttention";
constexpr auto kFlashAttentionScoreOpName = "FlashAttentionScore";
constexpr auto kFlashAttentionScoreGradOpName = "FlashAttentionScoreGrad";
constexpr auto kFusedInferAttentionScoreOpName = "FusedInferAttentionScore";
constexpr auto kPagedAttentionOpName = "PagedAttention";
constexpr auto kPagedAttentionMaskOpName = "PagedAttentionMask";
constexpr auto kReshapeAndCacheOpName = "ReshapeAndCache";
constexpr auto kRmsNormOpName = "RmsNorm";
constexpr auto kRmsNormGradOpName = "RmsNormGrad";
constexpr auto kRNNTLossOpName = "RNNTLoss";
constexpr auto kAllFiniteOpName = "AllFinite";
constexpr auto kWeightQuantMatmulQkvOpName = "WeightQuantMatmulQkv";
constexpr auto kWeightQuantMatmulFfnOpName = "WeightQuantMatmulFfn";
}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_NN_OP_NAME_H_
