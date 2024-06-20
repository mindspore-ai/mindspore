/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_BASE_NN_OPS_H_
#define MINDSPORE_CORE_BASE_NN_OPS_H_

#include <memory>
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ops/nn_op_name.h"
#include "utils/hash_map.h"
#include "ops/auto_generate/gen_ops_primitive.h"

namespace mindspore {
namespace prim {
// Loss
GVAR_DEF(PrimitivePtr, kPrimFlattenGrad, std::make_shared<Primitive>("FlattenGrad"));
GVAR_DEF(PrimitivePtr, kPrimSoftmaxV2, std::make_shared<Primitive>("SoftmaxV2"));
GVAR_DEF(PrimitivePtr, kPrimSoftmaxGrad, std::make_shared<Primitive>("SoftmaxGrad"));
GVAR_DEF(PrimitivePtr, kPrimSoftsign, std::make_shared<Primitive>("Softsign"));
GVAR_DEF(PrimitivePtr, kPrimSparseSoftmaxCrossEntropy, std::make_shared<Primitive>("SparseSoftmaxCrossEntropy"));
GVAR_DEF(PrimitivePtr, kPrimSoftmaxV2WithDropoutDoMaskV3, std::make_shared<Primitive>("SoftmaxV2WithDropoutDoMaskV3"));
GVAR_DEF(PrimitivePtr, kPrimLogSoftmaxV2, std::make_shared<Primitive>("LogSoftmaxV2"));
GVAR_DEF(PrimitivePtr, kPrimMultilabelMarginLoss, std::make_shared<Primitive>(kMultilabelMarginLossOpName));
GVAR_DEF(PrimitivePtr, kPrimMultilabelMarginLossGrad, std::make_shared<Primitive>(kMultilabelMarginLossGradOpName));
GVAR_DEF(PrimitivePtr, kPrimCTCLossV2, std::make_shared<Primitive>("CTCLossV2"));
GVAR_DEF(PrimitivePtr, kPrimCTCLossV2Grad, std::make_shared<Primitive>("CTCLossV2Grad"));
GVAR_DEF(PrimitivePtr, kPrimCTCLoss, std::make_shared<Primitive>(kCTCLossOpName));
GVAR_DEF(PrimitivePtr, kPrimTripletMarginLoss, std::make_shared<Primitive>(kTripletMarginLossOpName));
GVAR_DEF(PrimitivePtr, kPrimSmoothL1Loss, std::make_shared<Primitive>("SmoothL1Loss"));
GVAR_DEF(PrimitivePtr, kPrimSmoothL1LossV2, std::make_shared<Primitive>("SmoothL1LossV2"));
GVAR_DEF(PrimitivePtr, kPrimSmoothL1LossGrad, std::make_shared<Primitive>("SmoothL1LossGrad"));
GVAR_DEF(PrimitivePtr, kPrimSmoothL1LossGradV2, std::make_shared<Primitive>("SmoothL1LossGradV2"));
GVAR_DEF(PrimitivePtr, kPrimSoftMarginLoss, std::make_shared<Primitive>("SoftMarginLoss"));
GVAR_DEF(PrimitivePtr, kPrimSoftMarginLossGrad, std::make_shared<Primitive>("SoftMarginLossGrad"));
GVAR_DEF(PrimitivePtr, kPrimSoftmaxCrossEntropyWithLogits,
         std::make_shared<Primitive>("SoftmaxCrossEntropyWithLogits"));
GVAR_DEF(PrimitivePtr, kPrimL2Loss, std::make_shared<Primitive>("L2Loss"));
GVAR_DEF(PrimitivePtr, kPrimSigmoidCrossEntropyWithLogits,
         std::make_shared<Primitive>("SigmoidCrossEntropyWithLogits"));
GVAR_DEF(PrimitivePtr, kPrimSigmoidCrossEntropyWithLogitsV2,
         std::make_shared<Primitive>("SigmoidCrossEntropyWithLogitsV2"));
GVAR_DEF(PrimitivePtr, kPrimSigmoidCrossEntropyWithLogitsGrad,
         std::make_shared<Primitive>("SigmoidCrossEntropyWithLogitsGrad"));
GVAR_DEF(PrimitivePtr, kPrimSparseSoftmaxCrossEntropyWithLogits,
         std::make_shared<Primitive>("SparseSoftmaxCrossEntropyWithLogits"));
GVAR_DEF(PrimitivePtr, kPrimSparseSoftmaxCrossEntropyWithLogitsV2,
         std::make_shared<Primitive>("SparseSoftmaxCrossEntropyWithLogitsV2"));
GVAR_DEF(PrimitivePtr, kPrimMultiMarginLoss, std::make_shared<Primitive>(kMultiMarginLossOpName));
GVAR_DEF(PrimitivePtr, kPrimMultiMarginLossGrad, std::make_shared<Primitive>(kMultiMarginLossGradOpName));
GVAR_DEF(PrimitivePtr, kSoftmaxGradExt, std::make_shared<Primitive>("SoftmaxGradExt"));
GVAR_DEF(PrimitivePtr, kPrimOneHotD, std::make_shared<Primitive>("OneHotD"));

GVAR_DEF(PrimitivePtr, kPrimPdist, std::make_shared<Primitive>("Pdist"));
GVAR_DEF(PrimitivePtr, kPrimPdistGrad, std::make_shared<Primitive>("PdistGrad"));

// pad
GVAR_DEF(PrimitivePtr, kPrimPadV3, std::make_shared<Primitive>(kPadV3OpName));
GVAR_DEF(PrimitivePtr, kPrimPadV3Grad, std::make_shared<Primitive>(kPadV3GradOpName));
GVAR_DEF(PrimitivePtr, kPrimMirrorPadGrad, std::make_shared<Primitive>(kMirrorPadGradOpName));

// Norm
GVAR_DEF(PrimitivePtr, kPrimRenorm, std::make_shared<Primitive>(kRenormOpName));
GVAR_DEF(PrimitivePtr, kPrimNuclearNorm, std::make_shared<Primitive>(kNuclearNormOpName));
GVAR_DEF(PrimitivePtr, kPrimL2Normalize, std::make_shared<Primitive>("L2Normalize"));
GVAR_DEF(PrimitivePtr, kPrimL2NormalizeGrad, std::make_shared<Primitive>("L2NormalizeGrad"));
GVAR_DEF(PrimitivePtr, kPrimLayerNormXBackprop, std::make_shared<Primitive>("LayerNormXBackprop"));
GVAR_DEF(PrimitivePtr, kPrimLayerNormXBackpropV2, std::make_shared<Primitive>("LayerNormXBackpropV2"));
GVAR_DEF(PrimitivePtr, kPrimLayerNormBetaGammaBackprop, std::make_shared<Primitive>("LayerNormBetaGammaBackprop"));
GVAR_DEF(PrimitivePtr, kPrimLayerNormBetaGammaBackpropV2, std::make_shared<Primitive>("LayerNormBetaGammaBackpropV2"));
GVAR_DEF(PrimitivePtr, kPrimBNInfer, std::make_shared<Primitive>("BNInfer"));
GVAR_DEF(PrimitivePtr, kPrimBNInferGrad, std::make_shared<Primitive>("BNInferGrad"));
GVAR_DEF(PrimitivePtr, kPrimInstanceNorm, std::make_shared<Primitive>("InstanceNorm"));
GVAR_DEF(PrimitivePtr, kPrimInstanceNormGrad, std::make_shared<Primitive>("InstanceNormGrad"));
GVAR_DEF(PrimitivePtr, kPrimInstanceNormV2, std::make_shared<Primitive>("InstanceNormV2"));
GVAR_DEF(PrimitivePtr, kPrimInstanceNormV2Grad, std::make_shared<Primitive>("InstanceNormV2Grad"));
GVAR_DEF(PrimitivePtr, kPrimSyncBatchNorm, std::make_shared<Primitive>("SyncBatchNorm"));
GVAR_DEF(PrimitivePtr, kPrimSyncBatchNormGrad, std::make_shared<Primitive>("SyncBatchNormGrad"));
GVAR_DEF(PrimitivePtr, kPrimBNTrainingReduce, std::make_shared<Primitive>("BNTrainingReduce"));
GVAR_DEF(PrimitivePtr, kPrimBNTrainingReduceGrad, std::make_shared<Primitive>("BNTrainingReduceGrad"));
GVAR_DEF(PrimitivePtr, kPrimFusedBatchNorm, std::make_shared<Primitive>("FusedBatchNorm"));
GVAR_DEF(PrimitivePtr, kPrimRmsNorm, std::make_shared<Primitive>("RmsNorm"));
GVAR_DEF(PrimitivePtr, kPrimRmsNormGrad, std::make_shared<Primitive>("RmsNormGrad"));

GVAR_DEF(PrimitivePtr, kPrimWKV, std::make_shared<Primitive>("WKV"));
GVAR_DEF(PrimitivePtr, kPrimWKVGrad, std::make_shared<Primitive>("WKVGrad"));
GVAR_DEF(PrimitivePtr, kPrimDenseGrad, std::make_shared<Primitive>("DenseGrad"));
GVAR_DEF(PrimitivePtr, kPrimEmbeddingLookup, std::make_shared<Primitive>("EmbeddingLookup"));
GVAR_DEF(PrimitivePtr, kPrimEmbeddingLookupCommGrad, std::make_shared<Primitive>("EmbeddingLookupCommGrad"));
GVAR_DEF(PrimitivePtr, kPrimAudioSpectrogram, std::make_shared<Primitive>("AudioSpectrogram"));
GVAR_DEF(PrimitivePtr, kPrimCrop, std::make_shared<Primitive>("Crop"));
GVAR_DEF(PrimitivePtr, kPrimAttention, std::make_shared<Primitive>("Attention"));
GVAR_DEF(PrimitivePtr, kPrimLstm, std::make_shared<Primitive>("LSTM"));
GVAR_DEF(PrimitivePtr, kPrimLstmGrad, std::make_shared<Primitive>("LSTMGrad"));
GVAR_DEF(PrimitivePtr, kPrimLstmGradData, std::make_shared<Primitive>("LSTMGradData"));
GVAR_DEF(PrimitivePtr, kPrimLstmGradWeight, std::make_shared<Primitive>("LSTMGradWeight"));
GVAR_DEF(PrimitivePtr, kPrimFullConnection, std::make_shared<Primitive>("FullConnection"));
GVAR_DEF(PrimitivePtr, kPrimGroupConv2DGradInput, std::make_shared<Primitive>("GroupConv2DGradInput"));
GVAR_DEF(PrimitivePtr, kPrimDilation2D, std::make_shared<Primitive>("Dilation2D"));
GVAR_DEF(PrimitivePtr, kPrimDilation2DBackpropInput, std::make_shared<Primitive>("Dilation2DBackpropInput"));
GVAR_DEF(PrimitivePtr, kPrimDilation2DBackpropFilter, std::make_shared<Primitive>("Dilation2DBackpropFilter"));
GVAR_DEF(PrimitivePtr, kPrimDeformableOffsetsGrad, std::make_shared<Primitive>("DeformableOffsetsGrad"));
GVAR_DEF(PrimitivePtr, kPrimCustomNormalize, std::make_shared<Primitive>("CustomNormalize"));
GVAR_DEF(PrimitivePtr, kPrimDepthwiseConv2D, std::make_shared<Primitive>("DepthwiseConv2D"));
GVAR_DEF(PrimitivePtr, kPrimCTCGreedyDecoder, std::make_shared<Primitive>("CTCGreedyDecoder"));
GVAR_DEF(PrimitivePtr, kPrimDataFormatDimMap, std::make_shared<Primitive>("DataFormatDimMap"));
GVAR_DEF(PrimitivePtr, kPrimDataFormatVecPermute, std::make_shared<Primitive>("DataFormatVecPermute"));
GVAR_DEF(PrimitivePtr, kPrimDynamicStitch, std::make_shared<Primitive>("DynamicStitch"));
GVAR_DEF(PrimitivePtr, kPrimDetectionPostProcess, std::make_shared<Primitive>("DetectionPostProcess"));
GVAR_DEF(PrimitivePtr, kPrimBiasSubGrad, std::make_shared<Primitive>("BiasSubGrad"));
GVAR_DEF(PrimitivePtr, kPrimLrn, std::make_shared<Primitive>(kLRNOpName));
GVAR_DEF(PrimitivePtr, kPrimLrnGrad, std::make_shared<Primitive>("LRNGrad"));
GVAR_DEF(PrimitivePtr, kPrimDropoutGenMask, std::make_shared<Primitive>(kDropoutGenMaskOpName));
GVAR_DEF(PrimitivePtr, kPrimDropoutGenMaskV3, std::make_shared<Primitive>(kDropoutGenMaskV3OpName));
GVAR_DEF(PrimitivePtr, kPrimStatelessDropOutGenMask, std::make_shared<Primitive>(kStatelessDropOutGenMaskOpName));
GVAR_DEF(PrimitivePtr, kPrimDropoutDoMask, std::make_shared<Primitive>(kDropoutDoMaskOpName));
GVAR_DEF(PrimitivePtr, kPrimDropOutDoMask, std::make_shared<Primitive>("DropOutDoMask"));
GVAR_DEF(PrimitivePtr, kPrimDropoutDoMaskV3, std::make_shared<Primitive>(kDropoutDoMaskV3OpName));
GVAR_DEF(PrimitivePtr, kPrimDropOutDoMaskV3, std::make_shared<Primitive>("DropOutDoMaskV3"));
GVAR_DEF(PrimitivePtr, kPrimDropOutDoMaskV3D, std::make_shared<Primitive>("DropOutDoMaskV3D"));
GVAR_DEF(PrimitivePtr, kPrimDropoutGrad, std::make_shared<Primitive>(kDropoutGradOpName));
GVAR_DEF(PrimitivePtr, kPrimDropout2D, std::make_shared<Primitive>(kDropout2DOpName));
GVAR_DEF(PrimitivePtr, kPrimDropout3D, std::make_shared<Primitive>(kDropout3DOpName));
GVAR_DEF(PrimitivePtr, kPrimUniformInt, std::make_shared<Primitive>("UniformInt"));
GVAR_DEF(PrimitivePtr, kPrimUniformReal, std::make_shared<Primitive>("UniformReal"));
GVAR_DEF(PrimitivePtr, kPrimCudnnUniformReal, std::make_shared<Primitive>("CudnnUniformReal"));
GVAR_DEF(PrimitivePtr, kPrimSoftplus, std::make_shared<Primitive>("Softplus"));
GVAR_DEF(PrimitivePtr, kPrimQuantile, std::make_shared<Primitive>("Quantile"));
GVAR_DEF(PrimitivePtr, kPrimSoftplusGrad, std::make_shared<Primitive>("SoftplusGrad"));
GVAR_DEF(PrimitivePtr, kPrimBpropCut, std::make_shared<Primitive>("bprop_cut"));
GVAR_DEF(PrimitivePtr, kPrimFakeQuantPerLayer, std::make_shared<Primitive>("FakeQuantPerLayer"));
GVAR_DEF(PrimitivePtr, kPrimFakeQuantPerChannel, std::make_shared<Primitive>("FakeQuantPerChannel"));
GVAR_DEF(PrimitivePtr, kPrimFakeLearnedScaleQuantPerLayer,
         std::make_shared<Primitive>("FakeLearnedScaleQuantPerLayer"));
GVAR_DEF(PrimitivePtr, kPrimFakeLearnedScaleQuantPerChannel,
         std::make_shared<Primitive>("FakeLearnedScaleQuantPerChannel"));
GVAR_DEF(PrimitivePtr, kPrimFakeQuantWithMinMaxVars, std::make_shared<Primitive>("FakeQuantWithMinMaxVars"));
GVAR_DEF(PrimitivePtr, kPrimClipByNorm, std::make_shared<Primitive>("ClipByNorm"));
GVAR_DEF(PrimitivePtr, kPrimClipByNormNoDivSum, std::make_shared<Primitive>("ClipByNormNoDivSum"));
GVAR_DEF(PrimitivePtr, kPrimCustomExtractFeatures, std::make_shared<Primitive>("CustomExtractFeatures"));
GVAR_DEF(PrimitivePtr, kSquareSumV1, std::make_shared<Primitive>("SquareSumV1"));
GVAR_DEF(PrimitivePtr, kFusedMulAdd, std::make_shared<Primitive>("FusedMulAdd"));
GVAR_DEF(PrimitivePtr, kPrimSoftShrink, std::make_shared<Primitive>("SoftShrink"));
GVAR_DEF(PrimitivePtr, kPrimSoftShrinkGrad, std::make_shared<Primitive>("SoftShrinkGrad"));
GVAR_DEF(PrimitivePtr, kPrimHardSwish, std::make_shared<Primitive>("HardSwish"));
GVAR_DEF(PrimitivePtr, kPrimHardSwishGrad, std::make_shared<Primitive>("HardSwishGrad"));
GVAR_DEF(PrimitivePtr, kPrimDeformableOffsets, std::make_shared<Primitive>("DeformableOffsets"));
GVAR_DEF(PrimitivePtr, kPrimDeformableConv2d, std::make_shared<Primitive>("DeformableConv2d"));
GVAR_DEF(PrimitivePtr, kPrimLARSUpdate, std::make_shared<Primitive>("LARSUpdate"));
GVAR_DEF(PrimitivePtr, kPrimLarsV2Update, std::make_shared<Primitive>("LarsV2Update"));
GVAR_DEF(PrimitivePtr, kPrimBoundingBoxDecode, std::make_shared<Primitive>("BoundingBoxDecode"));
GVAR_DEF(PrimitivePtr, kPrimBoundingBoxEncode, std::make_shared<Primitive>("BoundingBoxEncode"));
GVAR_DEF(PrimitivePtr, kPrimROIAlign, std::make_shared<Primitive>("ROIAlign"));
GVAR_DEF(PrimitivePtr, kPrimROIAlignGrad, std::make_shared<Primitive>("ROIAlignGrad"));
GVAR_DEF(PrimitivePtr, kPrimBNTrainingUpdate, std::make_shared<Primitive>("BNTrainingUpdate"));
GVAR_DEF(PrimitivePtr, kPrimBNTrainingUpdateGrad, std::make_shared<Primitive>("BNTrainingUpdateGrad"));
GVAR_DEF(PrimitivePtr, kPrimMish, std::make_shared<Primitive>(kMishOpName));
GVAR_DEF(PrimitivePtr, kPrimBiasDropoutAdd, std::make_shared<Primitive>("BiasDropoutAdd"));
GVAR_DEF(PrimitivePtr, kPrimNthElement, std::make_shared<Primitive>("NthElement"));
GVAR_DEF(PrimitivePtr, kPrimIFMR, std::make_shared<Primitive>(kIFMROpName));
GVAR_DEF(PrimitivePtr, kPrimChannelShuffle, std::make_shared<Primitive>(kChannelShuffleOpName));
GVAR_DEF(PrimitivePtr, kPrimPromptFlashAttention, std::make_shared<Primitive>(kFlashPromptFlashAttentionOpName));
GVAR_DEF(PrimitivePtr, kPrimIncreFlashAttention, std::make_shared<Primitive>(kFlashIncreFlashAttentionOpName));
GVAR_DEF(PrimitivePtr, kPrimRNNTLoss, std::make_shared<Primitive>(kRNNTLossOpName));
GVAR_DEF(PrimitivePtr, kPrimMatmulQkv, std::make_shared<Primitive>(kMatmulQkvOpName));
GVAR_DEF(PrimitivePtr, kPrimMatmulFfn, std::make_shared<Primitive>(kMatmulFfnOpName));
GVAR_DEF(PrimitivePtr, kPrimAllFinite, std::make_shared<Primitive>(kAllFiniteOpName));
GVAR_DEF(PrimitivePtr, kPrimEmbeddingTableExport, std::make_shared<Primitive>("EmbeddingTableExport"));
GVAR_DEF(PrimitivePtr, kPrimEmbeddingTableImport, std::make_shared<Primitive>("EmbeddingTableImport"));
GVAR_DEF(PrimitivePtr, kPrimEmbeddingComputeVarExport, std::make_shared<Primitive>("EmbeddingComputeVarExport"));
GVAR_DEF(PrimitivePtr, kPrimEmbeddingComputeVarImport, std::make_shared<Primitive>("EmbeddingComputeVarImport"));
GVAR_DEF(PrimitivePtr, kPrimInitEmbeddingHashmap, std::make_shared<Primitive>("InitEmbeddingHashmap"));
GVAR_DEF(PrimitivePtr, kPrimInitPartitionMap, std::make_shared<Primitive>("InitPartitionMap"));
GVAR_DEF(PrimitivePtr, kPrimEmbeddingApplyAdam, std::make_shared<Primitive>("EmbeddingApplyAdam"));
GVAR_DEF(PrimitivePtr, kPrimEmbeddingApplyAdamW, std::make_shared<Primitive>("EmbeddingApplyAdamW"));
GVAR_DEF(PrimitivePtr, kPrimEmbeddingApplyAdaGrad, std::make_shared<Primitive>("EmbeddingApplyAdaGrad"));
GVAR_DEF(PrimitivePtr, kPrimEmbeddingApplyFtrl, std::make_shared<Primitive>("EmbeddingApplyFtrl"));
GVAR_DEF(PrimitivePtr, kPrimEmbeddingTableFind, std::make_shared<Primitive>("EmbeddingTableFind"));
GVAR_DEF(PrimitivePtr, kPrimEmbeddingTableFindAndInit, std::make_shared<Primitive>("EmbeddingTableFindAndInit"));
GVAR_DEF(PrimitivePtr, kPrimFakeRemoteLookupUniqued, std::make_shared<Primitive>("FakeRemoteLookupUniqued"));
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_NN_OPS_H_
