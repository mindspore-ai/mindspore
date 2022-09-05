/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_UTILS_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_UTILS_H_

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <algorithm>

#include "utils/log_adapter.h"
#include "ir/dtype/type.h"
#include "include/common/visible.h"

namespace mindspore {
// op name. Op which not exists in operator/ops.h, so define it's name here
constexpr auto kAdaptiveMaxPool3DGradOpName = "AdaptiveMaxPool3DGrad";
constexpr auto kFractionalAvgPoolGradOpName = "FractionalAvgPoolGrad";
constexpr auto kSegmentMaxOpName = "SegmentMax";
constexpr auto kSegmentMinOpName = "SegmentMin";
constexpr auto kSegmentSumOpName = "SegmentSum";
constexpr auto kSegmentMeanOpName = "SegmentMean";
constexpr auto kSegmentProdOpName = "SegmentProd";
constexpr auto kLuUnpackOpName = "LuUnpack";
constexpr auto kConcatOpName = "Concat";
constexpr auto kListDiffOpName = "ListDiff";
constexpr auto kUniqueOpName = "Unique";
constexpr auto kUniqueConsecutiveOpName = "UniqueConsecutive";
constexpr auto kMaskedSelectOpName = "MaskedSelect";
constexpr auto kComputeAccidentalHitsOpName = "ComputeAccidentalHits";
constexpr auto kCTCGreedyDecoderOpName = "CTCGreedyDecoder";
constexpr auto kDynamicStitchOpName = "DynamicStitch";
constexpr auto kFour2FiveOpName = "Four2Five";
constexpr auto kFive2FourOpName = "Five2Four";
constexpr auto kConv3DOpName = "Conv3D";
constexpr auto kConv3DBackpropFilterOpName = "Conv3DBackpropFilter";
constexpr auto kConv3DBackpropInputOpName = "Conv3DBackpropInput";
constexpr auto kConv2DOpName = "Conv2D";
constexpr auto kConvBN1OpName = "ConvBN1";
constexpr auto kBN2AddReluOpName = "BN2AddRelu";
constexpr auto kBN2ReLUOpName = "BN2Relu";
constexpr auto kBN2OpName = "BN2";
constexpr auto kFusedBN1OpName = "FusedBN1";
constexpr auto kFusedBN2OpName = "FusedBN2";
constexpr auto kFusedBN3OpName = "FusedBN3";
constexpr auto kBNGrad1OpName = "BNGrad1";
constexpr auto kBNGrad2OpName = "BNGrad2";
constexpr auto kBNGrad3OpName = "BNGrad3";
constexpr auto kBatchNorm = "BatchNorm";
constexpr auto kInstanceNorm = "InstanceNorm";
constexpr auto kBatchNormWithActivation = "BatchNormWithActivation";
constexpr auto kBatchNormWithAddAndActivation = "BatchNormWithAddAndActivation";
constexpr auto kBatchNormGradWithActivation = "BatchNormGradWithActivation";
constexpr auto kBatchNormGradWithAddAndActivation = "BatchNormGradWithAddAndActivation";
constexpr auto kClearZeroOpName = "ClearZero";
constexpr auto kAtomicAddrCleanOpName = "AtomicAddrClean";
constexpr auto kDynamicAtomicAddrCleanOpName = "DynamicAtomicAddrClean";
constexpr auto kGetNextOpName = "GetNext";
constexpr auto kInitDatasetQueueOpName = "InitDataSetQueue";
constexpr auto kEndOfSequence = "EndOfSequence";
constexpr auto kAllToAllVOpName = "AllToAllv";
constexpr auto kAllReduceOpName = "AllReduce";
constexpr auto kAllGatherOpName = "AllGather";
constexpr auto kHostAllGatherOpName = "HostAllGather";
constexpr auto kBroadcastOpName = "Broadcast";
constexpr auto kReceiveOpName = "Receive";
constexpr auto kHcomSendOpName = "Send";
constexpr auto kMuxReceiveOpName = "MuxReceive";
constexpr auto kMuxSendOpName = "MuxSend";
constexpr auto kReduceScatterOpName = "ReduceScatter";
constexpr auto kHostReduceScatterOpName = "HostReduceScatter";
constexpr auto kMemCpyAsyncOpName = "memcpy_async";
constexpr auto kTopKOpName = "TopK";
constexpr auto kLinSpaceOpName = "LinSpace";
constexpr auto kExtractImagePatchesOpName = "ExtractImagePatches";
constexpr auto kBNTrainingReduceOpName = "BNTrainingReduce";
constexpr auto kBNTrainingUpdateOpName = "BNTrainingUpdate";
constexpr auto kBNTrainingUpdateV2OpName = "BNTrainingUpdateV2";
constexpr auto kBNTrainingUpdateV3OpName = "BNTrainingUpdateV3";
constexpr auto kNonMaxSuppressionV3OpName = "NonMaxSuppressionV3";
constexpr auto kSimpleMeanGradOpName = "SimpleMeanGrad";
constexpr auto kMeanGradOpName = "MeanGrad";
constexpr auto kSliceOpName = "Slice";
constexpr auto kSliceGradOpName = "SliceGrad";
constexpr auto kCoalesceOpName = "Coalesce";
constexpr auto kTileOpName = "Tile";
constexpr auto kScatterNdOpName = "ScatterNd";
constexpr auto kStridedSliceAssignOpName = "StridedSliceAssign";
constexpr auto kStridedSliceOpName = "StridedSlice";
constexpr auto kEyeOpName = "Eye";
constexpr auto kStridedSliceGradOpName = "StridedSliceGrad";
constexpr auto kSparseGatherV2OpName = "SparseGatherV2";
constexpr auto kUnsortedSegmentProdOpName = "UnsortedSegmentProd";
constexpr auto kUnsortedSegmentMinOpName = "UnsortedSegmentMin";
constexpr auto kSortOpName = "Sort";
constexpr auto kFlattenGradOpName = "FlattenGrad";
constexpr auto kExpandDimsOpName = "ExpandDims";
constexpr auto kReshapeOpName = "Reshape";
constexpr auto kTransposeOpName = "Transpose";
constexpr auto kTransposeNODOpName = "TransposeNOD";
constexpr auto kSplitOpName = "Split";
constexpr auto kSplitVOpName = "SplitV";
constexpr auto kSparseApplyAdagradOpName = "SparseApplyAdagrad";
constexpr auto kMomentumOpName = "Momentum";
constexpr auto kApplyMomentumOpName = "ApplyMomentum";
constexpr auto kCombineMomentumOpName = "CombineMomentum";
constexpr auto kCombineMomentumWeightOpName = "CombineMomentumWeight";
constexpr auto kApplyAdadeltaOpName = "ApplyAdadelta";
constexpr auto kApplyAdagradOpName = "ApplyAdagrad";
constexpr auto kApplyAdagradDAName = "ApplyAdagradDA";
constexpr auto kApplyAdamOpName = "Adam";
constexpr auto kApplyAdaMaxOpName = "ApplyAdaMax";
constexpr auto kApplyAddSignOpName = "ApplyAddSign";
constexpr auto kApplyCenteredRMSPOpName = "ApplyCenteredRMSP";
constexpr auto kApplyCenteredRMSPropOpName = "ApplyCenteredRMSProp";
constexpr auto kApplyFtrlOpName = "ApplyFtrl";
constexpr auto kApplyFtrlV2OpName = "ApplyFtrlV2";
constexpr auto kApplyGradientDescentOpName = "ApplyGradientDescent";
constexpr auto kApplyPowerSignOpName = "ApplyPowerSign";
constexpr auto kApplyProximalAdagradOpName = "ApplyProximalAdagrad ";
constexpr auto kApplyProximalGradientDescentOpName = "ApplyProximalGradientDescent";
constexpr auto kApplyRMSPropOpName = "ApplyRMSProp";
constexpr auto kTransDataOpName = "TransData";
constexpr auto kTransDataRNNOpName = "TransDataRNN";
constexpr auto kStackInitOpName = "StackInit";
constexpr auto kStackPushOpName = "StackPush";
constexpr auto kStackPopOpName = "StackPop";
constexpr auto kStackOpName = "Stack";
constexpr auto kStackDestroyOpName = "StackDestroy";
constexpr auto kBNTrainingUpdateGradOpName = "BNTrainingUpdateGrad";
constexpr auto kBNTrainingReduceGradOpName = "BNTrainingReduceGrad";
constexpr auto kSquareSumV1OpName = "SquareSumV1";
constexpr auto kSquareSumV2OpName = "SquareSumV2";
constexpr auto kClipByNormNoDivSumOpName = "ClipByNormNoDivSum";
constexpr auto kPReluOpName = "PReLU";
constexpr auto kGreaterOpName = "Greater";
constexpr auto kSqrtOpName = "Sqrt";
constexpr auto kRsqrtOpName = "Rsqrt";
constexpr auto kRsqrtGradOpName = "RsqrtGrad";
constexpr auto kErfOpName = "Erf";
constexpr auto kDivOpName = "Div";
constexpr auto kRealDivOpName = "RealDiv";
constexpr auto kLambUpdateWithLROpName = "LambUpdateWithLR";
constexpr auto kLambNextMVWithDecayOpName = "LambNextMVWithDecay";
constexpr auto kLambNextMVWithDecayV1OpName = "LambNextMVWithDecayV1";
constexpr auto kClipByValueOpName = "ClipByValue";
constexpr auto kLambNextRightOpName = "LambNextRight";
constexpr auto kConfusionSoftmaxGradOpName = "ConfusionSoftmaxGrad";
constexpr auto kLambUpdateWithLrV2OpName = "LambUpdateWithLrV2";
constexpr auto kLayerNormXBackpropOpName = "LayerNormXBackprop";
constexpr auto kLayerNormXBackpropV2OpName = "LayerNormXBackpropV2";
constexpr auto kLayerNormBetaGammaBackpropOpName = "LayerNormBetaGammaBackprop";
constexpr auto kLayerNormBetaGammaBackpropV2OpName = "LayerNormBetaGammaBackpropV2";
constexpr auto kLambNextMVOpName = "LambNextMV";
constexpr auto kConfusionTransposeDOpName = "ConfusionTransposeD";
constexpr auto kAdamApplyOneWithDecayOpName = "AdamApplyOneWithDecay";
constexpr auto kAdamApplyOneWithDecayAssignOpName = "AdamApplyOneWithDecayAssign";
constexpr auto kBatchNormGradOpName = "BatchNormGrad";
constexpr auto kBNInferOpName = "BNInfer";
constexpr auto kAdamApplyOneOpName = "AdamApplyOne";
constexpr auto kAdamApplyOneAssignOpName = "AdamApplyOneAssign";
constexpr auto kResizeNearestNeighborGradOpName = "ResizeNearestNeighborGrad";
constexpr auto kFusedMulAddOpName = "FusedMulAdd";
constexpr auto kFusedMulAddNOpName = "FusedMulAddN";
constexpr auto kFusedMulApplyMomentumOpName = "FusedMulApplyMomentum";
constexpr auto kBiasAddOpName = "BiasAdd";
constexpr auto kConfusionMulGradOpName = "ConfusionMulGrad";
constexpr auto kStreamSwitchOpName = "StreamSwitch";
constexpr auto kStreamActiveOpName = "StreamActive";
constexpr auto kEndGraph = "EndGraph";
constexpr auto kAssignAddOpName = "AssignAdd";
constexpr auto kAssignSubOpName = "AssignSub";
constexpr auto kSendOpName = "StreamSend";
constexpr auto kRecvOpName = "StreamRecv";
constexpr auto kRpcSendOpName = "RpcSend";
constexpr auto kRpcRecvOpName = "RpcRecv";
constexpr auto kReluOpName = "ReLU";
constexpr auto kReluGradOpName = "ReluGrad";
constexpr auto kReluV2OpName = "ReLUV2";
constexpr auto kReluGradV2OpName = "ReluGradV2";
constexpr auto kAddOpName = "Add";
constexpr auto kAddNOpName = "AddN";
constexpr auto kResizeNearestNeighborV2OpName = "ResizeNearestNeighborV2";
constexpr auto kResizeNearestNeighborV2GradOpName = "ResizeNearestNeighborV2Grad";
constexpr auto kApplyRMSPropOpname = "ApplyRMSProp";
constexpr auto kCumsumOpName = "Cumsum";
constexpr auto kInplaceAddOpName = "InplaceAdd";
constexpr auto kInplaceSubOpName = "InplaceSub";
constexpr auto kResizeBilinearV2OpName = "kResizeBilinearV2";
constexpr auto kReduceProdOpName = "ReduceProd";
constexpr auto kCumprodOpName = "Cumprod";
constexpr auto kSpaceToBatchOpName = "SpaceToBatch";
constexpr auto kBatchToSpaceOpName = "BatchToSpace";
constexpr auto kSpaceToDepthOpName = "SpaceToDepth";
constexpr auto kPadOpName = "Pad";
constexpr auto kConv2DTransposeOpName = "Conv2DTranspose";
constexpr auto kConv2DBackpropInputOpName = "Conv2DBackpropInput";
constexpr auto kConv2DBackpropFilterOpName = "Conv2DBackpropFilter";
constexpr auto kDepthwiseConv2dNativeOpName = "DepthwiseConv2dNative";
constexpr auto kDepthwiseConv2dNativeBackpropInputOpName = "DepthwiseConv2dNativeBackpropInput";
constexpr auto kDepthwiseConv2dNativeBackpropFilterOpName = "DepthwiseConv2dNativeBackpropFilter";
constexpr auto kFusionOpConv2DBackpropInputReluGradV2Name = "FusionOp_Conv2DBackpropInput_ReluGradV2";
constexpr auto kFusionOpConv2DBackpropInputAddNReluGradV2Name = "FusionOp_Conv2DBackpropInput_AddN_ReluGradV2";
constexpr auto kLabelSetOpName = "LabelSet";
constexpr auto kLabelSwitchOpName = "LabelSwitch";
constexpr auto kLabelGotoOpName = "LabelGoto";
constexpr auto kBNInferGradOpName = "BNInferGrad";
constexpr auto kCallOpName = "call";
constexpr auto kPartialOpName = "partial";
constexpr auto kSwitchOpName = "Switch";
constexpr auto kReturnOpName = "Return";
constexpr auto kBpropCutOpName = "bprop_cut";
constexpr auto kLarsV2OpName = "LarsV2";
constexpr auto kLarsV2UpdateOpName = "LarsV2Update";
constexpr auto kSquareSumAllOpName = "SquareSumAll";
constexpr auto kNMSWithMaskOpName = "NMSWithMask";
constexpr auto kSoftmaxGradExtOpName = "SoftmaxGradExt";
constexpr auto kStridedReadOpName = "StridedRead";
constexpr auto kStridedWriteOpName = "StridedWrite";
constexpr auto kFusedAdamWeightDecayName = "FusedAdamWeightDecay";
constexpr auto kAdamWeightDecayName = "AdamWeightDecay";
constexpr auto kFusedCastAdamWeightDecayName = "FusedCastAdamWeightDecay";
constexpr auto kFusedAdamName = "FusedAdam";
constexpr auto kFusedAdaFactorName = "FusedAdaFactor";
constexpr auto kFusedAdaFactorWithGlobalNormName = "FusedAdaFactorWithGlobalNorm";
constexpr auto kFusedSparseAdamName = "FusedSparseAdam";
constexpr auto kFusedMatMulBiasAddName = "FusedMatMulBiasAdd";
constexpr auto kDeadNodeName = "DeadNode";
constexpr auto kPolyNodeName = "PolyNode";
constexpr auto kApplyAdagradV2OpName = "ApplyAdagradV2";
constexpr auto kSparseApplyAdagradV2OpName = "SparseApplyAdagradV2";
constexpr auto kSparseApplyFtrlOpName = "SparseApplyFtrl";
constexpr auto kSparseApplyFtrlV2OpName = "SparseApplyFtrlV2";
constexpr auto kApplyKerasMomentumOpName = "ApplyKerasMomentum";
constexpr auto kSparseApplyProximalAdagradOpName = "SparseApplyProximalAdagrad";
constexpr auto kSparseApplyRMSPropOpName = "SparseApplyRMSProp";
constexpr auto kSparseApplyAdadeltaOpName = "SparseApplyAdadelta";
constexpr auto kApplyAdamWithAmsgradOpName = "ApplyAdamWithAmsgrad";
constexpr auto kTensorMoveOpName = "TensorMove";
constexpr auto kTensorCopySlicesOpName = "TensorCopySlices";
constexpr auto kTensorScatterUpdateOpName = "TensorScatterUpdate";
constexpr auto kScatterNdUpdateOpName = "ScatterNdUpdate";
constexpr auto kPushOpName = "Push";
constexpr auto kPullOpName = "Pull";
constexpr auto kPushWeightOpName = "PushWeight";
constexpr auto kPullWeightOpName = "PullWeight";
constexpr auto kFusedPushWeightOpName = "FusedPushWeight";
constexpr auto kFusedPullWeightOpName = "FusedPullWeight";
constexpr auto kUpdateCacheOpName = "UpdateCache";
constexpr auto kCacheSwapTableOpName = "CacheSwapTable";
constexpr auto kEmbeddingLookupOpName = "EmbeddingLookup";
constexpr auto kEmbeddingLookupProxyOpName = "EmbeddingLookupProxy";
constexpr auto kGatherV2OpName = "Gather";
constexpr auto kPaddingOpName = "Padding";
constexpr auto kPoolingOpName = "Pooling";
constexpr auto kAvgPoolOpName = "AvgPool";
constexpr auto kAvgPoolGradOpName = "AvgPoolGrad";
constexpr auto kAvgPool3DOpName = "AvgPool3D";
constexpr auto kAvgPool3DGradOpName = "AvgPool3DGrad";
constexpr auto kAvgPoolGradVmOpName = "AvgPoolGradVm";
constexpr auto kMaxPoolOpName = "MaxPool";
constexpr auto kMaxPoolGradOpName = "MaxPoolGrad";
constexpr auto kMaxPool3DOpName = "MaxPool3D";
constexpr auto kMaxPool3DGradOpName = "MaxPool3DGrad";
constexpr auto kMaxPoolWithArgmaxOpName = "MaxPoolWithArgmax";
constexpr auto kMaxPoolGradWithArgmaxOpName = "MaxPoolGradWithArgmax";
constexpr auto kTensorAddOpName = "Add";
constexpr auto kKLDivLossOpName = "KLDivLoss";
constexpr auto kMaxPool3DGradGradOpName = "MaxPool3DGradGrad";
constexpr auto kCastOpName = "Cast";
constexpr auto kGreaterEqualOpName = "GreaterEqual";
constexpr auto kAbsOpName = "Abs";
constexpr auto kExpOpName = "Exp";
constexpr auto kNegOpName = "Neg";
constexpr auto kMinimumOpName = "Minimum";
constexpr auto kMaximumOpName = "Maximum";
constexpr auto kMulOpName = "Mul";
constexpr auto kSubOpName = "Sub";
constexpr auto kLogOpName = "Log";
constexpr auto kPowOpName = "Pow";
constexpr auto kReciprocalOpName = "Reciprocal";
constexpr auto kEqualOpName = "Equal";
constexpr auto kLessOpName = "Less";
constexpr auto kLessEqualOpName = "LessEqual";
constexpr auto kSquareOpName = "Square";
constexpr auto kSelectOpName = "Select";
constexpr auto kCSRSparseMatrixToSparseTensorOpName = "CSRSparseMatrixToSparseTensor";
constexpr auto kSparseSparseMinimumOpName = "SparseSparseMinimum";
constexpr auto kReduceSumOpName = "ReduceSum";
constexpr auto kReduceMinOpName = "ReduceMin";
constexpr auto kReduceMaxOpName = "ReduceMax";
constexpr auto kReduceMeanOpName = "ReduceMean";
constexpr auto kReduceAnyOpName = "ReduceAny";
constexpr auto kReduceAllOpName = "ReduceAll";
constexpr auto kRenormOpName = "Renorm";
constexpr auto kFusedWeightScaleApplyMomentum = "FusedWeightScaleApplyMomentum";
constexpr auto kFusedWeightApplyMomentum = "FusedWeightApplyMomentum";
constexpr auto kFusedScaleApplyMomentum = "FusedScaleApplyMomentum";
constexpr auto kBasicLSTMCellWeightGradOpName = "BasicLSTMCellWeightGrad";
constexpr auto kBasicLSTMCellInputGradOpName = "BasicLSTMCellInputGrad";
constexpr auto kBasicLSTMCellOpName = "BasicLSTMCell";
constexpr auto kDynamicRNNOpName = "DynamicRNN";
constexpr auto kLSTMOpName = "LSTM";
constexpr auto kLSTMGradOpName = "LSTMGrad";
constexpr auto kLSTMInputGradOpName = "LSTMInputGrad";
constexpr auto kDynamicGRUV2OpName = "DynamicGRUV2";
constexpr auto kGRUV2HiddenGradOpName = "GRUV2HiddenGrad";
constexpr auto kGRUV2HiddenGradCellOpName = "GRUV2HiddenGradCell";
constexpr auto kFusedSparseFtrlName = "FusedSparseFtrl";
constexpr auto kFusedSparseProximalAdagradName = "FusedSparseProximalAdagrad";
constexpr auto kFusedSparseLazyAdamName = "FusedSparseLazyAdam";
constexpr auto kSparseApplyFtrlName = "SparseApplyFtrl";
constexpr auto kSparseApplyFtrlV2Name = "SparseApplyFtrlV2";
constexpr auto kSGDName = "SGD";
constexpr auto kLARSUpdateName = "LARSUpdate";
constexpr auto kBasicLSTMCellCStateGradOpName = "BasicLSTMCellCStateGrad";
constexpr auto kBasicLSTMCellCStateGradV2OpName = "BasicLSTMCellCStateGradV2";
constexpr auto kMatMulOpName = "MatMul";
constexpr auto kMatMulV2OpName = "MatMulV2";
constexpr auto kBatchMatMulOpName = "BatchMatMul";
constexpr auto kBatchMatMulV2OpName = "BatchMatMulV2";
constexpr auto kBroadcastToOpName = "BroadcastTo";
constexpr auto kFusedAddReluV2Name = "FusedAddReluV2";
constexpr auto kFusedAddReluGradV2Name = "FusedAddReluGradV2";
constexpr auto kDropoutOpName = "Dropout";
constexpr auto kDropoutGradOpName = "DropoutGrad";
constexpr auto kDropoutGenMaskOpName = "DropoutGenMask";
constexpr auto kDropoutGenMaskV3OpName = "DropoutGenMaskV3";
constexpr auto kDropoutDoMaskOpName = "DropoutDoMask";
constexpr auto kDropoutDoMaskV3OpName = "DropoutDoMaskV3";
constexpr auto kSoftmaxV2WithDropoutDoMaskV3OpName = "SoftmaxV2WithDropoutDoMaskV3";
constexpr auto kGammaOpName = "Gamma";
constexpr auto kPoissonOpName = "Poisson";
constexpr auto kStandardLaplaceOpName = "StandardLaplace";
constexpr auto kStandardNormalOpName = "StandardNormal";
constexpr auto kUniformIntOpName = "UniformInt";
constexpr auto kUniformRealOpName = "UniformReal";
constexpr auto kSubAndFilterOpName = "SubAndFilter";
constexpr auto kPadAndShiftOpName = "PadAndShift";
constexpr auto kSparseSoftmaxCrossEntropyWithLogitsOpName = "SparseSoftmaxCrossEntropyWithLogits";
constexpr auto kOneHotOpName = "OneHot";
constexpr auto kSigmoidOpName = "Sigmoid";
constexpr auto kSoftmaxCrossEntropyWithLogitsOpName = "SoftmaxCrossEntropyWithLogits";
constexpr auto kUniformCandidateSamplerOpName = "UniformCandidateSampler";
constexpr auto kLogSoftmaxGradOpName = "LogSoftmaxGrad";
constexpr auto kLayerNormGradOpName = "LayerNormGrad";
constexpr auto kMinimumGradOpName = "MinimumGrad";
constexpr auto kMaximumGradOpName = "MaximumGrad";
constexpr auto kFusedDbnDwOpName = "FusedDbnDw";
constexpr auto kNPUAllocFloatStatusOpName = "NPUAllocFloatStatus";
constexpr auto kNPUGetFloatStatusOpName = "NPUGetFloatStatus";
constexpr auto kNPUClearFloatStatusOpName = "NPUClearFloatStatus";
constexpr auto kAssignOpName = "Assign";
constexpr auto kScatterAddOpName = "ScatterAdd";
constexpr auto kScatterUpdateOpName = "ScatterUpdate";
constexpr auto kEnvironCreateOpName = "EnvironCreate";
constexpr auto kEnvironSetOpName = "EnvironSet";
constexpr auto kEnvironGetOpName = "EnvironGet";
constexpr auto kEnvironDestroyAllOpName = "EnvironDestroyAll";
constexpr auto kGatherDGradV2OpName = "GatherDGradV2";
constexpr auto kNonDeterministicInts = "NonDeterministicInts";
constexpr auto kResizeBicubicOpName = "ResizeBicubic";
constexpr auto kUpdateStateOpName = "UpdateState";
constexpr auto kTruncatedNormal = "TruncatedNormal";
constexpr auto kPriorityReplayBufferCreate = "PriorityReplayBufferCreate";
constexpr auto kPriorityReplayBufferPush = "PriorityReplayBufferPush";
constexpr auto kPriorityReplayBufferSample = "PriorityReplayBufferSample";
constexpr auto kPriorityReplayBufferUpdate = "PriorityReplayBufferUpdate";
constexpr auto kPriorityReplayBufferDestroy = "PriorityReplayBufferDestroy";
constexpr auto kReservoirReplayBufferCreate = "ReservoirReplayBufferCreate";
constexpr auto kReservoirReplayBufferPush = "ReservoirReplayBufferPush";
constexpr auto kReservoirReplayBufferSample = "ReservoirReplayBufferSample";
constexpr auto kReservoirReplayBufferDestroy = "ReservoirReplayBufferDestroy";
constexpr auto kDenseToDenseSetOperation = "DenseToDenseSetOperation";
constexpr auto kResizeAreaOpName = "ResizeArea";
constexpr auto kNonZeroOpName = "NonZero";
constexpr auto kExtractGlimpse = "ExtractGlimpse";
constexpr auto kConv3DTransposeOpName = "Conv3DTranspose";
constexpr auto kParallelResizeBilinearGradOpName = "ParallelResizeBilinearGrad";
constexpr auto kArgminV2OpName = "ArgminV2";
constexpr auto kCentralizationOpName = "Centralization";
constexpr auto kGatherOpName = "Gather";
constexpr auto kGatherDOpName = "GatherD";
constexpr auto kEmbeddingLookupCommGradOpName = "EmbeddingLookupCommGrad";
constexpr auto kSubscalarOpName = "Subscalar";
constexpr auto kUnsortedSegmentSumOpName = "UnsortedSegmentSum";
constexpr auto kCumSumOpName = "CumSum";
constexpr auto kCumProdOpName = "CumProd";
constexpr auto kUnsortedSegmentMaxOpName = "UnsortedSegmentMax";
constexpr auto kCSRReduceSumOpName = "CSRReduceSum";
constexpr auto kCSRMVOpName = "CSRMV";
constexpr auto kCSRMMOpName = "CSRMM";
constexpr auto kCSRMulOpName = "CSRMul";
constexpr auto kCSRDivOpName = "CSRDiv";
constexpr auto kCSRGatherOpName = "CSRGather";
constexpr auto kCSR2COOOpName = "CSR2COO";
constexpr auto kCOO2CSROpName = "COO2CSR";
constexpr auto kRandomShuffle = "RandomShuffle";

// Communication world group
constexpr auto kNcclWorldGroup = "nccl_world_group";
constexpr auto kHcclWorldGroup = "hccl_world_group";
constexpr auto kSyncBnGroup = "sync_bn_group";
constexpr auto kRankID = "RANK_ID";

// Hcom Op Type
constexpr auto kHcomOpTypeAllReduce = "HcomAllReduce";
constexpr auto kHcomOpTypeAllGather = "HcomAllGather";
constexpr auto kHcomOpTypeBroadcast = "HcomBroadcast";
constexpr auto kHcomOpTypeSend = "HcomSend";
constexpr auto kHcomOpTypeReceive = "HcomReceive";
constexpr auto kHcomOpTypeReduceScatter = "HcomReduceScatter";

// attr key name
constexpr auto kAttrAlignCorners = "align_corners";
constexpr auto kAttrHalfPixelCenters = "half_pixel_centers";
constexpr auto kAttrInputNames = "input_names";
constexpr auto kAttrAttrNames = "attr_names";
constexpr auto kAttrBins = "bins";
constexpr auto kAttrMin = "min";
constexpr auto kAttrMax = "max";
constexpr auto kAttrIsAiCpuKernel = "is_AICPU_kernel";
constexpr auto kIsBackendCast = "is_backed_cast";
constexpr auto kAttrOutputNames = "output_names";
constexpr auto kAttrAsync = "async";
constexpr auto kAttrOffload = "offload";
constexpr auto kAttrOutIdx = "out_idx";
constexpr auto kAttrVisited = "visited";
constexpr auto kAttrReshapePaddingAxis = "reshape_padding_axis";
constexpr auto kAttrShape = "shape";
constexpr auto kAttrMomentum = "momentum";
constexpr auto kAttrEps = "eps";
constexpr auto kAttrEpsilon = "epsilon";
constexpr auto kAttrFactor = "factor";
constexpr auto kAttrIsRef = "isRef";
constexpr auto kAttrDataShape = "data_shape";
constexpr auto kAttrFormat = "format";
constexpr auto kAttrReshapeType = "reshape_type";
constexpr auto kAttrAxis = "axis";
constexpr auto kAttrAxes = "axes";
constexpr auto kAttrKeepDims = "keep_dims";
constexpr auto kAttrShapeGamma = "shape_gamma";
constexpr auto kAttrPerm = "perm";
constexpr auto kAttrTransposeFirst = "transpose_first";
constexpr auto kAttrAtomicAddMemSize = "automic_add_mem_size";
constexpr auto kAttrAtomicOutputIndexs = "atomic_output_clean_indexs";
constexpr auto kAttrNeedAtomic = "need_atomic";
constexpr auto kAttrAtomicWorkspaceIndexs = "atomic_workspace_clean_indexs";
constexpr auto kAttrSwitchCondition = "switch_condition";
constexpr auto kAttrDataType = "data_type";
constexpr auto kAttrDType = "dtype";
constexpr auto kAttrActiveTarget = "active_target";
constexpr auto kAttrActiveStreamId = "active_stream_id";
constexpr auto kAttrActiveStreamList = "active_stream_list";
constexpr auto kAttrTrueBranchStream = "true_branch_stream";
constexpr auto kAttrStreamSwitchKind = "stream_switch_kind";
constexpr auto kAttrEventId = "event_id";
constexpr auto kAttrLabelId = "label_id";
constexpr auto kAttrLogicId = "logic_id";
constexpr auto kAttrNodeInfo = "node_info";
constexpr auto kAttrNodeName = "node_name";
constexpr auto kAttrDynInput = "dynamic";
constexpr auto kAttrDynInputSizes = "dyn_input_sizes";
constexpr auto kAttrSrcFormat = "src_format";
constexpr auto kAttrDstFormat = "dst_format";
constexpr auto kAttrMultiples = "multiples";
constexpr auto kAttrFixPrecision = "fix_precision";
constexpr auto kAttrOutputPrecision = "output_precision";
constexpr auto kAttrOutputUsedNum = "output_used_num";
constexpr auto kAttrHasBias = "has_bias";
constexpr auto kAttrN = "n";
constexpr auto kAttrLabelForInsertStreamActive = "label_for_insert_stream_active";
constexpr auto kAttrFpBpEnd = "fpbp_end";
constexpr auto kAttrFusion = "fusion";
constexpr auto kAttrNotDelayFusion = "not_delay_fusion";
constexpr auto kAttrGroup = "group";
constexpr auto kAttrRankList = "rank_list";
constexpr auto kAttrGroups = "groups";
constexpr auto kAttrGroupBack = "group_back";
constexpr auto kAttrFracZGroup = "fracz_group";
constexpr auto kAttrFracZGroupIdx = "fracz_group_idx";
constexpr auto kAttrOp = "op";
constexpr auto kAttrDestRank = "dest_rank";
constexpr auto kAttrSrcRank = "src_rank";
constexpr auto kAttrSrTag = "sr_tag";
constexpr auto kAttrRootRank = "root_rank";
constexpr auto kAttrComm = "comm";
constexpr auto kAttrIsTraining = "is_training";
constexpr auto kAttrFusionId = "fusion_id";
constexpr auto kAttrDuplicated = "duplicated";
constexpr auto kAttrBucketId = "bucket_id";
constexpr auto kAttrGradOutputIndex = "grad_output_index";
constexpr auto kAttrLabelIndex = "label_index";
constexpr auto kAttrLabelSwitchList = "label_switch_list";
constexpr auto kAttrBeginMask = "begin_mask";
constexpr auto kAttrEndMask = "end_mask";
constexpr auto kAttrEllipsisMask = "ellipsis_mask";
constexpr auto kAttrNewAxisMask = "new_axis_mask";
constexpr auto kAttrShrinkAxisMask = "shrink_axis_mask";
constexpr auto kAttrDatadumpOriginalNames = "_datadump_original_names";
constexpr auto kAttrDatadumpIsMultiop = "_datadump_is_multiop";
constexpr auto kAttrNeedRecordEvent = "need_record_event";
constexpr auto kAttrStreamId = "stream_id";
constexpr auto kAttrRecomputeId = "recompute_id";
constexpr auto kAttrRecordEvent = "record_event";
constexpr auto kAttrWaitEvent = "wait_event";
constexpr auto kAttrRecordEventStream = "record_event_stream";
constexpr auto kAttrWaitEventStream = "wait_event_stream";
constexpr auto kAttrStream = "stream";
constexpr auto kAttrIndex = "index";
constexpr auto kAttrSplitDim = "split_dim";
constexpr auto kAttrNumSplit = "num_split";
constexpr auto kAttrReduction = "reduction";
constexpr auto kAttrOutputNum = "output_num";
constexpr auto kAttrOutputSize = "output_size";
constexpr auto kAttrScales = "scales";
constexpr auto kAttrSizeSplits = "size_splits";
constexpr auto kAttrOutputDefault = "output_default";
constexpr auto kAttrPrimitiveTarget = "primitive_target";
constexpr auto kAttrNotSupportOpForDevice = "not_support_op_for_device";
constexpr auto kAttrUseLocking = "use_locking";
constexpr auto kAttrReduceScatterFlag = "reduce_scatter_flag";
constexpr auto kAttrOffset = "offset";
constexpr auto kAttrCacheEnable = "cache_enable";
constexpr auto kAttrPsKey = "ps_key";
constexpr auto kAttrOptimizerType = "optim_type";
constexpr auto kAttrChildGraph = "child_graph";
constexpr auto kAttrInputNums = "inputNums";
constexpr auto kAttrT = "T";
constexpr auto kAttrNum = "num";
constexpr auto kAttrRecvType = "recv_type";
constexpr auto kAttrConcatDim = "concat_dim";
constexpr auto kAttrSplitCount = "split_count";
constexpr auto kAttrSendRankIds = "send_rank_ids";
constexpr auto kAttrRecvRankIds = "recv_rank_ids";
constexpr auto kAttrSendLens = "send_lens";
constexpr auto kAttrRecvLens = "recv_lens";
constexpr auto kAttrRankSize = "rank_size";
constexpr auto kAttrPadDimSize = "pad_dim_size";
constexpr auto kAttrPaddings = "paddings";
constexpr auto kAttrNumSegments = "num_segments";
constexpr auto kAttrStackOpName = "stack_op_name";
constexpr auto kAttrBegin = "begin";
constexpr auto kAttrEnd = "end";
constexpr auto kAttrSize = "size";
constexpr auto kAttrKsizes = "ksizes";
constexpr auto kAttrIsKernelDynamicImpl = "is_kernel_dynamic_impl";
constexpr auto kAttrIsKernelDynamicShape = "is_kernel_dynamic_shape";
constexpr auto kAttrIsDynamicShape = "is_dynamic_shape";
constexpr auto kAttrInputIsDynamicShape = "input_is_dynamic_shape";
constexpr auto kAttrOutputIsDynamicShape = "output_is_dynamic_shape";
constexpr auto kAttrPynativeNextOpName = "next_op";
constexpr auto kAttrPynativeNextIndex = "next_index";
constexpr auto kAttrCompileInfo = "compile_info";
constexpr auto kAttrFusionType = "fusion_type";
constexpr auto kAttrStride = "stride";
constexpr auto kAttrStrides = "strides";
constexpr auto kAttrKernelSize = "kernel_size";
constexpr auto kAttrDilation = "dilation";
constexpr auto kAttrPadMode = "pad_mode";
constexpr auto kAttPaddingMode = "padding_mode";
constexpr auto kAttrPad = "pad";
constexpr auto kAttrPadding = "padding";
constexpr auto kAttrMode = "mode";
constexpr auto kAttrWindow = "window";
constexpr auto kAttrCeilMode = "ceil_mode";
constexpr auto kAttrGlobalPooling = "global_pooling";
constexpr auto kAttrNonTask = "non_task";
constexpr auto kAttrIsGrad = "is_grad";
constexpr auto kAttrRecompute = "recompute";
constexpr auto kAttrSliceActivation = "slice_activation";
constexpr auto kAttrNeedCseAfterRecompute = "need_cse_after_recompute";
constexpr auto kAttrParallelDimInfo = "parallel_dim_info";
constexpr auto kAttrParallelFusionType = "parallel_fusion_type";
constexpr auto kAttrParallelTypeInfo = "parallel_type_info";
constexpr auto kAttrCompositeType = "composite_type";
constexpr auto kAttrStitch = "stitch";
constexpr auto kAttrTopoSortRhsFirst = "topo_sort_rhs_first";
constexpr auto kAttrIgnoreSideEffect = "ignore_side_effect";
constexpr auto kAttrSwitchLayer = "switch_layer";
constexpr auto kAttrReturn = "return";
constexpr auto kAttrRecursiveStart = "recursive_start";
constexpr auto kAttrRecursiveEnd = "recursive_end";
constexpr auto kAttrRecursive = "recursive";
constexpr auto kAttrMultiCallEnd = "multicall_end";
constexpr auto kAttrProfilingIterEnd = "PROFILING_ITER_END";
constexpr auto kAttrHiddenSize = "hidden_size";
constexpr auto kAttrInputSize = "input_size";
constexpr auto kAttrDstType = "dst_type";
constexpr auto kAttrDump = "dump";
constexpr auto kAttrSkipNopOpAddr = "skip_nop_op_addr";
constexpr auto kAttrSkipNopOpExecution = "skip_nop_op_execution";
constexpr auto kAttrFixedInputFormat = "fixed_input_format";
constexpr auto kAttrFixedOutputFormat = "fixed_output_format";
constexpr auto kAttrFixedInputDeviceShape = "fixed_input_device_shape";
constexpr auto kAttrFixedOutputDeviceShape = "fixed_output_device_shape";
constexpr auto kAttrFuncType = "func_type";
constexpr auto kNonMaxSuppressionWithOverlapsOpName = "NonMaxSuppressionWithOverlaps";
constexpr auto kAttrCustAicpu = "cust_aicpu";
constexpr auto kAttrIsInternalOutputNopNode = "is_internal_output_nop_node";
constexpr auto kAttrIsUBFusionOp = "is_ub_fusion_op";
constexpr auto kAttrNopOp = "nop_op";
constexpr auto kAttrPlaceHolderIndex = "placeholder_index";
constexpr auto kAttrMicro = "micro";
constexpr auto kAttrJsonFileName = "json_file_name";
constexpr auto kAttrNeedDropInput = "need_drop_input";
constexpr auto kAttrNeedConvertToValueNode = "need_convert_to_value_node";
constexpr auto kAttrSendSrcNodeName = "send_src_node_name";
constexpr auto kAttrSendDstNodeName = "send_dst_node_name";
constexpr auto kAttrSendDstRanks = "send_dst_ranks";
constexpr auto kAttrSendDstRoles = "send_dst_roles";
constexpr auto kAttrRecvSrcNodeName = "recv_src_node_name";
constexpr auto kAttrRecvDstNodeName = "recv_dst_node_name";
constexpr auto kAttrRecvSrcRanks = "recv_src_ranks";
constexpr auto kAttrRecvSrcRoles = "recv_src_roles";
constexpr auto kAttrInterProcessEdgeNames = "inter_process_edge_names";
constexpr auto kAttrInterProcessEdgeLabel = "inter_process_edge_label";
constexpr auto kAttrIsMuxRpcKernel = "is_mux_rpc_kernel";
constexpr auto kAttrGroupRankIds = "group_rank_ids";
constexpr auto kAttrReuseCommunication = "reuse_communication_node";
constexpr auto kAttrPrecisionFlag = "precision_flag";
constexpr auto kAttrDfmGroup = "deformable_groups";
constexpr auto kAttrModulated = "modulated";
constexpr auto kAttrDilations = "dilations";
constexpr auto kAttrDataFormat = "data_format";
constexpr auto kAttrPads = "pads";
constexpr auto kAttrKsize = "ksize";
constexpr auto kAttrOnlyUseFirstOutput = "only_use_first_output";
constexpr auto kAttrOnlyUseSecondOutput = "only_use_second_output";
constexpr auto kActualAbstract = "actual_abstract";
constexpr auto kAttrZeroInfinity = "zero_infinity";
constexpr auto kAttrBlank = "blank";
constexpr auto kAttrUpdateSlots = "update_slots";
constexpr auto kAttrLr = "lr";
constexpr auto kAttrNeedGradFlagOfInputs = "need_grad_flag_of_inputs";
constexpr auto kAttrIsCNodeNeedGrad = "is_cnode_need_grad";
constexpr auto kAttrJitLevel = "jit_level";
constexpr auto kAttrJitLevelO2 = "O2";
constexpr auto kAttrCellJitConfigDict = "_jit_config_dict";
constexpr auto kAttrBinaryOutput = "binary_output";
constexpr auto kAttrMinLength = "minlength";
constexpr auto kAttrMaxLength = "maxlength";

// FuncGraph Flags
constexpr auto kFlagsIsCutGraph = "is_cut_graph";
constexpr auto kFlagIsDynamicStructure = "is_dynamic_structure";
constexpr auto kFlagIsPynativeBpropGraph = "is_pynative_bprop_graph";
constexpr auto kFlagPyNativeRunInGraph = "pynative_run_in_graph";
constexpr auto kFlagNeedRenormalize = "need_renormalize";
constexpr auto kFlagEnableZeroCopyInGraph = "enable_zero_copy_in_graph";
// TODO(dsj): for ms_function running in graph_mode. should be delete later
constexpr auto kAttrMSFunction = "ms_function_graph";

// custom operator func type
constexpr auto kCustomTypeAOT = "aot";
constexpr auto kCustomTypeJULIA = "julia";
constexpr auto kCustomTypePyfunc = "pyfunc";
constexpr auto kCustomTypeTbe = "tbe";
constexpr auto kCustomTypeAICPU = "aicpu";
constexpr auto kCustomTypeHybrid = "hybrid";

// primal attr key name
constexpr auto kPrimalAttrForwardNodeName = "forward_node_name";

// attr value
constexpr auto kValueTargetSwitch = "target_switch";
constexpr auto kValueTargetOther = "target_other";
constexpr auto kValueTrue = "true";
constexpr auto kTensorValueIsType = "tensor_value_is_type";

// env key
constexpr auto kGraphOpRun = "GRAPH_OP_RUN";

// some size
const size_t kShape4dDims = 4;
const size_t kShape3dDims = 3;
const size_t kShape2dDims = 2;
const size_t kShape5dDims = 5;
const size_t kShape1dDims = 1;
const size_t kCubeSize = 16;
const size_t kCubeSize_C04 = 4;
const size_t kNiSize = 16;
const size_t kMemAlignSize = 512;
const size_t kBNChannelMultipleFactor = 4;
const int kParameterDataTensorMask = 0;
const int kParameterWeightTensorMask = 1;
const int kValueNodeTensorMask = 2;
constexpr auto kNCHWShapeSize = 4;

// define special index in special node
constexpr auto kDefaultStreamIndex = 0;
constexpr auto kWorldGroupStreamIndex = 1;
constexpr auto kAnfPrimitiveIndex = 0;
constexpr auto kFirstDataInputIndex = 1;
constexpr auto kRealInputNodeIndexInTupleGetItem = 1;
constexpr auto kInputNodeOutputIndexInTupleGetItem = 2;
constexpr auto kSparseGetAttrInputSize = 2;
constexpr auto kTupleGetItemInputSize = 3;

// index define of kTupleSetItem
constexpr auto kTupleSetItemTupleIndex = 1;
constexpr auto kTupleSetItemIndexIndex = 2;
constexpr auto kTupleSetItemValueIndex = 3;
constexpr auto kTupleSetItemInputSize = 4;
// index define of partial
constexpr auto kPartialMinInputSize = 2;
constexpr auto kPartialGraphIndex = 1;

// index define of switch
constexpr auto kSwitchInputSize = 4;
constexpr auto kSwitchTrueBranchIndex = 2;
constexpr auto kSwitchFalseBranchIndex = 3;
constexpr auto kSwitchBranchesNum = 2;

// index define of GridSampler & GridSamplerGrad
constexpr int kGridSamplerInputNum = 2;
constexpr int kGridSamplerOutputNum = 1;
constexpr int kGridSamplerGradInputNum = 3;
constexpr int kGridSamplerGradOutputNum = 2;

// index define of switch_layer
constexpr auto kSwitchLayerInputSize = 3;
constexpr auto kSwitchLayerSelectIndex = 1;
constexpr auto kSwitchLayerBranchesIndex = 2;

// index define of depend
constexpr auto kRealInputIndexInDepend = 1;
constexpr auto kDependAttachNodeIndex = 2;
constexpr auto kDependInputSize = 3;
// index define of UpdateState
constexpr auto kUpdateStateStateInput = 1;
constexpr auto kUpdateStateRealInput = 2;
// index define of Load
constexpr auto kLoadRealInput = 1;
constexpr auto kLoadStateInput = 2;
// time transfer unit
constexpr int kBasicTimeTransferUnit = 1000;
constexpr int kMaxVectorSize = 10000;
// index of input or output
constexpr size_t kIndex0 = 0;
constexpr size_t kIndex1 = 1;
constexpr size_t kIndex2 = 2;
constexpr size_t kIndex3 = 3;
constexpr size_t kIndex4 = 4;
constexpr size_t kIndex5 = 5;
constexpr size_t kIndex6 = 6;
constexpr size_t kIndex7 = 7;
constexpr size_t kIndex8 = 8;
constexpr size_t kIndex9 = 9;
constexpr size_t kIndex10 = 10;
constexpr size_t kIndex11 = 11;
constexpr size_t kIndex12 = 12;
constexpr size_t kIndex13 = 13;
constexpr size_t kIndex14 = 14;
constexpr size_t kIndex15 = 15;
constexpr size_t kIndex16 = 16;
// dim of shape
constexpr size_t kDim0 = 0;
constexpr size_t kDim1 = 1;
constexpr size_t kDim2 = 2;
constexpr size_t kDim3 = 3;
constexpr size_t kDim4 = 4;
constexpr size_t kDim5 = 5;
constexpr size_t kDim6 = 6;
// format
constexpr auto kOpFormat_DEFAULT = "DefaultFormat";
constexpr auto kOpFormat_ChannelFirst = "ChannelFirst";
constexpr auto kOpFormat_ChannelLast = "ChannelLast";
constexpr auto kOpFormat_NC1KHKWHWC0 = "NC1KHKWHWC0";
constexpr auto kOpFormat_ND = "ND";
constexpr auto kOpFormat_NCHW = "NCHW";
constexpr auto kOpFormat_NHWC = "NHWC";
constexpr auto kOpFormat_HWCN = "HWCN";
constexpr auto kOpFormat_CHWN = "CHWN";
constexpr auto kOpFormat_NC1HWC0 = "NC1HWC0";
constexpr auto kOpFormat_FRAC_Z = "FRACTAL_Z";
constexpr auto kOpFormat_FRACTAL_Z = "FRACTAL_Z";
constexpr auto kOpFormat_FRAC_NZ = "FRACTAL_NZ";
constexpr auto kOpFormat_C1HWNCoC0 = "C1HWNCoC0";
constexpr auto kOpFormat_NC1HWC0_C04 = "NC1HWC0_C04";
constexpr auto kOpFormat_FRACTAL_Z_C04 = "FRACTAL_Z_C04";
constexpr auto kOpFormat_NDHWC = "NDHWC";
constexpr auto kOpFormat_NCDHW = "NCDHW";
constexpr auto kOpFormat_DHWNC = "DHWNC";
constexpr auto kOpFormat_DHWCN = "DHWCN";
constexpr auto kOpFormat_NDC1HWC0 = "NDC1HWC0";
constexpr auto kOpFormat_FRACTAL_Z_3D = "FRACTAL_Z_3D";
constexpr auto kOpFormat_FRACTAL_ZN_LSTM = "FRACTAL_ZN_LSTM";
constexpr auto kOpFormat_FRACTAL_ZN_RNN = "FRACTAL_ZN_RNN";
constexpr auto kOpFormat_ND_RNN_BIAS = "ND_RNN_BIAS";
constexpr auto kSliceStart = "start";
constexpr auto kSliceStop = "stop";
constexpr auto kSliceStep = "step";

COMMON_EXPORT bool IsOneOfCustomAkgType(const std::string &name);
COMMON_EXPORT bool IsOneOfOperator(const std::string &name);
COMMON_EXPORT bool IsOneOfPosteriorOperator(const std::string &name);
COMMON_EXPORT bool IsOneOfCacheBlackList(const std::string &name);
COMMON_EXPORT bool IsOneOfNotSupportMultiThreadExec(const std::string &name);
COMMON_EXPORT bool IsOneOf3DFormat(const std::string &format);
COMMON_EXPORT bool IsOneOfNoPaddingFormat(const std::string &format);
COMMON_EXPORT bool IsOneOfDynamicShapeConstInputToAttr(const std::string &name);
COMMON_EXPORT bool IsOneOfDynamicShapeConstInputToAttrCPU(const std::string &name);
COMMON_EXPORT bool IsOneOfDynamicShapeConstInputToAttrGPU(const std::string &name);
COMMON_EXPORT bool IsOneOfComputeDepend(const std::string &name);
COMMON_EXPORT bool IsOneOfHWSpecialFormat(const std::string &format);
COMMON_EXPORT bool IsOneOfFormat(const std::string &format);
COMMON_EXPORT bool IsOneOfServerFormatC04(const std::string &format);

// The map between kernel's output and input ref relationship.
// Key is the output index while the value is input index which will be used as the reference of output.
using OutputInputRefMap = std::map<size_t, size_t>;

static inline uint64_t GetCurrentUSec() {
  constexpr int64_t const_num = 1000000;
  auto time_now = std::chrono::system_clock::now();
  auto tv_sec = std::chrono::duration_cast<std::chrono::seconds>(time_now.time_since_epoch()).count();
  auto tv_usec = std::chrono::duration_cast<std::chrono::microseconds>(time_now.time_since_epoch()).count();
  return static_cast<uint64_t>(tv_usec + tv_sec * const_num);
}

#define PROF_START(stage) uint64_t start_usec_##stage = mindspore::GetCurrentUSec()
#define PROF_END(stage)                                                                                     \
  do {                                                                                                      \
    uint64_t end_usec_##stage = mindspore::GetCurrentUSec();                                                \
    MS_LOG(INFO) << "[PROF]" << #stage << " costs " << (end_usec_##stage - start_usec_##stage) << " usec."; \
  } while (0)

#define PROF_MULTI_DEFINE(stage)       \
  do {                                 \
    static uint64_t total_##stage = 0; \
    static uint64_t count_##stage = 0; \
  } while (0)

#define PROF_LOCAL_DEFINE(stage) \
  do {                           \
    uint64_t total_##stage = 0;  \
    uint64_t count_##stage = 0;  \
  } while (0)

#define PROF_MULTI_START(stage) uint64_t start_usec_##stage = mindspore::GetCurrentUSec()

#define PROF_MULTI_END(stage)                                 \
  do {                                                        \
    ++count_##stage;                                          \
    uint64_t end_usec_##stage = mindspore::GetCurrentUSec();  \
    total_##stage += (end_usec_##stage - start_usec_##stage); \
  } while (0)

#define PROF_MULTI_PRINT(stage)                                                                             \
  do {                                                                                                      \
    MS_LOG(INFO) << #stage << " called " << count_##stage << " times, costs " << total_##stage << " usec."; \
  } while (0)
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_UTILS_H_
