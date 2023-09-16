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
#ifndef MINDSPORE_CORE_BASE_ASCEND_OP_NAME_H_
#define MINDSPORE_CORE_BASE_ASCEND_OP_NAME_H_

namespace mindspore {
constexpr auto kApplyCenteredRMSPOpName = "ApplyCenteredRMSP";
constexpr auto kApplyFtrlV2OpName = "ApplyFtrlV2";
constexpr auto kApplyFtrlV2DOpName = "ApplyFtrlV2D";
constexpr auto kApplyRMSPropDOpName = "ApplyRMSPropD";
constexpr auto kArgMaxDOpName = "ArgMaxD";
constexpr auto kArgMaxV2OpName = "ArgMaxV2";
constexpr auto kAtomicAddrCleanOpName = "AtomicAddrClean";
constexpr auto kBN2OpName = "BN2";
constexpr auto kBN2AddReluOpName = "BN2AddRelu";
constexpr auto kBN2ReluOpName = "BN2Relu";
constexpr auto kBNGrad1OpName = "BNGrad1";
constexpr auto kBNGrad2OpName = "BNGrad2";
constexpr auto kBNGrad3OpName = "BNGrad3";
constexpr auto kBNInferenceOpName = "BNInference";
constexpr auto kBNInferenceDOpName = "BNInferenceD";
constexpr auto kBNTrainingUpdateV2OpName = "BNTrainingUpdateV2";
constexpr auto kBNTrainingUpdateV3OpName = "BNTrainingUpdateV3";
constexpr auto kBasicLSTMCellOpName = "BasicLSTMCell";
constexpr auto kBasicLSTMCellCStateGradOpName = "BasicLSTMCellCStateGrad";
constexpr auto kBasicLSTMCellCStateGradV2OpName = "BasicLSTMCellCStateGradV2";
constexpr auto kBasicLSTMCellInputGradOpName = "BasicLSTMCellInputGrad";
constexpr auto kBasicLSTMCellWeightGradOpName = "BasicLSTMCellWeightGrad";
constexpr auto kClearZeroOpName = "ClearZero";
constexpr auto kConfusionMulGradOpName = "ConfusionMulGrad";
constexpr auto kConfusionSoftmaxGradOpName = "ConfusionSoftmaxGrad";
constexpr auto kConfusionTransposeDOpName = "ConfusionTransposeD";
constexpr auto kConv2DTransposeDOpName = "Conv2DTransposeD";
constexpr auto kConvBN1OpName = "ConvBN1";
constexpr auto kCropAndResizeDOpName = "CropAndResizeD";
constexpr auto kCumprodOpName = "Cumprod";
constexpr auto kCumprodDOpName = "CumprodD";
constexpr auto kCumulativeLogsumexpDOpName = "CumulativeLogsumexpD";
constexpr auto kDeadNodeOpName = "DeadNode";
constexpr auto kDepthwiseConv2DBackpropFilterOpName = "DepthwiseConv2DBackpropFilter";
constexpr auto kDepthwiseConv2DBackpropFilterDOpName = "DepthwiseConv2DBackpropFilterD";
constexpr auto kDepthwiseConv2DBackpropInputOpName = "DepthwiseConv2DBackpropInput";
constexpr auto kDepthwiseConv2DBackpropInputDOpName = "DepthwiseConv2DBackpropInputD";
constexpr auto kDropOutGenMaskV4OpName = "DropOutGenMaskV4";
constexpr auto kDynamicAtomicAddrCleanOpName = "DynamicAtomicAddrClean";
constexpr auto kEmbeddingLookupProxyOpName = "EmbeddingLookupProxy";
constexpr auto kFillV2DOpName = "FillV2D";
constexpr auto kFive2FourOpName = "Five2Four";
constexpr auto kFour2FiveOpName = "Four2Five";
constexpr auto kFusedBN1OpName = "FusedBN1";
constexpr auto kFusedBN2OpName = "FusedBN2";
constexpr auto kFusedBN3OpName = "FusedBN3";
constexpr auto kFusedCastAdamWeightDecayOpName = "FusedCastAdamWeightDecay";
constexpr auto kFusedDbnDwOpName = "FusedDbnDw";
constexpr auto kFusedMulAddNOpName = "FusedMulAddN";
constexpr auto kFusionOp_Conv2DBackpropInput_AddN_ReluGradV2OpName = "FusionOp_Conv2DBackpropInput_AddN_ReluGradV2";
constexpr auto kFusionOp_Conv2DBackpropInput_ReluGradV2OpName = "FusionOp_Conv2DBackpropInput_ReluGradV2";
constexpr auto kGRUV2HiddenGradOpName = "GRUV2HiddenGrad";
constexpr auto kGRUV2HiddenGradCellOpName = "GRUV2HiddenGradCell";
constexpr auto kHardShrinkOpName = "HardShrink";
constexpr auto kHardShrinkGradOpName = "HardShrinkGrad";
constexpr auto kHardSigmoidOpName = "HardSigmoid";
constexpr auto kHardSigmoidGradOpName = "HardSigmoidGrad";
constexpr auto kHostAllGatherOpName = "HostAllGather";
constexpr auto kHostReduceScatterOpName = "HostReduceScatter";
constexpr auto kIm2colOpName = "Im2col";
constexpr auto kKLDivOpName = "KLDiv";
constexpr auto kKlDivLossGradOpName = "KlDivLossGrad";
constexpr auto kLSTMInputGradOpName = "LSTMInputGrad";
constexpr auto kLambNextMVOpName = "LambNextMV";
constexpr auto kLambNextMVWithDecayOpName = "LambNextMVWithDecay";
constexpr auto kLambNextMVWithDecayV1OpName = "LambNextMVWithDecayV1";
constexpr auto kLambNextRightOpName = "LambNextRight";
constexpr auto kLambUpdateWithLROpName = "LambUpdateWithLR";
constexpr auto kLambUpdateWithLrV2OpName = "LambUpdateWithLrV2";
constexpr auto kLarsV2OpName = "LarsV2";
constexpr auto kMatMulBiasAddFusionOpName = "MatMulBiasAddFusion";
constexpr auto kMatMulBiasAddReluFusionOpName = "MatMulBiasAddReluFusion";
constexpr auto kMatrixDiagPartOpName = "MatrixDiagPart";
constexpr auto kMatrixDiagPartDOpName = "MatrixDiagPartD";
constexpr auto kMatrixSetDiagOpName = "MatrixSetDiag";
constexpr auto kMatrixSetDiagDOpName = "MatrixSetDiagD";
constexpr auto kMaxPoolExt2OpName = "MaxPoolExt2";
constexpr auto kMaxPoolV2OpName = "MaxPoolV2";
constexpr auto kMeanGradOpName = "MeanGrad";
constexpr auto kMemSetOpName = "MemSet";
constexpr auto kMuxReceiveOpName = "MuxReceive";
constexpr auto kMuxSendOpName = "MuxSend";
constexpr auto kNewIm2ColOpName = "NewIm2Col";
constexpr auto kPReluGradOpName = "PReluGrad";
constexpr auto kPoissonOpName = "Poisson";
constexpr auto kPriorityReplayBufferCreateOpName = "PriorityReplayBufferCreate";
constexpr auto kPriorityReplayBufferDestroyOpName = "PriorityReplayBufferDestroy";
constexpr auto kPriorityReplayBufferPushOpName = "PriorityReplayBufferPush";
constexpr auto kPriorityReplayBufferSampleOpName = "PriorityReplayBufferSample";
constexpr auto kPriorityReplayBufferUpdateOpName = "PriorityReplayBufferUpdate";
constexpr auto kPullWeightOpName = "PullWeight";
constexpr auto kPushWeightOpName = "PushWeight";
constexpr auto kRelu6OpName = "Relu6";
constexpr auto kResizeBilinearV2DOpName = "ResizeBilinearV2D";
constexpr auto kRpnProposalsOpName = "RpnProposals";
constexpr auto kRpnProposalsDOpName = "RpnProposalsD";
constexpr auto kShrinkOpName = "Shrink";
constexpr auto kSimpleMeanGradOpName = "SimpleMeanGrad";
constexpr auto kSliceDV2OpName = "SliceDV2";
constexpr auto kSoftmaxGradFusionOpName = "SoftmaxGradFusion";
constexpr auto kSoftmaxV2WithDropOutDoMaskV3DOpName = "SoftmaxV2WithDropOutDoMaskV3D";
constexpr auto kSparseApplyAdadeltaDOpName = "SparseApplyAdadeltaD";
constexpr auto kSparseApplyFtrlV2OpName = "SparseApplyFtrlV2";
constexpr auto kSquareSumV2OpName = "SquareSumV2";
constexpr auto kStreamActiveOpName = "StreamActive";
constexpr auto kStreamSwitchOpName = "StreamSwitch";
constexpr auto kStridedSliceAssignOpName = "StridedSliceAssign";
constexpr auto kStridedSliceAssignDOpName = "StridedSliceAssignD";
constexpr auto kStridedSliceDOpName = "StridedSliceD";
constexpr auto kSyncResizeBilinearV2OpName = "SyncResizeBilinearV2";
constexpr auto kSyncResizeBilinearV2GradOpName = "SyncResizeBilinearV2Grad";
constexpr auto kTransShapeOpName = "TransShape";
constexpr auto kUnsortedSegmentProdDOpName = "UnsortedSegmentProdD";
constexpr auto kUnstackWithNumOpName = "UnstackWithNum";
constexpr auto kCombineOptimizerOpName = "combine_optimizer";
constexpr auto kClipBoxesOpName = "kClipBoxes";
constexpr auto kClipBoxesDOpName = "kClipBoxesD";
constexpr auto kPartialOpName = "partial";
constexpr auto kRandomCache = "random_cache";

constexpr auto kNcclWorldGroup = "nccl_world_group";
constexpr auto kHcclWorldGroup = "hccl_world_group";
constexpr auto kSyncBnGroup = "sync_bn_group";
constexpr auto kRankID = "RANK_ID";

constexpr auto kHcomOpTypeAllToAllV = "HcomAllToAllV";
constexpr auto kHcomOpTypeAllReduce = "HcomAllReduce";
constexpr auto kHcomOpTypeReduce = "HcomReduce";
constexpr auto kHcomOpTypeAllGather = "HcomAllGather";
constexpr auto kHcomOpTypeBroadcast = "HcomBroadcast";
constexpr auto kHcomOpTypeSend = "HcomSend";
constexpr auto kHcomOpTypeReceive = "HcomReceive";
constexpr auto kHcomOpTypeReduceScatter = "HcomReduceScatter";
constexpr auto kHcomOpTypeBarrier = "HcomBarrier";

constexpr auto kEndGraph = "EndGraph";
constexpr auto kEndOfSequence = "EndOfSequence";
}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_ASCEND_OP_NAME_H_
