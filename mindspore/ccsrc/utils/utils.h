/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_UTILS_UTILS_H_
#define MINDSPORE_CCSRC_UTILS_UTILS_H_

#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <string>
#include <vector>
#include <set>

#include "utils/log_adapter.h"
#include "ir/dtype/type.h"

namespace mindspore {
// op name. Op which not exists in operator/ops.h, so define it's name here
constexpr auto kConcatOpName = "Concat";
constexpr auto kUniqueOpName = "Unique";
constexpr auto kComputeAccidentalHitsOpName = "ComputeAccidentalHits";
constexpr auto kCTCGreedyDecoderOpName = "CTCGreedyDecoder";
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
constexpr auto kGetNextOpName = "GetNext";
constexpr auto kInitDatasetQueueOpName = "InitDataSetQueue";
constexpr auto kEndOfSequence = "EndOfSequence";
constexpr auto kAllReduceOpName = "AllReduce";
constexpr auto kAllGatherOpName = "AllGather";
constexpr auto kHostAllGatherOpName = "HostAllGather";
constexpr auto kBroadcastOpName = "Broadcast";
constexpr auto kReceiveOpName = "Receive";
constexpr auto kHcomSendOpName = "Send";
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
constexpr auto kSimpleMeanGradOpName = "SimpleMeanGrad";
constexpr auto kMeanGradOpName = "MeanGrad";
constexpr auto kSliceOpName = "Slice";
constexpr auto kSliceGradOpName = "SliceGrad";
constexpr auto kTileOpName = "Tile";
constexpr auto kScatterNdOpName = "ScatterNd";
constexpr auto kStridedSliceAssignOpName = "StridedSliceAssign";
constexpr auto kStridedSliceOpName = "StridedSlice";
constexpr auto kStridedSliceGradOpName = "StridedSliceGrad";
constexpr auto kSparseGatherV2OpName = "SparseGatherV2";
constexpr auto kUnsortedSegmentProdOpName = "UnsortedSegmentProd";
constexpr auto kUnsortedSegmentMinOpName = "UnsortedSegmentMin";
constexpr auto kFlattenGradOpName = "FlattenGrad";
constexpr auto kExpandDimsOpName = "ExpandDims";
constexpr auto kReshapeOpName = "Reshape";
constexpr auto kTransposeOpName = "Transpose";
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
constexpr auto kBNTrainingUpdateGradOpName = "BNTrainingUpdateGrad";
constexpr auto kBNTrainingReduceGradOpName = "BNTrainingReduceGrad";
constexpr auto kSquareSumV1OpName = "SquareSumV1";
constexpr auto kSquareSumV2OpName = "SquareSumV2";
constexpr auto kClipByNormNoDivSumOpName = "ClipByNormNoDivSum";
constexpr auto kGreaterOpName = "Greater";
constexpr auto kSqrtOpName = "Sqrt";
constexpr auto kRsqrtOpName = "Rsqrt";
constexpr auto kErfOpName = "Erf";
constexpr auto kRealDivOpName = "RealDiv";
constexpr auto kLambUpdateWithLROpName = "LambUpdateWithLR";
constexpr auto kLambNextMVWithDecayOpName = "LambNextMVWithDecay";
constexpr auto kLambNextMVWithDecayV1OpName = "LambNextMVWithDecayV1";
constexpr auto kClipByValueOpName = "ClipByValue";
constexpr auto kLambNextRightOpName = "LambNextRight";
constexpr auto kConfusionSoftmaxGradOpName = "ConfusionSoftmaxGrad";
constexpr auto kLambUpdateWithLrV2OpName = "LambUpdateWithLrV2";
constexpr auto kLayerNormXBackpropOpName = "LayerNormXBackprop";
constexpr auto kLayerNormBetaGammaBackpropOpName = "LayerNormBetaGammaBackprop";
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
constexpr auto kAssignAddOpName = "AssignAdd";
constexpr auto kSendOpName = "StreamSend";
constexpr auto kRecvOpName = "StreamRecv";
constexpr auto kReluV2OpName = "ReLUV2";
constexpr auto kReluGradV2OpName = "ReluGradV2";
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
constexpr auto kLarsV2OpName = "LarsV2";
constexpr auto kLarsV2UpdateOpName = "LarsV2Update";
constexpr auto kSquareSumAllOpName = "SquareSumAll";
constexpr auto kNMSWithMaskOpName = "NMSWithMask";
constexpr auto kSoftmaxGradExtOpName = "SoftmaxGradExt";
constexpr auto kStridedReadOpName = "StridedRead";
constexpr auto kStridedWriteOpName = "StridedWrite";
constexpr auto kFusedAdamWeightDecayName = "FusedAdamWeightDecay";
constexpr auto kFusedAdamName = "FusedAdam";
constexpr auto kFusedSparseAdamName = "FusedSparseAdam";
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
constexpr auto kTensorScatterUpdateOpName = "TensorScatterUpdate";
constexpr auto kScatterNdUpdateOpName = "ScatterNdUpdate";
constexpr auto kPushOpName = "Push";
constexpr auto kPullOpName = "Pull";
constexpr auto kUpdateCacheOpName = "UpdateCache";
constexpr auto kCacheSwapTableOpName = "CacheSwapTable";
constexpr auto kEmbeddingLookupOpName = "EmbeddingLookup";
constexpr auto kEmbeddingLookupProxyOpName = "EmbeddingLookupProxy";
constexpr auto kGatherV2OpName = "Gather";
constexpr auto kPaddingOpName = "Padding";
constexpr auto kAvgPoolOpName = "AvgPool";
constexpr auto kAvgPoolGradOpName = "AvgPoolGrad";
constexpr auto kAvgPoolGradVmOpName = "AvgPoolGradVm";
constexpr auto kmaxPoolGradOpName = "MaxPoolGrad";
constexpr auto kMaxPoolWithArgmaxOpName = "MaxPoolWithArgmax";
constexpr auto kMaxPoolGradWithArgmaxOpName = "MaxPoolGradWithArgmax";
constexpr auto kTensorAddOpName = "Add";
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
constexpr auto kReduceSumOpName = "ReduceSum";
constexpr auto kReduceMinOpName = "ReduceMin";
constexpr auto kReduceMaxOpName = "ReduceMax";
constexpr auto kReduceMeanOpName = "ReduceMean";
constexpr auto kReduceAnyOpName = "ReduceAny";
constexpr auto kReduceAllOpName = "ReduceAll";
constexpr auto kFusedWeightScaleApplyMomentum = "FusedWeightScaleApplyMomentum";
constexpr auto kFusedWeightApplyMomentum = "FusedWeightApplyMomentum";
constexpr auto kFusedScaleApplyMomentum = "FusedScaleApplyMomentum";
constexpr auto kBasicLSTMCellWeightGradOpName = "BasicLSTMCellWeightGrad";
constexpr auto kBasicLSTMCellInputGradOpName = "BasicLSTMCellInputGrad";
constexpr auto kBasicLSTMCellOpName = "BasicLSTMCell";
constexpr auto kDynamicRNNOpName = "DynamicRNN";
constexpr auto kLSTMInputGradOpName = "LSTMInputGrad";
constexpr auto kDynamicGRUV2OpName = "DynamicGRUV2";
constexpr auto kGRUV2HiddenGradOpName = "GRUV2HiddenGrad";
constexpr auto kFusedSparseFtrlName = "FusedSparseFtrl";
constexpr auto kFusedSparseProximalAdagradName = "FusedSparseProximalAdagrad";
constexpr auto kFusedSparseLazyAdamName = "FusedSparseLazyAdam";
constexpr auto kSparseApplyFtrlName = "SparseApplyFtrl";
constexpr auto kSparseApplyFtrlV2Name = "SparseApplyFtrlV2";
constexpr auto kSGDName = "SGD";
constexpr auto kLARSUpdateName = "LARSUpdate";
constexpr auto kBasicLSTMCellCStateGradOpName = "BasicLSTMCellCStateGrad";
constexpr auto kBasicLSTMCellCStateGradV2OpName = "BasicLSTMCellCStateGradV2";
constexpr auto kMatMulV2OpName = "MatMulV2";
constexpr auto kMatMulOpName = "MatMul";
constexpr auto kBatchMatMulOpName = "BatchMatMul";
constexpr auto kBroadcastToOpName = "BroadcastTo";
constexpr auto kFusedAddReluV2Name = "FusedAddReluV2";
constexpr auto kFusedAddReluGradV2Name = "FusedAddReluGradV2";
constexpr auto kDropoutOpName = "Dropout";
constexpr auto kDropoutGradOpName = "DropoutGrad";
constexpr auto kDropoutGenMaskOpName = "DropoutGenMask";
constexpr auto kDropoutDoMaskOpName = "DropoutDoMask";
constexpr auto kSubAndFilterOpName = "SubAndFilter";
constexpr auto kPadAndShiftOpName = "PadAndShift";
constexpr auto kSparseSoftmaxCrossEntropyWithLogitsOpName = "SparseSoftmaxCrossEntropyWithLogits";
constexpr auto kOneHotOpName = "OneHot";
constexpr auto kSoftmaxCrossEntropyWithLogitsOpName = "SoftmaxCrossEntropyWithLogits";
constexpr auto kUniformCandidateSamplerOpName = "UniformCandidateSampler";

// Communication world group
constexpr auto kNcclWorldGroup = "nccl_world_group";
constexpr auto kHcclWorldGroup = "hccl_world_group";
constexpr auto kSyncBnGroup = "sync_bn_group";

// Hcom Op Type
constexpr auto kHcomOpTypeAllReduce = "HcomAllReduce";
constexpr auto kHcomOpTypeAllGather = "HcomAllGather";
constexpr auto kHcomOpTypeBroadcast = "HcomBroadcast";
constexpr auto kHcomOpTypeSend = "HcomSend";
constexpr auto kHcomOpTypeReceive = "HcomReceive";
constexpr auto kHcomOpTypeReduceScatter = "HcomReduceScatter";

// attr key name
constexpr auto kAttrInputNames = "input_names";
constexpr auto kAttrIsAICPUKernel = "is_AICPU_kernel";
constexpr auto kIsBackendCast = "is_backed_cast";
constexpr auto kAttrOutputNames = "output_names";
constexpr auto kAttrVisited = "visited";
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
constexpr auto kAttrKeepDims = "keep_dims";
constexpr auto kAttrShapeGamma = "shape_gamma";
constexpr auto kAttrPerm = "perm";
constexpr auto kAttrTransposeFirst = "transpose_first";
constexpr auto kAttrAtomicAddMemSize = "automic_add_mem_size";
constexpr auto kAttrAtomicOutputIndexs = "atomic_output_clean_indexs";
constexpr auto kAttrAtomicWorkspaceIndexs = "atomic_workspace_clean_indexs";
constexpr auto kAttrSwitchCondition = "switch_condition";
constexpr auto kAttrDataType = "data_type";
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
constexpr auto kAttrGroup = "group";
constexpr auto kAttrOp = "op";
constexpr auto kAttrDestRank = "dest_rank";
constexpr auto kAttrSrcRank = "src_rank";
constexpr auto kAttrSrTag = "sr_tag";
constexpr auto kAttrRootRank = "root_rank";
constexpr auto kAttrIsTraining = "is_training";
constexpr auto kAttrFusionId = "fusion_id";
constexpr auto kAttrBucketId = "bucket_id";
constexpr auto kAttrGradOutputIndex = "grad_output_index";
constexpr auto kAttrLabelIndex = "label_index";
constexpr auto kAttrLabelSwitchList = "label_switch_list";
constexpr auto kAttrNewAxisMask = "new_axis_mask";
constexpr auto kAttrShrinkAxisMask = "shrink_axis_mask";
constexpr auto kAttrDatadumpOriginalNames = "_datadump_original_names";
constexpr auto kAttrDatadumpIsMultiop = "_datadump_is_multiop";
constexpr auto kAttrNeedRecordEvent = "need_record_event";
constexpr auto kAttrStreamId = "stream_id";
constexpr auto kAttrRecordEvent = "record_event";
constexpr auto kAttrWaitEvent = "wait_event";
constexpr auto kAttrRecordEventStream = "record_event_stream";
constexpr auto kAttrWaitEventStream = "wait_event_stream";
constexpr auto kAttrIndex = "index";
constexpr auto kAttrSplitDim = "split_dim";
constexpr auto kAttrNumSplit = "num_split";
constexpr auto kAttrReduction = "reduction";
constexpr auto kAttrOutputNum = "output_num";
constexpr auto kAttrSizeSplits = "size_splits";
constexpr auto kAttrOutputDefault = "output_default";
constexpr auto kAttrPrimitiveTarget = "primitive_target";
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
constexpr auto kAttrRankSize = "rank_size";
constexpr auto kAttrPadDimSize = "pad_dim_size";
constexpr auto kAttrPaddings = "paddings";
constexpr auto kAttrNumSegments = "num_segments";
constexpr auto kAttrBegin = "begin";
constexpr auto kAttrSize = "size";
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
constexpr auto kAttrPad = "pad";
constexpr auto kAttrPadding = "padding";
constexpr auto kAttrNonTask = "non_task";
constexpr auto kAttrIsGrad = "is_grad";
constexpr auto kAttrRecompute = "recompute";
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

// attr value
constexpr auto kValueTargetSwitch = "target_switch";
constexpr auto kValueTargetOther = "target_other";

// some size
const size_t kShape4dDims = 4;
const size_t kShape2dDims = 2;
const size_t kShape5dDims = 5;
const size_t kShape1dDims = 1;
const size_t kCubeSize = 16;
const size_t kMemAlignSize = 512;
const size_t kBNChannelMultipleFactor = 4;
const int kParameterDataTensorMask = 0;
const int kParameterWeightTensorMask = 1;
const int kValueNodeTensorMask = 2;

// define special index in special node
constexpr auto kAnfPrimitiveIndex = 0;
constexpr auto kFirstDataInputIndex = 1;
constexpr auto kRealInputNodeIndexInTupleGetItem = 1;
constexpr auto kInputNodeOutputIndexInTupleGetItem = 2;
constexpr auto kTupleGetItemInputSize = 3;
constexpr auto kSwitchInputSize = 4;
constexpr auto kFirstBranchInSwitch = 2;
constexpr auto kCallKernelGraphIndex = 1;
constexpr auto kSwitchTrueKernelGraphIndex = 2;
constexpr auto kSwitchFalseKernelGraphIndex = 3;
constexpr auto kMakeTupleInSwitchLayerIndex = 2;
// index define of control depend
constexpr auto kControlDependPriorIndex = 1;
constexpr auto kControlDependBehindIndex = 2;
constexpr auto kControlDependMode = "depend_mode";
// index define of depend
constexpr auto kRealInputIndexInDepend = 1;
constexpr auto kDependAttachNodeIndex = 2;
constexpr auto kDependInputSize = 3;
// index define of UpdateState
constexpr auto kUpdateStateStateInput = 1;
constexpr auto kUpdateStateRealInput = 2;
// format
constexpr auto kOpFormat_DEFAULT = "DefaultFormat";
constexpr auto kOpFormat_NC1KHKWHWC0 = "NC1KHKWHWC0";
constexpr auto kOpFormat_ND = "ND";
constexpr auto kOpFormat_NCHW = "NCHW";
constexpr auto kOpFormat_NHWC = "NHWC";
constexpr auto kOpFormat_HWCN = "HWCN";
constexpr auto kOpFormat_NC1HWC0 = "NC1HWC0";
constexpr auto kOpFormat_FRAC_Z = "FracZ";
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

const std::set<std::string> kOpFormatList = {kOpFormat_DEFAULT,      kOpFormat_NC1KHKWHWC0,
                                             kOpFormat_ND,           kOpFormat_NCHW,
                                             kOpFormat_NHWC,         kOpFormat_HWCN,
                                             kOpFormat_NC1HWC0,      kOpFormat_FRAC_Z,
                                             kOpFormat_C1HWNCoC0,    kOpFormat_FRAC_NZ,
                                             kOpFormat_NC1HWC0_C04,  kOpFormat_FRACTAL_Z_C04,
                                             kOpFormat_NDHWC,        kOpFormat_FRACTAL_ZN_LSTM,
                                             kOpFormat_NDC1HWC0,     kOpFormat_NCDHW,
                                             kOpFormat_FRACTAL_Z_3D, kOpFormat_DHWNC,
                                             kOpFormat_DHWCN};
const std::set<std::string> kDefaultCompatibleFormat = {kOpFormat_ND, kOpFormat_NCHW, kOpFormat_NHWC, kOpFormat_HWCN,
                                                        kOpFormat_NCDHW};
const std::set<std::string> kOptOperatorSet = {kMomentumOpName,
                                               kApplyMomentumOpName,
                                               kApplyAdadeltaOpName,
                                               kApplyAdagradOpName,
                                               kApplyAdagradDAName,
                                               kApplyAdamOpName,
                                               kApplyAdaMaxOpName,
                                               kApplyAddSignOpName,
                                               kApplyCenteredRMSPOpName,
                                               kApplyFtrlOpName,
                                               kApplyFtrlV2OpName,
                                               kApplyGradientDescentOpName,
                                               kApplyPowerSignOpName,
                                               kApplyProximalAdagradOpName,
                                               kApplyProximalGradientDescentOpName,
                                               kApplyRMSPropOpName,
                                               kFusedAdamWeightDecayName,
                                               kFusedAdamName,
                                               kFusedSparseAdamName,
                                               kFusedWeightScaleApplyMomentum,
                                               kFusedScaleApplyMomentum,
                                               kApplyCenteredRMSPropOpName,
                                               kFusedSparseFtrlName,
                                               kFusedSparseProximalAdagradName,
                                               kFusedSparseLazyAdamName,
                                               kSparseApplyFtrlName,
                                               kSparseApplyFtrlV2Name,
                                               kSGDName,
                                               kLARSUpdateName,
                                               kCombineMomentumWeightOpName,
                                               kCombineMomentumOpName,
                                               kSparseApplyProximalAdagradOpName};

const std::set<std::string> kPosteriorOperatorSet = {kPullOpName};

const std::set<std::string> kOpCacheAllowList = {kUniformCandidateSamplerOpName, kInitDatasetQueueOpName,
                                                 kGetNextOpName};

const std::set<std::string> kHWSpecialFormatSet = {
  kOpFormat_FRACTAL_Z_3D, kOpFormat_NC1KHKWHWC0,   kOpFormat_NC1HWC0,         kOpFormat_FRAC_NZ,  kOpFormat_C1HWNCoC0,
  kOpFormat_NC1HWC0_C04,  kOpFormat_FRACTAL_Z_C04, kOpFormat_FRACTAL_ZN_LSTM, kOpFormat_NDC1HWC0, kOpFormat_FRAC_Z};

const std::set<TypeId> kFloatDataTypeSet = {kNumberTypeFloat16, kNumberTypeFloat32};

const std::set<std::string> kComputeDepend = {kUniqueOpName, kComputeAccidentalHitsOpName, kSubAndFilterOpName,
                                              kPadAndShiftOpName, kCTCGreedyDecoderOpName};

const std::set<std::string> k3DFormatSet = {kOpFormat_NCDHW, kOpFormat_NDC1HWC0, kOpFormat_FRACTAL_Z_3D,
                                            kOpFormat_NDHWC, kOpFormat_DHWCN,    kOpFormat_DHWNC};

const std::set<std::string> DynamicShapeConstInputToAttr = {
  kCastOpName,      kExpandDimsOpName, kReshapeOpName,   kEmbeddingLookupOpName, kTransposeOpName, kReduceSumOpName,
  kReduceMinOpName, kReduceMeanOpName, kReduceMaxOpName, kReduceAllOpName,       kReduceAnyOpName, kConcatOpName};

static inline void ChangeFileMode(const std::string &file_name, mode_t mode) {
  try {
    if (chmod(file_name.c_str(), mode) != 0) {
      MS_LOG(DEBUG) << "Change file `" << file_name << "` to mode " << std::oct << mode << " fail.";
    }
  } catch (std::exception &e) {
    MS_LOG(DEBUG) << "File `" << file_name << "` change mode failed! May be not exist.";
  }
}

static inline uint64_t GetCurrentUSec() {
  struct timeval tv;
  int ret = gettimeofday(&tv, nullptr);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "Fail gettimeofday, ret = " << ret;
  }
  return static_cast<uint64_t>(tv.tv_usec + tv.tv_sec * 1000000);
}

#define PROF_START(stage) uint64_t start_usec_##stage = mindspore::GetCurrentUSec()
#define PROF_END(stage)                                                                         \
  do {                                                                                          \
    uint64_t end_usec_##stage = mindspore::GetCurrentUSec();                                    \
    MS_LOG(INFO) << #stage << " costs " << (end_usec_##stage - start_usec_##stage) << " usec."; \
  } while (0)

#define PROF_MULTI_DEFINE(stage)     \
  static uint64_t total_##stage = 0; \
  static uint64_t count_##stage = 0;

#define PROF_LOCAL_DEFINE(stage) \
  uint64_t total_##stage = 0;    \
  uint64_t count_##stage = 0;

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
#endif  // MINDSPORE_CCSRC_UTILS_UTILS_H_
