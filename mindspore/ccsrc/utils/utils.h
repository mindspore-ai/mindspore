/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_UTILS_UTILS_H_
#define MINDSPORE_MINDSPORE_CCSRC_UTILS_UTILS_H_

#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <string>
#include <vector>
#include <set>

#include "utils/log_adapter.h"

namespace mindspore {
// op name. Op which not exists in operator/ops.h, so define it's name here
constexpr auto kFour2FiveOpName = "Four2Five";
constexpr auto kFive2FourOpName = "Five2Four";
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
constexpr auto kClearZeroOpName = "ClearZero";
constexpr auto kAtomicAddrCleanOpName = "AtomicAddrClean";
constexpr auto kGetNextOpName = "GetNext";
constexpr auto kAllReduceOpName = "AllReduce";
constexpr auto kAllGatherOpName = "AllGather";
constexpr auto kHostAllGatherOpName = "HostAllGather";
constexpr auto kBroadcastOpName = "Broadcast";
constexpr auto kReduceScatterOpName = "ReduceScatter";
constexpr auto kHostReduceScatterOpName = "HostReduceScatter";
constexpr auto kMemCpyAsyncOpName = "memcpy_async";
constexpr auto kTopKOpName = "TopK";
constexpr auto kExtractImagePatchesOpName = "ExtractImagePatches";
constexpr auto kBNTrainingReduceOpName = "BNTrainingReduce";
constexpr auto kBNTrainingUpdateOpName = "BNTrainingUpdate";
constexpr auto kBNTrainingUpdateV2OpName = "BNTrainingUpdateV2";
constexpr auto kSimpleMeanGradOpName = "SimpleMeanGrad";
constexpr auto kMeanGradOpName = "MeanGrad";
constexpr auto kSliceOpName = "Slice";
constexpr auto kSliceGradOpName = "SliceGrad";
constexpr auto kTileOpName = "Tile";
constexpr auto kScatterNdOpName = "ScatterNd";
constexpr auto kStridedSliceAssignOpName = "StridedSliceAssign";
constexpr auto kStridedSliceOpName = "StridedSlice";
constexpr auto kStridedSliceGradOpName = "StridedSliceGrad";
constexpr auto kUnsortedSegmentProdOpName = "UnsortedSegmentProd";
constexpr auto kUnsortedSegmentMinOpName = "UnsortedSegmentMin";
constexpr auto kFlattenGradOpName = "FlattenGrad";
constexpr auto kExpandDimsOpName = "ExpandDims";
constexpr auto kSplitOpName = "Split";
constexpr auto kSparseApplyAdagradOpName = "SparseApplyAdagrad";
constexpr auto kMomentumOpName = "Momentum";
constexpr auto kApplyMomentumOpName = "ApplyMomentum";
constexpr auto kApplyAdadeltaOpName = "ApplyAdadelta";
constexpr auto kApplyAdagradOpName = "ApplyAdagrad";
constexpr auto kApplyAdagradDAName = "ApplyAdagradDA";
constexpr auto kApplyAdamOpName = "Adam";
constexpr auto kApplyAdaMaxOpName = "ApplyAdaMax";
constexpr auto kApplyAddSignOpName = "ApplyAddSign";
constexpr auto kApplyCenteredRMSPOpName = "ApplyCenteredRMSP";
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
constexpr auto kBatchNormGradOpName = "BatchNormGrad";
constexpr auto kBNInferOpName = "BNInfer";
constexpr auto kAdamApplyOneOpName = "AdamApplyOne";
constexpr auto kResizeNearestNeighborGradOpName = "ResizeNearestNeighborGrad";
constexpr auto kFusedMulAddOpName = "FusedMulAdd";
constexpr auto kFusedMulAddNOpName = "FusedMulAddN";
constexpr auto kFusedMulApplyMomentumOpName = "FusedMulApplyMomentum";
constexpr auto kBiasAddOpName = "BiasAdd";
constexpr auto kConfusionMulGradOpName = "ConfusionMulGrad";
constexpr auto kStreamSwitchOpName = "StreamSwitch";
constexpr auto kStreamActiveOpName = "StreamActive";
constexpr auto kAssignAddOpName = "AssignAdd";
constexpr auto kSendOpName = "Send";
constexpr auto kRecvOpName = "Recv";
constexpr auto kReluV2OpName = "ReLUV2";
constexpr auto kReluGradV2OpName = "ReluGradV2";
constexpr auto kAddNOpName = "AddN";
constexpr auto kResizeNearestNeighborV2OpName = "ResizeNearestNeighborV2";
constexpr auto kResizeNearestNeighborV2GradOpName = "ResizeNearestNeighborV2Grad";
constexpr auto kApplyRMSPropOpname = "ApplyRMSProp";
constexpr auto kCumsumOpName = "Cumsum";
constexpr auto kResizeBilinearV2OpName = "kResizeBilinearV2";
constexpr auto kReduceProdOpName = "ReduceProd";
constexpr auto kCumprodOpName = "Cumprod";
constexpr auto kSpaceToBatchOpName = "SpaceToBatch";
constexpr auto kBatchToSpaceOpName = "BatchToSpace";
constexpr auto kPadOpName = "Pad";
constexpr auto kConv2DBackpropInputOpName = "Conv2DBackpropInput";
constexpr auto kFusionOpConv2DBackpropInputReluGradV2Name = "FusionOp_Conv2DBackpropInput_ReluGradV2";
constexpr auto kFusionOpConv2DBackpropInputAddNReluGradV2Name = "FusionOp_Conv2DBackpropInput_AddN_ReluGradV2";
constexpr auto kLabelSetOpName = "LabelSet";
constexpr auto kLabelSwitchOpName = "LabelSwitch";
constexpr auto kLabelGotoOpName = "LabelGoto";
constexpr auto kBNInferGradOpName = "BNInferGrad";
constexpr auto kCallOpName = "call";
constexpr auto kPartialOpName = "partial";
constexpr auto kSwitchOpName = "switch";
constexpr auto kReturnOpName = "return";
constexpr auto kLarsV2OpName = "LarsV2";
constexpr auto kLarsV2UpdateOpName = "LarsV2Update";
constexpr auto kSquareSumAllOpName = "SquareSumAll";
constexpr auto kNMSWithMaskOpName = "NMSWithMask";

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
constexpr auto kAttrAxis = "axis";
constexpr auto kAttrKeepDims = "keep_dims";
constexpr auto kAttrShapeGamma = "shape_gamma";
constexpr auto kAttrPerm = "perm";
constexpr auto kAttrTransposeFirst = "transpose_first";
constexpr auto kAttrAutomicAddMemSize = "automic_add_mem_size";
constexpr auto kAttrAutomicOutputIndexs = "atomic_output_clean_indexs";
constexpr auto kAttrAutomicWorkspaceSize = "atomic_workspace_clean_size";
constexpr auto kAttrSwitchCondition = "switch_condition";
constexpr auto kAttrDataType = "data_type";
constexpr auto kAttrActiveTarget = "active_target";
constexpr auto kAttrActiveStreamList = "active_stream_list";
constexpr auto kAttrTrueBranchStream = "true_branch_stream";
constexpr auto kAttrEventId = "event_id";
constexpr auto kAttrDynInput = "dynamic";
constexpr auto kAttrDynInputSizes = "dyn_input_sizes";
constexpr auto kAttrSrcFormat = "src_format";
constexpr auto kAttrOutputUsedNum = "output_used_num";
constexpr auto kAttrHasBias = "has_bias";
constexpr auto kAttrN = "n";
constexpr auto kAttrLabelForInsertStreamActive = "label_for_insert_stream_active";
constexpr auto kAttrFusion = "fusion";
constexpr auto kAttrGroup = "group";
constexpr auto kAttrOp = "op";
constexpr auto kAttrIsTraining = "is_training";
constexpr auto kAttrFusionId = "fusion_id";
constexpr auto kAttrLabelIndex = "label_index";
constexpr auto kAttrLabelSwitchList = "label_switch_list";
constexpr auto kAttrNewAxisMask = "new_axis_mask";
constexpr auto kAttrShrinkAxisMask = "shrink_axis_mask";

// attr value
constexpr auto kValueTargetSwitch = "target_switch";
constexpr auto kValueTargetOther = "target_other";

// some size
const size_t kShape4dDims = 4;
const size_t kShape5dDims = 5;
const size_t kCubeSize = 16;
const size_t kMemAlignSize = 512;
const int kParameterDataTensorMask = 0;
const int kParameterWeightTensorMask = 1;
const int kValueNodeTensorMask = 2;

// define special index in special node
constexpr auto kAnfPrimitiveIndex = 0;
constexpr auto kAnfPartialFuncGraphIndex = 1;
constexpr auto kRealInputNodeIndexInTupleGetItem = 1;
constexpr auto kInputNodeOutputIndexInTupleGetItem = 2;
constexpr auto kTupleGetItemInputSize = 3;
// index define of control depend
constexpr auto kControlDependPriorIndex = 1;
constexpr auto kControlDependBehindIndex = 2;
// index define of depend
constexpr auto kRealInputIndexInDepend = 1;
constexpr auto kDependAttachNodeIndex = 2;
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
const std::set<std::string> kOpFormatList = {
  kOpFormat_DEFAULT,     kOpFormat_NC1KHKWHWC0,   kOpFormat_ND,     kOpFormat_NCHW,      kOpFormat_NHWC,
  kOpFormat_HWCN,        kOpFormat_NC1HWC0,       kOpFormat_FRAC_Z, kOpFormat_C1HWNCoC0, kOpFormat_FRAC_NZ,
  kOpFormat_NC1HWC0_C04, kOpFormat_FRACTAL_Z_C04, kOpFormat_NDHWC};
const std::set<std::string> kDefaultCompatibleFormat = {kOpFormat_ND, kOpFormat_NCHW, kOpFormat_NHWC, kOpFormat_HWCN};
const std::set<std::string> kOptOperatorSet = {
  kMomentumOpName,       kApplyMomentumOpName,        kApplyAdadeltaOpName,
  kApplyAdagradOpName,   kApplyAdagradDAName,         kApplyAdamOpName,
  kApplyAdaMaxOpName,    kApplyAddSignOpName,         kApplyCenteredRMSPOpName,
  kApplyFtrlOpName,      kApplyFtrlV2OpName,          kApplyGradientDescentOpName,
  kApplyPowerSignOpName, kApplyProximalAdagradOpName, kApplyProximalGradientDescentOpName,
  kApplyRMSPropOpName,
};

const std::set<std::string> kNeedTransFormatSet = {kOpFormat_FRAC_Z,       kOpFormat_NC1KHKWHWC0, kOpFormat_NC1HWC0,
                                                   kOpFormat_FRAC_NZ,      kOpFormat_C1HWNCoC0,   kOpFormat_NC1HWC0_C04,
                                                   kOpFormat_FRACTAL_Z_C04};

static inline void ChangeFileMode(const std::string &file_name, mode_t mode) {
  if (access(file_name.c_str(), F_OK) != 0) {
    MS_LOG(DEBUG) << "File `" << file_name << "` does not exist.";
    return;
  }
  if (chmod(file_name.c_str(), mode) != 0) {
    MS_LOG(WARNING) << "Change file `" << file_name << "` to mode " << std::oct << mode << " fail.";
  }
}
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_UTILS_UTILS_H_
