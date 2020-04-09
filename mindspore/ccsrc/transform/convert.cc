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

#include "transform/convert.h"

#include <inttypes.h>
#include <algorithm>
#include <stack>
#include "utils/utils.h"

#include "operator/ops.h"
#include "utils/log_adapter.h"
#include "utils/graph_utils.h"
#include "utils/symbolic.h"
#include "utils/config_manager.h"
#include "utils/convert_utils.h"
#include "./common.h"

namespace mindspore {
namespace transform {
using std::endl;

#define ADPT_DESC_ONE(T) std::make_shared<OpAdapterDesc>(std::make_shared<OpAdapter<T>>())
#define ADPT_DESC_TWO(T, I) \
  std::make_shared<OpAdapterDesc>(std::make_shared<OpAdapter<T>>(), std::make_shared<OpAdapter<I>>())
#define GET_MACRO(_1, _2, DESC, ...) DESC
#define ADPT_DESC(...) GET_MACRO(__VA_ARGS__, ADPT_DESC_TWO, ADPT_DESC_ONE, ...)(__VA_ARGS__)

using ge::Operator;
using mindspore::kAnyValue;
using std::make_shared;
using std::shared_ptr;
using std::string;
using std::vector;

const char kNameCustomOp[] = "CustomOp";
const char kNameConst[] = "Const";
const char kNameParam[] = "parameter";
const char kNameRandomUniform[] = "RandomUniform";
const char kNameSimpleMean[] = "SimpleMean";
const char kNameSimpleMeanGrad[] = "SimpleMeanGrad";
const char kNameAllReduce[] = "AllReduce";
const char kNameBroadcast[] = "Broadcast";
const char kNameAllgather[] = "AllGather";
const char kNameReduceScatter[] = "ReduceScatter";
const char kNameReduceSum[] = "ReduceSum";
const char kNameIsFinite[] = "isFinite";
const char kNameReciprocal[] = "Reciprocal";
const char kNameRsqrt[] = "Rsqrt";
const char kNameRsqrtGrad[] = "RsqrtGrad";
const char kNameSqrt[] = "Sqrt";
const char kNameSquare[] = "Square";
const char kNameSquaredDifference[] = "SquaredDifference";
const char kNamePow[] = "Pow";
const char kNameBatchMatMul[] = "BatchMatMul";
const char kNameStridedSlice[] = "StridedSlice";
const char kNameStridedSliceGrad[] = "StridedSliceGrad";
const char kNameExpandDims[] = "ExpandDims";
const char kNameLog[] = "Log";
const char kNameLogicalAnd[] = "LogicalAnd";
const char kNameLogicalNot[] = "LogicalNot";
const char kNameLogicalOr[] = "LogicalOr";
const char kNameExp[] = "Exp";
const char kNameLessEqual[] = "LessEqual";
const char kNameGreaterEqual[] = "GreaterEqual";
const char kNameEqual[] = "Equal";
const char kNameNotEqual[] = "NotEqual";
const char kNameFlattenGrad[] = "FlattenGrad";
const char kNameConvolution[] = "Convolution";
const char kNameBiasAdd[] = "BiasAdd";
const char kNameMaxPoolGrad[] = "MaxPoolGrad";
const char kNameAvgPoolGrad[] = "AvgPoolGrad";
const char kNameMaxPoolGradWithArgmax[] = "MaxPoolGradWithArgmax";
const char kNameApplyMomentum[] = "ApplyMomentum";
const char kNameDropoutDoMask[] = "DropoutDoMask";
const char kNameResizeBilinear[] = "ResizeBilinear";
const char kNameResizeBilinearGrad[] = "ResizeBilinearGrad";
const char kNameZerosLike[] = "ZerosLike";
const char kNameOnesLike[] = "OnesLike";
const char kNameTruncatedNormal[] = "TruncatedNormal";
const char kNameSpaceToBatchNd[] = "SpaceToBatchNd";
const char kNameConfusionMatrix[] = "ConfusionMatrix";
const char kNameResizeNearestNeighborD[] = "ResizeNearestNeighbor";
const char kNameResizeNearestNeighborGrad[] = "ResizeNearestNeighborGrad";
const char kNameApplyAdam[] = "Adam";
const char kNameReLU6[] = "ReLU6";
const char kNameReLU6Grad[] = "ReLU6Grad";
const char kNameElu[] = "Elu";
const char kNameEluGrad[] = "EluGrad";
const char kNameScatterNdUpdate[] = "ScatterNdUpdate";
const char kNameNMSWithMask[] = "NMSWithMask";
const char kNameCheckValid[] = "CheckValid";
const char kNameSmoothL1Loss[] = "SmoothL1Loss";
const char kNameSmoothL1LossGrad[] = "SmoothL1LossGrad";
const char kNameSGD[] = "SGD";
const char kNameSigmoidCrossEntropyWithLogits[] = "SigmoidCrossEntropyWithLogits";
const char kNameSigmoidCrossEntropyWithLogitsGrad[] = "SigmoidCrossEntropyWithLogitsGrad";
const char kNameScatterNdD[] = "ScatterNd";
const char kNamePadD[] = "Pad";
const char kNameMirrorPad[] = "MirrorPad";
const char kNameMirrorPadGrad[] = "MirrorPadGrad";
const char kNameGatherNd[] = "GatherNd";
const char kNameArgmax[] = "Argmax";
const char kNameArgmin[] = "Argmin";
const char kNameArgMaxWithValue[] = "ArgMaxWithValue";
const char kNameArgMinWithValue[] = "ArgMinWithValue";
const char kNameReduceProd[] = "ReduceProd";
const char kNameCumProd[] = "CumProd";
const char kNameDiagpart[] = "Diagpart";
const char kNameSplitD[] = "Split";
const char kNameBatchToSpaceNd[] = "BatchToSpaceNd";
const char kNameFloor[] = "Floor";
const char kNameNPUGetFloatStatus[] = "NPUGetFloatStatus";
const char kNameAssignAdd[] = "AssignAdd";
const char kNameAssignSub[] = "AssignSub";
const char kNameNPUAllocFloatStatus[] = "NPUAllocFloatStatus";
const char kNameNPUClearFloatStatus[] = "NPUClearFloatStatus";
const char kNameReshape[] = "Reshape";
const char kNameRealDiv[] = "RealDiv";
const char kNameTile[] = "Tile";
const char kNameCos[] = "Cos";
const char kNameACos[] = "ACos";
const char kNameACosGrad[] = "ACosGrad";
const char kNameFloorDiv[] = "FloorDiv";
const char kNameSin[] = "Sin";
const char kNamePrelu[] = "PReLU";
const char kNamePreluGrad[] = "PReLUGrad";
const char kNameSigmoid[] = "Sigmoid";
const char kNameSigmoidGrad[] = "SigmoidGrad";
const char kNameL2Normalize[] = "L2Normalize";
const char kNameL2NormalizeGrad[] = "L2NormalizeGrad";
const char kNameSoftmax[] = "Softmax";
const char kNameIOU[] = "IOU";
const char kNameBoundingBoxDecode[] = "BoundingBoxDecode";
const char kNameBoundingBoxEncode[] = "BoundingBoxEncode";
const char kNameSlice[] = "Slice";
const char kNameAddN[] = "AddN";
const char kNameLess[] = "Less";
const char kNameGreater[] = "Greater";
const char kNamePack[] = "Pack";
const char kNameUnpack[] = "Unpack";
const char kNameMerge[] = "Merge";
const char kNameGeSwitch[] = "GeSwitch";

const char kNameHuberLoss[] = "HuberLoss";
const char kNameCumSum[] = "CumSum";
const char kNameHuberLossGrad[] = "HuberLossGrad";
const char kNameSparseSoftmaxCrossEntropy[] = "SparseSoftmaxCrossEntropy";
const char kNameSparseSoftmaxCrossEntropyGrad[] = "SparseSoftmaxCrossEntropyGrad";
const char kNameTopK[] = "TopK";
const char kNameSoftmaxGrad[] = "SoftmaxGrad";
const char kNameMaxPool[] = "MaxPool";
const char kNameAvgPool[] = "AvgPool";
const char kNameMaxPoolWithArgmax[] = "MaxPoolWithArgmax";
const char kNameBatchNorm[] = "BatchNorm";
const char kNameBatchNormGrad[] = "BatchNormGrad";
const char kNameROIAlign[] = "ROIAlign";
const char kNameROIAlignGrad[] = "ROIAlignGrad";
const char kNameRandomChoiceWithMask[] = "RandomChoiceWithMask";
const char kNameAbs[] = "Abs";
const char kNameAbsGrad[] = "AbsGrad";
const char kNameBinaryCrossEntropy[] = "BinaryCrossEntropy";
const char kNameBinaryCrossEntropyGrad[] = "BinaryCrossEntropyGrad";
const char kNameSparseApplyAdagrad[] = "SparseApplyAdagrad";
const char kNameAcosh[] = "Acosh";
const char kNameFloorMod[] = "FloorMod";
const char kNameSpaceToDepth[] = "SpaceToDepth";
const char kNameDepthToSpace[] = "DepthToSpace";
const char kNameSign[] = "Sign";
const char kNameLARSUpdate[] = "LARSUpdate";
const char kNameRound[] = "Round";
const char kNamePrint[] = "Print";
const char kNameApplyFtrl[] = "ApplyFtrl";
const char kNameDiag[] = "Diag";
const char kNameDiagPart[] = "DiagPart";
const char kNameSpaceToBatch[] = "SpaceToBatch";
const char kNameBatchToSpace[] = "BatchToSpace";
const char kNameAtan2[] = "Atan2";
const char kNameApplyRMSProp[] = "ApplyRMSProp";
const char kNameApplyCenteredRMSProp[] = "ApplyCenteredRMSProp";

// -----------------OpAdapter initialization--------------
std::unordered_map<std::string, OpAdapterDescPtr> &DfGraphConvertor::get_adpt_map() {
  static std::unordered_map<std::string, OpAdapterDescPtr> adpt_map = {
    {string(kNameCustomOp), ADPT_DESC(Operator)},
    {string(kNameIOU), ADPT_DESC(Iou)},
    {string(kNameGreaterEqual), ADPT_DESC(GreaterEqual)},
    {string(kNameSlice), ADPT_DESC(SliceD)},
    {string(kNameApplyMomentum), ADPT_DESC(ApplyMomentum)},
    {string(kNameMaxPool), ADPT_DESC(MaxPool)},
    {string(kNameAvgPool), ADPT_DESC(AvgPool)},
    {string(kNameMaxPoolWithArgmax), ADPT_DESC(MaxPoolWithArgmax)},
    {string(kNameTopK), ADPT_DESC(TopKV2)},
    {string(kNamePack), ADPT_DESC(Pack)},
    {string(kNameUnpack), ADPT_DESC(Unpack)},
    {string(kNameSplitD), ADPT_DESC(SplitD)},
    {string(kNameAllReduce), ADPT_DESC(HcomAllReduce)},
    {string(kNameBroadcast), ADPT_DESC(HcomBroadcast)},
    {string(kNameAllgather), ADPT_DESC(HcomAllGather)},
    {string(kNameReduceScatter), ADPT_DESC(HcomReduceScatter)},
    {string(kNameMaxPoolGrad), ADPT_DESC(MaxPoolGrad)},
    {string(kNameAvgPoolGrad), ADPT_DESC(AvgPoolGrad)},
    {string(kNameMaxPoolGradWithArgmax), ADPT_DESC(MaxPoolGradWithArgmax)},
    {prim::kPrimAssign->name(), ADPT_DESC(Assign)},
    {prim::kPrimStateSetItem->name(), ADPT_DESC(Assign)},
    {prim::kPrimReluGrad->name(), ADPT_DESC(ReluGrad)},
    {prim::kPrimFusedBatchNormGrad->name(), ADPT_DESC(FusedBatchNormGrad)},
    {prim::kPrimBiasAddGrad->name(), ADPT_DESC(BiasAddGrad)},
    {prim::kPrimConv2D->name(), ADPT_DESC(Conv2D)},
    {prim::kPrimConv2DBackpropInput->name(), ADPT_DESC(Conv2DBackpropInputD)},
    {prim::kPrimConv2DBackpropFilter->name(), ADPT_DESC(Conv2DBackpropFilterD)},
    {prim::kPrimDepthwiseConv2dNative->name(), ADPT_DESC(DepthwiseConv2D)},
    {prim::kPrimDepthwiseConv2dNativeBackpropFilter->name(), ADPT_DESC(DepthwiseConv2DBackpropFilterD)},
    {prim::kPrimDepthwiseConv2dNativeBackpropInput->name(), ADPT_DESC(DepthwiseConv2DBackpropInputD)},
    {prim::kPrimFusedBatchNorm->name(), ADPT_DESC(FusedBatchNorm, BatchNorm)},
    {string(kNameBatchNorm), ADPT_DESC(BatchNorm)},
    {string(kNameBatchNormGrad), ADPT_DESC(BatchNormGrad)},
    {string(kNameReshape), ADPT_DESC(Reshape)},
    {string(kNameFlattenGrad), ADPT_DESC(Reshape)},
    {prim::kPrimFlatten->name(), ADPT_DESC(Flatten)},
    {string(kNameAddN), ADPT_DESC(AddN)},
    {string(kNameLess), ADPT_DESC(Less)},
    {string(kNameSqrt), ADPT_DESC(Sqrt)},
    {string(kNameRsqrt), ADPT_DESC(Rsqrt)},
    {string(kNameSquare), ADPT_DESC(Square)},
    {prim::kPrimTanh->name(), ADPT_DESC(Tanh)},
    {prim::kPrimTanhGrad->name(), ADPT_DESC(TanhGrad)},
    {string(kNameResizeNearestNeighborD), ADPT_DESC(ResizeNearestNeighborD)},
    {string(kNameResizeNearestNeighborGrad), ADPT_DESC(ResizeNearestNeighborGrad)},
    {string(kNameApplyAdam), ADPT_DESC(ApplyAdam)},
    {string(kNameReLU6), ADPT_DESC(Relu6)},
    {string(kNameReLU6Grad), ADPT_DESC(Relu6Grad)},
    {string(kNameElu), ADPT_DESC(Elu)},
    {string(kNameEluGrad), ADPT_DESC(EluGrad)},
    {string(kNameResizeBilinearGrad), ADPT_DESC(ResizeBilinearGrad)},
    {string(kNameResizeBilinear), ADPT_DESC(ResizeBilinearD)},
    {string(kNameZerosLike), ADPT_DESC(ZerosLike)},
    {string(kNameOnesLike), ADPT_DESC(OnesLike)},
    {string(kNameScatterNdUpdate), ADPT_DESC(ScatterNdUpdate)},
    {string(kNameNMSWithMask), ADPT_DESC(NMSWithMask)},
    {string(kNameCheckValid), ADPT_DESC(CheckValid)},
    {string(kNameSmoothL1Loss), ADPT_DESC(SmoothL1Loss)},
    {string(kNameSmoothL1LossGrad), ADPT_DESC(SmoothL1LossGrad)},
    {string(kNameSigmoidCrossEntropyWithLogits), ADPT_DESC(SigmoidCrossEntropyWithLogits)},
    {string(kNameSigmoidCrossEntropyWithLogitsGrad), ADPT_DESC(SigmoidCrossEntropyWithLogitsGrad)},
    {string(kNameScatterNdD), ADPT_DESC(ScatterNdD)},
    {string(kNamePadD), ADPT_DESC(PadD)},
    {string(kNameMirrorPad), ADPT_DESC(MirrorPad)},
    {string(kNameMirrorPadGrad), ADPT_DESC(MirrorPadGrad)},
    {string(kNameGatherNd), ADPT_DESC(GatherNd)},
    {string(kNameArgmax), ADPT_DESC(ArgMaxD)},
    {string(kNameArgmin), ADPT_DESC(ArgMinD)},
    {string(kNameArgMaxWithValue), ADPT_DESC(ArgMaxWithValue)},
    {string(kNameArgMinWithValue), ADPT_DESC(ArgMinWithValue)},
    {prim::kPrimReduceSum->name(), ADPT_DESC(ReduceSumD)},
    {prim::kPrimReduceMean->name(), ADPT_DESC(ReduceMeanD)},
    {prim::kPrimReduceAll->name(), ADPT_DESC(ReduceAll)},
    {prim::kPrimReduceMin->name(), ADPT_DESC(ReduceMinD)},
    {prim::kPrimReduceMax->name(), ADPT_DESC(ReduceMaxD)},
    {string(kNameLARSUpdate), ADPT_DESC(LarsV2Update)},
    {string(kNameReduceProd), ADPT_DESC(ReduceProdD)},
    {string(kNameCumProd), ADPT_DESC(CumprodD)},
    {string(kNameMerge), ADPT_DESC(Merge)},
    {string(kNameGeSwitch), ADPT_DESC(Switch)},
    {string(kNameCumSum), ADPT_DESC(CumsumD)},

    {prim::kPrimMul->name(), ADPT_DESC(Mul)},
    {string(kNameTile), ADPT_DESC(TileD)},
    {prim::kPrimOneHot->name(), ADPT_DESC(OneHot)},

    {prim::kPrimGatherV2->name(), ADPT_DESC(GatherV2D)},
    {string(kNameCos), ADPT_DESC(Cos)},
    {string(kNameACos), ADPT_DESC(Acos)},
    {string(kNameACosGrad), ADPT_DESC(AcosGrad)},
    {string(kNameFloor), ADPT_DESC(Floor)},
    {string(kNameFloorDiv), ADPT_DESC(FloorDiv)},
    {string(kNameSin), ADPT_DESC(Sin)},
    {string(kNameExp), ADPT_DESC(Exp)},
    {string(kNameBoundingBoxEncode), ADPT_DESC(BoundingBoxEncode)},
    {string(kNameBoundingBoxDecode), ADPT_DESC(BoundingBoxDecode)},

    {prim::kPrimCast->name(), ADPT_DESC(Cast)},
    {string(kNameRealDiv), ADPT_DESC(RealDiv)},
    {prim::kPrimNeg->name(), ADPT_DESC(Neg)},
    {prim::kPrimTranspose->name(), ADPT_DESC(TransposeD)},
    {prim::kPrimSub->name(), ADPT_DESC(Sub)},
    {string(kNameReciprocal), ADPT_DESC(Reciprocal)},
    {prim::kPrimDropoutGenMask->name(), ADPT_DESC(DropOutGenMask)},
    {string(kNameAssignAdd), ADPT_DESC(AssignAdd)},
    {string(kNameAssignSub), ADPT_DESC(AssignSub)},
    {prim::kPrimConcat->name(), ADPT_DESC(ConcatD)},
    {string(kNamePow), ADPT_DESC(Pow)},
    {string(kNameExp), ADPT_DESC(Exp)},
    {string(kNameEqual), ADPT_DESC(Equal)},
    {string(kNameNotEqual), ADPT_DESC(NotEqual)},
    {string(kNameLog), ADPT_DESC(Log)},
    {string(kNameLogicalAnd), ADPT_DESC(LogicalAnd)},
    {string(kNameLogicalNot), ADPT_DESC(LogicalNot)},
    {string(kNameLogicalOr), ADPT_DESC(LogicalOr)},
    {string(kNameGreater), ADPT_DESC(Greater)},
    {prim::kPrimMaximum->name(), ADPT_DESC(Maximum)},
    {prim::kPrimRelu->name(), ADPT_DESC(Relu)},
    {string(kNamePrelu), ADPT_DESC(PRelu)},
    {string(kNamePreluGrad), ADPT_DESC(PReluGrad)},
    {string(kNameSigmoid), ADPT_DESC(Sigmoid)},
    {string(kNameSigmoidGrad), ADPT_DESC(SigmoidGrad)},
    {string(kNameSGD), ADPT_DESC(SGD)},
    {prim::kPrimLogSoftmaxGrad->name(), ADPT_DESC(LogSoftmaxGrad)},
    {prim::kPrimMaximumGrad->name(), ADPT_DESC(MaximumGrad)},
    {prim::kPrimMinimumGrad->name(), ADPT_DESC(MinimumGrad)},
    {string(kNameL2Normalize), ADPT_DESC(L2Normalize)},
    {string(kNameL2NormalizeGrad), ADPT_DESC(L2NormalizeGrad)},

    {prim::kPrimMinimum->name(), ADPT_DESC(Minimum)},
    {prim::kPrimSelect->name(), ADPT_DESC(Select)},
    {string(kNameLessEqual), ADPT_DESC(LessEqual)},
    {prim::kPrimLogSoftmax->name(), ADPT_DESC(LogSoftmax)},
    {string(kNameTruncatedNormal), ADPT_DESC(TruncatedNormal)},
    {string(kNameStridedSliceGrad), ADPT_DESC(StridedSliceGrad)},
    {prim::kPrimGelu->name(), ADPT_DESC(Gelu)},
    {prim::kPrimGeluGrad->name(), ADPT_DESC(GeluGrad)},
    {string(kNameStridedSlice), ADPT_DESC(StridedSlice)},
    {prim::kPrimUnsortedSegmentSum->name(), ADPT_DESC(UnsortedSegmentSumD)},
    {string(kNameExpandDims), ADPT_DESC(ExpandDims)},
    {prim::kPrimSqueeze->name(), ADPT_DESC(Squeeze)},
    {prim::kPrimLayerNorm->name(), ADPT_DESC(LayerNorm)},
    {prim::kPrimLayerNormGrad->name(), ADPT_DESC(LayerNormGrad)},
    {string(kNameBatchMatMul), ADPT_DESC(BatchMatMul)},
    {string(kNameDropoutDoMask), ADPT_DESC(DropOutDoMask)},

    {string(kNameNPUGetFloatStatus), ADPT_DESC(NPUGetFloatStatus)},
    {string(kNameNPUAllocFloatStatus), ADPT_DESC(NPUAllocFloatStatus)},
    {string(kNameNPUClearFloatStatus), ADPT_DESC(NPUClearFloatStatus)},

    {string(kNameRandomChoiceWithMask), ADPT_DESC(RandomChoiceWithMask)},
    {prim::kPrimSoftmaxCrossEntropyWithLogits->name(), ADPT_DESC(SoftmaxCrossEntropyWithLogits)},

    {prim::kPrimScalarSummary->name(), ADPT_DESC(Summary)},
    {prim::kPrimImageSummary->name(), ADPT_DESC(Summary)},
    {prim::kPrimTensorSummary->name(), ADPT_DESC(Summary)},
    {prim::kPrimTensorAdd->name(),
     std::make_shared<OpAdapterDesc>(std::make_shared<OpAdapter<Add>>(ExtraAttr({{"mode", MakeValue(1)}})),
                                     std::make_shared<OpAdapter<Add>>(ExtraAttr({{"mode", MakeValue(1)}})))},
    {string(kNameBiasAdd), ADPT_DESC(BiasAdd)},
    {prim::kPrimRelu->name(), ADPT_DESC(Relu)},

    {prim::kPrimMatMul->name(), ADPT_DESC(MatMul)},

    {string(kNameConst), ADPT_DESC(Constant, Const)},
    {string(kNameSoftmax), ADPT_DESC(Softmax)},
    {string(kNameSoftmaxGrad), ADPT_DESC(SoftmaxGrad)},
    {string(kNameParam), ADPT_DESC(Data)},
    {string(kNameROIAlign), ADPT_DESC(ROIAlign)},
    {string(kNameROIAlignGrad), ADPT_DESC(ROIAlignGrad)},
    {string(kNameAbs), ADPT_DESC(Abs)},
    {string(kNameAbsGrad), ADPT_DESC(AbsGrad)},
    {string(kNameBinaryCrossEntropy), ADPT_DESC(BinaryCrossEntropy)},
    {string(kNameBinaryCrossEntropyGrad), ADPT_DESC(BinaryCrossEntropyGrad)},
    {string(kNameSparseApplyAdagrad), ADPT_DESC(SparseApplyAdagradD)},
    {string(kNameAcosh), ADPT_DESC(Acosh)},
    {string(kNameFloorMod), ADPT_DESC(FloorMod)},
    {string(kNameSpaceToDepth), ADPT_DESC(SpaceToDepth)},
    {string(kNameDepthToSpace), ADPT_DESC(DepthToSpace)},
    {string(kNameSign), ADPT_DESC(Sign)},
    {string(kNameRound), ADPT_DESC(Round)},
    {string(kNameApplyFtrl), ADPT_DESC(ApplyFtrl)},
    {string(kNameDiag), ADPT_DESC(Diag)},
    {string(kNameDiagPart), ADPT_DESC(DiagPart)},
    {string(kNameSpaceToBatch), ADPT_DESC(SpaceToBatchD)},
    {string(kNameBatchToSpace), ADPT_DESC(BatchToSpaceD)},
    {string(kNameAtan2), ADPT_DESC(Atan2)},
    {string(kNameApplyRMSProp), ADPT_DESC(ApplyRMSPropD)},
    {string(kNameApplyCenteredRMSProp), ADPT_DESC(ApplyCenteredRMSProp)}};
#ifdef ENABLE_GE
  adpt_map[string(kNamePrint)] = ADPT_DESC(Print);
#endif
  return adpt_map;
}

// ---------------implement of DfGraphConvertor-------------
PrimType GetCNodeFuncType(const CNodePtr cnode) {
  if (cnode->inputs().empty()) {
    return kPrimTypeUnknown;
  }

  AnfNodePtr valuenode = cnode->input(0);
  if (IsValueNode<Primitive>(valuenode)) {
    // check whether the valuenode is primitive
    return GetValueNode<PrimitivePtr>(valuenode)->prim_type();
  }
  return kPrimTypeUnknown;
}

OpAdapterPtr DfGraphConvertor::FindAdapter(const AnfNodePtr node, bool train) {
  if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();

    std::string name = kNameCustomOp;
    if (!IsCustomCNode(cnode)) {
      name = GetCNodeFuncName(cnode);
    }

    auto it_adpt = get_adpt_map().find(name);
    if (it_adpt != get_adpt_map().end()) {
      return it_adpt->second->Get(train);
    } else {
      MS_LOG(ERROR) << "Can't find OpAdapter for " << name;
    }
  }

  if (node->isa<ValueNode>()) {
    return get_adpt_map()[kNameConst]->Get(train);
  }
  if (node->isa<Parameter>()) {
    return get_adpt_map()[kNameParam]->Get(train);
  }
  return OpAdapterPtr(nullptr);
}

void DfGraphConvertor::InitLoopVar(std::vector<ge::Operator> *init_input) {
  if (this->training_) {
    GeTensorDesc desc(GeShape(), ge::FORMAT_NCHW, ge::DT_INT64);
    auto var_iter_num = std::make_shared<Variable>("npu_runconfig/iterations_per_loop");
    auto var_loop_cond = std::make_shared<Variable>("npu_runconfig/loop_cond");
    auto var_one = std::make_shared<Variable>("npu_runconfig/one");
    auto var_zero = std::make_shared<Variable>("npu_runconfig/zero");
    (void)var_iter_num->update_output_desc_y(desc);
    (void)var_loop_cond->update_output_desc_y(desc);
    (void)var_one->update_output_desc_y(desc);
    (void)var_zero->update_output_desc_y(desc);
    vars_["npu_runconfig/iterations_per_loop"] = var_iter_num;
    vars_["npu_runconfig/loop_cond"] = var_loop_cond;
    vars_["npu_runconfig/one"] = var_one;
    vars_["npu_runconfig/zero"] = var_zero;

    int64_t value = 0;
    auto const_iter_num = std::make_shared<Constant>("const/npu_runconfig/iterations_per_loop");
    if (ConfigManager::GetInstance().dataset_mode() == DS_SINK_MODE) {
      value = ConfigManager::GetInstance().iter_num();
    } else {
      MS_LOG(INFO) << "Run with normal(non-sink) mode, the iterator number will always be 1";
      value = 1;
      ConfigManager::GetInstance().set_iter_num(value);
    }
    value -= 1;  // iteration start from 0, the max iteration number for n loop should be n-1
    (void)const_iter_num->set_attr_value(GeTensor(desc, reinterpret_cast<uint8_t *>(&value), sizeof(int64_t)));

    auto const_loop_cond = std::make_shared<Constant>("const/npu_runconfig/loop_cond");
    value = 0;
    (void)const_loop_cond->set_attr_value(GeTensor(desc, reinterpret_cast<uint8_t *>(&value), sizeof(int64_t)));

    auto const_one = std::make_shared<Constant>("const/npu_runconfig/one");
    value = 1;
    (void)const_one->set_attr_value(GeTensor(desc, reinterpret_cast<uint8_t *>(&value), sizeof(int64_t)));

    auto const_zero = std::make_shared<Constant>("const/npu_runconfig/zero");
    value = 0;
    (void)const_zero->set_attr_value(GeTensor(desc, reinterpret_cast<uint8_t *>(&value), sizeof(int64_t)));

    (void)const_iter_num->update_output_desc_y(desc);
    (void)const_loop_cond->update_output_desc_y(desc);
    (void)const_one->update_output_desc_y(desc);
    (void)const_zero->update_output_desc_y(desc);

    auto assign_iter_num = std::make_shared<Assign>("assign/npu_runconfig/iterations_per_loop");
    (void)assign_iter_num->set_input_ref(*var_iter_num).set_input_value(*const_iter_num);
    auto assign_loop_cond = std::make_shared<Assign>("assign/npu_runconfig/loop_cond");
    (void)assign_loop_cond->set_input_ref(*var_loop_cond).set_input_value(*const_loop_cond);
    auto assign_one = std::make_shared<Assign>("assign/npu_runconfig/one");
    (void)assign_one->set_input_ref(*var_one).set_input_value(*const_one);
    auto assign_zero = std::make_shared<Assign>("assign/npu_runconfig/zero");
    (void)assign_zero->set_input_ref(*var_zero).set_input_value(*const_zero);

    init_input->push_back(*var_iter_num);
    init_input->push_back(*var_loop_cond);
    init_input->push_back(*var_one);
    init_input->push_back(*var_zero);
    init_ops_.push_back(var_iter_num);
    init_ops_.push_back(var_loop_cond);
    init_ops_.push_back(var_one);
    init_ops_.push_back(var_zero);
    init_ops_.push_back(const_iter_num);
    init_ops_.push_back(const_loop_cond);
    init_ops_.push_back(const_one);
    init_ops_.push_back(const_zero);
    init_ops_.push_back(assign_iter_num);
    init_ops_.push_back(assign_loop_cond);
    init_ops_.push_back(assign_one);
    init_ops_.push_back(assign_zero);
  }
}

OpAdapterPtr DfGraphConvertor::FindAdapter(const std::string &name, bool train) {
  auto it = get_adpt_map().find(name);
  if (it != get_adpt_map().end()) {
    return it->second->Get(train);
  }
  MS_LOG(ERROR) << "Can't find OpAdapter for " << name;
  return transform::OpAdapterPtr(nullptr);
}

void DfGraphConvertor::DrawParamInitSubGraph(const std::string &name, const AnfNodePtr &it) {
  // draw init subgraph
  init_sout_ << "op_assign" << it.get() << "[label=<";
  init_sout_ << "<table border='1' cellborder='1'>" << endl;
  init_sout_ << "<tr>";
  init_sout_ << "<td port='1'>resource</td>";
  init_sout_ << "<td port='2'>value</td>";
  init_sout_ << "</tr>" << endl;
  init_sout_ << "<tr><td colspan=\"2\">"
             << "\"assign_" << name << "\"</td></tr>" << endl;
  init_sout_ << "</table>> shape=plaintext]" << endl;
  init_sout_ << "param" << it.get() << "[shape=octagon, label=\"" << name << "\"]" << endl;
  init_sout_ << "const" << it.get() << "[label= \"" << name << "_const"
             << "\" shape=ellipse]" << endl;
  init_sout_ << "param" << it.get() << "->"
             << "op_assign" << it.get() << ":1" << endl;
  init_sout_ << "const" << it.get() << "->"
             << "op_assign" << it.get() << ":2" << endl;
}

void DfGraphConvertor::SetupParamInitSubGraph(const TensorOrderMap &tensors, std::vector<ge::Operator> *init_input) {
  DfGraphPtr init_graph = std::make_shared<DfGraph>("init");
  std::vector<AnfNodePtr> nodes = TopoSort(anf_graph_->get_return());

  for (auto &it : nodes) {
    if (it->isa<ValueNode>()) {
      if (IsValueNode<SymbolicKeyInstance>(it)) {
        auto symbolic = GetValueNode<SymbolicKeyInstancePtr>(it);
        auto name = std::static_pointer_cast<Parameter>(symbolic->node())->name();
        auto iter = vars_.find(name);  // get correspoding varaible op
        if (iter != vars_.end()) {
          op_cache_[it.get()] = iter->second;
          // #ifdef DRAW_GE_GRAPH
          compute_sout_ << op_draw_name_[params_[name].get()] << " -> " << op_draw_name_[it.get()]
                        << "[style=\"dotted\"]" << endl;
          // #endif
        }
      } else if (IsValueNode<RefKey>(it)) {
        auto refkey = GetValueNode<RefKeyPtr>(it);
        auto name = refkey->tag();
        auto iter = vars_.find(name);  // get correspoding varaible op
        if (iter != vars_.end()) {
          op_cache_[it.get()] = iter->second;
          compute_sout_ << op_draw_name_[params_[name].get()] << " -> " << op_draw_name_[it.get()]
                        << "[style=\"dotted\"]" << endl;
        }
      }
    }
  }

  for (auto &it : tensors) {
    if (vars_.find(it.first) == vars_.end()) {
      MS_LOG(WARNING) << "Init parameter " << it.first << " didn't appear in graph.";
      vars_[it.first] = nullptr;
    }
  }

  // set up init sub graph
  if (init_input->size()) {
    // init sub graph needs no input
    MS_LOG(INFO) << "Build data init subgraph.";
    (void)init_graph->SetInputs(*init_input);
    this->init_graph_ = init_graph;
  } else {
    this->init_graph_ = nullptr;
  }
}

void DfGraphConvertor::MakeDatasetHandler(const std::string &name, const size_t &input_idx, const AnfNodePtr &it) {
  MS_LOG(INFO) << "The " << name << " is the " << input_idx << "(st/nd/th) input";
  if (ConfigManager::GetInstance().dataset_mode() == DS_SINK_MODE) {
    auto getnext_idx = static_cast<int64_t>(input_idx);
    DatasetGraphParam param = ConfigManager::GetInstance().dataset_param();
    if (!param.input_indexes().empty() && input_idx <= param.input_indexes().size()) {
      getnext_idx = param.input_indexes()[input_idx] - 1;  // input_idx start from 0.
      MS_LOG(INFO) << "remap input_index:" << input_idx << " to getnext_index:" << getnext_idx << ".";
    }
    // use iterator_getnext op with output_name instead of data op in BuildGraph.
    out_handle_cache_[it.get()] = OutHandler(dataset_iter_getnext_, "y" + std::to_string(getnext_idx));
  }
}

void DfGraphConvertor::SetupBroadcast(const std::shared_ptr<HcomBroadcast> &broadcast,
                                      const std::vector<GeTensorDesc> &broadcast_desc,
                                      const DfGraphPtr &broadcast_graph, std::vector<ge::Operator> broadcast_input) {
  MS_LOG(INFO) << "build broadcast subgraph";
  if (broadcast_desc.size() != broadcast_input.size()) {
    MS_LOG(EXCEPTION) << "Desc number of BroadCast is not equal to number of Input";
  }
  (void)broadcast->create_dynamic_input_x(static_cast<unsigned int>(broadcast_input.size()));
  (void)broadcast->create_dynamic_output_y(static_cast<unsigned int>(broadcast_desc.size()));
  for (unsigned int i = 0; i < broadcast_input.size(); i++) {
    (void)broadcast->set_dynamic_input_x(i, broadcast_input[i]);
    (void)broadcast->update_dynamic_output_desc_y(i, broadcast_desc[i]);
  }
  (void)broadcast_graph->SetInputs(broadcast_input);
  this->broadcast_graph_ = broadcast_graph;
}

void DfGraphConvertor::InitParamWithData(const TensorOrderMap &tensors) {
  int index = 0;
  std::vector<Operator> init_input;
  for (auto it : tensors) {
    std::string name = it.first;
    auto node_itor = params_.find(name);
    // if name not in params_, create a node in graph
    if (node_itor == params_.end()) {
      MS_LOG(WARNING) << "" << name << " is not in params, and create a new node.";
      ParameterPtr param = anf_graph_->add_parameter();
      name = name + "_temp";
      param->set_name(name);
      (void)ConvertParameter(param);
      node_itor = params_.find(name);
    }
    auto node = node_itor->second;
    auto op_itor = op_cache_.find(node.get());
    if (op_itor == op_cache_.end()) {
      MS_LOG(EXCEPTION) << "Can not find op for node " << node->ToString() << ".";
    }
    auto adpt = FindAdapter(kNameParam, training_);
    if (adpt == nullptr) continue;
    auto param_op = adpt->generate(name + "_data");
    MS_LOG(INFO) << "Add parameter " << name << " as input, index " << index << ".";
    (void)std::static_pointer_cast<Data>(param_op)->set_attr_index(index++);

    if (!training_) {
      auto adpt_const = FindAdapter(kNameConst, training_);
      if (adpt_const == nullptr) continue;
      auto const_op = adpt_const->generate(name + "_const");
      (void)adpt_const->setAttr(const_op, "value", it.second);

      auto const_op_desc = TransformUtil::GetGeTensorDesc(it.second->shape_c(), it.second->data_type(), kOpFormat_NCHW);
      if (const_op_desc == nullptr) {
        MS_LOG(ERROR) << "Create variable " << name << " ouptut descriptor failed!";
        continue;
      }
      (void)std::static_pointer_cast<Constant>(const_op)->update_output_desc_y(*const_op_desc);

      vars_[name] = const_op;
      op_itor->second = const_op;
      continue;
    }

    // create tensor descriptor for output descriptor
    auto desc = TransformUtil::GetGeTensorDesc(it.second->shape_c(), it.second->data_type(), kOpFormat_NCHW);
    if (desc == nullptr) {
      MS_LOG(ERROR) << "Create variable " << name << " ouptut descriptor failed!";
      continue;
    }

    // we need three variable ops for each graph with same name
    // build init subgraph
    auto init_var = std::make_shared<Variable>(name);
    auto assign_op = std::make_shared<Assign>("assign_" + name);
    (void)init_var->update_output_desc_y(*desc);
    (void)assign_op->set_input_ref(*init_var).set_input_value(*param_op);
    init_input.push_back(*init_var);
    init_ops_.push_back(param_op);
    init_ops_.push_back(assign_op);
    init_ops_.push_back(init_var);

    auto variable = std::make_shared<Variable>(name);
    (void)variable->update_output_desc_y(*desc);
    // do not use read variable while variable sink
    MS_LOG(DEBUG) << "InitParam, op_name = " << name << ", var = " << variable->GetName() << ".";
    op_itor->second = variable;  // replace parameter with variable
    vars_[name] = variable;      // prevent the variable operator from being freed
    DrawParamInitSubGraph(name, node);
  }
  InitLoopVar(&init_input);
  SetupParamInitSubGraph(tensors, &init_input);
}

// convert all parameter need initialize to variable
DfGraphConvertor &DfGraphConvertor::InitParam(const TensorOrderMap &tensors) {
  size_t input_idx = 0;
  if (error_ != 0) {
    return *this;
  }
  if (anf_graph_ == nullptr || anf_graph_->output() == nullptr) {
    error_ = INVALID_ARGUMENT;
    MS_LOG(ERROR) << "Invalid AnfGraph in InitParam.";
    return *this;
  }

  // Processing input with MakeDatasetHandler
  for (auto &it : anf_graph_->parameters()) {
    auto op_itor = op_cache_.find(it.get());  // converted node
    if (it->isa<Parameter>() && op_itor != op_cache_.end()) {
      string name = std::static_pointer_cast<Parameter>(it)->name();
      auto tensor_itor = tensors.find(name);  // in init value map
      if (tensor_itor == tensors.end()) {
        DfGraphConvertor::MakeDatasetHandler(name, input_idx, it);
        input_idx++;
      }
    }
  }
  InitParamWithData(tensors);
  init_sout_ << "}" << endl;
  return *this;
}

#if (defined ENABLE_GE)
void DfGraphConvertor::BuildSaveCheckpointGraph() {
  std::vector<Operator> graph_inputs;
  ge::op::Save save_op("save_parms");
  int save_op_is_active = 0;
  size_t index = 0;
  string name;

  int32_t count_size = std::count_if(vars_.begin(), vars_.end(), [](const std::pair<std::string, OperatorPtr> &it) {
    return (it.second == nullptr || it.first.find("/") != std::string::npos);
  });

  (void)save_op.create_dynamic_input_tensors(vars_.size() - static_cast<size_t>(count_size));

  // for each "parameter" in anf graph excluding "input"
  for (const auto &it : vars_) {
    name = it.first;
    if (it.second == nullptr || name.find("/") != std::string::npos) continue;
    Variable variable(name);
    (void)variable.update_output_desc_y(it.second->GetOutputDesc(0));
    (void)save_op.set_dynamic_input_tensors(index++, variable);

    graph_inputs.push_back(variable);

    if (save_op_is_active == 0) {
      checkpoint_sout_ << "op_save" << &save_op << "[label=<";
      checkpoint_sout_ << "<table border='1' cellborder='1'>" << endl;
      checkpoint_sout_ << "<tr><td port='1'>tensor</td></tr>" << endl;
      checkpoint_sout_ << "<tr><td colspan=\"1\">"
                       << "\"saveop"
                       << "\"</td></tr>" << endl;
      checkpoint_sout_ << "</table>> shape=plaintext]" << endl;
    }

    checkpoint_sout_ << "param" << it.second << "[shape=octagon, label=\"" << name << "\"]" << endl;

    checkpoint_sout_ << "param" << it.second << "->"
                     << "op_save" << &save_op << ":1" << endl;
    save_op_is_active = 1;
  }
  if (save_op_is_active) {
    std::vector<Operator> graph_output;
    graph_output.emplace_back(save_op);
    DfGraphPtr checkpoint_graph = std::make_shared<DfGraph>("checkpoint");
    (void)checkpoint_graph->SetInputs(graph_inputs);
    (void)checkpoint_graph->SetOutputs(graph_output);
    this->save_ckp_graph_ = checkpoint_graph;
  } else {
    this->save_ckp_graph_ = nullptr;
  }

  checkpoint_sout_ << "}" << endl;
  return;
}
#endif

DfGraphConvertor &DfGraphConvertor::GenerateBroadcastGraph(const TensorOrderMap &tensors) {
  if (error_ != 0) {
    return *this;
  }
  if (anf_graph_ == nullptr || anf_graph_->output() == nullptr) {
    error_ = INVALID_ARGUMENT;
    MS_LOG(ERROR) << "Invalid AnfGraph in generate broadcast graph";
    return *this;
  }

  DfGraphPtr broadcast_graph = std::make_shared<DfGraph>("broadcast");
  // collect the operators create for broadcast sub graph, in order to avoid auto release
  std::vector<Operator> broadcast_input;
  std::vector<GeTensorDesc> broadcast_desc;
  auto broadcast = std::make_shared<HcomBroadcast>("broadcast_parameter");
  (void)broadcast->set_attr_root_rank(0);
  (void)broadcast->set_attr_group("hccl_world_group");
  broadcast_ops_.push_back(broadcast);

  // find every parameter, build broadcast subgraph (or initialize the parameter with constant)
  for (auto &it : anf_graph_->parameters()) {
    auto op_itor = op_cache_.find(it.get());  // converted node
    if (it->isa<Parameter>() && op_itor != op_cache_.end()) {
      string name = std::static_pointer_cast<Parameter>(it)->name();
      auto tensor_itor = tensors.find(name);  // in init tensor map
      if (tensor_itor != tensors.end()) {
        auto tensor = tensor_itor->second;
        auto shape_ge = tensor->shape_c();

        // create tensor descriptor for output descriptor
        auto desc = TransformUtil::GetGeTensorDesc(shape_ge, tensor->data_type(), kOpFormat_NCHW);
        if (desc == nullptr) {
          MS_LOG(ERROR) << "Create variable " << name << " ouptut descriptor failed!";
          continue;
        }

        // build broadcast subgraph
        if (distribute_) {
          auto broadcast_var = std::make_shared<Variable>(name);
          (void)broadcast_var->update_output_desc_y(*desc);
          broadcast_input.push_back(*broadcast_var);
          broadcast_desc.push_back(*desc);
          broadcast_ops_.push_back(broadcast_var);
        }
      }
    }
  }

  // set up broadcast sub graph
  if (!broadcast_input.empty()) {
    DfGraphConvertor::SetupBroadcast(broadcast, broadcast_desc, broadcast_graph, broadcast_input);
  } else {
    this->broadcast_graph_ = nullptr;
  }
  return *this;
}

DfGraphConvertor &DfGraphConvertor::GenerateCheckpointGraph() {
  if (error_ != 0) {
    MS_LOG(ERROR) << "Generate checkpoint graph failed, found error code " << error_ << ".";
    return *this;
  }
  if (anf_graph_ == nullptr || anf_graph_->output() == nullptr) {
    error_ = INVALID_ARGUMENT;
    MS_LOG(ERROR) << "Invalid AnfGraph in GenerateCheckpointGraph";
    return *this;
  }
#if (defined ENABLE_GE)
  BuildSaveCheckpointGraph();
  // Restoring from checkpoint file is done by pyfront, not in graph now.
#endif
  return *this;
}

DfGraphConvertor &DfGraphConvertor::ConvertAllNode() {
  if (error_ != 0) {
    return *this;
  }
  if (anf_graph_ == nullptr || anf_graph_->output() == nullptr) {
    MS_LOG(ERROR) << "Invalid AnfGraph";
    error_ = FAILED;
    return *this;
  }

  compute_sout_.clear();
  compute_sout_ << "digraph {" << endl;
  init_sout_.clear();
  init_sout_ << "digraph {" << endl;
  checkpoint_sout_.clear();
  checkpoint_sout_ << "digraph {" << endl;
  restore_checkpoint_sout_.clear();
  restore_checkpoint_sout_ << "digraph {" << endl;

  // Convert all anf node to Operator
  MS_LOG(DEBUG) << "convert all node";
  std::vector<AnfNodePtr> nodes = TopoSort(anf_graph_->get_return());
  for (auto &it : nodes) {
    (void)Convert(it);
    if (this->error_ != 0) {
      MS_LOG(ERROR) << "failed to convert node: " << it->DebugString() << ".";
    }
  }

  // Create dataset iterator and iterator_getnext node
  if (ConfigManager::GetInstance().dataset_mode() == DS_SINK_MODE) {
    DatasetGraphParam param = ConfigManager::GetInstance().dataset_param();
    MS_LOG(INFO) << "Dataset param is " << param.ToString() << ".";
    // GetNext
    auto iter_getnext_op = make_shared<ge::op::GetNext>("get_next_tmp");
    (void)iter_getnext_op->set_attr_output_types(param.ge_types());
    (void)iter_getnext_op->set_attr_output_shapes(param.shapes());
    (void)iter_getnext_op->set_attr_channel_name(param.queue_name());

    // save iter_getnext_op for later use
    dataset_iter_getnext_ = iter_getnext_op;
  }

  // return the data flow graph
  return *this;
}

void DfGraphConvertor::TraceOutputFromTupleGetItem(const AnfNodePtr &anf_out) {
  auto it = out_handle_cache_.find(anf_out.get());
  if (it != out_handle_cache_.end()) {
    OutHandler handle = it->second;
    auto op = handle.op;
    if (op != nullptr) {
      MS_LOG(INFO) << "op name: " << op->GetName() << ", op type: " << op->GetOpType() << ", out_name: " << handle.out;
      graph_outputs_.emplace_back(std::make_pair(*op, handle.out));
    } else {
      MS_LOG(EXCEPTION) << "tuple_getitem: " << anf_out->fullname_with_scope() << " is not converted";
    }
  } else {
    // invalid tuple_getitem e.g. tuple_getitem(tuple_getitem())/tuple_getitem(depend())/tuple_getitem(make_tuple())
    MS_LOG(WARNING) << "Invalid tuple_getitem: " << anf_out->fullname_with_scope();
  }
}

void DfGraphConvertor::TraceOutput(const AnfNodePtr node) {
  AnfNodePtr anf_out = node;
  AnfNodePtr pre_node = nullptr;

  // trace Parameter node
  TraceOutputFromParameter(anf_out);
  // then trace cnode
  if (!node->isa<CNode>()) {
    return;
  }

  // trace tuple_getitem
  while (anf_out->isa<CNode>() && IsPrimitiveCNode(anf_out, prim::kPrimTupleGetItem)) {
    pre_node = anf_out;
    anf_out = anf_out->cast<CNodePtr>()->input(1);
  }
  // trace every element of make_tuple
  auto c = anf_out->cast<CNodePtr>();
  std::string name = "";
  if (anf_out->isa<CNode>()) {
    name = GetCNodeFuncName(c);
  }

  if (name == "make_tuple") {
    for (unsigned int i = 1; i < c->inputs().size(); i++) {
      TraceOutput(c->input(i));
    }
  } else if (name == "depend") {
    if (c->inputs().size() < 3) {  // "depend" primitive have 3 inputs
      MS_LOG(EXCEPTION) << "length of inputs is " << c->inputs().size() << ", which is less than 3";
    }
    TraceOutput(c->input(1));
  } else if (name == "tuple_getitem") {
    TraceOutputFromTupleGetItem(anf_out);
  } else {
    // add outputs;
    auto op = Convert(anf_out);
    std::string index;
    if (op != nullptr) {
      if ((pre_node != nullptr) && IsPrimitiveCNode(pre_node, prim::kPrimTupleGetItem)) {
        auto item = out_handle_cache_.find(pre_node.get());
        if (item != out_handle_cache_.end()) {
          index = item->second.out;
        } else {
          MS_LOG(WARNING) << "Can't get operater: " << anf_out->fullname_with_scope() << " 's output item";
        }
      }
      MS_LOG(INFO) << "Add graph output: " << anf_out->fullname_with_scope() << ":" << index;
      graph_outputs_.emplace_back(make_pair(*op, index));
    }
  }
}

void DfGraphConvertor::TraceOutputFromParameter(const AnfNodePtr &anf_out) {
  if (anf_out->isa<Parameter>()) {
    MS_LOG(INFO) << "Add graph output: " << anf_out->fullname_with_scope();
    auto it = out_handle_cache_.find(anf_out.get());
    if (it != out_handle_cache_.end()) {
      // For dataset graph mode, input parameter is converted to a "iterator_get_next:yn" OutHandler.
      OutHandler handle = it->second;
      auto op = handle.op;
      MS_LOG(INFO) << "op name: " << op->GetName() << ", op type: " << op->GetOpType() << ", out_name: " << handle.out;
      graph_outputs_.emplace_back(make_pair(*op, handle.out));
    } else {
      // common parameter case
      auto op = Convert(anf_out);
      if (op != nullptr) {
        MS_LOG(INFO) << "op name: " << op->GetName() << ", op type: " << op->GetOpType();
        graph_outputs_.emplace_back(std::make_pair(*op, ""));
      }
    }
  }
}

void SetupDatasetIterGetNextNode(const OperatorPtr &op) {
  if (ConfigManager::GetInstance().dataset_mode() == DS_SINK_MODE) {
    DatasetGraphParam param = ConfigManager::GetInstance().dataset_param();
    size_t output_num = param.ge_types().size();
    MS_LOG(INFO) << "Set iterator_getnext op's output num = " << output_num << ".";
    // set iterator_getnext op's output num
    shared_ptr<ge::op::GetNext> iter_getnext = std::static_pointer_cast<ge::op::GetNext>(op);
    (void)iter_getnext->create_dynamic_output_y(static_cast<unsigned int>(output_num));

    for (uint32_t i = 0; i < output_num; i++) {
      ge::TensorDesc desc(GeShape(param.shapes()[i]), ge::FORMAT_NCHW, (ge::DataType)param.ge_types()[i]);
      // we don't SetRealDimCnt here since GE do not use this output's real-dim
      (void)iter_getnext->update_dynamic_output_desc_y((i), desc);
    }
  }
  return;
}

DfGraphConvertor &DfGraphConvertor::BuildGraph() {
  SetupDatasetIterGetNextNode(dataset_iter_getnext_);

  if (error_ != 0) {
    return *this;
  }

  // update tuple_out_handle_cache_
  for (auto it : tuple_out_handle_cache_) {
    std::size_t len = it.second->size();
    for (std::size_t i = 0; i < len; i++) {
      OutHandler handle = (*it.second)[i];
      if (handle.op) {
        string name = handle.op->GetName();
        if (vars_.count(name)) {
          OperatorPtr new_op = vars_[name];
          if (new_op != nullptr) {
            MS_LOG(INFO) << "update tuple_out_handle_cache_ " << name;
            (*it.second)[i] = OutHandler(new_op, handle.out);
          }
        }
      }
    }
  }

  // set up dependices
  MS_LOG(DEBUG) << "set up dependices";
  std::vector<AnfNodePtr> nodes = ::mindspore::TopoSort(anf_graph_->get_return());
  for (auto &it : nodes) {
    SetNodeInput(it);
    SetOpControlInput(it);
    UpdateOpDesc(it);
  }

  if (error_ == 0) {
    df_graph_ = make_shared<DfGraph>(anf_graph_->ToString());
  } else {
    return *this;
  }

  // set graph input according to the order from anf graph
  std::vector<Operator> inputs;
  if (ConfigManager::GetInstance().dataset_mode() == DS_SINK_MODE) {
    inputs.push_back(*dataset_iter_getnext_);
  } else {
    auto params = anf_graph_->parameters();
    int index = 0;
    for (auto &it : params) {
      auto name = std::static_pointer_cast<Parameter>(it)->name();
      //  the parameters which has not been converted to var
      if (vars_.find(name) == vars_.end()) {
        auto op = Convert(it);
        MS_EXCEPTION_IF_NULL(op);
        MS_LOG(INFO) << "add not var input " << it->ToString() << ", index " << index;
        if (op == nullptr) {
          MS_LOG(ERROR) << "Convert graph failed!";
          return *this;
        }
        UpdateDataOpDesc(it, op);

        MS_LOG(INFO) << "add input " << it->ToString() << ", index " << index;
        (void)std::static_pointer_cast<Data>(op)->set_attr_index(index++);
        inputs.push_back(*op);
      } else if (vars_[name] != nullptr) {
        MS_LOG(INFO) << "add var input " << it->ToString();
        auto op = Convert(it);
        MS_EXCEPTION_IF_NULL(op);
        inputs.push_back(*op);
      }
    }
  }

  // Add const nodes as graph input for some operator work with constant
  std::transform(graph_const_inputs_.begin(), graph_const_inputs_.end(), std::back_inserter(inputs),
                 [](OperatorPtr x) { return *x; });

  MS_LOG(INFO) << "set graph input num: " << inputs.size();
  (void)df_graph_->SetInputs(inputs);

  // set graph output
  // set the value of finale return apply node as the output of dataflow graph
  MS_LOG(DEBUG) << "set output";
  graph_outputs_.clear();
  TraceOutput(anf_graph_->get_return()->input(1));
  MS_LOG(INFO) << "set graph output num: " << graph_outputs_.size();
  (void)df_graph_->SetOutputs(graph_outputs_);

  compute_sout_ << "}" << endl;
  // For the graph(e.g. eval_subgraph) whose IterNum is 1, donot set NeedIteration flag.
  if (ConfigManager::GetInstance().iter_num() > 1) {
    df_graph_->SetNeedIteration(true);
  }
  return *this;
}

void DfGraphConvertor::UpdateDataOpDesc(const AnfNodePtr &it, const OperatorPtr &op) const {
  auto node = std::static_pointer_cast<AnfNode>(it);
  if (node == nullptr) {
    MS_LOG(ERROR) << "Update data op descriptor failed! Invalid node.";
    return;
  }
  auto normal_shape_ptr = dyn_cast<abstract::Shape>(node->Shape());
  vector<int> shape;
  if (normal_shape_ptr == nullptr) {
    MS_LOG(INFO) << "Invalid shape to update data op descriptor.";
    return;
  }
  shape = normal_shape_ptr->shape();
  if (node->Type() == nullptr) {
    MS_LOG(INFO) << "Invalid type to update data op descriptor.";
    return;
  }
  TypeId me_type = node->Type()->type_id();
  if (kObjectTypeTensorType == me_type) {
    me_type = dyn_cast<TensorType>(node->Type())->element()->type_id();
  }
  std::ostringstream buf;
  buf << "[" << shape << "]";
  MS_LOG(INFO) << "input shape is " << buf.str() << ", type is " << me_type;
  auto desc = TransformUtil::GetGeTensorDesc(shape, me_type, "NCHW");
  if (desc == nullptr) {
    MS_LOG(ERROR) << "Update data op descriptor failed! TensorDesc is null.";
  } else {
    (void)std::static_pointer_cast<Data>(op)->update_input_desc_data(*desc);
    (void)std::static_pointer_cast<Data>(op)->update_output_desc_out(*desc);
  }
}

DfGraphPtr DfGraphConvertor::GetComputeGraph() { return df_graph_; }

DfGraphPtr DfGraphConvertor::GetInitGraph() { return init_graph_; }

DfGraphPtr DfGraphConvertor::GetSaveCheckpointGraph() { return save_ckp_graph_; }

DfGraphPtr DfGraphConvertor::GetBroadcastGraph() { return broadcast_graph_; }

void DfGraphConvertor::SetOpControlInput(const AnfNodePtr node) {
  if (control_depend_cache_.find(node.get()) == control_depend_cache_.end()) {
    return;
  }

  std::vector<ControlEdge> control_edges = control_depend_cache_[node.get()];
  if ((control_edges.empty())) {
    MS_LOG(ERROR) << "Get control depend node's src or dest operator failed";
    return;
  }

  for (auto &item : control_edges) {
    (void)item.dest_op->AddControlInput(*item.src_op);
  }
}

void DfGraphConvertor::SetOpInput(const OpAdapterPtr &adpt, const CNodePtr &node) {
  OperatorPtr src = Convert(node);
  auto &inputs = node->inputs();
  for (size_t i = 1; i < inputs.size(); i++) {
    auto pred = inputs[i];
    while (pred->isa<CNode>() && GetCNodeFuncName(pred->cast<CNodePtr>()) == "depend") {
      pred = pred->cast<CNodePtr>()->input(1);
    }
    // skip the None input
    if (IsValueNode<None>(pred)) {
      continue;
    }
    // find in out_hadnle_cache_ first
    auto it = out_handle_cache_.find(pred.get());
    if (it != out_handle_cache_.end()) {
      int ret = adpt->setInput(src, SizeToInt(i), it->second);
      if (ret == 0) {
        if (pred->isa<CNode>() && GetCNodeFuncName(pred->cast<CNodePtr>()) == "tuple_getitem") {
          compute_sout_ << op_draw_name_[pred->cast<CNodePtr>()->input(1).get()] << " -> " << op_draw_name_[node.get()]
                        << ":" << i << endl;
        } else if (pred->isa<Parameter>()) {
          compute_sout_ << op_draw_name_[pred.get()] << " -> " << op_draw_name_[node.get()] << ":" << i << endl;
        } else {
          // don't draw anything.
          MS_LOG(INFO) << "DRAW_GE_GRAPH: Shouldn't have this case.";
        }
        AddGraphConstInput(it->second.op);
      }
    } else if (tuple_out_handle_cache_.find(pred.get()) != tuple_out_handle_cache_.end()) {
      std::shared_ptr<std::vector<OutHandler>> handler_vec = tuple_out_handle_cache_[pred.get()];
      int ret = adpt->setInput(src, SizeToInt(i), handler_vec);
      if ((ret == 0) && pred->isa<CNode>() && (pred->cast<CNodePtr>()->inputs().size() == handler_vec->size() + 1)) {
        for (unsigned int j = 0; j < handler_vec->size(); j++) {
          compute_sout_ << op_draw_name_[pred->cast<CNodePtr>()->input(j + 1).get()] << " -> "
                        << op_draw_name_[node.get()] << ":" << i << endl;
          AddGraphConstInput(handler_vec->at(j).op);
        }
      } else {
        MS_LOG(WARNING) << "Convert tuple node setInput failed : " << node->ToString();
      }
    } else {
      auto op = Convert(pred);
      int ret = adpt->setInput(src, SizeToInt(i), op);
      if (ret == 0) {
        compute_sout_ << op_draw_name_[pred.get()] << " -> " << op_draw_name_[node.get()] << ":" << i << endl;
        AddGraphConstInput(op);
      }
    }
  }
}

void DfGraphConvertor::AddGraphConstInput(const OperatorPtr &op) {
  if (op->GetOpType() == "Constant") {
    graph_const_inputs_.push_back(op);
  }
}

void DfGraphConvertor::SetNodeInput(const AnfNodePtr node) {
  if (!node->isa<CNode>()) {
    return;
  }
  if (op_cache_.find(node.get()) == op_cache_.end()) {
    return;
  }
  auto cnode = node->cast<CNodePtr>();
  OpAdapterPtr adpt = FindAdapter(cnode, training_);
  if (adpt == nullptr) {
    error_ = NOT_FOUND;
    return;
  }

  // get Operator from op_cache_, use adapter to set Inputs
  DfGraphConvertor::SetOpInput(adpt, cnode);
}

// Update GE op's shape and type info
void DfGraphConvertor::UpdateOpDesc(const AnfNodePtr node) {
  if (nullptr == node || !node->isa<CNode>()) {
    return;
  }

  if (op_cache_.find(node.get()) == op_cache_.end()) {
    return;
  }

  OpAdapterPtr adpt = FindAdapter(node, training_);
  if (adpt == nullptr) {
    error_ = NOT_FOUND;
    return;
  }

  // get Operator from op_cache_
  OperatorPtr op = Convert(node);

  adpt->updateOutputDesc(op, node->Shape(), node->Type(), node);
}

OperatorPtr DfGraphConvertor::Convert(const AnfNodePtr node) {
  if (node == nullptr) {
    MS_LOG(ERROR) << "node is nullptr";
    error_ = NOT_FOUND;
    return nullptr;
  }
  // find in cache
  if (op_cache_.count(node.get())) {
    return op_cache_[node.get()];
  }

  // do not convert primitive node
  if (IsValueNode<Primitive>(node)) {
    return nullptr;
  }

  // convert a new one
  if (node->isa<CNode>()) {
    return ConvertCNode(node->cast<CNodePtr>());
  }
  if (node->isa<Parameter>()) {
    return ConvertParameter(node);
  }
  if (node->isa<ValueNode>()) {
    return ConvertValueNode(node->cast<ValueNodePtr>());
  }

  MS_LOG(ERROR) << "Invalide AnfNode";
  error_ = INVALID_ARGUMENT;
  return nullptr;
}

void DfGraphConvertor::ConvertMakeTuple(const CNodePtr node) {
  std::shared_ptr<std::vector<OutHandler>> tuple_items = std::make_shared<std::vector<OutHandler>>();
  // convert each tuple item to a OutHandler
  for (size_t i = 1; i < node->inputs().size(); i++) {
    AnfNodePtr item = node->input(i);
    OperatorPtr op = Convert(item);
    if (op != nullptr) {
      tuple_items->emplace_back(OutHandler(op, ""));
    } else if (out_handle_cache_.find(item.get()) != out_handle_cache_.end()) {
      tuple_items->push_back(out_handle_cache_[item.get()]);
    } else {
      MS_LOG(WARNING) << "This anf node is not supported as a tuple item : " << item->ToString();
      return;
    }
  }

  tuple_out_handle_cache_[node.get()] = tuple_items;
}

AnfNodePtr DfGraphConvertor::TraceTupleGetItem(const CNodePtr &node, unsigned int *index) {
  const int TUPLE_GET_ITEM_INDEX = 2;
  if (node->inputs().size() < 3) {  // "tuple_getitem" primitive must have 3 inputs
    MS_LOG(EXCEPTION) << "length of inputs of TupleGetItem is less than 3";
  }
  auto index_node = node->inputs()[TUPLE_GET_ITEM_INDEX];
  if (!index_node->isa<ValueNode>()) {
    error_ = INVALID_ARGUMENT;
    MS_LOG(EXCEPTION) << "can't convert get item with non-constant index";
  }
  *index = IntToUint(GetValue<int>(GetValueNode(index_node)));
  return node->inputs()[1];
}

AnfNodePtr DfGraphConvertor::TraceDepend(const CNodePtr &node) {
  auto cnode = node->cast<CNodePtr>();
  if (cnode->inputs().size() < 3) {  // "depend" primitive have 3 inputs
    MS_LOG(EXCEPTION) << "length of inputs of depend is less than 3";
  }
  return cnode->inputs()[1];
}

AnfNodePtr DfGraphConvertor::TraceMakeTuple(const CNodePtr &node, unsigned int index) {
  if (index + 1 >= node->inputs().size()) {
    MS_LOG(EXCEPTION) << "length of make_tuple is less than index: " << index;
  }
  return node->inputs()[index + 1];
}

OutHandler DfGraphConvertor::GetHandler(const AnfNodePtr &node, const std::stack<unsigned int> &index_stack,
                                        AnfNode *const draw_index) {
  if (node == nullptr) {
    MS_LOG(ERROR) << "Get nullptr while trace real op";
    return OutHandler(nullptr, "");
  }
  std::ostringstream ss;
  ss << "op" << node.get();
  if (index_stack.empty()) {
    op_draw_name_[draw_index] = ss.str();
    return OutHandler(Convert(node), "");
  } else {
    OpAdapterPtr adpt = FindAdapter(node, training_);
    if (nullptr == adpt) {
      MS_LOG(ERROR) << "Can not get node output as adpt is nullptr!";
      error_ = NOT_FOUND;
      return OutHandler(nullptr, "");
    }
    OperatorPtr op = Convert(node);
    if (op == nullptr) {
      error_ = NOT_FOUND;
      MS_LOG(ERROR) << "Can not convert node for trace real op";
      return OutHandler(nullptr, "");
    }
    op_draw_name_[draw_index] = ss.str();
    return adpt->getOutput(Convert(node), UintToInt(index_stack.top()));
  }
}

// get the real operator through maketuple tuple_getitem depend
OutHandler DfGraphConvertor::TraceRealOp(AnfNodePtr node) {
  bool flag = IsPrimitiveCNode(node, prim::kPrimTupleGetItem) || IsPrimitiveCNode(node, prim::kPrimMakeTuple) ||
              IsPrimitiveCNode(node, prim::kPrimDepend);
  std::stack<unsigned int> index_stack;
  auto draw_index = node.get();
  while (flag) {
    flag = false;
    if (IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
      unsigned int index;
      node = TraceTupleGetItem(node->cast<CNodePtr>(), &index);
      index_stack.push(index);
      flag = true;
    } else if (IsPrimitiveCNode(node, prim::kPrimMakeTuple)) {
      if (index_stack.empty()) {
        MS_LOG(ERROR) << "TraceRealOp find a make_tuple node";
        return OutHandler(nullptr, "");
      } else {
        node = TraceMakeTuple(node->cast<CNodePtr>(), index_stack.top());
        index_stack.pop();
        flag = true;
      }
    } else if (IsPrimitiveCNode(node, prim::kPrimDepend)) {
      node = TraceDepend(node->cast<CNodePtr>());
      flag = true;
    }
  }
  return GetHandler(node, index_stack, draw_index);
}

void DfGraphConvertor::ConvertTupleGetItem(const CNodePtr node) {
  auto handle = TraceRealOp(node);
  if (handle.op == nullptr) {
    MS_LOG(ERROR) << "Failed to trace tuple get item";
    return;
  }
  out_handle_cache_[node.get()] = handle;
}

// Get the real op for tuple_getitem through make tuple, or depend
AnfNodePtr DfGraphConvertor::GetRealOpNode(AnfNodePtr node) {
  const int TUPLE_GET_ITEM_INDEX = 2;
  if (IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
    auto node_inputs = node->cast<CNodePtr>()->inputs();
    if (node_inputs.size() != 3) {  // "tuple_getitem" primitive must have 3 inputs
      MS_LOG(ERROR) << "tuple get item node not correct!";
      error_ = FAILED;
      return node;
    }
    MS_EXCEPTION_IF_NULL(node_inputs[TUPLE_GET_ITEM_INDEX]);
    if (!node_inputs[TUPLE_GET_ITEM_INDEX]->isa<ValueNode>()) {
      error_ = INVALID_ARGUMENT;
      MS_LOG(EXCEPTION) << "can't convert get item with non-constant index";
    }
    auto value_ptr = GetValueNode(node_inputs[TUPLE_GET_ITEM_INDEX])->cast<Int32ImmPtr>();
    if (value_ptr == nullptr) {
      MS_LOG(ERROR) << "Can not convert get item as value is nullptr!";
      error_ = FAILED;
      return node;
    }
    int index = value_ptr->value();

    // make_tuple apply inputs:make_tuple, [tuple_items,]
    if (IsPrimitiveCNode(node_inputs[1], prim::kPrimMakeTuple)) {
      auto tuple_inputs = node->cast<CNodePtr>()->inputs();
      if (tuple_inputs.size() < IntToSize(index + 1)) {
        MS_LOG(ERROR) << "make tuple input items node not correct! size:" << tuple_inputs.size()
                      << ", item index:" << index;
        error_ = FAILED;
        return node;
      }
      return GetRealOpNode(tuple_inputs[IntToSize(index + 1)]);
    }
    return GetRealOpNode(node_inputs[1]);
  }

  // depend apply inputs: depend,output,depended_node
  if (IsPrimitiveCNode(node, prim::kPrimDepend)) {
    auto depend_inputs = node->cast<CNodePtr>()->inputs();
    if (depend_inputs.size() != 3) {  // "depend" primitive have 3 inputs
      MS_LOG(ERROR) << "depend input items not correct";
      error_ = FAILED;
      return node;
    }
    return GetRealOpNode(depend_inputs[1]);
  }
  return node;
}

// convert the anf node to corresponding operator list
std::vector<OperatorPtr> DfGraphConvertor::ConvertDependNode(const AnfNodePtr node) {
  if (IsPrimitiveCNode(node, prim::kPrimMakeTuple)) {
    std::vector<OperatorPtr> op_lists;
    auto node_inputs = node->cast<CNodePtr>()->inputs();
    for (size_t index = 1; index < node_inputs.size(); index++) {
      auto op = Convert(GetRealOpNode(node_inputs[index]));
      if (op == nullptr) {
        MS_LOG(ERROR) << "Convert control depend node to operator failed";
        error_ = FAILED;
        return std::vector<OperatorPtr>({});
      }
      op_lists.push_back(op);
    }
    return op_lists;
  }

  auto op = Convert(GetRealOpNode(node));
  if (op == nullptr) {
    MS_LOG(ERROR) << "Convert control depend node to operator failed";
    error_ = FAILED;
    return std::vector<OperatorPtr>({});
  }
  return std::vector<OperatorPtr>({op});
}

// get the anf node list for depend
std::vector<AnfNodePtr> DfGraphConvertor::GetDependNodes(const AnfNodePtr &node) {
  std::vector<AnfNodePtr> nodes;
  // for make tuple, should control depend on the tuple items
  if (IsPrimitiveCNode(node, prim::kPrimMakeTuple)) {
    auto node_inputs = node->cast<CNodePtr>()->inputs();
    for (size_t index = 1; index < node_inputs.size(); index++) {
      nodes.push_back(GetRealOpNode(node_inputs[index]));
    }
    return nodes;
  }

  // for parameter ,find the apply that used the parameter as the control depended node
  if (node->isa<Parameter>()) {
    auto uses = node->func_graph()->manager()->node_users()[node];
    for (auto &use : uses) {
      auto use_node = use.first;
      if ((use_node->isa<CNode>()) && (!IsPrimitiveCNode(use_node, prim::kPrimControlDepend))) {
        nodes.push_back(GetRealOpNode(use_node));
      }
    }
    return nodes;
  }
  nodes.push_back(GetRealOpNode(node));
  return nodes;
}

void DfGraphConvertor::DrawControlDepend(const AnfNodePtr &src_node, const AnfNodePtr &dest_node) {
#ifdef DRAW_GE_GRAPH
  auto src_depend_nodes = GetDependNodes(src_node);
  auto dst_depend_nodes = GetDependNodes(dest_node);
  if (src_depend_nodes.size() == 1 && dst_depend_nodes.size() > 1) {
    for (auto &item : dst_depend_nodes) {
      compute_sout_ << op_draw_name_[src_depend_nodes[0].get()] << " -> " << op_draw_name_[item.get()]
                    << "[style=\"dotted\"]" << endl;
    }
  } else if (src_depend_nodes.size() > 1 && dst_depend_nodes.size() == 1) {
    for (auto &item : src_depend_nodes) {
      compute_sout_ << op_draw_name_[item.get()] << " -> " << op_draw_name_[dst_depend_nodes[0].get()]
                    << "[style=\"dotted\"]" << endl;
    }
  } else if (src_depend_nodes.size() == 1 && dst_depend_nodes.size() == 1) {
    compute_sout_ << op_draw_name_[src_depend_nodes[0].get()] << " -> " << op_draw_name_[dst_depend_nodes[0].get()]
                  << "[style=\"dotted\"]" << endl;
  }
#endif
}

void DfGraphConvertor::GetDependOnParameterUse(const CNodePtr &node, const AnfNodePtr &src_node,
                                               const AnfNodePtr &dest_node,
                                               const std::shared_ptr<std::vector<OperatorPtr>> &src_ops_list,
                                               const std::shared_ptr<std::vector<OperatorPtr>> &dst_ops_list) {
  if (src_node->isa<Parameter>()) {
    auto uses = node->func_graph()->manager()->node_users()[src_node];
    for (auto &use : uses) {
      auto use_node = use.first;
      if ((use_node->isa<CNode>()) && (!IsPrimitiveCNode(use_node, prim::kPrimControlDepend)) &&
          (!IsPrimitiveCNode(use_node, prim::kPrimMakeTuple))) {
        auto converted_list = ConvertDependNode(use_node);
        src_ops_list->insert(src_ops_list->end(), converted_list.begin(), converted_list.end());
      }
    }
  }

  if (dest_node->isa<Parameter>()) {
    auto uses = node->func_graph()->manager()->node_users()[dest_node];
    for (auto &use : uses) {
      auto use_node = use.first;
      if ((use_node->isa<CNode>()) && (!IsPrimitiveCNode(use_node, prim::kPrimControlDepend)) &&
          (!IsPrimitiveCNode(use_node, prim::kPrimMakeTuple))) {
        auto converted_list = ConvertDependNode(use_node);
        dst_ops_list->insert(dst_ops_list->end(), converted_list.begin(), converted_list.end());
      }
    }
  }
}

bool DfGraphConvertor::GetControlDependList(const CNodePtr &node,
                                            const std::shared_ptr<std::vector<OperatorPtr>> &src_ops_list,
                                            const std::shared_ptr<std::vector<OperatorPtr>> &dst_ops_list) {
  const int CONTROL_DEPEND_INDEX = 0;
  const int SRC_NODE_INDEX = 1;
  const int DEST_NODE_INDEX = 2;
  const int DEPEND_MODE_NORMAL_USE = 0;
  const int DEPEND_MODE_ON_PARAMETER_USE = 1;

  auto node_inputs = node->inputs();
  if (node_inputs.size() <= DEST_NODE_INDEX) {
    MS_LOG(WARNING) << "Control depend node input size error";
    return false;
  }
  auto src_node = node_inputs[SRC_NODE_INDEX];
  auto dest_node = node_inputs[DEST_NODE_INDEX];
  if ((src_node == nullptr) || (dest_node == nullptr)) {
    MS_LOG(ERROR) << "Control depend node miss src or dest node";
    error_ = FAILED;
    return false;
  }
  AnfNodePtr fn = node_inputs[CONTROL_DEPEND_INDEX];
  PrimitivePtr prim_ptr = GetValueNode<PrimitivePtr>(fn);
  ValuePtr mode_ptr = prim_ptr->GetAttr("depend_mode");
  int depend_mode = DEPEND_MODE_NORMAL_USE;
  if (mode_ptr != nullptr) {
    auto mode_int = mode_ptr->cast<Int32ImmPtr>();
    MS_EXCEPTION_IF_NULL(mode_int);
    depend_mode = mode_int->value();
    MS_LOG(DEBUG) << "depend_mode = " << depend_mode;
  }
  if (depend_mode == DEPEND_MODE_ON_PARAMETER_USE) {
    GetDependOnParameterUse(node, src_node, dest_node, src_ops_list, dst_ops_list);
  }

  if (src_node->isa<CNode>()) {
    auto converted_list = ConvertDependNode(src_node);
    src_ops_list->insert(src_ops_list->end(), converted_list.begin(), converted_list.end());
  }

  if (dest_node->isa<CNode>()) {
    auto converted_list = ConvertDependNode(dest_node);
    dst_ops_list->insert(dst_ops_list->end(), converted_list.begin(), converted_list.end());
  }
  if (src_ops_list->empty() || dst_ops_list->empty()) {
    MS_LOG(WARNING) << "Control depend node's src or dest node is not a apply node, ignore it";
    error_ = SUCCESS;
  }
  return true;
}

void DfGraphConvertor::ConvertControlDependNode(const CNodePtr node) {
  const int SRC_NODE_INDEX = 1;
  const int DEST_NODE_INDEX = 2;
  if (control_depend_cache_.find(node.get()) != control_depend_cache_.end()) {
    return;
  }
  auto node_inputs = node->inputs();
  if (node_inputs.size() <= DEST_NODE_INDEX) {
    MS_LOG(WARNING) << "Control depend node input size error";
    return;
  }
  auto src_node = node_inputs[SRC_NODE_INDEX];
  auto dest_node = node_inputs[DEST_NODE_INDEX];
  if ((src_node == nullptr) || (dest_node == nullptr)) {
    MS_LOG(ERROR) << "Control depend node miss src or dest node";
    error_ = FAILED;
    return;
  }
  std::shared_ptr<std::vector<OperatorPtr>> src_ops_list = std::make_shared<std::vector<OperatorPtr>>();
  std::shared_ptr<std::vector<OperatorPtr>> dst_ops_list = std::make_shared<std::vector<OperatorPtr>>();
  if (!GetControlDependList(node, src_ops_list, dst_ops_list)) {
    MS_LOG(ERROR) << "Get depend list failed";
    error_ = FAILED;
    return;
  }
  std::vector<ControlEdge> control_edges;
  if (src_ops_list->size() == 1 && dst_ops_list->size() > 1) {
    (void)std::transform(dst_ops_list->begin(), dst_ops_list->end(), std::back_inserter(control_edges),
                         [src_ops_list](const OperatorPtr &op) -> ControlEdge {
                           return {(*src_ops_list)[0], op};
                         });
  } else if (src_ops_list->size() > 1 && dst_ops_list->size() == 1) {
    (void)std::transform(src_ops_list->begin(), src_ops_list->end(), std::back_inserter(control_edges),
                         [dst_ops_list](const OperatorPtr &op) -> ControlEdge {
                           return {op, (*dst_ops_list)[0]};
                         });
  } else if (src_ops_list->size() == 1 && dst_ops_list->size() == 1) {
    control_edges.push_back({(*src_ops_list)[0], (*dst_ops_list)[0]});
  } else {
    MS_LOG(ERROR) << "Convert control depend node to operator failed, depend src:" << src_ops_list->size()
                  << " -> dst:" << dst_ops_list->size();
    error_ = FAILED;
    return;
  }
  control_depend_cache_[node.get()] = control_edges;

#ifdef DRAW_GE_GRAPH
  DrawControlDepend(src_node, dest_node);
#endif
}

bool DfGraphConvertor::CheckCNode(const std::string &name, const CNodePtr node) {
  // ignore apply node of return
  if (name == "return" || name == "depend") {
    return false;
  }

  // make_tuple is used for a dynamic_input, convert it to a vector of OutHandlers
  if (name == "make_tuple") {
    ConvertMakeTuple(node);
    return false;
  }

  // As for nodes with multi outputs, convert tuple_getitem to OutHandle
  if (name == "tuple_getitem") {
    ConvertTupleGetItem(node);
    return false;
  }

  if (name == "ControlDepend") {
    ConvertControlDependNode(node);
    return false;
  }

  return true;
}

OperatorPtr DfGraphConvertor::ConvertCNode(const CNodePtr node) {
  std::string name = GetCNodeFuncName(node);
  if (!CheckCNode(name, node)) {
    return nullptr;
  }

  // get corresponding OpAdapter
  OpAdapterPtr adpt = FindAdapter(node, training_);
  if (adpt == nullptr) {
    error_ = NOT_FOUND;
    return nullptr;
  }

  // get operator
  OperatorPtr op = nullptr;
  auto it_op = op_cache_.find(node.get());
  if (it_op != op_cache_.end()) {
    op = it_op->second;
  } else {
    op = adpt->generate(node);
  }

  // set attribute for primitive
  (void)adpt->setAttr(op, node);

  // add into cache
  (void)op_cache_.insert(std::make_pair(node.get(), op));

  DrawCNode(node, adpt);

  return op_cache_[node.get()];
}

OperatorPtr DfGraphConvertor::ConvertParameter(const AnfNodePtr node) {
  // convert Parameter in ANF to variable in DataFlow
  auto op = FindAdapter(node, training_)->generate(node);
  op_cache_[node.get()] = op;

  // build index for parameter using name
  std::string name = std::static_pointer_cast<Parameter>(node)->name();
  params_[name] = node;

  std::ostringstream ss;
  ss << "op" << node.get();
  op_draw_name_[node.get()] = ss.str();
  compute_sout_ << ss.str() << "[shape=octagon, label=\"" << name << "\"]" << endl;
  return op_cache_[node.get()];
}

Status DfGraphConvertor::TryConvertValueNodeToMultiConst(const ValueNodePtr node) {
  MS_EXCEPTION_IF_NULL(node);
  ValuePtr value = node->value();
  MS_EXCEPTION_IF_NULL(value);
  if (!value->isa<ValueList>() && !value->isa<ValueTuple>()) {
    return FAILED;
  }

  auto vec = value->isa<ValueTuple>() ? value->cast<ValueTuplePtr>()->value() : value->cast<ValueListPtr>()->value();
  if (vec.empty()) {
    return FAILED;
  }

  std::shared_ptr<std::vector<OutHandler>> tuple_items = std::make_shared<std::vector<OutHandler>>();
  for (size_t i = 0; i < vec.size(); i++) {
    MS_EXCEPTION_IF_NULL(vec[i]);
    if (vec[i]->isa<MeTensor>()) {
      GeTensorPtr ge_tensor = transform::TransformUtil::ConvertTensor(vec[i]->cast<MeTensorPtr>(), kOpFormat_NCHW);
      auto const_op = std::make_shared<Constant>(node->fullname_with_scope() + "/const/inputs/" + std::to_string(i));
      (void)const_op->set_attr_value(*ge_tensor);
      (void)const_op->update_output_desc_y(ge_tensor->GetTensorDesc());
      tuple_items->emplace_back(OutHandler(const_op, ""));
    } else {
      return FAILED;
    }
  }
  if (tuple_items->empty()) {
    return FAILED;
  }

  tuple_out_handle_cache_[node.get()] = tuple_items;
  return SUCCESS;
}

OperatorPtr DfGraphConvertor::ConvertValueNode(const ValueNodePtr node) {
  // convert valuenode in ANF to Const in DataFlow
  // find paramerte referenced by SymbolicKeyInstance of valuenode
  std::ostringstream ss;
  ss << "op" << node.get();
  op_draw_name_[node.get()] = ss.str();
  compute_sout_ << ss.str() << "[label= \"" << node->value()->ToString() << "\" shape=ellipse]" << endl;

  if (TryConvertValueNodeToMultiConst(node) == SUCCESS) {
    MS_LOG(INFO) << "Convert value node to multi Constant OP success";
    return nullptr;
  }

  OpAdapterPtr adpt = FindAdapter(node, training_);
  if (adpt == nullptr) {
    error_ = NOT_FOUND;
    return nullptr;
  }
  auto op = adpt->generate(node);
  // set const's attrs
  if (adpt->setAttr(op, "value", node->value()) != 0) {
    MS_LOG(WARNING) << "set attr value for const failed";
  }

#if (defined ENABLE_GE)
  auto const_op = std::static_pointer_cast<Constant>(op);
  if (const_op == nullptr) {
    MS_LOG(ERROR) << "Get Constant operator failed";
    return nullptr;
  }
  auto ge_tensor = const_op->get_attr_value();
  auto ge_desc = ge_tensor.GetTensorDesc();
  (void)const_op->update_output_desc_y(ge_desc);
#endif

  op_cache_[node.get()] = op;
  return op_cache_[node.get()];
}

void DfGraphConvertor::DrawCNode(const CNodePtr node, const OpAdapterPtr adpt) {
  if (nullptr == adpt || nullptr == node) {
    MS_LOG(ERROR) << "Failed to draw apply node as adpt or node is nullptr!";
    return;
  }
  std::ostringstream ss;
  ss << "op" << node.get();
  op_draw_name_[node.get()] = ss.str();

  compute_sout_ << ss.str() << "[label=<";
  compute_sout_ << "<table border='1' cellborder='1'>" << endl;

  auto input_map = adpt->getInputMap();
  auto dyn_input_map = adpt->getDynInputMap();
  if (input_map.size() + dyn_input_map.size() > 0) {
    compute_sout_ << "<tr>";
    for (auto &it : input_map) {
      compute_sout_ << "<td port='" << it.first << "'>" << it.second.name << "</td>";
    }
    for (auto &it : dyn_input_map) {
      compute_sout_ << "<td port='" << it.first << "'>" << it.second.name << "</td>";
    }
    compute_sout_ << "</tr>" << endl;
  }

  compute_sout_ << "<tr><td colspan=\"" << (input_map.size() + dyn_input_map.size()) << "\">\"" << node->ToString()
                << ":" << GetCNodeFuncName(node) << "\"</td></tr>" << endl;

  // print attrs' values
  auto atts = adpt->GetAttrsFromDrawGraph();
  for (auto &it : atts) {
    compute_sout_ << "<tr><td colspan=\"" << (input_map.size() + dyn_input_map.size()) << "\">\"" << it
                  << "\"</td></tr>";
  }

  adpt->clearAttrVect();

  compute_sout_ << "</table>> shape=plaintext]" << endl;
}
}  // namespace transform
}  // namespace mindspore
