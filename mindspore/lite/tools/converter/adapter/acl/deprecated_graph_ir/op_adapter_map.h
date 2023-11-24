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

#ifndef MINDSPORE_CCSRC_INCLUDE_TRANSFORM_GRAPH_IR_OP_ADAPTER_MAP_H_
#define MINDSPORE_CCSRC_INCLUDE_TRANSFORM_GRAPH_IR_OP_ADAPTER_MAP_H_

#include <memory>
#include <string>

#include "utils/hash_map.h"

namespace mindspore {
namespace transform {
constexpr const char kNameCustomOp[] = "Custom";
constexpr const char kNameConst[] = "Const";
constexpr const char kNameParam[] = "parameter";
constexpr const char kNameRandomUniform[] = "RandomUniform";
constexpr const char kNameUniformReal[] = "UniformReal";
constexpr const char kNameLogNormalReverse[] = "LogNormalReverse";
constexpr const char kNameSimpleMean[] = "SimpleMean";
constexpr const char kNameSimpleMeanGrad[] = "SimpleMeanGrad";
constexpr const char kNameAllReduce[] = "AllReduce";
constexpr const char kNameBroadcast[] = "Broadcast";
constexpr const char kNameBroadcastTo[] = "BroadcastTo";
constexpr const char kNameBroadcastToD[] = "BroadcastToD";
constexpr const char kNameBlackmanWindow[] = "BlackmanWindow";
constexpr const char kNameBartlettWindow[] = "BartlettWindow";
constexpr const char kNameAllgather[] = "AllGather";
constexpr const char kNameAllToAllv[] = "AllToAllv";
constexpr const char kNameReduceScatter[] = "ReduceScatter";
constexpr const char kNameReduceSum[] = "ReduceSum";
constexpr const char kNameIsFinite[] = "IsFinite";
constexpr const char kNameReciprocal[] = "Reciprocal";
constexpr const char kNameRsqrt[] = "Rsqrt";
constexpr const char kNameSqrt[] = "Sqrt";
constexpr const char kNameSquare[] = "Square";
constexpr const char kNameSquaredDifference[] = "SquaredDifference";
constexpr const char kNamePow[] = "Pow";
constexpr const char kNameBatchMatMul[] = "BatchMatMul";
constexpr const char kNameBatchMatMulV2[] = "BatchMatMulV2";
constexpr const char kNameBincount[] = "Bincount";
constexpr const char kNameStridedSlice[] = "StridedSlice";
constexpr const char kNameStridedSliceGrad[] = "StridedSliceGrad";
constexpr const char kNameExpandDims[] = "ExpandDims";
constexpr const char kNameLog[] = "Log";
constexpr const char kNameLogicalAnd[] = "LogicalAnd";
constexpr const char kNameLogicalNot[] = "LogicalNot";
constexpr const char kNameLogicalOr[] = "LogicalOr";
constexpr const char kNameListDiff[] = "ListDiff";
constexpr const char kNameExp[] = "Exp";
constexpr const char kNameLessEqual[] = "LessEqual";
constexpr const char kNameGreaterEqual[] = "GreaterEqual";
constexpr const char kNameApproximateEqual[] = "ApproximateEqual";
constexpr const char kNameEqual[] = "Equal";
constexpr const char kNameNotEqual[] = "NotEqual";
constexpr const char kNameFlattenGrad[] = "FlattenGrad";
constexpr const char kNameFillDiagonal[] = "FillDiagonal";
constexpr const char kNameEye[] = "Eye";
constexpr const char kNameConvolution[] = "Convolution";
constexpr const char kNameMaxPool3D[] = "MaxPool3D";
constexpr const char kNameMaxPool3DGrad[] = "MaxPool3DGrad";
constexpr const char kNameConv3DTranspose[] = "Conv3DTranspose";
constexpr const char kNameConv3D[] = "Conv3D";
constexpr const char kNameConv3DBackpropInput[] = "Conv3DBackpropInput";
constexpr const char kNameConv3DBackpropFilter[] = "Conv3DBackpropFilter";
constexpr const char kNameBiasAdd[] = "BiasAdd";
constexpr const char kNameMaxPoolGrad[] = "MaxPoolGrad";
constexpr const char kNameRsqrtGrad[] = "RsqrtGrad";
constexpr const char kNameSqrtGrad[] = "SqrtGrad";
constexpr const char kNameReciprocalGrad[] = "ReciprocalGrad";
constexpr const char kNameAvgPoolGrad[] = "AvgPoolGrad";
constexpr const char kNameAvgPoolGradD[] = "AvgPoolGradD";
constexpr const char kNameAvgPoolGradGe[] = "AvgPoolGradGe";
constexpr const char kNameMaxPoolGradWithArgmax[] = "MaxPoolGradWithArgmax";
constexpr const char kNameMaxPoolGradWithArgmaxV2[] = "MaxPoolGradWithArgmaxV2";
constexpr const char kNameApplyMomentum[] = "ApplyMomentum";
constexpr const char kNameDropoutDoMask[] = "DropoutDoMask";
constexpr const char kNameDropout2D[] = "Dropout2D";
constexpr const char kNameDropOutDoMaskV3[] = "DropOutDoMaskV3";
constexpr const char kNameDropOutDoMaskV3D[] = "DropOutDoMaskV3D";
constexpr const char kNameDropOutGenMaskV4[] = "DropOutGenMaskV4";
constexpr const char kNameResizeBilinear[] = "ResizeBilinear";
constexpr const char kNameResizeBilinearV2[] = "ResizeBilinearV2";
constexpr const char kNameResizeBilinearGrad[] = "ResizeBilinearGrad";
constexpr const char kNameZerosLike[] = "ZerosLike";
constexpr const char kNameOnesLike[] = "OnesLike";
constexpr const char kNameTruncatedNormal[] = "TruncatedNormal";
constexpr const char kNameSpaceToBatchND[] = "SpaceToBatchND";
constexpr const char kNameConfusionMatrix[] = "ConfusionMatrix";
constexpr const char kNameResizeNearestNeighbor[] = "ResizeNearestNeighbor";
constexpr const char kNameResizeNearestNeighborGrad[] = "ResizeNearestNeighborGrad";
constexpr const char kNameAdam[] = "Adam";
constexpr const char kNameApplyAdam[] = "ApplyAdam";
constexpr const char kNameApplyAdamD[] = "ApplyAdamD";
constexpr const char kNameApplyAdagrad[] = "ApplyAdagrad";
constexpr const char kNameApplyAdadelta[] = "ApplyAdadelta";
constexpr const char kNameApplyAdaMax[] = "ApplyAdaMax";
constexpr const char kNameApplyGradientDescent[] = "ApplyGradientDescent";
constexpr const char kNameApplyPowerSign[] = "ApplyPowerSign";
constexpr const char kNameApplyPowerSignD[] = "ApplyPowerSignD";
constexpr const char kNameApplyProximalGradientDescent[] = "ApplyProximalGradientDescent";
constexpr const char kNameExtractImagePatches[] = "ExtractImagePatches";
constexpr const char kNameReLU6[] = "ReLU6";
constexpr const char kNameReLU6Grad[] = "ReLU6Grad";
constexpr const char kNameSoftplus[] = "Softplus";
constexpr const char kNameSoftplusGrad[] = "SoftplusGrad";
constexpr const char kNameElu[] = "Elu";
constexpr const char kNameEluGrad[] = "EluGrad";
constexpr const char kNameTensorScatterUpdate[] = "TensorScatterUpdate";
constexpr const char kNameTensorScatterElements[] = "TensorScatterElements";
constexpr const char kNameTensorScatterAdd[] = "TensorScatterAdd";
constexpr const char kNameTriu[] = "Triu";
constexpr const char kNameScatterElements[] = "ScatterElements";
constexpr const char kNameNonZero[] = "NonZero";
constexpr const char kNameNonZeroWithValue[] = "NonZeroWithValue";
constexpr const char kNameNonZeroWithValueShape[] = "NonZeroWithValueShape";
constexpr const char kNameScatterUpdate[] = "ScatterUpdate";
constexpr const char kNameScatterNdUpdate[] = "ScatterNdUpdate";
constexpr const char kNameNMSWithMask[] = "NMSWithMask";
constexpr const char kNameCheckValid[] = "CheckValid";
constexpr const char kNameSmoothL1Loss[] = "SmoothL1Loss";
constexpr const char kNameSmoothL1LossGrad[] = "SmoothL1LossGrad";
constexpr const char kNameSGD[] = "SGD";
constexpr const char kNameSigmoidCrossEntropyWithLogits[] = "SigmoidCrossEntropyWithLogits";
constexpr const char kNameSigmoidCrossEntropyWithLogitsGrad[] = "SigmoidCrossEntropyWithLogitsGrad";
constexpr const char kNameSigmoidCrossEntropyWithLogitsV2[] = "BCEWithLogitsLoss";
constexpr const char kNameScatterNd[] = "ScatterNd";
constexpr const char kNameScatterNdD[] = "ScatterNdD";
constexpr const char kNamePadD[] = "PadD";
constexpr const char kNamePadV1[] = "PadV1";
constexpr const char kNameMirrorPad[] = "MirrorPad";
constexpr const char kNameMirrorPadGrad[] = "MirrorPadGrad";
constexpr const char kNameGatherNd[] = "GatherNd";
constexpr const char kNameGatherD[] = "GatherD";
constexpr const char kNameGatherV2D[] = "GatherV2D";
constexpr const char kNameArgmax[] = "Argmax";
constexpr const char kNameArgmin[] = "Argmin";
constexpr const char kNameArgMaxWithValue[] = "ArgMaxWithValue";
constexpr const char kNameArgMinWithValue[] = "ArgMinWithValue";
constexpr const char kNameReduceProd[] = "ReduceProd";
constexpr const char kNameDynamicReduceProd[] = "DynamicReduceProd";
constexpr const char kNameCumprod[] = "Cumprod";
constexpr const char kNameCumProd[] = "CumProd";
constexpr const char kNameCumprodD[] = "CumprodD";
constexpr const char kNameDiagpart[] = "Diagpart";
constexpr const char kNameSplit[] = "Split";
constexpr const char kNameBatchToSpaceNd[] = "BatchToSpaceND";
constexpr const char kNameBatchToSpaceNdV2[] = "BatchToSpaceNDV2";
constexpr const char kNameFloor[] = "Floor";
constexpr const char kNameAssign[] = "Assign";
constexpr const char kNameAssignAdd[] = "AssignAdd";
constexpr const char kNameAssignSub[] = "AssignSub";
constexpr const char kNameNPUGetFloatStatus[] = "NPUGetFloatStatus";
constexpr const char kNameNPUAllocFloatStatus[] = "NPUAllocFloatStatus";
constexpr const char kNameNPUClearFloatStatus[] = "NPUClearFloatStatus";
constexpr const char kNameNPUGetFloatStatusV2[] = "NPUGetFloatStatusV2";
constexpr const char kNameNPUClearFloatStatusV2[] = "NPUClearFloatStatusV2";
constexpr const char kNameReshape[] = "Reshape";
constexpr const char kNameTransShape[] = "TransShape";
constexpr const char kNameDiv[] = "Div";
constexpr const char kNameDivNoNan[] = "DivNoNan";
constexpr const char kNameRealDiv[] = "RealDiv";
constexpr const char kNameBitwiseAnd[] = "BitwiseAnd";
constexpr const char kNameBitwiseOr[] = "BitwiseOr";
constexpr const char kNameBitwiseXor[] = "BitwiseXor";
constexpr const char kNameBesselI0e[] = "BesselI0e";
constexpr const char kNameBesselI1e[] = "BesselI1e";
constexpr const char kNameBNTrainingReduce[] = "BNTrainingReduce";
constexpr const char kNameBNTrainingReduceGrad[] = "BNTrainingReduceGrad";
constexpr const char kNameBNTrainingUpdate[] = "BNTrainingUpdate";
constexpr const char kNameBNTrainingUpdateGrad[] = "BNTrainingUpdateGrad";
constexpr const char kNameErf[] = "Erf";
constexpr const char kNameErfc[] = "Erfc";
constexpr const char kNameExpm1[] = "Expm1";
constexpr const char kNameInplaceAddD[] = "InplaceAdd";
constexpr const char kNameInplaceSubD[] = "InplaceSub";
constexpr const char kNameInplaceUpdateD[] = "InplaceUpdate";
constexpr const char kNameInTopK[] = "InTopK";
constexpr const char kNameInTopKD[] = "InTopKD";
constexpr const char kNameInv[] = "Inv";
constexpr const char kNameInvGrad[] = "InvGrad";
constexpr const char kNameInvert[] = "Invert";
constexpr const char kNameLinSpace[] = "LinSpace";
constexpr const char kNameLog1p[] = "Log1p";
constexpr const char kNameLRN[] = "LRN";
constexpr const char kNameLRNGrad[] = "LRNGrad";
constexpr const char kNameLSTMInputGrad[] = "LSTMInputGrad";
constexpr const char kNameMatMul[] = "MatMul";
constexpr const char kNameMatrixDiag[] = "MatrixDiag";
constexpr const char kNameMatrixDiagV3[] = "MatrixDiagV3";
constexpr const char kNameMatrixDiagPartD[] = "MatrixDiagPartD";
constexpr const char kNameMatrixSetDiagD[] = "MatrixSetDiagD";
constexpr const char kNameMaxPool3DGradGrad[] = "MaxPool3DGradGrad";
constexpr const char kNameMaxPoolGradGrad[] = "MaxPoolGradGrad";
constexpr const char kNameMaxPoolGradGradWithArgmax[] = "MaxPoolGradGradWithArgmax";
constexpr const char kNameMish[] = "Mish";
constexpr const char kNameMulNoNan[] = "MulNoNan";
constexpr const char kNameParallelConcat[] = "ParallelConcat";
constexpr const char kNamePopulationCount[] = "PopulationCount";
constexpr const char kNameReduceAny[] = "ReduceAny";
constexpr const char kNameReduceAnyD[] = "ReduceAnyD";
constexpr const char kNameReluGradV2[] = "ReluGradV2";
constexpr const char kNameCeil[] = "Ceil";
constexpr const char kNameCosineEmbeddingLoss[] = "CosineEmbeddingLoss";
constexpr const char kNameXdivy[] = "Xdivy";
constexpr const char kNameMod[] = "Mod";
constexpr const char kNameRint[] = "Rint";
constexpr const char kNameIf[] = "If";
constexpr const char kNameScatterAdd[] = "ScatterAdd";
constexpr const char kNameScatterSub[] = "ScatterSub";
constexpr const char kNameScatterMul[] = "ScatterMul";
constexpr const char kNameScatterDiv[] = "ScatterDiv";
constexpr const char kNameScatterMin[] = "ScatterMin";
constexpr const char kNameScatterMax[] = "ScatterMax";
constexpr const char kNameScatterNdAdd[] = "ScatterNdAdd";
constexpr const char kNameScatterNdSub[] = "ScatterNdSub";
constexpr const char kNameScatterNonAliasingAdd[] = "ScatterNonAliasingAdd";
constexpr const char kNameSeLU[] = "SeLU";
constexpr const char kNameSoftsign[] = "Softsign";
constexpr const char kNameSort[] = "Sort";
constexpr const char kNameSpaceToBatchNDD[] = "SpaceToBatchNDD";
constexpr const char kNameSparseApplyFtrlV2[] = "SparseApplyFtrlV2";
constexpr const char kNameSparseApplyProximalAdagrad[] = "SparseApplyProximalAdagrad";
constexpr const char kNameTruncateDiv[] = "TruncateDiv";
constexpr const char kNameTruncateMod[] = "TruncateMod";
constexpr const char kNameUnsortedSegmentMax[] = "UnsortedSegmentMax";
constexpr const char kNameUnsortedSegmentProd[] = "UnsortedSegmentProd";
constexpr const char kNameWtsARQ[] = "WtsARQ";
constexpr const char kNameXlogy[] = "Xlogy";
constexpr const char kNameReLUV2[] = "ReLUV2";
constexpr const char kNameAccumulateNV2[] = "AccumulateNV2";
constexpr const char kNameConfusionMulGrad[] = "ConfusionMulGrad";
constexpr const char kNameActsULQ[] = "ActsULQ";
constexpr const char kNameActsULQInputGrad[] = "ActsULQInputGrad";
constexpr const char kNameActULQClampMaxGrad[] = "ActULQClampMaxGrad";
constexpr const char kNameActULQClampMinGrad[] = "ActULQClampMinGrad";
constexpr const char kNameHistogramFixedWidthD[] = "HistogramFixedWidth";
constexpr const char kNameIFMR[] = "IFMR";
constexpr const char kNameCentralization[] = "Centralization";
constexpr const char kNameApplyAdagradV2[] = "ApplyAdagradV2";
constexpr const char kNameApplyAdagradV2D[] = "ApplyAdagradV2D";
constexpr const char kNameApplyAddSign[] = "ApplyAddSign";
constexpr const char kNameSparseApplyAdagradV2[] = "SparseApplyAdagradV2";
constexpr const char kNameDataFormatDimMap[] = "DataFormatDimMap";
constexpr const char kNameTile[] = "Tile";
constexpr const char kNameTileD[] = "TileD";
constexpr const char kNameCos[] = "Cos";
constexpr const char kNameCosh[] = "Cosh";
constexpr const char kNameACos[] = "ACos";
constexpr const char kNameACosGrad[] = "ACosGrad";
constexpr const char kNameAcosGrad[] = "AcosGrad";
constexpr const char kNameFloorDiv[] = "FloorDiv";
constexpr const char kNameSin[] = "Sin";
constexpr const char kNameSinh[] = "Sinh";
constexpr const char kNameAsin[] = "Asin";
constexpr const char kNameAsinGrad[] = "AsinGrad";
constexpr const char kNameAsinh[] = "Asinh";
constexpr const char kNameAsinhGrad[] = "AsinhGrad";
constexpr const char kNamePrelu[] = "PReLU";
constexpr const char kNamePreluGrad[] = "PReLUGrad";
constexpr const char kNameSigmoid[] = "Sigmoid";
constexpr const char kNameSigmoidGrad[] = "SigmoidGrad";
constexpr const char kNameHSwish[] = "HSwish";
constexpr const char kNameHSwishGrad[] = "HSwishGrad";
constexpr const char kNameHSigmoid[] = "HSigmoid";
constexpr const char kNameL2Normalize[] = "L2Normalize";
constexpr const char kNameL2NormalizeGrad[] = "L2NormalizeGrad";
constexpr const char kNameSoftmax[] = "Softmax";
constexpr const char kNameIOU[] = "IOU";
constexpr const char kNameBoundingBoxDecode[] = "BoundingBoxDecode";
constexpr const char kNameBoundingBoxEncode[] = "BoundingBoxEncode";
constexpr const char kNameSlice[] = "Slice";
constexpr const char kNameAddN[] = "AddN";
constexpr const char kNameLess[] = "Less";
constexpr const char kNameGreater[] = "Greater";
constexpr const char kNameUnpack[] = "Unpack";
constexpr const char kNameMerge[] = "Merge";
constexpr const char kNameGeSwitch[] = "GeSwitch";

constexpr const char kNameHuberLoss[] = "HuberLoss";
constexpr const char kNameCumSum[] = "CumSum";
constexpr const char kNameCumsumD[] = "CumsumD";
constexpr const char kNameCumsum[] = "Cumsum";
constexpr const char kNameHuberLossGrad[] = "HuberLossGrad";
constexpr const char kNameSparseSoftmaxCrossEntropy[] = "SparseSoftmaxCrossEntropy";
constexpr const char kNameSparseSoftmaxCrossEntropyGrad[] = "SparseSoftmaxCrossEntropyGrad";
constexpr const char kNameNLLLoss[] = "NLLLoss";
constexpr const char kNameNLLLossGrad[] = "NLLLossGrad";
constexpr const char kNameTopK[] = "TopK";
constexpr const char kNameSoftmaxGrad[] = "SoftmaxGrad";
constexpr const char kNameMaxPool[] = "MaxPool";
constexpr const char kNameAvgPool[] = "AvgPool";
constexpr const char kNameMaxPoolWithArgmax[] = "MaxPoolWithArgmax";
constexpr const char kNameMaxPoolWithArgmaxV2[] = "MaxPoolWithArgmaxV2";
constexpr const char kNameBatchNorm[] = "BatchNorm";
constexpr const char kNameBatchNormGrad[] = "BatchNormGrad";
constexpr const char kNameROIAlign[] = "ROIAlign";
constexpr const char kNameROIAlignGrad[] = "ROIAlignGrad";
constexpr const char kNameRandomChoiceWithMask[] = "RandomChoiceWithMask";
constexpr const char kNameAbs[] = "Abs";
constexpr const char kNameAbsGrad[] = "AbsGrad";
constexpr const char kNameBinaryCrossEntropy[] = "BinaryCrossEntropy";
constexpr const char kNameBinaryCrossEntropyGrad[] = "BinaryCrossEntropyGrad";
constexpr const char kNameSparseApplyAdagrad[] = "SparseApplyAdagrad";
constexpr const char kNameSparseApplyAdagradD[] = "SparseApplyAdagradD";
constexpr const char kNameSparseApplyFtrlD[] = "SparseApplyFtrlD";
constexpr const char kNameApplyProximalAdagrad[] = "ApplyProximalAdagrad";
constexpr const char kNameAcosh[] = "Acosh";
constexpr const char kNameAcoshGrad[] = "AcoshGrad";
constexpr const char kNameFloorMod[] = "FloorMod";
constexpr const char kNameSpaceToDepth[] = "SpaceToDepth";
constexpr const char kNameDepthToSpace[] = "DepthToSpace";
constexpr const char kNameSign[] = "Sign";
constexpr const char kNameLARSUpdate[] = "LARSUpdate";
constexpr const char kNameRound[] = "Round";
constexpr const char kNamePrint[] = "Print";
constexpr const char kNameStringFormat[] = "StringFormat";
constexpr const char kNameApplyFtrl[] = "ApplyFtrl";
constexpr const char kNameDiag[] = "Diag";
constexpr const char kNameDiagPart[] = "DiagPart";
constexpr const char kNameDiagPartD[] = "DiagPartD";
constexpr const char kNameSpaceToBatch[] = "SpaceToBatch";
constexpr const char kNameBatchToSpace[] = "BatchToSpace";
constexpr const char kNameTan[] = "Tan";
constexpr const char kNameAtan[] = "Atan";
constexpr const char kNameAtanGrad[] = "AtanGrad";
constexpr const char kNameAtanh[] = "Atanh";
constexpr const char kNameAtan2[] = "Atan2";
constexpr const char kNameApplyRMSProp[] = "ApplyRMSProp";
constexpr const char kNameApplyCenteredRMSProp[] = "ApplyCenteredRMSProp";
constexpr const char kNameBasicLSTMCell[] = "BasicLSTMCell";
constexpr const char kNameBasicLSTMCellInputGrad[] = "BasicLSTMCellInputGrad";
constexpr const char kNameBasicLSTMCellWeightGrad[] = "BasicLSTMCellWeightGrad";
constexpr const char kNameBasicLSTMCellCStateGrad[] = "BasicLSTMCellCStateGrad";
constexpr const char kNameDynamicRNN[] = "DynamicRNN";
constexpr const char kNameDynamicRNNGrad[] = "DynamicRNNGrad";
constexpr const char kNameDynamicGRUV2[] = "DynamicGRUV2";
constexpr const char kNameDynamicGRUV2Grad[] = "DynamicGRUV2Grad";
constexpr const char kNameL2Loss[] = "L2Loss";
constexpr const char kNameCTCLoss[] = "CTCLoss";
constexpr const char kNameRange[] = "Range";
constexpr const char kNameSquareSumAll[] = "SquareSumAll";
constexpr const char kNameAscendQuant[] = "Quant";
constexpr const char kNameAscendDequant[] = "Dequant";
constexpr const char kNameAscendAntiQuant[] = "AscendAntiQuant";
constexpr const char kNameCropAndResize[] = "CropAndResize";
constexpr const char kNameReverseSequence[] = "ReverseSequence";
constexpr const char kNameEditDistance[] = "EditDistance";
constexpr const char kNameCase[] = "Case";
constexpr const char kNameAssert[] = "Assert";
constexpr const char kNameCTCGreedyDecoder[] = "CTCGreedyDecoder";
constexpr const char kNameReverseV2[] = "ReverseV2";
constexpr const char kNameReverseV2D[] = "ReverseV2D";
constexpr const char kNameLambApplyWeightAssign[] = "LambApplyWeightAssign";
constexpr const char kNameLambApplyOptimizerAssign[] = "LambApplyOptimizerAssign";
constexpr const char kNameScale[] = "Scale";
constexpr const char kNameEltwise[] = "Eltwise";
constexpr const char kNameFullConnection[] = "FullConnection";
constexpr const char kNameFusedBatchNorm[] = "FusedBatchNorm";
constexpr const char kNamePooling[] = "Pooling";
constexpr const char kNameMaxPoolV3[] = "MaxPoolV3";
constexpr const char kNameAvgPoolV2[] = "AvgPoolV2";
constexpr const char kNameShape[] = "Shape";
constexpr const char kNameTensorShape[] = "TensorShape";
constexpr const char kNameDynamicShape[] = "DynamicShape";
constexpr const char kNameGather[] = "Gather";
constexpr const char kNameUnsqueeze[] = "Unsqueeze";
constexpr const char kNamePadV3[] = "PadV3";
constexpr const char kNamePadV3Grad[] = "PadV3Grad";
constexpr const char kNamePadV2[] = "PadV2";
constexpr const char kNameGlobalAvgPool[] = "GlobalAveragePool";
constexpr const char kNameAdaptiveMaxPool2D[] = "AdaptiveMaxPool2D";
constexpr const char kNameAdaptiveMaxPool2d[] = "AdaptiveMaxPool2d";
constexpr const char kNameDilation2DBackpropInput[] = "Dilation2DBackpropInput";
constexpr const char kNameStridedSliceV2[] = "StridedSliceV2";
constexpr const char kNameBNInference[] = "BNInference";
constexpr const char kNameDeconvolution[] = "Deconvolution";
constexpr const char kNameUpsample[] = "Upsample";
constexpr const char kNameConv2DTransposeD[] = "Conv2DTransposeD";
constexpr const char kNameArgMaxV2[] = "ArgMaxV2";
constexpr const char kNameResizeNearestNeighborV2[] = "ResizeNearestNeighborV2";
constexpr const char kNameResizeNearestNeighborV2D[] = "ResizeNearestNeighborV2D";
constexpr const char kNameResizeNearestNeighborV2Grad[] = "ResizeNearestNeighborV2Grad";
constexpr const char kNameConv2DBackpropInputD[] = "Conv2DBackpropInputD";
constexpr const char kNameConv2DBackpropInput[] = "Conv2DBackpropInput";
constexpr const char kNameConv2DBackpropInputV2[] = "Conv2DBackpropInputV2";
constexpr const char kNameConv2DBackpropFilterD[] = "Conv2DBackpropFilterD";
constexpr const char kNameConv2DBackpropFilter[] = "Conv2DBackpropFilter";
constexpr const char kNameConcatV2[] = "ConcatV2";
constexpr const char kNameFillV1[] = "FillV1";
constexpr const char kNameTensorArray[] = "TensorArray";
constexpr const char kNameTensorArrayWrite[] = "TensorArrayWrite";
constexpr const char kNameTensorArrayGather[] = "TensorArrayGather";
constexpr const char kNameTensorMove[] = "TensorMove";
constexpr const char kNameWKV[] = "WKV";
constexpr const char kNameWKVGrad[] = "WKVGrad";
constexpr const char kNameWhile[] = "While";
constexpr const char kNameKMeansCentroids[] = "KMeansCentroids";
constexpr const char kNameIsNan[] = "IsNan";
constexpr const char kNameKLDivLoss[] = "KLDivLoss";
constexpr const char kNameKLDiv[] = "KLDiv";
constexpr const char kNameGetShape[] = "GetShape";
constexpr const char kNameKlDivLossGrad[] = "KLDivLossGrad";
constexpr const char kNameRandomStandardNormal[] = "RandomStandardNormal";
constexpr const char kNameStandardNormal[] = "StandardNormal";
constexpr const char kNameUnsortedSegmentSum[] = "UnsortedSegmentSum";
constexpr const char kNameSpaceToBatchTF[] = "SpaceToBatchTF";
constexpr const char kNameBatchToSpaceTF[] = "BatchToSpaceTF";
constexpr const char kNameMaskedSelect[] = "MaskedSelect";
constexpr const char kNamePartitionedCall[] = "PartitionedCall";
constexpr const char kNameRangeV2[] = "RangeV2";
constexpr const char kNameOCRDetectionPreHandle[] = "OCRDetectionPreHandle";
constexpr const char kNameOCRFindContours[] = "OCRFindContours";
constexpr const char kNameBatchDilatePolys[] = "BatchDilatePolys";
constexpr const char kNameResizeAndClipPolys[] = "ResizeAndClipPolys";
constexpr const char kNameOCRDetectionPostHandle[] = "OCRDetectionPostHandle";
constexpr const char kNameOCRIdentifyPreHandle[] = "OCRIdentifyPreHandle";
constexpr const char kNameBatchEnqueue[] = "BatchEnqueue";
constexpr const char kNameDequeue[] = "Dequeue";
constexpr const char kNameOCRRecognitionPreHandle[] = "OCRRecognitionPreHandle";
constexpr const char kNameStringUpper[] = "StringUpper";
constexpr const char kNameStringLength[] = "StringLength";
constexpr const char kNameDecodeImage[] = "DecodeImage";
constexpr const char kNameDecodeBase64[] = "DecodeBase64";
constexpr const char kNameMakeTuple[] = "MakeTuple";
constexpr const char kNameMakeList[] = "make_list";
constexpr const char kNameTupleGetItem[] = "TupleGetItem";
constexpr const char kNameListGetItem[] = "ListGetItem";
constexpr const char kNameLoad[] = "Load";
constexpr const char kNameDepend[] = "Depend";
constexpr const char kNameReturn[] = "Return";
constexpr const char kNameIdentity[] = "Identity";
constexpr const char kNameUpdateState[] = "UpdateState";
constexpr const char kNameTransData[] = "TransData";
constexpr const char kNameWhere[] = "Where";
constexpr const char kNameSelectV2[] = "SelectV2";
constexpr const char kNameAsStrided[] = "AsStrided";
constexpr const char kNameViewCopy[] = "ViewCopy";
constexpr const char kNameSend[] = "Send";
constexpr const char kNameReceive[] = "Receive";
constexpr const char kNameIndexAdd[] = "IndexAdd";
constexpr const char kNameIndexFill[] = "IndexFill";
constexpr const char kNameUnique[] = "Unique";
constexpr const char kNameDynamicBroadcastGradientArgs[] = "DynamicBroadcastGradientArgs";
constexpr const char kNameDynamicStitch[] = "DynamicStitch";
constexpr const char kNameThreshold[] = "Threshold";
constexpr const char kNameCosineSimilarity[] = "CosineSimilarity";
constexpr const char kNameLayerNormXBackpropV2[] = "LayerNormXBackpropV2";
constexpr const char kNameLayerNormBetaGammaBackpropV2[] = "LayerNormBetaGammaBackpropV2";
constexpr const char kNameGRUV2HiddenGradCell[] = "GRUV2HiddenGradCell";
constexpr const char kNameTopKV2[] = "TopKV2";
constexpr const char kNameGridSampler2D[] = "GridSampler2D";
constexpr const char kNameLeftShift[] = "LeftShift";
constexpr const char kNameRightShift[] = "RightShift";
constexpr const char kNameReduceLogSumExp[] = "ReduceLogSumExp";
constexpr const char kNameReduceLogSum[] = "ReduceLogSum";
constexpr const char kNameSize[] = "Size";
constexpr const char kNameTfIdfVectorizer[] = "TfIdfVectorizer";
constexpr const char kNameMVNV2[] = "MVNV2";
constexpr const char kNameCommonGRU[] = "CommonGRU";
constexpr const char kNameTril[] = "Tril";
constexpr const char kNameConv2DTransposeV2[] = "Conv2DTransposeV2";
constexpr const char kNameGridSampler3D[] = "GridSampler3D";
constexpr const char kNameResizeArea[] = "ResizeArea";
constexpr const char kNameResizeBicubic[] = "ResizeBicubic";
constexpr const char kNameIm2col[] = "Im2col";
constexpr const char kNameAffineGrid[] = "AffineGrid";
constexpr const char kNameFFN[] = "FFN";
constexpr const char kNameBlendFaceBgPartOne[] = "BlendFaceBgPartOne";
constexpr const char kNameNonZeroV2[] = "NonZeroV2";
constexpr const char kNameResize[] = "Resize";
constexpr const char kNameAdaptiveAvgPool[] = "AdaptiveAvgPool";
constexpr const char kNamePromptFlashAttention[] = "PromptFlashAttention";
constexpr const char kNameFlashAttentionScore[] = "FlashAttentionScore";
constexpr const char kNameFlashAttentionScoreGrad[] = "FlashAttentionScoreGrad";
constexpr const char kNameEnvironCreate[] = "EnvironCreate";
constexpr const char kNameEnvironDestroyAll[] = "EnvironDestroyAll";
constexpr const char kNameEnvironGet[] = "EnvironGet";
constexpr const char kNameEnvironSet[] = "EnvironSet";

class OpAdapterDesc;

class OpAdapterMap {
 public:
  static mindspore::HashMap<std::string, std::shared_ptr<OpAdapterDesc>> &get();
};
}  // namespace transform
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_TRANSFORM_GRAPH_IR_OP_ADAPTER_MAP_H_