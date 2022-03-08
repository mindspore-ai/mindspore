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

#ifndef MINDSPORE_CORE_BASE_CORE_OPS_H_
#define MINDSPORE_CORE_BASE_CORE_OPS_H_

#include <iostream>
#include <string>
#include <memory>
#include "ir/anf.h"
#include "ir/primitive.h"
#include "utils/flags.h"
#include "utils/hash_map.h"

namespace mindspore {
namespace prim {
GVAR_DEF(ValuePtr, kValueOne, std::make_shared<Int64Imm>(1));
#define COMMA ,
GVAR_DEF(mindspore::HashMap<std::string COMMA ValuePtr>, kSideEffectPropagate,
         {{mindspore::GRAPH_FLAG_SIDE_EFFECT_PROPAGATE COMMA kValueOne}});
#undef COMMA
constexpr auto kGetNext = "GetNext";
constexpr auto kGather = "Gather";
constexpr auto kAddcdiv = "Addcdiv";
constexpr auto kAddcmul = "Addcmul";
constexpr auto kCdist = "Cdist";
constexpr auto kCdistGrad = "CdistGrad";
// Arithmetic
constexpr auto kScalarAdd = "ScalarAdd";
constexpr auto kScalarSub = "ScalarSub";
constexpr auto kScalarMul = "ScalarMul";
constexpr auto kScalarDiv = "ScalarDiv";
constexpr auto kScalarFloordiv = "ScalarFloordiv";
constexpr auto kScalarMod = "ScalarMod";
constexpr auto kScalarPow = "ScalarPow";
constexpr auto kScalarTrunc = "ScalarTrunc";
constexpr auto kScalarFloor = "ScalarFloor";
constexpr auto kScalarUadd = "ScalarUadd";
constexpr auto kScalarUsub = "ScalarUsub";
constexpr auto kExp = "Exp";
constexpr auto kEqual = "Equal";
constexpr auto kNotEqual = "NotEqual";
constexpr auto kNeg = "Neg";
constexpr auto kSub = "Sub";
constexpr auto kMul = "Mul";
constexpr auto kMulNoNan = "MulNoNan";
constexpr auto kACos = "ACos";
constexpr auto kACosGrad = "ACosGrad";
constexpr auto kRealDiv = "RealDiv";
constexpr auto kReciprocal = "Reciprocal";
constexpr auto kLog = "Log";
constexpr auto kLogicalXor = "LogicalXor";
constexpr auto kSelect = "Select";
constexpr auto kAdd = "Add";
constexpr auto kBiasAdd = "BiasAdd";
constexpr auto kTile = "Tile";
constexpr auto kAcosh = "Acosh";
constexpr auto kAcoshGrad = "AcoshGrad";
constexpr auto kBiasAddGrad = "BiasAddGrad";
constexpr auto kMatrixInverse = "MatrixInverse";
constexpr auto kMatrixDeterminant = "MatrixDeterminant";
constexpr auto kLogMatrixDeterminant = "LogMatrixDeterminant";
constexpr auto kCos = "Cos";
constexpr auto kAsinh = "Asinh";
constexpr auto kAsinhGrad = "AsinhGrad";
constexpr auto kAbs = "Abs";
constexpr auto kAsin = "Asin";
constexpr auto kAsinGrad = "AsinGrad";
constexpr auto kTrunc = "Trunc";
constexpr auto kLpNorm = "LpNorm";
constexpr auto kSquare = "Square";
constexpr auto kRealInner = "RealInner";
constexpr auto kReal = "Real";
constexpr auto kImag = "Imag";
constexpr auto kConj = "Conj";
constexpr auto kGer = "Ger";

// Math
constexpr auto kCross = "Cross";

// Arrays
constexpr auto kDynamicShape = "DynamicShape";
constexpr auto kTensorShape = "TensorShape";
constexpr auto kStack = "Stack";
constexpr auto kUnstack = "Unstack";
constexpr auto kTupleGetItem = "TupleGetItem";
constexpr auto kSliceGetItem = "SliceGetItem";
constexpr auto kGeLU = "GeLU";
constexpr auto kGLU = "GLU";
constexpr auto kReLU = "ReLU";
constexpr auto kReLU6 = "ReLU6";
constexpr auto kReLUV2 = "ReLUV2";
constexpr auto kReLUGrad = "ReluGrad";
constexpr auto kReLUGradV2 = "ReluGradV2";
constexpr auto kRint = "Rint";
constexpr auto kGeLUGrad = "GeLUGrad";
constexpr auto kFastGeLU = "FastGeLU";
constexpr auto kFastGeLUGrad = "FastGeLUGrad";
constexpr auto kStridedSlice = "StridedSlice";
constexpr auto kStridedSliceGrad = "StridedSliceGrad";
constexpr auto kCoalesce = "Coalesce";
constexpr auto kZerosLike = "ZerosLike";
constexpr auto kOnes = "Ones";
constexpr auto kOnesLike = "OnesLike";
constexpr auto kIdentity = "Identity";
constexpr auto kDiag = "Diag";
constexpr auto kDiagPart = "DiagPart";
constexpr auto kDynamicBroadcastGradientArgs = "DynamicBroadcastGradientArgs";
constexpr auto kTranspose = "Transpose";
constexpr auto kSplitV = "SplitV";
constexpr auto kDynamicBroadcastTo = "DynamicBroadcastTo";
constexpr auto kReshape = "Reshape";
constexpr auto kLstsq = "Lstsq";
constexpr auto kLowerBound = "LowerBound";
constexpr auto kUpperBound = "UpperBound";
constexpr auto kCummax = "Cummax";

// NN
constexpr auto kCTCLoss = "CTCLoss";
constexpr auto kLayerNorm = "LayerNorm";
constexpr auto kLayerNormGrad = "LayerNormGrad";
constexpr auto kDropoutGenMask = "DropoutGenMask";
constexpr auto kDropoutDoMask = "DropoutDoMask";
constexpr auto kDropoutDoMaskV3 = "DropoutDoMaskV3";
constexpr auto kDropout = "Dropout";
constexpr auto kDropoutGrad = "DropoutGrad";
constexpr auto kConv2DTranspose = "Conv2DTranspose";
constexpr auto kSparseApplyAdadelta = "SparseApplyAdadelta";
constexpr auto kRoll = "Roll";
constexpr auto kTanh = "Tanh";
constexpr auto kGridSampler3D = "GridSampler3D";
constexpr auto kGridSampler3DGrad = "GridSampler3DGrad";

// CSRTensor
constexpr auto kMakeCSRTensor = "MakeCSRTensor";
constexpr auto kCSRTensorGetValues = "CSRTensorGetValues";
constexpr auto kCSRTensorGetIndptr = "CSRTensorGetIndptr";
constexpr auto kCSRTensorGetIndices = "CSRTensorGetIndices";
constexpr auto kCSRTensorGetDenseShape = "CSRTensorGetDenseShape";
constexpr auto kIsCSRFunc = "IsCSRFunc";

// COOTensor
constexpr auto kMakeCOOTensor = "MakeCOOTensor";
constexpr auto kCOOTensorGetValues = "COOTensorGetValues";
constexpr auto kCOOTensorGetIndices = "COOTensorGetIndices";
constexpr auto kCOOTensorGetDenseShapes = "COOTensorGetDenseShape";
constexpr auto kCOOTensorDenseMatmul = "COOTensorDenseMatmul";

// Sparse ops
constexpr auto kSparseTensorDenseMatmul = "SparseTensorDenseMatmul";
constexpr auto kCSRDenseMul = "CSRDenseMul";
constexpr auto kCSRReduceSum = "CSRReduceSum";
constexpr auto kCSRMV = "CSRMV";
constexpr auto kCSRMul = "CSRMul";
constexpr auto kCSRGather = "CSRGather";
constexpr auto kCSR2COO = "CSR2COO";
constexpr auto kCOO2CSR = "COO2CSR";
constexpr auto kCSRDiv = "CSRDiv";

// Meta Function Graph
constexpr auto kJ = "J";
constexpr auto kVmap = "Vmap";

// Others
constexpr auto kMakeTuple = "MakeTuple";
constexpr auto kAssign = "Assign";
constexpr auto kAssignAdd = "AssignAdd";
constexpr auto kAssignSub = "AssignSub";
constexpr auto kEnvironCreate = "EnvironCreate";
constexpr auto kEnvironSet = "EnvironSet";
constexpr auto kEnvironGet = "EnvironGet";
constexpr auto kEnvironAdd = "EnvironAdd";
constexpr auto kEnvironDestroyAll = "EnvironDestroyAll";

//
// Here list all primitives used in backend or some special primitives used by core.
// GetNext
GVAR_DEF(PrimitivePtr, kPrimGetNext, std::make_shared<Primitive>(kGetNext));

// Arithmetic
GVAR_DEF(PrimitivePtr, kPrimScalarAdd, std::make_shared<Primitive>(kScalarAdd));
GVAR_DEF(PrimitivePtr, kPrimScalarSub, std::make_shared<Primitive>(kScalarSub));
GVAR_DEF(PrimitivePtr, kPrimScalarMul, std::make_shared<Primitive>(kScalarMul));
GVAR_DEF(PrimitivePtr, kPrimScalarDiv, std::make_shared<Primitive>(kScalarDiv));
GVAR_DEF(PrimitivePtr, kPrimScalarFloordiv, std::make_shared<Primitive>(kScalarFloordiv));
GVAR_DEF(PrimitivePtr, kPrimScalarMod, std::make_shared<Primitive>(kScalarMod));
GVAR_DEF(PrimitivePtr, kPrimScalarPow, std::make_shared<Primitive>(kScalarPow));
GVAR_DEF(PrimitivePtr, kPrimScalarTrunc, std::make_shared<Primitive>(kScalarTrunc));
GVAR_DEF(PrimitivePtr, kPrimScalarFloor, std::make_shared<Primitive>(kScalarFloor));
GVAR_DEF(PrimitivePtr, kPrimScalarUadd, std::make_shared<Primitive>(kScalarUadd));
GVAR_DEF(PrimitivePtr, kPrimScalarUsub, std::make_shared<Primitive>(kScalarUsub));
GVAR_DEF(PrimitivePtr, kPrimScalarExp, std::make_shared<Primitive>("scalar_exp"));
GVAR_DEF(PrimitivePtr, kPrimScalarLog, std::make_shared<Primitive>("scalar_log"));
GVAR_DEF(PrimitivePtr, kPrimScalarSin, std::make_shared<Primitive>("scalar_sin"));
GVAR_DEF(PrimitivePtr, kPrimScalarCos, std::make_shared<Primitive>("scalar_cos"));
GVAR_DEF(PrimitivePtr, kPrimScalarTan, std::make_shared<Primitive>("scalar_tan"));
GVAR_DEF(PrimitivePtr, kPrimTrunc, std::make_shared<Primitive>(kTrunc));

// Comparisons
GVAR_DEF(PrimitivePtr, kPrimScalarEq, std::make_shared<Primitive>("scalar_eq"));
GVAR_DEF(PrimitivePtr, kPrimScalarLt, std::make_shared<Primitive>("scalar_lt"));
GVAR_DEF(PrimitivePtr, kPrimScalarGt, std::make_shared<Primitive>("scalar_gt"));
GVAR_DEF(PrimitivePtr, kPrimScalarNe, std::make_shared<Primitive>("scalar_ne"));
GVAR_DEF(PrimitivePtr, kPrimScalarLe, std::make_shared<Primitive>("scalar_le"));
GVAR_DEF(PrimitivePtr, kPrimScalarGe, std::make_shared<Primitive>("scalar_ge"));
GVAR_DEF(PrimitivePtr, kPrimBoolNot, std::make_shared<Primitive>("bool_not"));
GVAR_DEF(PrimitivePtr, kPrimBoolAnd, std::make_shared<Primitive>("bool_and"));
GVAR_DEF(PrimitivePtr, kPrimBoolOr, std::make_shared<Primitive>("bool_or"));
GVAR_DEF(PrimitivePtr, kPrimBoolEq, std::make_shared<Primitive>("bool_eq"));
GVAR_DEF(PrimitivePtr, kPrimGreater, std::make_shared<Primitive>("Greater"));
GVAR_DEF(PrimitivePtr, kPrimGreaterEqual, std::make_shared<Primitive>("GreaterEqual"));
GVAR_DEF(PrimitivePtr, kPrimLess, std::make_shared<Primitive>("Less"));
GVAR_DEF(PrimitivePtr, kPrimLessEqual, std::make_shared<Primitive>("LessEqual"));
GVAR_DEF(PrimitivePtr, kPrimEqual, std::make_shared<Primitive>(kEqual));
GVAR_DEF(PrimitivePtr, kPrimNotEqual, std::make_shared<Primitive>(kNotEqual));
GVAR_DEF(PrimitivePtr, kPrimLogicalAnd, std::make_shared<Primitive>("LogicalAnd"));
GVAR_DEF(PrimitivePtr, kPrimLogicalOr, std::make_shared<Primitive>("LogicalOr"));
GVAR_DEF(PrimitivePtr, kPrimLogicalNot, std::make_shared<Primitive>("LogicalNot"));
GVAR_DEF(PrimitivePtr, kPrimLogicalXor, std::make_shared<Primitive>(kLogicalXor));
GVAR_DEF(PrimitivePtr, kPrimEqualCount, std::make_shared<Primitive>("EqualCount"));
GVAR_DEF(PrimitivePtr, kPrimApproximateEqual, std::make_shared<Primitive>("ApproximateEqual"));

GVAR_DEF(PrimitivePtr, kPrimDistribute, std::make_shared<Primitive>("distribute"));
GVAR_DEF(PrimitivePtr, kPrimIm2Col, std::make_shared<Primitive>("im2col"));
GVAR_DEF(PrimitivePtr, kPrimCol2Im, std::make_shared<Primitive>("col2im"));
GVAR_DEF(PrimitivePtr, kPrimIm2ColV1, std::make_shared<Primitive>("im2col_v1"));
GVAR_DEF(PrimitivePtr, kPrimCol2ImV1, std::make_shared<Primitive>("col2im_v1"));

GVAR_DEF(PrimitivePtr, kPrimLabelGoto, std::make_shared<Primitive>("LabelGoto"));
GVAR_DEF(PrimitivePtr, kPrimLabelSwitch, std::make_shared<Primitive>("LabelSwitch"));
GVAR_DEF(PrimitivePtr, kPrimLabelSet, std::make_shared<Primitive>("LabelSet"));

// Stack ops
GVAR_DEF(PrimitivePtr, kPrimStackInit, std::make_shared<Primitive>("StackInit"));
GVAR_DEF(PrimitivePtr, kPrimStackDestroy, std::make_shared<Primitive>("StackDestroy"));
GVAR_DEF(PrimitivePtr, kPrimStackPush, std::make_shared<Primitive>("StackPush"));
GVAR_DEF(PrimitivePtr, kPrimStackPop, std::make_shared<Primitive>("StackPop"));

// Arrays
GVAR_DEF(PrimitivePtr, kPrimIdentitys, std::make_shared<Primitive>(kIdentity));
GVAR_DEF(PrimitivePtr, kPrimDynamicBroadcastTo, std::make_shared<Primitive>(kDynamicBroadcastTo));
GVAR_DEF(PrimitivePtr, kPrimCummin, std::make_shared<Primitive>("Cummin"));
GVAR_DEF(PrimitivePtr, kPrimBroadcastTo, std::make_shared<Primitive>("BroadcastTo"));
GVAR_DEF(PrimitivePtr, kPrimScalarToArray, std::make_shared<Primitive>("scalar_to_array"));
GVAR_DEF(PrimitivePtr, kPrimTopK, std::make_shared<Primitive>("TopK"));
GVAR_DEF(PrimitivePtr, kPrimArrayToScalar, std::make_shared<Primitive>("array_to_scalar"));
GVAR_DEF(PrimitivePtr, kPrimBroadcastShape, std::make_shared<Primitive>("broadcast_shape"));
GVAR_DEF(PrimitivePtr, kPrimArrayMap, std::make_shared<Primitive>("array_map"));
GVAR_DEF(PrimitivePtr, kPrimArrayReduce, std::make_shared<Primitive>("array_reduce"));
GVAR_DEF(PrimitivePtr, kPrimCast, std::make_shared<Primitive>("Cast"));
GVAR_DEF(PrimitivePtr, kPrimConcat, std::make_shared<Primitive>("Concat"));
GVAR_DEF(PrimitivePtr, kPrimSqueeze, std::make_shared<Primitive>("Squeeze"));
GVAR_DEF(PrimitivePtr, kPrimUnsqueeze, std::make_shared<Primitive>("Unsqueeze"));
GVAR_DEF(PrimitivePtr, kPrimTranspose, std::make_shared<Primitive>(kTranspose));
GVAR_DEF(PrimitivePtr, kPrimGatherV2, std::make_shared<Primitive>("GatherV2"));
GVAR_DEF(PrimitivePtr, kPrimGatherD, std::make_shared<Primitive>("GatherD"));
GVAR_DEF(PrimitivePtr, kPrimGather, std::make_shared<Primitive>("Gather"));
GVAR_DEF(PrimitivePtr, kPrimGatherNd, std::make_shared<Primitive>("GatherNd"));
GVAR_DEF(PrimitivePtr, kPrimSparseGatherV2, std::make_shared<Primitive>("SparseGatherV2"));
GVAR_DEF(PrimitivePtr, kPrimCoalesce, std::make_shared<Primitive>(kCoalesce));
GVAR_DEF(PrimitivePtr, kPrimSparseToDense, std::make_shared<Primitive>("SparseToDense"));
GVAR_DEF(PrimitivePtr, kPrimShape, std::make_shared<Primitive>("Shape"));
GVAR_DEF(PrimitivePtr, kPrimStridedSlice, std::make_shared<Primitive>(kStridedSlice));
GVAR_DEF(PrimitivePtr, kPrimStridedSliceGrad, std::make_shared<Primitive>(kStridedSliceGrad));
GVAR_DEF(PrimitivePtr, kPrimTensorShape, std::make_shared<Primitive>(kTensorShape));
GVAR_DEF(PrimitivePtr, kPrimDynamicShape, std::make_shared<Primitive>(kDynamicShape));
GVAR_DEF(PrimitivePtr, kPrimEmbeddingLookup, std::make_shared<Primitive>("EmbeddingLookup"));
GVAR_DEF(PrimitivePtr, kPrimEmbeddingLookupCommGrad, std::make_shared<Primitive>("EmbeddingLookupCommGrad"));
GVAR_DEF(PrimitivePtr, kPrimSize, std::make_shared<Primitive>("Size"));
GVAR_DEF(PrimitivePtr, kPrimArgMax, std::make_shared<Primitive>("Argmax"));
GVAR_DEF(PrimitivePtr, kPrimArgMin, std::make_shared<Primitive>("Argmin"));
GVAR_DEF(PrimitivePtr, kPrimPack, std::make_shared<Primitive>("Pack"));
GVAR_DEF(PrimitivePtr, kPrimStack, std::make_shared<Primitive>(kStack));
GVAR_DEF(PrimitivePtr, kPrimUnpack, std::make_shared<Primitive>("Unpack"));
GVAR_DEF(PrimitivePtr, kPrimUnstack, std::make_shared<Primitive>(kUnstack));
GVAR_DEF(PrimitivePtr, kPrimUnsortedSegmentMax, std::make_shared<Primitive>("UnsortedSegmentMax"));
GVAR_DEF(PrimitivePtr, kPrimUnsortedSegmentSum, std::make_shared<Primitive>("UnsortedSegmentSum"));
GVAR_DEF(PrimitivePtr, kPrimUnsortedSegmentMin, std::make_shared<Primitive>("UnsortedSegmentMin"));
GVAR_DEF(PrimitivePtr, kPrimConcatOffset, std::make_shared<Primitive>("ConcatOffset"));
GVAR_DEF(PrimitivePtr, kPrimReshape, std::make_shared<Primitive>("Reshape"));
GVAR_DEF(PrimitivePtr, kPrimSubAndFilter, std::make_shared<Primitive>("SubAndFilter"));
GVAR_DEF(PrimitivePtr, kPrimMapCacheIdx, std::make_shared<Primitive>("MapCacheIdx"));
GVAR_DEF(PrimitivePtr, kPrimUpdateCache, std::make_shared<Primitive>("UpdateCache"));
GVAR_DEF(PrimitivePtr, kPrimComputeAccidentalHits, std::make_shared<Primitive>("ComputeAccidentalHits"));
GVAR_DEF(PrimitivePtr, kPrimCacheSwapTable, std::make_shared<Primitive>("CacheSwapTable"));
GVAR_DEF(PrimitivePtr, kPrimDynamicAssign, std::make_shared<Primitive>("DynamicAssign"));
GVAR_DEF(PrimitivePtr, kPrimPadAndShift, std::make_shared<Primitive>("PadAndShift"));
GVAR_DEF(PrimitivePtr, kPrimSlice, std::make_shared<Primitive>("Slice"));
GVAR_DEF(PrimitivePtr, kPrimSliceGrad, std::make_shared<Primitive>("SliceGrad"));
GVAR_DEF(PrimitivePtr, kPrimSliceFusion, std::make_shared<Primitive>("SliceFusion"));
GVAR_DEF(PrimitivePtr, kPrimTile, std::make_shared<Primitive>(kTile));
GVAR_DEF(PrimitivePtr, kPrimAddN, std::make_shared<Primitive>("AddN"));
GVAR_DEF(PrimitivePtr, kPrimAccumulateNV2, std::make_shared<Primitive>("AccumulateNV2"));
GVAR_DEF(PrimitivePtr, kPrimTransData, std::make_shared<Primitive>("TransData"));
GVAR_DEF(PrimitivePtr, kPrimTransDataRNN, std::make_shared<Primitive>("TransDataRNN"));
GVAR_DEF(PrimitivePtr, kPrimNMSWithMask, std::make_shared<Primitive>("NMSWithMask"));
GVAR_DEF(PrimitivePtr, kPrimPad, std::make_shared<Primitive>("Pad"));
GVAR_DEF(PrimitivePtr, kPrimArgMaxWithValue, std::make_shared<Primitive>("ArgMaxWithValue"));
GVAR_DEF(PrimitivePtr, kPrimUnique, std::make_shared<Primitive>("Unique"));
GVAR_DEF(PrimitivePtr, kPrimUniqueGrad, std::make_shared<Primitive>("UniqueGrad"));
GVAR_DEF(PrimitivePtr, kPrimExtractImagePatches, std::make_shared<Primitive>("ExtractImagePatches"));
GVAR_DEF(PrimitivePtr, kPrimDynamicRNN, std::make_shared<Primitive>("DynamicRNN"));
GVAR_DEF(PrimitivePtr, kPrimDynamicRNNGrad, std::make_shared<Primitive>("DynamicRNNGrad"));
GVAR_DEF(PrimitivePtr, kPrimDynamicGRUV2, std::make_shared<Primitive>("DynamicGRUV2"));
GVAR_DEF(PrimitivePtr, kPrimDynamicGRUV2Grad, std::make_shared<Primitive>("DynamicGRUV2Grad"));
GVAR_DEF(PrimitivePtr, kPrimScatterAdd, std::make_shared<Primitive>("ScatterAdd"));
GVAR_DEF(PrimitivePtr, kPrimScatterSub, std::make_shared<Primitive>("ScatterSub"));
GVAR_DEF(PrimitivePtr, kPrimScatterMul, std::make_shared<Primitive>("ScatterMul"));
GVAR_DEF(PrimitivePtr, kPrimScatterDiv, std::make_shared<Primitive>("ScatterDiv"));
GVAR_DEF(PrimitivePtr, kPrimScatterMax, std::make_shared<Primitive>("ScatterMax"));
GVAR_DEF(PrimitivePtr, kPrimScatterMin, std::make_shared<Primitive>("ScatterMin"));
GVAR_DEF(PrimitivePtr, kPrimScatterNdAdd, std::make_shared<Primitive>("ScatterNdAdd"));
GVAR_DEF(PrimitivePtr, kPrimScatterUpdate, std::make_shared<Primitive>("ScatterUpdate"));
GVAR_DEF(PrimitivePtr, kPrimScatterElements, std::make_shared<Primitive>("ScatterElements"));
GVAR_DEF(PrimitivePtr, kPrimTensorCopySlices, std::make_shared<Primitive>("TensorCopySlices"));
GVAR_DEF(PrimitivePtr, kPrimMapUniform, std::make_shared<Primitive>("MapUniform"));
GVAR_DEF(PrimitivePtr, kPrimSplit, std::make_shared<Primitive>("Split"));
GVAR_DEF(PrimitivePtr, kPrimSplitV, std::make_shared<Primitive>(kSplitV));
GVAR_DEF(PrimitivePtr, kPrimSequenceMask, std::make_shared<Primitive>("SequenceMask"));
GVAR_DEF(PrimitivePtr, kPrimRange, std::make_shared<Primitive>("Range"));
GVAR_DEF(PrimitivePtr, kPrimSpaceToBatchND, std::make_shared<Primitive>("SpaceToBatchND"));
GVAR_DEF(PrimitivePtr, kPrimBatchToSpaceND, std::make_shared<Primitive>("BatchToSpaceND"));
GVAR_DEF(PrimitivePtr, kPrimDepthToSpace, std::make_shared<Primitive>("DepthToSpace"));
GVAR_DEF(PrimitivePtr, kPrimBatchToSpace, std::make_shared<Primitive>("BatchToSpace"));
GVAR_DEF(PrimitivePtr, kPrimSpaceToBatch, std::make_shared<Primitive>("SpaceToBatch"));
GVAR_DEF(PrimitivePtr, kPrimScatterNd, std::make_shared<Primitive>("ScatterNd"));
GVAR_DEF(PrimitivePtr, kPrimScatterNdUpdate, std::make_shared<Primitive>("ScatterNdUpdate"));
GVAR_DEF(PrimitivePtr, kPrimScatterNonAliasingAdd, std::make_shared<Primitive>("ScatterNonAliasingAdd"));
GVAR_DEF(PrimitivePtr, kPrimConstantOfShape, std::make_shared<Primitive>("ConstantOfShape"));
GVAR_DEF(PrimitivePtr, kPrimSquaredDifference, std::make_shared<Primitive>("SquaredDifference"));
GVAR_DEF(PrimitivePtr, kPrimReverseV2, std::make_shared<Primitive>("ReverseV2"));
GVAR_DEF(PrimitivePtr, kPrimReverseSequence, std::make_shared<Primitive>("ReverseSequence"));
GVAR_DEF(PrimitivePtr, kPrimRank, std::make_shared<Primitive>("Rank"));
GVAR_DEF(PrimitivePtr, kPrimResizeBilinear, std::make_shared<Primitive>("ResizeBilinear"));
GVAR_DEF(PrimitivePtr, kPrimParallelResizeBilinear, std::make_shared<Primitive>("ParallelResizeBilinear"));
GVAR_DEF(PrimitivePtr, kPrimParallelResizeBilinearGrad, std::make_shared<Primitive>("ParallelResizeBilinearGrad"));
GVAR_DEF(PrimitivePtr, kPrimResizeGrad, std::make_shared<Primitive>("ResizeGrad"));
GVAR_DEF(PrimitivePtr, kPrimResizeNearestNeighbor, std::make_shared<Primitive>("ResizeNearestNeighbor"));
GVAR_DEF(PrimitivePtr, kPrimResizeNearestNeighborGrad, std::make_shared<Primitive>("ResizeNearestNeighborGrad"));
GVAR_DEF(PrimitivePtr, kPrimDynamicResizeNearestNeighbor, std::make_shared<Primitive>("DynamicResizeNearestNeighbor"));
GVAR_DEF(PrimitivePtr, kPrimSort, std::make_shared<Primitive>("Sort"));
GVAR_DEF(PrimitivePtr, kPrimMaskedFill, std::make_shared<Primitive>("MaskedFill"));
GVAR_DEF(PrimitivePtr, kPrimMaskedSelect, std::make_shared<Primitive>("MaskedSelect"));
GVAR_DEF(PrimitivePtr, kPrimDiag, std::make_shared<Primitive>(kDiag));
GVAR_DEF(PrimitivePtr, kPrimDiagPart, std::make_shared<Primitive>(kDiagPart));
GVAR_DEF(PrimitivePtr, kPrimNonZero, std::make_shared<Primitive>("NonZero"));
GVAR_DEF(PrimitivePtr, kPrimRealInner, std::make_shared<Primitive>(kRealInner));
GVAR_DEF(PrimitivePtr, kPrimReal, std::make_shared<Primitive>(kReal));
GVAR_DEF(PrimitivePtr, kPrimImag, std::make_shared<Primitive>(kImag));
GVAR_DEF(PrimitivePtr, kPrimConj, std::make_shared<Primitive>(kConj));
GVAR_DEF(PrimitivePtr, kPrimExtractVolumePatches, std::make_shared<Primitive>("ExtractVolumePatches"));
GVAR_DEF(PrimitivePtr, kPrimLstsq, std::make_shared<Primitive>(kLstsq));
GVAR_DEF(PrimitivePtr, kPrimLowerBound, std::make_shared<Primitive>(kLowerBound));
GVAR_DEF(PrimitivePtr, kPrimUpperBound, std::make_shared<Primitive>(kUpperBound));
GVAR_DEF(PrimitivePtr, kPrimCummax, std::make_shared<Primitive>(kCummax));

// NN
GVAR_DEF(PrimitivePtr, kPrimCeLU, std::make_shared<Primitive>("CeLU"));
GVAR_DEF(PrimitivePtr, kPrimAdam, std::make_shared<Primitive>("Adam"));
GVAR_DEF(PrimitivePtr, kPrimApplyAdaMax, std::make_shared<Primitive>("ApplyAdaMax"));
GVAR_DEF(PrimitivePtr, kPrimAudioSpectrogram, std::make_shared<Primitive>("AudioSpectrogram"));
GVAR_DEF(PrimitivePtr, kPrimFlatten, std::make_shared<Primitive>("Flatten"));
GVAR_DEF(PrimitivePtr, kPrimCrop, std::make_shared<Primitive>("Crop"));
GVAR_DEF(PrimitivePtr, kPrimFlattenGrad, std::make_shared<Primitive>("FlattenGrad"));
GVAR_DEF(PrimitivePtr, kPrimSoftmax, std::make_shared<Primitive>("Softmax"));
GVAR_DEF(PrimitivePtr, kPrimSoftsign, std::make_shared<Primitive>("Softsign"));
GVAR_DEF(PrimitivePtr, kPrimSparseSoftmaxCrossEntropy, std::make_shared<Primitive>("SparseSoftmaxCrossEntropy"));
GVAR_DEF(PrimitivePtr, kPrimSoftmaxV2WithDropoutDoMaskV3, std::make_shared<Primitive>("SoftmaxV2WithDropoutDoMaskV3"));
GVAR_DEF(PrimitivePtr, kPrimLogSoftmax, std::make_shared<Primitive>("LogSoftmax"));
GVAR_DEF(PrimitivePtr, kPrimLogSoftmaxGrad, std::make_shared<Primitive>("LogSoftmaxGrad"));
GVAR_DEF(PrimitivePtr, kPrimLstm, std::make_shared<Primitive>("LSTM"));
GVAR_DEF(PrimitivePtr, kPrimTan, std::make_shared<Primitive>("Tan"));
GVAR_DEF(PrimitivePtr, kPrimAtan2, std::make_shared<Primitive>("Atan2"));
GVAR_DEF(PrimitivePtr, kPrimAtan, std::make_shared<Primitive>("Atan"));
GVAR_DEF(PrimitivePtr, kPrimAsin, std::make_shared<Primitive>(kAsin));
GVAR_DEF(PrimitivePtr, kPrimSinh, std::make_shared<Primitive>("Sinh"));
GVAR_DEF(PrimitivePtr, kPrimCosh, std::make_shared<Primitive>("Cosh"));
GVAR_DEF(PrimitivePtr, kPrimTanh, std::make_shared<Primitive>(kTanh));
GVAR_DEF(PrimitivePtr, kPrimAsinh, std::make_shared<Primitive>(kAsinh));
GVAR_DEF(PrimitivePtr, kPrimAcosh, std::make_shared<Primitive>(kAcosh));
GVAR_DEF(PrimitivePtr, kPrimAtanh, std::make_shared<Primitive>("Atanh"));
GVAR_DEF(PrimitivePtr, kPrimApplyGradientDescent, std::make_shared<Primitive>("ApplyGradientDescent"));
GVAR_DEF(PrimitivePtr, kPrimApplyPowerSignD, std::make_shared<Primitive>("ApplyPowerSign"));
GVAR_DEF(PrimitivePtr, kPrimBesselI0e, std::make_shared<Primitive>("BesselI0e"));
GVAR_DEF(PrimitivePtr, kPrimBesselI1e, std::make_shared<Primitive>("BesselI1e"));
GVAR_DEF(PrimitivePtr, kPrimTanhGrad, std::make_shared<Primitive>("TanhGrad"));
GVAR_DEF(PrimitivePtr, kPrimPooling, std::make_shared<Primitive>("Pooling"));
GVAR_DEF(PrimitivePtr, kPrimPoolingGrad, std::make_shared<Primitive>("PoolingGrad"));
GVAR_DEF(PrimitivePtr, kPrimROIPooling, std::make_shared<Primitive>("ROIPooling"));
GVAR_DEF(PrimitivePtr, kPrimMaxPool, std::make_shared<Primitive>("MaxPool"));
GVAR_DEF(PrimitivePtr, kPrimMaxPoolGrad, std::make_shared<Primitive>("MaxPoolGrad"));
GVAR_DEF(PrimitivePtr, kPrimMaxPoolWithArgmax, std::make_shared<Primitive>("MaxPoolWithArgmax"));
GVAR_DEF(PrimitivePtr, kPrimMaxPoolGradWithArgmax, std::make_shared<Primitive>("MaxPoolGradWithArgmax"));
GVAR_DEF(PrimitivePtr, kPrimApplyCenteredRMSProp, std::make_shared<Primitive>("ApplyCenteredRMSProp"));
GVAR_DEF(PrimitivePtr, kPrimAvgPool, std::make_shared<Primitive>("AvgPool"));
GVAR_DEF(PrimitivePtr, kPrimAvgPool3D, std::make_shared<Primitive>("AvgPool3D"));
GVAR_DEF(PrimitivePtr, kPrimAvgPoolGrad, std::make_shared<Primitive>("AvgPoolGrad"));
GVAR_DEF(PrimitivePtr, kPrimAvgPool3DGrad, std::make_shared<Primitive>("AvgPool3DGrad"));
GVAR_DEF(PrimitivePtr, kPrimAvgPoolGradVm, std::make_shared<Primitive>("AvgPoolGradVm"));
GVAR_DEF(PrimitivePtr, kPrimFusedSparseAdam, std::make_shared<Primitive>("FusedSparseAdam"));
GVAR_DEF(PrimitivePtr, kPrimFusedBatchNorm, std::make_shared<Primitive>("FusedBatchNorm"));
GVAR_DEF(PrimitivePtr, kPrimConv2D, std::make_shared<Primitive>("Conv2D"));
GVAR_DEF(PrimitivePtr, kPrimConv3D, std::make_shared<Primitive>("Conv3D"));
GVAR_DEF(PrimitivePtr, kPrimCTCLossV2, std::make_shared<Primitive>("CTCLossV2"));
GVAR_DEF(PrimitivePtr, kPrimCTCLossV2Grad, std::make_shared<Primitive>("CTCLossV2Grad"));
GVAR_DEF(PrimitivePtr, kPrimCTCLoss, std::make_shared<Primitive>(kCTCLoss));
GVAR_DEF(PrimitivePtr, kPrimFullConnection, std::make_shared<Primitive>("FullConnection"));
GVAR_DEF(PrimitivePtr, kPrimConv2DTranspose, std::make_shared<Primitive>(kConv2DTranspose));
GVAR_DEF(PrimitivePtr, kPrimConv3DTranspose, std::make_shared<Primitive>("Conv3DTranspose"));
GVAR_DEF(PrimitivePtr, kPrimRoll, std::make_shared<Primitive>(kRoll));
GVAR_DEF(PrimitivePtr, kPrimGroupConv2DGradInput, std::make_shared<Primitive>("GroupConv2DGradInput"));
GVAR_DEF(PrimitivePtr, kPrimBatchNorm, std::make_shared<Primitive>("BatchNorm"));
GVAR_DEF(PrimitivePtr, kPrimBatchNormGrad, std::make_shared<Primitive>("BatchNormGrad"));
GVAR_DEF(PrimitivePtr, kPrimSyncBatchNorm, std::make_shared<Primitive>("SyncBatchNorm"));
GVAR_DEF(PrimitivePtr, kPrimSyncBatchNormGrad, std::make_shared<Primitive>("SyncBatchNormGrad"));
GVAR_DEF(PrimitivePtr, kPrimBNTrainingReduce, std::make_shared<Primitive>("BNTrainingReduce"));
GVAR_DEF(PrimitivePtr, kPrimBNTrainingReduceGrad, std::make_shared<Primitive>("BNTrainingReduceGrad"));
GVAR_DEF(PrimitivePtr, kPrimReluGrad, std::make_shared<Primitive>(kReLUGrad));
GVAR_DEF(PrimitivePtr, kPrimReluGradV2, std::make_shared<Primitive>("ReluGradV2"));
GVAR_DEF(PrimitivePtr, kPrimRelu6Grad, std::make_shared<Primitive>("ReLU6Grad"));
GVAR_DEF(PrimitivePtr, kPrimConv2DBackpropInput, std::make_shared<Primitive>("Conv2DBackpropInput"));
GVAR_DEF(PrimitivePtr, kPrimConv2DBackpropFilter, std::make_shared<Primitive>("Conv2DBackpropFilter"));
GVAR_DEF(PrimitivePtr, kPrimConv3DBackpropInput, std::make_shared<Primitive>("Conv3DBackpropInput"));
GVAR_DEF(PrimitivePtr, kPrimConv3DBackpropFilter, std::make_shared<Primitive>("Conv3DBackpropFilter"));
GVAR_DEF(PrimitivePtr, kPrimCustomNormalize, std::make_shared<Primitive>("CustomNormalize"));
GVAR_DEF(PrimitivePtr, kPrimDepthwiseConv2dNative, std::make_shared<Primitive>("DepthwiseConv2dNative"));
GVAR_DEF(PrimitivePtr, kPrimCTCGreedyDecoder, std::make_shared<Primitive>("CTCGreedyDecoder"));
GVAR_DEF(PrimitivePtr, kPrimDataFormatDimMap, std::make_shared<Primitive>("DataFormatDimMap"));
GVAR_DEF(PrimitivePtr, kPrimDynamicStitch, std::make_shared<Primitive>("DynamicStitch"));
GVAR_DEF(PrimitivePtr, kPrimDepthwiseConv2dNativeBackpropFilter,
         std::make_shared<Primitive>("DepthwiseConv2dNativeBackpropFilter"));
GVAR_DEF(PrimitivePtr, kPrimDepthwiseConv2dNativeBackpropInput,
         std::make_shared<Primitive>("DepthwiseConv2dNativeBackpropInput"));
GVAR_DEF(PrimitivePtr, kPrimDetectionPostProcess, std::make_shared<Primitive>("DetectionPostProcess"));
GVAR_DEF(PrimitivePtr, kPrimBiasAddGrad, std::make_shared<Primitive>(kBiasAddGrad));
GVAR_DEF(PrimitivePtr, kPrimBiasAdd, std::make_shared<Primitive>(kBiasAdd));
GVAR_DEF(PrimitivePtr, kPrimBiasSubGrad, std::make_shared<Primitive>("BiasSubGrad"));
GVAR_DEF(PrimitivePtr, kPrimBinaryCrossEntropy, std::make_shared<Primitive>("BinaryCrossEntropy"));
GVAR_DEF(PrimitivePtr, kPrimBinaryCrossEntropyGrad, std::make_shared<Primitive>("BinaryCrossEntropyGrad"));
GVAR_DEF(PrimitivePtr, kPrimSmoothL1Loss, std::make_shared<Primitive>("SmoothL1Loss"));
GVAR_DEF(PrimitivePtr, kPrimSmoothL1LossGrad, std::make_shared<Primitive>("SmoothL1LossGrad"));
GVAR_DEF(PrimitivePtr, kPrimSoftMarginLoss, std::make_shared<Primitive>("SoftMarginLoss"));
GVAR_DEF(PrimitivePtr, kPrimSoftMarginLossGrad, std::make_shared<Primitive>("SoftMarginLossGrad"));
GVAR_DEF(PrimitivePtr, kPrimSoftmaxCrossEntropyWithLogits,
         std::make_shared<Primitive>("SoftmaxCrossEntropyWithLogits"));
GVAR_DEF(PrimitivePtr, kPrimL2Loss, std::make_shared<Primitive>("L2Loss"));
GVAR_DEF(PrimitivePtr, kPrimSigmoidCrossEntropyWithLogits,
         std::make_shared<Primitive>("SigmoidCrossEntropyWithLogits"));
GVAR_DEF(PrimitivePtr, kPrimSigmoidCrossEntropyWithLogitsGrad,
         std::make_shared<Primitive>("SigmoidCrossEntropyWithLogitsGrad"));
GVAR_DEF(PrimitivePtr, kPrimSparseSoftmaxCrossEntropyWithLogits,
         std::make_shared<Primitive>("SparseSoftmaxCrossEntropyWithLogits"));
GVAR_DEF(PrimitivePtr, kPrimMomentum, std::make_shared<Primitive>("Momentum"));
GVAR_DEF(PrimitivePtr, kPrimApplyMomentum, std::make_shared<Primitive>("ApplyMomentum"));
GVAR_DEF(PrimitivePtr, kPrimApplyFtrl, std::make_shared<Primitive>("ApplyFtrl"));
GVAR_DEF(PrimitivePtr, kPrimLrn, std::make_shared<Primitive>("LRN"));
GVAR_DEF(PrimitivePtr, kPrimLayerNorm, std::make_shared<Primitive>(kLayerNorm));
GVAR_DEF(PrimitivePtr, kPrimLayerNormGrad, std::make_shared<Primitive>(kLayerNormGrad));
GVAR_DEF(PrimitivePtr, kPrimLayerNormXBackprop, std::make_shared<Primitive>("LayerNormXBackprop"));
GVAR_DEF(PrimitivePtr, kPrimLayerNormXBackpropV2, std::make_shared<Primitive>("LayerNormXBackpropV2"));
GVAR_DEF(PrimitivePtr, kPrimLayerNormBetaGammaBackprop, std::make_shared<Primitive>("LayerNormBetaGammaBackprop"));
GVAR_DEF(PrimitivePtr, kPrimLayerNormBetaGammaBackpropV2, std::make_shared<Primitive>("LayerNormBetaGammaBackpropV2"));
GVAR_DEF(PrimitivePtr, kPrimLog1p, std::make_shared<Primitive>("Log1p"));
GVAR_DEF(PrimitivePtr, kPrimDropoutGenMask, std::make_shared<Primitive>(kDropoutGenMask));
GVAR_DEF(PrimitivePtr, kPrimDropoutDoMask, std::make_shared<Primitive>(kDropoutDoMask));
GVAR_DEF(PrimitivePtr, kPrimDropoutDoMaskV3, std::make_shared<Primitive>(kDropoutDoMaskV3));
GVAR_DEF(PrimitivePtr, kPrimDropoutGrad, std::make_shared<Primitive>(kDropoutGrad));
GVAR_DEF(PrimitivePtr, kPrimDropout, std::make_shared<Primitive>(kDropout));
GVAR_DEF(PrimitivePtr, kPrimUniformReal, std::make_shared<Primitive>("UniformReal"));
GVAR_DEF(PrimitivePtr, kPrimCudnnUniformReal, std::make_shared<Primitive>("CudnnUniformReal"));
GVAR_DEF(PrimitivePtr, kPrimOneHot, std::make_shared<Primitive>("OneHot"));
GVAR_DEF(PrimitivePtr, kPrimGeLU, std::make_shared<Primitive>(kGeLU));
GVAR_DEF(PrimitivePtr, kPrimGeLUGrad, std::make_shared<Primitive>(kGeLUGrad));
GVAR_DEF(PrimitivePtr, kPrimFastGeLU, std::make_shared<Primitive>(kFastGeLU));
GVAR_DEF(PrimitivePtr, kPrimFastGeLUGrad, std::make_shared<Primitive>(kFastGeLUGrad));
GVAR_DEF(PrimitivePtr, kPrimRelu, std::make_shared<Primitive>(kReLU));
GVAR_DEF(PrimitivePtr, kPrimElu, std::make_shared<Primitive>("Elu"));
GVAR_DEF(PrimitivePtr, kPrimEluGrad, std::make_shared<Primitive>("EluGrad"));
GVAR_DEF(PrimitivePtr, kPrimRelu6, std::make_shared<Primitive>(kReLU6));
GVAR_DEF(PrimitivePtr, kPrimReluV2, std::make_shared<Primitive>(kReLUV2));
GVAR_DEF(PrimitivePtr, kPrimPRelu, std::make_shared<Primitive>("PReLU"));
GVAR_DEF(PrimitivePtr, kPrimSelu, std::make_shared<Primitive>("SeLU"));
GVAR_DEF(PrimitivePtr, kPrimSoftplus, std::make_shared<Primitive>("Softplus"));
GVAR_DEF(PrimitivePtr, kPrimSoftplusGrad, std::make_shared<Primitive>("SoftplusGrad"));
GVAR_DEF(PrimitivePtr, kPrimZeros, std::make_shared<Primitive>("Zeros"));
GVAR_DEF(PrimitivePtr, kPrimZerosLike, std::make_shared<Primitive>(kZerosLike));
GVAR_DEF(PrimitivePtr, kPrimOnes, std::make_shared<Primitive>(kOnes));
GVAR_DEF(PrimitivePtr, kPrimOnesLike, std::make_shared<Primitive>(kOnesLike));
GVAR_DEF(PrimitivePtr, kPrimBpropCut, std::make_shared<Primitive>("bprop_cut"));
GVAR_DEF(PrimitivePtr, kPrimFakeQuantPerLayer, std::make_shared<Primitive>("FakeQuantPerLayer"));
GVAR_DEF(PrimitivePtr, kPrimFakeQuantPerChannel, std::make_shared<Primitive>("FakeQuantPerChannel"));
GVAR_DEF(PrimitivePtr, kPrimFakeLearnedScaleQuantPerLayer,
         std::make_shared<Primitive>("FakeLearnedScaleQuantPerLayer"));
GVAR_DEF(PrimitivePtr, kPrimFakeLearnedScaleQuantPerChannel,
         std::make_shared<Primitive>("FakeLearnedScaleQuantPerChannel"));
GVAR_DEF(PrimitivePtr, kPrimFakeQuantWithMinMaxVars, std::make_shared<Primitive>("FakeQuantWithMinMaxVars"));
GVAR_DEF(PrimitivePtr, kPrimApplyRMSProp, std::make_shared<Primitive>("ApplyRMSProp"));
GVAR_DEF(PrimitivePtr, kPrimSparseApplyFtrl, std::make_shared<Primitive>("SparseApplyFtrl"));
GVAR_DEF(PrimitivePtr, kPrimSparseApplyProximalAdagrad, std::make_shared<Primitive>("SparseApplyProximalAdagrad"));
GVAR_DEF(PrimitivePtr, kPrimFusedAdam, std::make_shared<Primitive>("FusedAdam"));
GVAR_DEF(PrimitivePtr, kPrimFusedAdamWeightDecay, std::make_shared<Primitive>("FusedAdamWeightDecay"));
GVAR_DEF(PrimitivePtr, kPrimSGD, std::make_shared<Primitive>("SGD"));
GVAR_DEF(PrimitivePtr, kPrimBCEWithLogitsLoss, std::make_shared<Primitive>("BCEWithLogitsLoss"));
GVAR_DEF(PrimitivePtr, kPrimClipByNormNoDivSum, std::make_shared<Primitive>("ClipByNormNoDivSum"));
GVAR_DEF(PrimitivePtr, kPrimTensorMove, std::make_shared<Primitive>("TensorMove"));
GVAR_DEF(PrimitivePtr, kPrimL2Normalize, std::make_shared<Primitive>("L2Normalize"));
GVAR_DEF(PrimitivePtr, kPrimCustomExtractFeatures, std::make_shared<Primitive>("CustomExtractFeatures"));
GVAR_DEF(PrimitivePtr, kLambApplyOptimizerAssign, std::make_shared<Primitive>("LambApplyOptimizerAssign"));
GVAR_DEF(PrimitivePtr, kLambApplyWeightAssign, std::make_shared<Primitive>("LambApplyWeightAssign"));
GVAR_DEF(PrimitivePtr, kSoftmaxGradExt, std::make_shared<Primitive>("SoftmaxGradExt"));
GVAR_DEF(PrimitivePtr, kPrimSparseApplyAdadelta, std::make_shared<Primitive>(kSparseApplyAdadelta));
GVAR_DEF(PrimitivePtr, kSquareSumV1, std::make_shared<Primitive>("SquareSumV1"));
GVAR_DEF(PrimitivePtr, kFusedMulAdd, std::make_shared<Primitive>("FusedMulAdd"));
GVAR_DEF(PrimitivePtr, kPrimSoftShrink, std::make_shared<Primitive>("SoftShrink"));
GVAR_DEF(PrimitivePtr, kPrimSoftShrinkGrad, std::make_shared<Primitive>("SoftShrinkGrad"));
GVAR_DEF(PrimitivePtr, kPrimHShrink, std::make_shared<Primitive>("HShrink"));
GVAR_DEF(PrimitivePtr, kPrimHShrinkGrad, std::make_shared<Primitive>("HShrinkGrad"));
GVAR_DEF(PrimitivePtr, kPrimHSVToRGB, std::make_shared<Primitive>("HSVToRGB"));
GVAR_DEF(PrimitivePtr, kPrimApplyAdagradDA, std::make_shared<Primitive>("ApplyAdagradDA"));
GVAR_DEF(PrimitivePtr, kPrimApplyAdagradV2, std::make_shared<Primitive>("ApplyAdagradV2"));
GVAR_DEF(PrimitivePtr, kPrimApplyProximalGradientDescent, std::make_shared<Primitive>("ApplyProximalGradientDescent"));
GVAR_DEF(PrimitivePtr, kPrimSparseApplyRMSProp, std::make_shared<Primitive>("SparseApplyRMSProp"));
GVAR_DEF(PrimitivePtr, kPrimApplyKerasMomentum, std::make_shared<Primitive>("ApplyKerasMomentum"));
GVAR_DEF(PrimitivePtr, kPrimLARSUpdate, std::make_shared<Primitive>("LARSUpdate"));
GVAR_DEF(PrimitivePtr, kPrimBoundingBoxDecode, std::make_shared<Primitive>("BoundingBoxDecode"));
GVAR_DEF(PrimitivePtr, kPrimROIAlign, std::make_shared<Primitive>("ROIAlign"));
GVAR_DEF(PrimitivePtr, kPrimApplyAddSign, std::make_shared<Primitive>("ApplyAddSign"));
GVAR_DEF(PrimitivePtr, kPrimApplyAdagrad, std::make_shared<Primitive>("ApplyAdagrad"));
GVAR_DEF(PrimitivePtr, kPrimApplyAdadelta, std::make_shared<Primitive>("ApplyAdadelta"));
GVAR_DEF(PrimitivePtr, kPrimApplyAdamWithAmsgrad, std::make_shared<Primitive>("ApplyAdamWithAmsgrad"));
GVAR_DEF(PrimitivePtr, kPrimGridSampler3D, std::make_shared<Primitive>(kGridSampler3D));
GVAR_DEF(PrimitivePtr, kPrimGridSampler3DGrad, std::make_shared<Primitive>(kGridSampler3DGrad));
GVAR_DEF(PrimitivePtr, kPrimBNTrainingUpdate, std::make_shared<Primitive>("BNTrainingUpdate"));
GVAR_DEF(PrimitivePtr, kPrimBNTrainingUpdateGrad, std::make_shared<Primitive>("BNTrainingUpdateGrad"));

// Comm ops
GVAR_DEF(PrimitivePtr, kPrimMirror, std::make_shared<Primitive>("_MirrorOperator"));
GVAR_DEF(PrimitivePtr, kPrimMirrorMiniStep, std::make_shared<Primitive>("_MirrorMiniStepOperator"));
GVAR_DEF(PrimitivePtr, kPrimMiniStepAllGather, std::make_shared<Primitive>("_MiniStepAllGather"));
GVAR_DEF(PrimitivePtr, kPrimMicroStepAllGather, std::make_shared<Primitive>("_MicroStepAllGather"));
GVAR_DEF(PrimitivePtr, kPrimVirtualDiv, std::make_shared<Primitive>("_VirtualDiv"));
GVAR_DEF(PrimitivePtr, kPrimVirtualAdd, std::make_shared<Primitive>("_VirtualAdd"));
GVAR_DEF(PrimitivePtr, kPrimVirtualDataset, std::make_shared<Primitive>("_VirtualDataset"));
GVAR_DEF(PrimitivePtr, kPrimVirtualOutput, std::make_shared<Primitive>("_VirtualOutput"));
GVAR_DEF(PrimitivePtr, kPrimSend, std::make_shared<Primitive>("Send"));
GVAR_DEF(PrimitivePtr, kPrimReceive, std::make_shared<Primitive>("Receive"));
GVAR_DEF(PrimitivePtr, kPrimRpcSend, std::make_shared<Primitive>("RpcSend"));
GVAR_DEF(PrimitivePtr, kPrimRpcRecv, std::make_shared<Primitive>("RpcRecv"));
GVAR_DEF(PrimitivePtr, kPrimAllReduce, std::make_shared<Primitive>("AllReduce"));
GVAR_DEF(PrimitivePtr, kPrimNeighborExchange, std::make_shared<Primitive>("NeighborExchange"));
GVAR_DEF(PrimitivePtr, kPrimNeighborExchangeV2, std::make_shared<Primitive>("NeighborExchangeV2"));
GVAR_DEF(PrimitivePtr, kPrimNeighborExchangeV2Grad, std::make_shared<Primitive>("NeighborExchangeV2Grad"));
GVAR_DEF(PrimitivePtr, kPrimAllToAll, std::make_shared<Primitive>("AlltoAll"));
GVAR_DEF(PrimitivePtr, kPrimAllToAllv, std::make_shared<Primitive>("AllToAllv"));
GVAR_DEF(PrimitivePtr, kPrimAllSwap, std::make_shared<Primitive>("_AllSwap"));
GVAR_DEF(PrimitivePtr, kPrimBroadcast, std::make_shared<Primitive>("Broadcast"));
GVAR_DEF(PrimitivePtr, kPrimAllGather, std::make_shared<Primitive>("AllGather"));
GVAR_DEF(PrimitivePtr, kPrimReduceScatter, std::make_shared<Primitive>("ReduceScatter"));
GVAR_DEF(PrimitivePtr, kPrimMemCpyAsync, std::make_shared<Primitive>("memcpy_async"));
GVAR_DEF(PrimitivePtr, kPrimFill, std::make_shared<Primitive>("Fill"));
GVAR_DEF(PrimitivePtr, kPrimFusedPushWeight, std::make_shared<Primitive>("FusedPushWeight"));
GVAR_DEF(PrimitivePtr, kPrimFusedPullWeight, std::make_shared<Primitive>("FusedPullWeight"));
GVAR_DEF(PrimitivePtr, kPrimInitDataSetQueue, std::make_shared<Primitive>("InitDataSetQueue"));
GVAR_DEF(PrimitivePtr, kPrimVirtualAssignAdd, std::make_shared<Primitive>("_VirtualAssignAdd"));
GVAR_DEF(PrimitivePtr, kPrimVirtualAccuGrad, std::make_shared<Primitive>("_VirtualAccuGrad"));
GVAR_DEF(PrimitivePtr, kPrimMirrorMicroStep, std::make_shared<Primitive>("_MirrorMicroStepOperator"));
GVAR_DEF(PrimitivePtr, kPrimApplyProximalAdagrad, std::make_shared<Primitive>("ApplyProximalAdagrad"));

// Quant ops
GVAR_DEF(PrimitivePtr, kPrimBatchNormFold, std::make_shared<Primitive>("BatchNormFold"));
GVAR_DEF(PrimitivePtr, kPrimFakeQuantWithMinMaxVarsPerChannel,
         std::make_shared<Primitive>("FakeQuantWithMinMaxVarsPerChannel"));
// Control ops
GVAR_DEF(PrimitivePtr, kPrimMerge, std::make_shared<Primitive>("Merge"));
// RowTensor
GVAR_DEF(PrimitivePtr, kPrimMakeRowTensor, std::make_shared<Primitive>("MakeRowTensor"));
GVAR_DEF(PrimitivePtr, kPrimRowTensorGetValues, std::make_shared<Primitive>("RowTensorGetValues"));
GVAR_DEF(PrimitivePtr, kPrimRowTensorGetIndices, std::make_shared<Primitive>("RowTensorGetIndices"));
GVAR_DEF(PrimitivePtr, kPrimRowTensorGetDenseShape, std::make_shared<Primitive>("RowTensorGetDenseShape"));
GVAR_DEF(PrimitivePtr, kPrimRowTensorAdd, std::make_shared<Primitive>("RowTensorAdd"));

// COOTensor
GVAR_DEF(PrimitivePtr, kPrimMakeCOOTensor, std::make_shared<Primitive>(kMakeCOOTensor));
GVAR_DEF(PrimitivePtr, kPrimCOOTensorGetValues, std::make_shared<Primitive>(kCOOTensorGetValues));
GVAR_DEF(PrimitivePtr, kPrimCOOTensorGetIndices, std::make_shared<Primitive>(kCOOTensorGetIndices));
GVAR_DEF(PrimitivePtr, kPrimCOOTensorGetDenseShape, std::make_shared<Primitive>(kCOOTensorGetDenseShapes));

// CSRTensor
GVAR_DEF(PrimitivePtr, kPrimMakeCSRTensor, std::make_shared<Primitive>(kMakeCSRTensor));
GVAR_DEF(PrimitivePtr, kPrimCSRTensorGetValues, std::make_shared<Primitive>(kCSRTensorGetValues));
GVAR_DEF(PrimitivePtr, kPrimCSRTensorGetIndptr, std::make_shared<Primitive>(kCSRTensorGetIndptr));
GVAR_DEF(PrimitivePtr, kPrimCSRTensorGetIndices, std::make_shared<Primitive>(kCSRTensorGetIndices));
GVAR_DEF(PrimitivePtr, kPrimCSRTensorGetDenseShape, std::make_shared<Primitive>(kCSRTensorGetDenseShape));
GVAR_DEF(PrimitivePtr, kPrimIsCSRFunc, std::make_shared<Primitive>(kIsCSRFunc));

// Sparse ops
GVAR_DEF(PrimitivePtr, kPrimSparseTensorDenseMatmul, std::make_shared<Primitive>(kSparseTensorDenseMatmul));
GVAR_DEF(PrimitivePtr, kPrimCOOTensorDenseMatmul, std::make_shared<Primitive>(kCOOTensorDenseMatmul));
GVAR_DEF(PrimitivePtr, kPrimCSRDenseMul, std::make_shared<Primitive>(kCSRDenseMul));
GVAR_DEF(PrimitivePtr, kPrimCSRReduceSum, std::make_shared<Primitive>(kCSRReduceSum));
GVAR_DEF(PrimitivePtr, kPrimCSRMV, std::make_shared<Primitive>(kCSRMV));
GVAR_DEF(PrimitivePtr, kPrimCSRMul, std::make_shared<Primitive>(kCSRMul));
GVAR_DEF(PrimitivePtr, kPrimCSRGather, std::make_shared<Primitive>(kCSRGather));
GVAR_DEF(PrimitivePtr, kPrimCSR2COO, std::make_shared<Primitive>(kCSR2COO));
GVAR_DEF(PrimitivePtr, kPrimCOO2CSR, std::make_shared<Primitive>(kCOO2CSR));
GVAR_DEF(PrimitivePtr, kPrimCSRDiv, std::make_shared<Primitive>(kCSRDiv));

// TensorList
GVAR_DEF(PrimitivePtr, kPrimTensorListFromTensor, std::make_shared<Primitive>("TensorListFromTensor"));
GVAR_DEF(PrimitivePtr, kPrimTensorListReserve, std::make_shared<Primitive>("TensorListReserve"));
GVAR_DEF(PrimitivePtr, kPrimTensorListStack, std::make_shared<Primitive>("TensorListStack"));
GVAR_DEF(PrimitivePtr, kPrimTensorListSetItem, std::make_shared<Primitive>("TensorListSetItem"));

// Maths
GVAR_DEF(PrimitivePtr, kPrimCross, std::make_shared<Primitive>(kCross));
GVAR_DEF(PrimitivePtr, kPrimBesselI0, std::make_shared<Primitive>("BesselI0"));
GVAR_DEF(PrimitivePtr, kPrimBesselI1, std::make_shared<Primitive>("BesselI1"));
GVAR_DEF(PrimitivePtr, kPrimGer, std::make_shared<Primitive>("Ger"));
GVAR_DEF(PrimitivePtr, kPrimCeil, std::make_shared<Primitive>("Ceil"));
GVAR_DEF(PrimitivePtr, kPrimLuSolve, std::make_shared<Primitive>("LuSolve"));
GVAR_DEF(PrimitivePtr, kPrimCholeskyInverse, std::make_shared<Primitive>("CholeskyInverse"));
GVAR_DEF(PrimitivePtr, kPrimTensorAdd, std::make_shared<Primitive>("TensorAdd"));
GVAR_DEF(PrimitivePtr, kPrimAdd, std::make_shared<Primitive>(kAdd));
GVAR_DEF(PrimitivePtr, kPrimAddcdiv, std::make_shared<Primitive>(kAddcdiv));
GVAR_DEF(PrimitivePtr, kPrimAddcmul, std::make_shared<Primitive>(kAddcmul));
GVAR_DEF(PrimitivePtr, kPrimMatMul, std::make_shared<Primitive>("MatMul"));
GVAR_DEF(PrimitivePtr, kPrimMatMulV2, std::make_shared<Primitive>("MatMulV2"));
GVAR_DEF(PrimitivePtr, kPrimMatrixDiag, std::make_shared<Primitive>("MatrixDiag"));
GVAR_DEF(PrimitivePtr, kPrimMatrixDiagPart, std::make_shared<Primitive>("MatrixDiagPartV3"));
GVAR_DEF(PrimitivePtr, kPrimBatchMatMul, std::make_shared<Primitive>("BatchMatMul"));
GVAR_DEF(PrimitivePtr, kPrimBatchMatMulV2, std::make_shared<Primitive>("BatchMatMulV2"));
GVAR_DEF(PrimitivePtr, kPrimMaximumGrad, std::make_shared<Primitive>("MaximumGrad"));
GVAR_DEF(PrimitivePtr, kPrimMinimumGrad, std::make_shared<Primitive>("MinimumGrad"));
GVAR_DEF(PrimitivePtr, kPrimReduce, std::make_shared<Primitive>("Reduce"));
GVAR_DEF(PrimitivePtr, kPrimReduceMean, std::make_shared<Primitive>("ReduceMean"));
GVAR_DEF(PrimitivePtr, kPrimReduceSum, std::make_shared<Primitive>("ReduceSum"));
GVAR_DEF(PrimitivePtr, kPrimReduceAll, std::make_shared<Primitive>("ReduceAll"));
GVAR_DEF(PrimitivePtr, kPrimReduceAny, std::make_shared<Primitive>("ReduceAny"));
GVAR_DEF(PrimitivePtr, kPrimReduceMax, std::make_shared<Primitive>("ReduceMax"));
GVAR_DEF(PrimitivePtr, kPrimReduceMin, std::make_shared<Primitive>("ReduceMin"));
GVAR_DEF(PrimitivePtr, kPrimCentralization, std::make_shared<Primitive>("Centralization"));
GVAR_DEF(PrimitivePtr, kPrimNeg, std::make_shared<Primitive>(kNeg));
GVAR_DEF(PrimitivePtr, kPrimSin, std::make_shared<Primitive>("Sin"));
GVAR_DEF(PrimitivePtr, kPrimCos, std::make_shared<Primitive>(kCos));
GVAR_DEF(PrimitivePtr, kPrimSub, std::make_shared<Primitive>(kSub));
GVAR_DEF(PrimitivePtr, kPrimMul, std::make_shared<Primitive>(kMul));
GVAR_DEF(PrimitivePtr, kPrimMulNoNan, std::make_shared<Primitive>(kMulNoNan));
GVAR_DEF(PrimitivePtr, kPrimDiv, std::make_shared<Primitive>("Div"));
GVAR_DEF(PrimitivePtr, kPrimMod, std::make_shared<Primitive>("Mod"));
GVAR_DEF(PrimitivePtr, kPrimFloor, std::make_shared<Primitive>("Floor"));
GVAR_DEF(PrimitivePtr, kPrimInvert, std::make_shared<Primitive>("Invert"));
GVAR_DEF(PrimitivePtr, kPrimDivNoNan, std::make_shared<Primitive>("DivNoNan"));
GVAR_DEF(PrimitivePtr, kPrimMinimum, std::make_shared<Primitive>("Minimum"));
GVAR_DEF(PrimitivePtr, kPrimMaximum, std::make_shared<Primitive>("Maximum"));
GVAR_DEF(PrimitivePtr, kPrimSquare, std::make_shared<Primitive>(kSquare));
GVAR_DEF(PrimitivePtr, kPrimCumSum, std::make_shared<Primitive>("CumSum"));
GVAR_DEF(PrimitivePtr, kPrimCumProd, std::make_shared<Primitive>("CumProd"));
GVAR_DEF(PrimitivePtr, kPrimSubscalar, std::make_shared<Primitive>("Subscalar"));
GVAR_DEF(PrimitivePtr, kPrimInplaceAdd, std::make_shared<Primitive>("InplaceAdd"));
GVAR_DEF(PrimitivePtr, kPrimLpNorm, std::make_shared<Primitive>(kLpNorm));
GVAR_DEF(PrimitivePtr, kPrimInplaceSub, std::make_shared<Primitive>("InplaceSub"));
GVAR_DEF(PrimitivePtr, kPrimPow, std::make_shared<Primitive>("Pow"));
GVAR_DEF(PrimitivePtr, kPrimPower, std::make_shared<Primitive>("Power"));
GVAR_DEF(PrimitivePtr, kPrimRealDiv, std::make_shared<Primitive>(kRealDiv));
GVAR_DEF(PrimitivePtr, kPrimFloorDiv, std::make_shared<Primitive>("FloorDiv"));
GVAR_DEF(PrimitivePtr, kPrimTruncateDiv, std::make_shared<Primitive>("TruncateDiv"));
GVAR_DEF(PrimitivePtr, kPrimSqrt, std::make_shared<Primitive>("Sqrt"));
GVAR_DEF(PrimitivePtr, kPrimTruncateMod, std::make_shared<Primitive>("TruncateMod"));
GVAR_DEF(PrimitivePtr, kPrimSqrtGrad, std::make_shared<Primitive>("SqrtGrad"));
GVAR_DEF(PrimitivePtr, kPrimReciprocal, std::make_shared<Primitive>(kReciprocal));
GVAR_DEF(PrimitivePtr, kPrimReciprocalGrad, std::make_shared<Primitive>("ReciprocalGrad"));
GVAR_DEF(PrimitivePtr, kPrimExpandDims, std::make_shared<Primitive>("ExpandDims"));
GVAR_DEF(PrimitivePtr, kPrimAbs, std::make_shared<Primitive>(kAbs));
GVAR_DEF(PrimitivePtr, kPrimAbsGrad, std::make_shared<Primitive>("AbsGrad"));
GVAR_DEF(PrimitivePtr, kPrimRint, std::make_shared<Primitive>("Rint"));
GVAR_DEF(PrimitivePtr, kPrimRound, std::make_shared<Primitive>("Round"));
GVAR_DEF(PrimitivePtr, kPrimExp, std::make_shared<Primitive>(kExp));
GVAR_DEF(PrimitivePtr, kPrimExpm1, std::make_shared<Primitive>("Expm1"));
GVAR_DEF(PrimitivePtr, kPrimLog, std::make_shared<Primitive>(kLog));
GVAR_DEF(PrimitivePtr, kPrimRsqrt, std::make_shared<Primitive>("Rsqrt"));
GVAR_DEF(PrimitivePtr, kPrimRsqrtGrad, std::make_shared<Primitive>("RsqrtGrad"));
GVAR_DEF(PrimitivePtr, kPrimLinSpace, std::make_shared<Primitive>("LinSpace"));
GVAR_DEF(PrimitivePtr, kPrimNonMaxSuppression, std::make_shared<Primitive>("NonMaxSuppression"));
GVAR_DEF(PrimitivePtr, kPrimSign, std::make_shared<Primitive>("Sign"));
GVAR_DEF(PrimitivePtr, kPrimACos, std::make_shared<Primitive>(kACos));
GVAR_DEF(PrimitivePtr, kPrimAsinGrad, std::make_shared<Primitive>(kAsinGrad));
GVAR_DEF(PrimitivePtr, kPrimACosGrad, std::make_shared<Primitive>(kACosGrad));
GVAR_DEF(PrimitivePtr, kPrimAtanGrad, std::make_shared<Primitive>("AtanGrad"));
GVAR_DEF(PrimitivePtr, kPrimAsinhGrad, std::make_shared<Primitive>(kAsinhGrad));
GVAR_DEF(PrimitivePtr, kPrimAcoshGrad, std::make_shared<Primitive>("AcoshGrad"));
GVAR_DEF(PrimitivePtr, kPrimFloorMod, std::make_shared<Primitive>("FloorMod"));
GVAR_DEF(PrimitivePtr, kPrimCdist, std::make_shared<Primitive>(kCdist));
GVAR_DEF(PrimitivePtr, kPrimCdistGrad, std::make_shared<Primitive>(kCdistGrad));
GVAR_DEF(PrimitivePtr, kPrimWhere, std::make_shared<Primitive>("Where"));
GVAR_DEF(PrimitivePtr, kPrimMatrixInverse, std::make_shared<Primitive>(kMatrixInverse));
GVAR_DEF(PrimitivePtr, kPrimMatrixDeterminant, std::make_shared<Primitive>(kMatrixDeterminant));
GVAR_DEF(PrimitivePtr, kPrimLogMatrixDeterminant, std::make_shared<Primitive>(kLogMatrixDeterminant));
GVAR_DEF(PrimitivePtr, kPrimIndexAdd, std::make_shared<Primitive>("IndexAdd"));
GVAR_DEF(PrimitivePtr, kPrimIdentityMath, std::make_shared<Primitive>("Identity", kSideEffectPropagate));
GVAR_DEF(PrimitivePtr, kPrimInvGrad, std::make_shared<Primitive>("InvGrad"));
GVAR_DEF(PrimitivePtr, kPrimErfinv, std::make_shared<Primitive>("Erfinv"));
GVAR_DEF(PrimitivePtr, kPrimIsNan, std::make_shared<Primitive>("IsNan"));
GVAR_DEF(PrimitivePtr, kPrimIsInf, std::make_shared<Primitive>("IsInf"));
GVAR_DEF(PrimitivePtr, kPrimIsFinite, std::make_shared<Primitive>("IsFinite"));
GVAR_DEF(PrimitivePtr, kPrimIsClose, std::make_shared<Primitive>("IsClose"));
GVAR_DEF(PrimitivePtr, kPrimLerp, std::make_shared<Primitive>("Lerp"));
GVAR_DEF(PrimitivePtr, kPrimSquareSumAll, std::make_shared<Primitive>("SquareSumAll"));
GVAR_DEF(PrimitivePtr, kPrimComplex, std::make_shared<Primitive>("Complex"));
GVAR_DEF(PrimitivePtr, kPrimXdivy, std::make_shared<Primitive>("Xdivy"));
GVAR_DEF(PrimitivePtr, kPrimXlogy, std::make_shared<Primitive>("Xlogy"));
GVAR_DEF(PrimitivePtr, kPrimInv, std::make_shared<Primitive>("Inv"));
GVAR_DEF(PrimitivePtr, kPrimBitwiseOr, std::make_shared<Primitive>("BitwiseOr"));
GVAR_DEF(PrimitivePtr, kPrimBitwiseAnd, std::make_shared<Primitive>("BitwiseAnd"));
GVAR_DEF(PrimitivePtr, kPrimBitwiseXor, std::make_shared<Primitive>("BitwiseXor"));
GVAR_DEF(PrimitivePtr, kPrimEinsum, std::make_shared<Primitive>("Einsum"));
GVAR_DEF(PrimitivePtr, kPrimEinsumGrad, std::make_shared<Primitive>("EinsumGrad"));

// Image
GVAR_DEF(PrimitivePtr, kPrimNonMaxSuppressionV3, std::make_shared<Primitive>("NonMaxSuppressionV3"));

// Statements
GVAR_DEF(PrimitivePtr, kPrimReturn, std::make_shared<Primitive>("Return"));
GVAR_DEF(PrimitivePtr, kPrimUnroll, std::make_shared<Primitive>("Unroll"));
GVAR_DEF(PrimitivePtr, kPrimSwitch, std::make_shared<Primitive>("Switch"));
GVAR_DEF(PrimitivePtr, kPrimSwitchLayer, std::make_shared<Primitive>("switch_layer"));
GVAR_DEF(PrimitivePtr, kPrimAssign, std::make_shared<Primitive>(kAssign));
GVAR_DEF(PrimitivePtr, kPrimAssignAdd, std::make_shared<Primitive>(kAssignAdd));
GVAR_DEF(PrimitivePtr, kPrimAssignSub, std::make_shared<Primitive>(kAssignSub));
GVAR_DEF(PrimitivePtr, kPrimSelect, std::make_shared<Primitive>(kSelect));
GVAR_DEF(PrimitivePtr, kPrimCall, std::make_shared<Primitive>("call"));

GVAR_DEF(PrimitivePtr, kPrimMakeTuple, std::make_shared<Primitive>(kMakeTuple));
GVAR_DEF(PrimitivePtr, kPrimMakeSlice, std::make_shared<Primitive>("make_slice"));
GVAR_DEF(PrimitivePtr, kPrimTupleGetItem, std::make_shared<Primitive>(kTupleGetItem));
GVAR_DEF(PrimitivePtr, kPrimSliceGetItem, std::make_shared<Primitive>(kSliceGetItem));
GVAR_DEF(PrimitivePtr, kPrimArrayGetItem, std::make_shared<Primitive>("array_getitem"));
GVAR_DEF(PrimitivePtr, kPrimTupleSetItem, std::make_shared<Primitive>("tuple_setitem"));
GVAR_DEF(PrimitivePtr, kPrimArraySetItem, std::make_shared<Primitive>("array_setitem"));
GVAR_DEF(PrimitivePtr, kPrimGetAttr, std::make_shared<Primitive>("getattr"));
GVAR_DEF(PrimitivePtr, kPrimTupleLen, std::make_shared<Primitive>("tuple_len"));
GVAR_DEF(PrimitivePtr, kPrimArrayLen, std::make_shared<Primitive>("array_len"));
GVAR_DEF(PrimitivePtr, kPrimTileShape, std::make_shared<Primitive>("tile_shape"));
GVAR_DEF(PrimitivePtr, kPrimGenerateShapeIndex, std::make_shared<Primitive>("generate_shape_index"));
GVAR_DEF(PrimitivePtr, kPrimGenerateInverseIndex, std::make_shared<Primitive>("generate_inverse_index"));

// Debug ops
GVAR_DEF(PrimitivePtr, kPrimAssert, std::make_shared<Primitive>("Assert"));
#ifndef ENABLE_SECURITY
GVAR_DEF(PrimitivePtr, kPrimScalarSummary, std::make_shared<Primitive>("ScalarSummary"));
GVAR_DEF(PrimitivePtr, kPrimImageSummary, std::make_shared<Primitive>("ImageSummary"));
GVAR_DEF(PrimitivePtr, kPrimTensorSummary, std::make_shared<Primitive>("TensorSummary"));
GVAR_DEF(PrimitivePtr, kPrimHistogramSummary, std::make_shared<Primitive>("HistogramSummary"));
#endif
GVAR_DEF(PrimitivePtr, kPrimDebug, std::make_shared<Primitive>("Debug"));

// Dynamic shape testing
GVAR_DEF(PrimitivePtr, kPrimGpuConvertToDynamicShape, std::make_shared<Primitive>("GpuConvertToDynamicShape"));
GVAR_DEF(PrimitivePtr, kPrimErrorOnDynamicShapeInput, std::make_shared<Primitive>("ErrorOnDynamicShapeInput"));

// Other miscellaneous
GVAR_DEF(PrimitivePtr, kPrimDepend, std::make_shared<Primitive>("Depend", kSideEffectPropagate));
GVAR_DEF(PrimitivePtr, kPrimIOU, std::make_shared<Primitive>("IOU"));
GVAR_DEF(PrimitivePtr, kPrimReformat, std::make_shared<Primitive>("Reformat"));
GVAR_DEF(PrimitivePtr, kPrimLoad, std::make_shared<Primitive>("Load"));
GVAR_DEF(PrimitivePtr, kPrimUpdateState, std::make_shared<Primitive>("UpdateState"));
GVAR_DEF(PrimitivePtr, kPrimPartial, std::make_shared<Primitive>("Partial", kSideEffectPropagate));
GVAR_DEF(PrimitivePtr, kPrimIdentity, std::make_shared<Primitive>("identity", kSideEffectPropagate));
GVAR_DEF(PrimitivePtr, kPrimHookBackward, std::make_shared<Primitive>("HookBackward"));
GVAR_DEF(PrimitivePtr, kPrimCellBackwardHook, std::make_shared<Primitive>("CellBackwardHook"));
GVAR_DEF(PrimitivePtr, kPrimPrintShapeType, std::make_shared<Primitive>("PrintShapeType"));
GVAR_DEF(PrimitivePtr, kPrimSameTypeShape, std::make_shared<Primitive>("SameTypeShape"));
GVAR_DEF(PrimitivePtr, kPrimPrint, std::make_shared<Primitive>("Print"));
GVAR_DEF(PrimitivePtr, kPrimIs_, std::make_shared<Primitive>("is_"));
GVAR_DEF(PrimitivePtr, kPrimIsNot, std::make_shared<Primitive>("is_not"));
GVAR_DEF(PrimitivePtr, kPrimInDict, std::make_shared<Primitive>("in_dict"));
GVAR_DEF(PrimitivePtr, kPrimNotInDict, std::make_shared<Primitive>("not_in_dict"));
GVAR_DEF(PrimitivePtr, kPrimIsConsant, std::make_shared<Primitive>("is_constant"));
GVAR_DEF(PrimitivePtr, kPrimEquivFormat, std::make_shared<Primitive>("EquivFormat"));
GVAR_DEF(PrimitivePtr, kPrimLshProjection, std::make_shared<Primitive>("LshProjection"));
GVAR_DEF(PrimitivePtr, kPrimHashtableLookup, std::make_shared<Primitive>("HashtableLookup"));
GVAR_DEF(PrimitivePtr, kPrimCustomPredict, std::make_shared<Primitive>("CustomPredict"));
GVAR_DEF(PrimitivePtr, kPrimPriorBox, std::make_shared<Primitive>("PriorBox"));
GVAR_DEF(PrimitivePtr, kPrimQuantDTypeCast, std::make_shared<Primitive>("QuantDTypeCast"));
GVAR_DEF(PrimitivePtr, kPrimWhile, std::make_shared<Primitive>("While"));
GVAR_DEF(PrimitivePtr, kPrimPull, std::make_shared<Primitive>("Pull"));
GVAR_DEF(PrimitivePtr, kPrimPush, std::make_shared<Primitive>("Push"));
GVAR_DEF(PrimitivePtr, kPrimNPUAllocFloatStatus, std::make_shared<Primitive>("NPUAllocFloatStatus"));
GVAR_DEF(PrimitivePtr, kPrimPyFunc, std::make_shared<Primitive>("PyFunc"));

// Structures
GVAR_DEF(PrimitivePtr, kPrimMakeList, std::make_shared<Primitive>("make_list"));
GVAR_DEF(PrimitivePtr, kPrimMakeKeywordArg, std::make_shared<Primitive>("make_keyword_arg"));
GVAR_DEF(PrimitivePtr, kPrimListGetItem, std::make_shared<Primitive>("list_getitem"));
GVAR_DEF(PrimitivePtr, kPrimListSetItem, std::make_shared<Primitive>("list_setitem"));
GVAR_DEF(PrimitivePtr, kPrimDictGetItem, std::make_shared<Primitive>("dict_getitem"));
GVAR_DEF(PrimitivePtr, kPrimDictSetItem, std::make_shared<Primitive>("dict_setitem"));
GVAR_DEF(PrimitivePtr, kPrimDictGetKeys, std::make_shared<Primitive>("dict_getkeys"));
GVAR_DEF(PrimitivePtr, kPrimDictGetValues, std::make_shared<Primitive>("dict_getvalues"));
GVAR_DEF(PrimitivePtr, kPrimDictItems, std::make_shared<Primitive>("dict_items"));
GVAR_DEF(PrimitivePtr, kPrimListAppend, std::make_shared<Primitive>("list_append"));
GVAR_DEF(PrimitivePtr, kPrimListLen, std::make_shared<Primitive>("list_len"));

// Other miscellaneous
GVAR_DEF(PrimitivePtr, kPrimEnvironCreate, std::make_shared<Primitive>(kEnvironCreate));
GVAR_DEF(PrimitivePtr, kPrimEnvironSet, std::make_shared<Primitive>(kEnvironSet));
GVAR_DEF(PrimitivePtr, kPrimEnvironGet, std::make_shared<Primitive>(kEnvironGet));
GVAR_DEF(PrimitivePtr, kPrimEnvironAdd, std::make_shared<Primitive>(kEnvironAdd));
GVAR_DEF(PrimitivePtr, kPrimEnvironDestroyAll, std::make_shared<Primitive>(kEnvironDestroyAll));
GVAR_DEF(PrimitivePtr, kPrimMakeRefKey, std::make_shared<Primitive>("MakeRefKey"));
GVAR_DEF(PrimitivePtr, kPrimGetRefKey, std::make_shared<Primitive>("get_ref_key"));
GVAR_DEF(PrimitivePtr, kPrimMakeRef, std::make_shared<Primitive>("make_ref"));
GVAR_DEF(PrimitivePtr, kPrimGetRefValue, std::make_shared<Primitive>("get_ref_value"));

// Python interpreter runner
GVAR_DEF(PrimitivePtr, kPrimPyInterpret, std::make_shared<Primitive>("PyInterpret"));

// Other primitive not used by backend but used in core;
GVAR_DEF(PrimitivePtr, kPrimStateSetItem, std::make_shared<Primitive>("state_setitem"));
GVAR_DEF(PrimitivePtr, kPrimJ, std::make_shared<Primitive>(kJ, kSideEffectPropagate));
GVAR_DEF(PrimitivePtr, kPrimVmap, std::make_shared<Primitive>(kVmap, kSideEffectPropagate));
GVAR_DEF(PrimitivePtr, kPrimShard, std::make_shared<Primitive>("Shard", kSideEffectPropagate));

// Used to build graph which have keyword arguments
GVAR_DEF(PrimitivePtr, kPrimExtractKeywordArg, std::make_shared<Primitive>("extract_keyword_arg"));
GVAR_DEF(PrimitivePtr, kPrimMakeDict, std::make_shared<Primitive>("make_dict"));

// GraphKernel ops
GVAR_DEF(PrimitivePtr, kPrimInplaceAssign, std::make_shared<Primitive>("InplaceAssign"));

// Custom
GVAR_DEF(PrimitivePtr, kPrimCustom, std::make_shared<Primitive>("Custom"));

// Only used in lite
GVAR_DEF(PrimitivePtr, kPrimLeakyRelu, std::make_shared<Primitive>("LeakyRelu"));
GVAR_DEF(PrimitivePtr, kPrimConstant, std::make_shared<Primitive>("Constant"));
GVAR_DEF(PrimitivePtr, kPrimLocalResponseNormalization, std::make_shared<Primitive>("LocalResponseNormalization"));
GVAR_DEF(PrimitivePtr, kPrimFftReal, std::make_shared<Primitive>("FftReal"));
GVAR_DEF(PrimitivePtr, kPrimMfcc, std::make_shared<Primitive>("Mfcc"));
GVAR_DEF(PrimitivePtr, kPrimRfft, std::make_shared<Primitive>("Rfft"));
GVAR_DEF(PrimitivePtr, kPrimFftImag, std::make_shared<Primitive>("FftImag"));
GVAR_DEF(PrimitivePtr, kPrimSkipGram, std::make_shared<Primitive>("SkipGram"));
GVAR_DEF(PrimitivePtr, kPrimConv2DFusion, std::make_shared<Primitive>("Conv2DFusion"));
GVAR_DEF(PrimitivePtr, kPrimConv2dTransposeFusion, std::make_shared<Primitive>("Conv2dTransposeFusion"));
GVAR_DEF(PrimitivePtr, kPrimDepthWiseConv2DFusion, std::make_shared<Primitive>("DepthWiseConv2DFusion"));
GVAR_DEF(PrimitivePtr, kPrimAddFusion, std::make_shared<Primitive>("AddFusion"));
GVAR_DEF(PrimitivePtr, kPrimScaleFusion, std::make_shared<Primitive>("ScaleFusion"));
GVAR_DEF(PrimitivePtr, kPrimSubFusion, std::make_shared<Primitive>("SubFusion"));
GVAR_DEF(PrimitivePtr, kPrimMulFusion, std::make_shared<Primitive>("MulFusion"));
GVAR_DEF(PrimitivePtr, kPrimSigmoid, std::make_shared<Primitive>("Sigmoid"));
GVAR_DEF(PrimitivePtr, kPrimSigmoidGrad, std::make_shared<Primitive>("SigmoidGrad"));
GVAR_DEF(PrimitivePtr, kPrimHSigmoid, std::make_shared<Primitive>("HSigmoid"));
GVAR_DEF(PrimitivePtr, kPrimHSigmoidGrad, std::make_shared<Primitive>("HSigmoidGrad"));
GVAR_DEF(PrimitivePtr, kPrimClip, std::make_shared<Primitive>("Clip"));
GVAR_DEF(PrimitivePtr, kPrimHardTanh, std::make_shared<Primitive>("HardTanh"));
GVAR_DEF(PrimitivePtr, kPrimDepthWiseConv2DTransposeFusion,
         std::make_shared<Primitive>("DepthWiseConv2DTransposeFusion"));
GVAR_DEF(PrimitivePtr, kPrimArgMinFusion, std::make_shared<Primitive>("ArgMinFusion"));
GVAR_DEF(PrimitivePtr, kPrimArgMaxFusion, std::make_shared<Primitive>("ArgMaxFusion"));
GVAR_DEF(PrimitivePtr, kPrimSpaceToDepth, std::make_shared<Primitive>("SpaceToDepth"));
GVAR_DEF(PrimitivePtr, kPrimPadFusion, std::make_shared<Primitive>("PadFusion"));
GVAR_DEF(PrimitivePtr, kPrimPowFusion, std::make_shared<Primitive>("PowFusion"));
GVAR_DEF(PrimitivePtr, kPrimResize, std::make_shared<Primitive>("Resize"));
GVAR_DEF(PrimitivePtr, kPrimArgMinWithValue, std::make_shared<Primitive>("ArgMinWithValue"));
GVAR_DEF(PrimitivePtr, kPrimIf, std::make_shared<Primitive>("If"));
GVAR_DEF(PrimitivePtr, kPrimAvgPoolFusion, std::make_shared<Primitive>("AvgPoolFusion"));
GVAR_DEF(PrimitivePtr, kPrimMaxPoolFusion, std::make_shared<Primitive>("MaxPoolFusion"));
GVAR_DEF(PrimitivePtr, kPrimActivation, std::make_shared<Primitive>("Activation"));
GVAR_DEF(PrimitivePtr, kPrimPReLUFusion, std::make_shared<Primitive>("PReLUFusion"));
GVAR_DEF(PrimitivePtr, kPrimTopKFusion, std::make_shared<Primitive>("TopKFusion"));
GVAR_DEF(PrimitivePtr, kPrimTileFusion, std::make_shared<Primitive>("TileFusion"));
GVAR_DEF(PrimitivePtr, kPrimReduceFusion, std::make_shared<Primitive>("ReduceFusion"));
GVAR_DEF(PrimitivePtr, kPrimLayerNormFusion, std::make_shared<Primitive>("LayerNormFusion"));
GVAR_DEF(PrimitivePtr, kPrimDType, std::make_shared<Primitive>("DType"));
GVAR_DEF(PrimitivePtr, kPrimDivFusion, std::make_shared<Primitive>("DivFusion"));
GVAR_DEF(PrimitivePtr, kPrimErf, std::make_shared<Primitive>("Erf"));
GVAR_DEF(PrimitivePtr, kPrimErfc, std::make_shared<Primitive>("Erfc"));
GVAR_DEF(PrimitivePtr, kPrimSplice, std::make_shared<Primitive>("Splice"));
GVAR_DEF(PrimitivePtr, kPrimAffine, std::make_shared<Primitive>("Affine"));
GVAR_DEF(PrimitivePtr, kPrimEltwise, std::make_shared<Primitive>("Eltwise"));
GVAR_DEF(PrimitivePtr, kPrimMatMulFusion, std::make_shared<Primitive>("MatMulFusion"));
GVAR_DEF(PrimitivePtr, kPrimDynamicQuant, std::make_shared<Primitive>("DynamicQuant"));

// Type introspection
GVAR_DEF(PrimitivePtr, kPrimTypeOf, std::make_shared<Primitive>("typeof"));
GVAR_DEF(PrimitivePtr, kPrimHasType, std::make_shared<Primitive>("hastype"));

GVAR_DEF(PrimitivePtr, kPrimResolve, std::make_shared<Primitive>("resolve"));
GVAR_DEF(PrimitivePtr, kPrimEmbed, std::make_shared<Primitive>("embed"));
GVAR_DEF(PrimitivePtr, kPrimRefToEmbed, std::make_shared<Primitive>("RefToEmbed"));
GVAR_DEF(PrimitivePtr, kPrimCreateInstance, std::make_shared<Primitive>("create_instance"));

// Other miscellaneous
GVAR_DEF(PrimitivePtr, kPrimGetRefOrigin, std::make_shared<Primitive>("get_ref_origin"));
GVAR_DEF(PrimitivePtr, kPrimInsertGradientOf, std::make_shared<Primitive>("InsertGradientOf"));
GVAR_DEF(PrimitivePtr, kPrimCheckBprop, std::make_shared<Primitive>("CheckBprop"));
GVAR_DEF(PrimitivePtr, kPrimMixedPrecisionCast, std::make_shared<Primitive>("mixed_precision_cast"));
GVAR_DEF(PrimitivePtr, kPrimMakeRecord, std::make_shared<Primitive>("make_record"));

// Structures
GVAR_DEF(PrimitivePtr, kPrimListMap, std::make_shared<Primitive>("list_map"));
GVAR_DEF(PrimitivePtr, kPrimListReduce, std::make_shared<Primitive>("list_reduce"));
GVAR_DEF(PrimitivePtr, kPrimTupleReversed, std::make_shared<Primitive>("tuple_reversed"));
GVAR_DEF(PrimitivePtr, kPrimReducedShape, std::make_shared<Primitive>("reduced_shape"));
GVAR_DEF(PrimitivePtr, kPrimTupleDiv, std::make_shared<Primitive>("tuple_div"));
GVAR_DEF(PrimitivePtr, kPrimTupleToArray, std::make_shared<Primitive>("tuple_to_array"));
GVAR_DEF(PrimitivePtr, kPrimShapeMul, std::make_shared<Primitive>("shape_mul"));
GVAR_DEF(PrimitivePtr, kPrimTupleEqual, std::make_shared<Primitive>("tuple_equal"));
GVAR_DEF(PrimitivePtr, kPrimListEqual, std::make_shared<Primitive>("list_equal"));
GVAR_DEF(PrimitivePtr, kPrimMakeRange, std::make_shared<Primitive>("make_range"));
GVAR_DEF(PrimitivePtr, kPrimStopGradient, std::make_shared<Primitive>("stop_gradient"));
GVAR_DEF(PrimitivePtr, kPrimStringEqual, std::make_shared<Primitive>("string_equal"));
GVAR_DEF(PrimitivePtr, kPrimStringConcat, std::make_shared<Primitive>("string_concat"));
GVAR_DEF(PrimitivePtr, kPrimDictLen, std::make_shared<Primitive>("dict_len"));
GVAR_DEF(PrimitivePtr, kPrimFakeBprop, std::make_shared<Primitive>("fake_bprop"));
GVAR_DEF(PrimitivePtr, kPrimBroadcastGradientArgs, std::make_shared<Primitive>("BroadcastGradientArgs"));
GVAR_DEF(PrimitivePtr, kPrimDynamicBroadcastGradientArgs, std::make_shared<Primitive>(kDynamicBroadcastGradientArgs));

// Random
GVAR_DEF(PrimitivePtr, kPrimStandardNormal, std::make_shared<Primitive>("StandardNormal"));

// RL Ops
GVAR_DEF(PrimitivePtr, kPrimTensorArrayStack, std::make_shared<Primitive>("TensorArrayStack"));
GVAR_DEF(PrimitivePtr, kPrimTensorArray, std::make_shared<Primitive>("TensorArray"));
GVAR_DEF(PrimitivePtr, kPrimTensorArrayWrite, std::make_shared<Primitive>("TensorArrayWrite"));
GVAR_DEF(PrimitivePtr, kPrimTensorArrayGather, std::make_shared<Primitive>("TensorArrayGather"));

class DoSignaturePrimitive : public Primitive {
 public:
  explicit DoSignaturePrimitive(const std::string &name, const ValuePtr &function)
      : Primitive("S-Prim-" + name), function_(function) {}

  ~DoSignaturePrimitive() override = default;

  MS_DECLARE_PARENT(DoSignaturePrimitive, Primitive)

  const ValuePtr function() const { return function_; }

 private:
  ValuePtr function_;
};
using DoSignaturePrimitivePtr = std::shared_ptr<DoSignaturePrimitive>;
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_CORE_OPS_H_
