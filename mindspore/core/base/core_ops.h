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
MS_CORE_API inline const ValuePtr kValueOne = std::make_shared<Int64Imm>(1);
MS_CORE_API inline const mindspore::HashMap<std::string, ValuePtr> kSideEffectPropagate = {
  {mindspore::GRAPH_FLAG_SIDE_EFFECT_PROPAGATE, kValueOne},
};

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
constexpr auto kAbs = "Abs";
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

// Meta Function Graph
constexpr auto kJ = "J";

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
MS_CORE_API inline const PrimitivePtr kPrimGetNext = std::make_shared<Primitive>(kGetNext);

// Arithmetic
MS_CORE_API inline const PrimitivePtr kPrimScalarAdd = std::make_shared<Primitive>(kScalarAdd);
MS_CORE_API inline const PrimitivePtr kPrimScalarSub = std::make_shared<Primitive>(kScalarSub);
MS_CORE_API inline const PrimitivePtr kPrimScalarMul = std::make_shared<Primitive>(kScalarMul);
MS_CORE_API inline const PrimitivePtr kPrimScalarDiv = std::make_shared<Primitive>(kScalarDiv);
MS_CORE_API inline const PrimitivePtr kPrimScalarFloordiv = std::make_shared<Primitive>(kScalarFloordiv);
MS_CORE_API inline const PrimitivePtr kPrimScalarMod = std::make_shared<Primitive>(kScalarMod);
MS_CORE_API inline const PrimitivePtr kPrimScalarPow = std::make_shared<Primitive>(kScalarPow);
MS_CORE_API inline const PrimitivePtr kPrimScalarTrunc = std::make_shared<Primitive>(kScalarTrunc);
MS_CORE_API inline const PrimitivePtr kPrimScalarFloor = std::make_shared<Primitive>(kScalarFloor);
MS_CORE_API inline const PrimitivePtr kPrimScalarUadd = std::make_shared<Primitive>(kScalarUadd);
MS_CORE_API inline const PrimitivePtr kPrimScalarUsub = std::make_shared<Primitive>(kScalarUsub);
MS_CORE_API inline const PrimitivePtr kPrimScalarExp = std::make_shared<Primitive>("scalar_exp");
MS_CORE_API inline const PrimitivePtr kPrimScalarLog = std::make_shared<Primitive>("scalar_log");
MS_CORE_API inline const PrimitivePtr kPrimScalarSin = std::make_shared<Primitive>("scalar_sin");
MS_CORE_API inline const PrimitivePtr kPrimScalarCos = std::make_shared<Primitive>("scalar_cos");
MS_CORE_API inline const PrimitivePtr kPrimScalarTan = std::make_shared<Primitive>("scalar_tan");
MS_CORE_API inline const PrimitivePtr kPrimTrunc = std::make_shared<Primitive>(kTrunc);

// Comparisons
MS_CORE_API inline const PrimitivePtr kPrimScalarEq = std::make_shared<Primitive>("scalar_eq");
MS_CORE_API inline const PrimitivePtr kPrimScalarLt = std::make_shared<Primitive>("scalar_lt");
MS_CORE_API inline const PrimitivePtr kPrimScalarGt = std::make_shared<Primitive>("scalar_gt");
MS_CORE_API inline const PrimitivePtr kPrimScalarNe = std::make_shared<Primitive>("scalar_ne");
MS_CORE_API inline const PrimitivePtr kPrimScalarLe = std::make_shared<Primitive>("scalar_le");
MS_CORE_API inline const PrimitivePtr kPrimScalarGe = std::make_shared<Primitive>("scalar_ge");
MS_CORE_API inline const PrimitivePtr kPrimBoolNot = std::make_shared<Primitive>("bool_not");
MS_CORE_API inline const PrimitivePtr kPrimBoolAnd = std::make_shared<Primitive>("bool_and");
MS_CORE_API inline const PrimitivePtr kPrimBoolOr = std::make_shared<Primitive>("bool_or");
MS_CORE_API inline const PrimitivePtr kPrimBoolEq = std::make_shared<Primitive>("bool_eq");
MS_CORE_API inline const PrimitivePtr kPrimGreater = std::make_shared<Primitive>("Greater");
MS_CORE_API inline const PrimitivePtr kPrimGreaterEqual = std::make_shared<Primitive>("GreaterEqual");
MS_CORE_API inline const PrimitivePtr kPrimLess = std::make_shared<Primitive>("Less");
MS_CORE_API inline const PrimitivePtr kPrimLessEqual = std::make_shared<Primitive>("LessEqual");
MS_CORE_API inline const PrimitivePtr kPrimEqual = std::make_shared<Primitive>(kEqual);
MS_CORE_API inline const PrimitivePtr kPrimNotEqual = std::make_shared<Primitive>(kNotEqual);
MS_CORE_API inline const PrimitivePtr kPrimLogicalAnd = std::make_shared<Primitive>("LogicalAnd");
MS_CORE_API inline const PrimitivePtr kPrimLogicalOr = std::make_shared<Primitive>("LogicalOr");
MS_CORE_API inline const PrimitivePtr kPrimLogicalNot = std::make_shared<Primitive>("LogicalNot");
MS_CORE_API inline const PrimitivePtr kPrimEqualCount = std::make_shared<Primitive>("EqualCount");
MS_CORE_API inline const PrimitivePtr kPrimApproximateEqual = std::make_shared<Primitive>("ApproximateEqual");

MS_CORE_API inline const PrimitivePtr kPrimDistribute = std::make_shared<Primitive>("distribute");
MS_CORE_API inline const PrimitivePtr kPrimIm2Col = std::make_shared<Primitive>("im2col");
MS_CORE_API inline const PrimitivePtr kPrimCol2Im = std::make_shared<Primitive>("col2im");
MS_CORE_API inline const PrimitivePtr kPrimIm2ColV1 = std::make_shared<Primitive>("im2col_v1");
MS_CORE_API inline const PrimitivePtr kPrimCol2ImV1 = std::make_shared<Primitive>("col2im_v1");

MS_CORE_API inline const PrimitivePtr kPrimLabelGoto = std::make_shared<Primitive>("LabelGoto");
MS_CORE_API inline const PrimitivePtr kPrimLabelSwitch = std::make_shared<Primitive>("LabelSwitch");
MS_CORE_API inline const PrimitivePtr kPrimLabelSet = std::make_shared<Primitive>("LabelSet");

// Stack ops
MS_CORE_API inline const PrimitivePtr kPrimStackInit = std::make_shared<Primitive>("StackInit");
MS_CORE_API inline const PrimitivePtr kPrimStackDestroy = std::make_shared<Primitive>("StackDestroy");
MS_CORE_API inline const PrimitivePtr kPrimStackPush = std::make_shared<Primitive>("StackPush");
MS_CORE_API inline const PrimitivePtr kPrimStackPop = std::make_shared<Primitive>("StackPop");

// Arrays
MS_CORE_API inline const PrimitivePtr kPrimDynamicBroadcastTo = std::make_shared<Primitive>(kDynamicBroadcastTo);
MS_CORE_API inline const PrimitivePtr kPrimCummin = std::make_shared<Primitive>("Cummin");
MS_CORE_API inline const PrimitivePtr kPrimBroadcastTo = std::make_shared<Primitive>("BroadcastTo");
MS_CORE_API inline const PrimitivePtr kPrimScalarToArray = std::make_shared<Primitive>("scalar_to_array");
MS_CORE_API inline const PrimitivePtr kPrimTopK = std::make_shared<Primitive>("TopK");
MS_CORE_API inline const PrimitivePtr kPrimArrayToScalar = std::make_shared<Primitive>("array_to_scalar");
MS_CORE_API inline const PrimitivePtr kPrimBroadcastShape = std::make_shared<Primitive>("broadcast_shape");
MS_CORE_API inline const PrimitivePtr kPrimArrayMap = std::make_shared<Primitive>("array_map");
MS_CORE_API inline const PrimitivePtr kPrimArrayReduce = std::make_shared<Primitive>("array_reduce");
MS_CORE_API inline const PrimitivePtr kPrimCast = std::make_shared<Primitive>("Cast");
MS_CORE_API inline const PrimitivePtr kPrimConcat = std::make_shared<Primitive>("Concat");
MS_CORE_API inline const PrimitivePtr kPrimSqueeze = std::make_shared<Primitive>("Squeeze");
MS_CORE_API inline const PrimitivePtr kPrimUnsqueeze = std::make_shared<Primitive>("Unsqueeze");
MS_CORE_API inline const PrimitivePtr kPrimTranspose = std::make_shared<Primitive>(kTranspose);
MS_CORE_API inline const PrimitivePtr kPrimGatherV2 = std::make_shared<Primitive>("GatherV2");
MS_CORE_API inline const PrimitivePtr kPrimGatherD = std::make_shared<Primitive>("GatherD");
MS_CORE_API inline const PrimitivePtr kPrimGather = std::make_shared<Primitive>("Gather");
MS_CORE_API inline const PrimitivePtr kPrimGatherNd = std::make_shared<Primitive>("GatherNd");
MS_CORE_API inline const PrimitivePtr kPrimSparseGatherV2 = std::make_shared<Primitive>("SparseGatherV2");
MS_CORE_API inline const PrimitivePtr kPrimCoalesce = std::make_shared<Primitive>(kCoalesce);
MS_CORE_API inline const PrimitivePtr kPrimSparseToDense = std::make_shared<Primitive>("SparseToDense");
MS_CORE_API inline const PrimitivePtr kPrimShape = std::make_shared<Primitive>("Shape");
MS_CORE_API inline const PrimitivePtr kPrimStridedSlice = std::make_shared<Primitive>(kStridedSlice);
MS_CORE_API inline const PrimitivePtr kPrimStridedSliceGrad = std::make_shared<Primitive>(kStridedSliceGrad);
MS_CORE_API inline const PrimitivePtr kPrimDynamicShape = std::make_shared<Primitive>(kDynamicShape);
MS_CORE_API inline const PrimitivePtr kPrimEmbeddingLookup = std::make_shared<Primitive>("EmbeddingLookup");
MS_CORE_API inline const PrimitivePtr kPrimEmbeddingLookupCommGrad =
  std::make_shared<Primitive>("EmbeddingLookupCommGrad");
MS_CORE_API inline const PrimitivePtr kPrimSize = std::make_shared<Primitive>("Size");
MS_CORE_API inline const PrimitivePtr kPrimArgMax = std::make_shared<Primitive>("Argmax");
MS_CORE_API inline const PrimitivePtr kPrimArgMin = std::make_shared<Primitive>("Argmin");
MS_CORE_API inline const PrimitivePtr kPrimPack = std::make_shared<Primitive>("Pack");
MS_CORE_API inline const PrimitivePtr kPrimStack = std::make_shared<Primitive>(kStack);
MS_CORE_API inline const PrimitivePtr kPrimUnpack = std::make_shared<Primitive>("Unpack");
MS_CORE_API inline const PrimitivePtr kPrimUnstack = std::make_shared<Primitive>(kUnstack);
MS_CORE_API inline const PrimitivePtr kPrimUnsortedSegmentMax = std::make_shared<Primitive>("UnsortedSegmentMax");
MS_CORE_API inline const PrimitivePtr kPrimUnsortedSegmentSum = std::make_shared<Primitive>("UnsortedSegmentSum");
MS_CORE_API inline const PrimitivePtr kPrimUnsortedSegmentMin = std::make_shared<Primitive>("UnsortedSegmentMin");
MS_CORE_API inline const PrimitivePtr kPrimConcatOffset = std::make_shared<Primitive>("ConcatOffset");
MS_CORE_API inline const PrimitivePtr kPrimReshape = std::make_shared<Primitive>("Reshape");
MS_CORE_API inline const PrimitivePtr kPrimSubAndFilter = std::make_shared<Primitive>("SubAndFilter");
MS_CORE_API inline const PrimitivePtr kPrimMapCacheIdx = std::make_shared<Primitive>("MapCacheIdx");
MS_CORE_API inline const PrimitivePtr kPrimUpdateCache = std::make_shared<Primitive>("UpdateCache");
MS_CORE_API inline const PrimitivePtr kPrimComputeAccidentalHits = std::make_shared<Primitive>("ComputeAccidentalHits");
MS_CORE_API inline const PrimitivePtr kPrimCacheSwapTable = std::make_shared<Primitive>("CacheSwapTable");
MS_CORE_API inline const PrimitivePtr kPrimDynamicAssign = std::make_shared<Primitive>("DynamicAssign");
MS_CORE_API inline const PrimitivePtr kPrimPadAndShift = std::make_shared<Primitive>("PadAndShift");
MS_CORE_API inline const PrimitivePtr kPrimSlice = std::make_shared<Primitive>("Slice");
MS_CORE_API inline const PrimitivePtr kPrimSliceGrad = std::make_shared<Primitive>("SliceGrad");
MS_CORE_API inline const PrimitivePtr kPrimSliceFusion = std::make_shared<Primitive>("SliceFusion");
MS_CORE_API inline const PrimitivePtr kPrimTile = std::make_shared<Primitive>(kTile);
MS_CORE_API inline const PrimitivePtr kPrimAddN = std::make_shared<Primitive>("AddN");
MS_CORE_API inline const PrimitivePtr kPrimAccumulateNV2 = std::make_shared<Primitive>("AccumulateNV2");
MS_CORE_API inline const PrimitivePtr kPrimTransData = std::make_shared<Primitive>("TransData");
MS_CORE_API inline const PrimitivePtr kPrimTransDataRNN = std::make_shared<Primitive>("TransDataRNN");
MS_CORE_API inline const PrimitivePtr kPrimNMSWithMask = std::make_shared<Primitive>("NMSWithMask");
MS_CORE_API inline const PrimitivePtr kPrimPad = std::make_shared<Primitive>("Pad");
MS_CORE_API inline const PrimitivePtr kPrimArgMaxWithValue = std::make_shared<Primitive>("ArgMaxWithValue");
MS_CORE_API inline const PrimitivePtr kPrimUnique = std::make_shared<Primitive>("Unique");
MS_CORE_API inline const PrimitivePtr kPrimUniqueGrad = std::make_shared<Primitive>("UniqueGrad");
MS_CORE_API inline const PrimitivePtr kPrimExtractImagePatches = std::make_shared<Primitive>("ExtractImagePatches");
MS_CORE_API inline const PrimitivePtr kPrimDynamicRNN = std::make_shared<Primitive>("DynamicRNN");
MS_CORE_API inline const PrimitivePtr kPrimDynamicRNNGrad = std::make_shared<Primitive>("DynamicRNNGrad");
MS_CORE_API inline const PrimitivePtr kPrimDynamicGRUV2 = std::make_shared<Primitive>("DynamicGRUV2");
MS_CORE_API inline const PrimitivePtr kPrimDynamicGRUV2Grad = std::make_shared<Primitive>("DynamicGRUV2Grad");
MS_CORE_API inline const PrimitivePtr kPrimScatterAdd = std::make_shared<Primitive>("ScatterAdd");
MS_CORE_API inline const PrimitivePtr kPrimScatterSub = std::make_shared<Primitive>("ScatterSub");
MS_CORE_API inline const PrimitivePtr kPrimScatterMul = std::make_shared<Primitive>("ScatterMul");
MS_CORE_API inline const PrimitivePtr kPrimScatterDiv = std::make_shared<Primitive>("ScatterDiv");
MS_CORE_API inline const PrimitivePtr kPrimScatterMax = std::make_shared<Primitive>("ScatterMax");
MS_CORE_API inline const PrimitivePtr kPrimScatterMin = std::make_shared<Primitive>("ScatterMin");
MS_CORE_API inline const PrimitivePtr kPrimScatterNdAdd = std::make_shared<Primitive>("ScatterNdAdd");
MS_CORE_API inline const PrimitivePtr kPrimScatterUpdate = std::make_shared<Primitive>("ScatterUpdate");
MS_CORE_API inline const PrimitivePtr kPrimScatterElements = std::make_shared<Primitive>("ScatterElements");
MS_CORE_API inline const PrimitivePtr kPrimTensorCopySlices = std::make_shared<Primitive>("TensorCopySlices");
MS_CORE_API inline const PrimitivePtr kPrimMapUniform = std::make_shared<Primitive>("MapUniform");
MS_CORE_API inline const PrimitivePtr kPrimSplit = std::make_shared<Primitive>("Split");
MS_CORE_API inline const PrimitivePtr kPrimSplitV = std::make_shared<Primitive>(kSplitV);
MS_CORE_API inline const PrimitivePtr kPrimSequenceMask = std::make_shared<Primitive>("SequenceMask");
MS_CORE_API inline const PrimitivePtr kPrimRange = std::make_shared<Primitive>("Range");
MS_CORE_API inline const PrimitivePtr kPrimSpaceToBatchND = std::make_shared<Primitive>("SpaceToBatchND");
MS_CORE_API inline const PrimitivePtr kPrimBatchToSpaceND = std::make_shared<Primitive>("BatchToSpaceND");
MS_CORE_API inline const PrimitivePtr kPrimDepthToSpace = std::make_shared<Primitive>("DepthToSpace");
MS_CORE_API inline const PrimitivePtr kPrimBatchToSpace = std::make_shared<Primitive>("BatchToSpace");
MS_CORE_API inline const PrimitivePtr kPrimSpaceToBatch = std::make_shared<Primitive>("SpaceToBatch");
MS_CORE_API inline const PrimitivePtr kPrimScatterNd = std::make_shared<Primitive>("ScatterNd");
MS_CORE_API inline const PrimitivePtr kPrimScatterNdUpdate = std::make_shared<Primitive>("ScatterNdUpdate");
MS_CORE_API inline const PrimitivePtr kPrimScatterNonAliasingAdd = std::make_shared<Primitive>("ScatterNonAliasingAdd");
MS_CORE_API inline const PrimitivePtr kPrimConstantOfShape = std::make_shared<Primitive>("ConstantOfShape");
MS_CORE_API inline const PrimitivePtr kPrimSquaredDifference = std::make_shared<Primitive>("SquaredDifference");
MS_CORE_API inline const PrimitivePtr kPrimReverseV2 = std::make_shared<Primitive>("ReverseV2");
MS_CORE_API inline const PrimitivePtr kPrimReverseSequence = std::make_shared<Primitive>("ReverseSequence");
MS_CORE_API inline const PrimitivePtr kPrimRank = std::make_shared<Primitive>("Rank");
MS_CORE_API inline const PrimitivePtr kPrimResizeBilinear = std::make_shared<Primitive>("ResizeBilinear");
MS_CORE_API inline const PrimitivePtr kPrimParallelResizeBilinear =
  std::make_shared<Primitive>("ParallelResizeBilinear");
MS_CORE_API inline const PrimitivePtr kPrimParallelResizeBilinearGrad =
  std::make_shared<Primitive>("ParallelResizeBilinearGrad");
MS_CORE_API inline const PrimitivePtr kPrimResizeGrad = std::make_shared<Primitive>("ResizeGrad");
MS_CORE_API inline const PrimitivePtr kPrimResizeNearestNeighbor = std::make_shared<Primitive>("ResizeNearestNeighbor");
MS_CORE_API inline const PrimitivePtr kPrimResizeNearestNeighborGrad =
  std::make_shared<Primitive>("ResizeNearestNeighborGrad");
MS_CORE_API inline const PrimitivePtr kPrimDynamicResizeNearestNeighbor =
  std::make_shared<Primitive>("DynamicResizeNearestNeighbor");
MS_CORE_API inline const PrimitivePtr kPrimSort = std::make_shared<Primitive>("Sort");
MS_CORE_API inline const PrimitivePtr kPrimMaskedFill = std::make_shared<Primitive>("MaskedFill");
MS_CORE_API inline const PrimitivePtr kPrimMaskedSelect = std::make_shared<Primitive>("MaskedSelect");
MS_CORE_API inline const PrimitivePtr kPrimDiag = std::make_shared<Primitive>(kDiag);
MS_CORE_API inline const PrimitivePtr kPrimDiagPart = std::make_shared<Primitive>(kDiagPart);
MS_CORE_API inline const PrimitivePtr kPrimNonZero = std::make_shared<Primitive>("NonZero");
MS_CORE_API inline const PrimitivePtr kPrimRealInner = std::make_shared<Primitive>(kRealInner);
MS_CORE_API inline const PrimitivePtr kPrimReal = std::make_shared<Primitive>(kReal);
MS_CORE_API inline const PrimitivePtr kPrimImag = std::make_shared<Primitive>(kImag);
MS_CORE_API inline const PrimitivePtr kPrimConj = std::make_shared<Primitive>(kConj);
MS_CORE_API inline const PrimitivePtr kPrimExtractVolumePatches = std::make_shared<Primitive>("ExtractVolumePatches");
MS_CORE_API inline const PrimitivePtr kPrimLstsq = std::make_shared<Primitive>(kLstsq);
MS_CORE_API inline const PrimitivePtr kPrimLowerBound = std::make_shared<Primitive>(kLowerBound);
MS_CORE_API inline const PrimitivePtr kPrimUpperBound = std::make_shared<Primitive>(kUpperBound);
MS_CORE_API inline const PrimitivePtr kPrimCummax = std::make_shared<Primitive>(kCummax);

// NN
MS_CORE_API inline const PrimitivePtr kPrimCeLU = std::make_shared<Primitive>("CeLU");
MS_CORE_API inline const PrimitivePtr kPrimAdam = std::make_shared<Primitive>("Adam");
MS_CORE_API inline const PrimitivePtr kPrimApplyAdaMax = std::make_shared<Primitive>("ApplyAdaMax");
MS_CORE_API inline const PrimitivePtr kPrimAudioSpectrogram = std::make_shared<Primitive>("AudioSpectrogram");
MS_CORE_API inline const PrimitivePtr kPrimFlatten = std::make_shared<Primitive>("Flatten");
MS_CORE_API inline const PrimitivePtr kPrimCrop = std::make_shared<Primitive>("Crop");
MS_CORE_API inline const PrimitivePtr kPrimFlattenGrad = std::make_shared<Primitive>("FlattenGrad");
MS_CORE_API inline const PrimitivePtr kPrimSoftmax = std::make_shared<Primitive>("Softmax");
MS_CORE_API inline const PrimitivePtr kPrimSparseSoftmaxCrossEntropy =
  std::make_shared<Primitive>("SparseSoftmaxCrossEntropy");
MS_CORE_API inline const PrimitivePtr kPrimSoftmaxV2WithDropoutDoMaskV3 =
  std::make_shared<Primitive>("SoftmaxV2WithDropoutDoMaskV3");
MS_CORE_API inline const PrimitivePtr kPrimLogSoftmax = std::make_shared<Primitive>("LogSoftmax");
MS_CORE_API inline const PrimitivePtr kPrimLogSoftmaxGrad = std::make_shared<Primitive>("LogSoftmaxGrad");
MS_CORE_API inline const PrimitivePtr kPrimLstm = std::make_shared<Primitive>("LSTM");
MS_CORE_API inline const PrimitivePtr kPrimTan = std::make_shared<Primitive>("Tan");
MS_CORE_API inline const PrimitivePtr kPrimAtan2 = std::make_shared<Primitive>("Atan2");
MS_CORE_API inline const PrimitivePtr kPrimAtan = std::make_shared<Primitive>("Atan");
MS_CORE_API inline const PrimitivePtr kPrimAsin = std::make_shared<Primitive>("Asin");
MS_CORE_API inline const PrimitivePtr kPrimSinh = std::make_shared<Primitive>("Sinh");
MS_CORE_API inline const PrimitivePtr kPrimCosh = std::make_shared<Primitive>("Cosh");
MS_CORE_API inline const PrimitivePtr kPrimTanh = std::make_shared<Primitive>(kTanh);
MS_CORE_API inline const PrimitivePtr kPrimAsinh = std::make_shared<Primitive>("Asinh");
MS_CORE_API inline const PrimitivePtr kPrimAcosh = std::make_shared<Primitive>(kAcosh);
MS_CORE_API inline const PrimitivePtr kPrimAtanh = std::make_shared<Primitive>("Atanh");
MS_CORE_API inline const PrimitivePtr kPrimApplyGradientDescent = std::make_shared<Primitive>("ApplyGradientDescent");
MS_CORE_API inline const PrimitivePtr kPrimApplyPowerSignD = std::make_shared<Primitive>("ApplyPowerSign");
MS_CORE_API inline const PrimitivePtr kPrimBesselI0e = std::make_shared<Primitive>("BesselI0e");
MS_CORE_API inline const PrimitivePtr kPrimBesselI1e = std::make_shared<Primitive>("BesselI1e");
MS_CORE_API inline const PrimitivePtr kPrimTanhGrad = std::make_shared<Primitive>("TanhGrad");
MS_CORE_API inline const PrimitivePtr kPrimPooling = std::make_shared<Primitive>("Pooling");
MS_CORE_API inline const PrimitivePtr kPrimPoolingGrad = std::make_shared<Primitive>("PoolingGrad");
MS_CORE_API inline const PrimitivePtr kPrimROIPooling = std::make_shared<Primitive>("ROIPooling");
MS_CORE_API inline const PrimitivePtr kPrimMaxPool = std::make_shared<Primitive>("MaxPool");
MS_CORE_API inline const PrimitivePtr kPrimMaxPoolGrad = std::make_shared<Primitive>("MaxPoolGrad");
MS_CORE_API inline const PrimitivePtr kPrimMaxPoolWithArgmax = std::make_shared<Primitive>("MaxPoolWithArgmax");
MS_CORE_API inline const PrimitivePtr kPrimMaxPoolGradWithArgmax = std::make_shared<Primitive>("MaxPoolGradWithArgmax");
MS_CORE_API inline const PrimitivePtr kPrimApplyCenteredRMSProp = std::make_shared<Primitive>("ApplyCenteredRMSProp");
MS_CORE_API inline const PrimitivePtr kPrimAvgPool = std::make_shared<Primitive>("AvgPool");
MS_CORE_API inline const PrimitivePtr kPrimAvgPool3D = std::make_shared<Primitive>("AvgPool3D");
MS_CORE_API inline const PrimitivePtr kPrimAvgPoolGrad = std::make_shared<Primitive>("AvgPoolGrad");
MS_CORE_API inline const PrimitivePtr kPrimAvgPool3DGrad = std::make_shared<Primitive>("AvgPool3DGrad");
MS_CORE_API inline const PrimitivePtr kPrimAvgPoolGradVm = std::make_shared<Primitive>("AvgPoolGradVm");
MS_CORE_API inline const PrimitivePtr kPrimFusedSparseAdam = std::make_shared<Primitive>("FusedSparseAdam");
MS_CORE_API inline const PrimitivePtr kPrimFusedBatchNorm = std::make_shared<Primitive>("FusedBatchNorm");
MS_CORE_API inline const PrimitivePtr kPrimConv2D = std::make_shared<Primitive>("Conv2D");
MS_CORE_API inline const PrimitivePtr kPrimConv3D = std::make_shared<Primitive>("Conv3D");
MS_CORE_API inline const PrimitivePtr kPrimCTCLossV2 = std::make_shared<Primitive>("CTCLossV2");
MS_CORE_API inline const PrimitivePtr kPrimCTCLossV2Grad = std::make_shared<Primitive>("CTCLossV2Grad");
MS_CORE_API inline const PrimitivePtr kPrimCTCLoss = std::make_shared<Primitive>(kCTCLoss);
MS_CORE_API inline const PrimitivePtr kPrimFullConnection = std::make_shared<Primitive>("FullConnection");
MS_CORE_API inline const PrimitivePtr kPrimConv2DTranspose = std::make_shared<Primitive>(kConv2DTranspose);
MS_CORE_API inline const PrimitivePtr kPrimConv3DTranspose = std::make_shared<Primitive>("Conv3DTranspose");
MS_CORE_API inline const PrimitivePtr kPrimRoll = std::make_shared<Primitive>(kRoll);
MS_CORE_API inline const PrimitivePtr kPrimGroupConv2DGradInput = std::make_shared<Primitive>("GroupConv2DGradInput");
MS_CORE_API inline const PrimitivePtr kPrimBatchNorm = std::make_shared<Primitive>("BatchNorm");
MS_CORE_API inline const PrimitivePtr kPrimBatchNormGrad = std::make_shared<Primitive>("BatchNormGrad");
MS_CORE_API inline const PrimitivePtr kPrimSyncBatchNorm = std::make_shared<Primitive>("SyncBatchNorm");
MS_CORE_API inline const PrimitivePtr kPrimSyncBatchNormGrad = std::make_shared<Primitive>("SyncBatchNormGrad");
MS_CORE_API inline const PrimitivePtr kPrimBNTrainingReduce = std::make_shared<Primitive>("BNTrainingReduce");
MS_CORE_API inline const PrimitivePtr kPrimBNTrainingReduceGrad = std::make_shared<Primitive>("BNTrainingReduceGrad");
MS_CORE_API inline const PrimitivePtr kPrimReluGrad = std::make_shared<Primitive>(kReLUGrad);
MS_CORE_API inline const PrimitivePtr kPrimReluGradV2 = std::make_shared<Primitive>("ReluGradV2");
MS_CORE_API inline const PrimitivePtr kPrimRelu6Grad = std::make_shared<Primitive>("ReLU6Grad");
MS_CORE_API inline const PrimitivePtr kPrimConv2DBackpropInput = std::make_shared<Primitive>("Conv2DBackpropInput");
MS_CORE_API inline const PrimitivePtr kPrimConv2DBackpropFilter = std::make_shared<Primitive>("Conv2DBackpropFilter");
MS_CORE_API inline const PrimitivePtr kPrimConv3DBackpropInput = std::make_shared<Primitive>("Conv3DBackpropInput");
MS_CORE_API inline const PrimitivePtr kPrimConv3DBackpropFilter = std::make_shared<Primitive>("Conv3DBackpropFilter");
MS_CORE_API inline const PrimitivePtr kPrimCustomNormalize = std::make_shared<Primitive>("CustomNormalize");
MS_CORE_API inline const PrimitivePtr kPrimDepthwiseConv2dNative = std::make_shared<Primitive>("DepthwiseConv2dNative");
MS_CORE_API inline const PrimitivePtr kPrimCTCGreedyDecoder = std::make_shared<Primitive>("CTCGreedyDecoder");
MS_CORE_API inline const PrimitivePtr kPrimDataFormatDimMap = std::make_shared<Primitive>("DataFormatDimMap");
MS_CORE_API inline const PrimitivePtr kPrimDynamicStitch = std::make_shared<Primitive>("DynamicStitch");
MS_CORE_API inline const PrimitivePtr kPrimDepthwiseConv2dNativeBackpropFilter =
  std::make_shared<Primitive>("DepthwiseConv2dNativeBackpropFilter");
MS_CORE_API inline const PrimitivePtr kPrimDepthwiseConv2dNativeBackpropInput =
  std::make_shared<Primitive>("DepthwiseConv2dNativeBackpropInput");
MS_CORE_API inline const PrimitivePtr kPrimDetectionPostProcess = std::make_shared<Primitive>("DetectionPostProcess");
MS_CORE_API inline const PrimitivePtr kPrimBiasAddGrad = std::make_shared<Primitive>(kBiasAddGrad);
MS_CORE_API inline const PrimitivePtr kPrimBiasAdd = std::make_shared<Primitive>(kBiasAdd);
MS_CORE_API inline const PrimitivePtr kPrimBiasSubGrad = std::make_shared<Primitive>("BiasSubGrad");
MS_CORE_API inline const PrimitivePtr kPrimBinaryCrossEntropy = std::make_shared<Primitive>("BinaryCrossEntropy");
MS_CORE_API inline const PrimitivePtr kPrimBinaryCrossEntropyGrad =
  std::make_shared<Primitive>("BinaryCrossEntropyGrad");
MS_CORE_API inline const PrimitivePtr kPrimSmoothL1Loss = std::make_shared<Primitive>("SmoothL1Loss");
MS_CORE_API inline const PrimitivePtr kPrimSmoothL1LossGrad = std::make_shared<Primitive>("SmoothL1LossGrad");
MS_CORE_API inline const PrimitivePtr kPrimSoftMarginLoss = std::make_shared<Primitive>("SoftMarginLoss");
MS_CORE_API inline const PrimitivePtr kPrimSoftMarginLossGrad = std::make_shared<Primitive>("SoftMarginLossGrad");
MS_CORE_API inline const PrimitivePtr kPrimSoftmaxCrossEntropyWithLogits =
  std::make_shared<Primitive>("SoftmaxCrossEntropyWithLogits");
MS_CORE_API inline const PrimitivePtr kPrimL2Loss = std::make_shared<Primitive>("L2Loss");
MS_CORE_API inline const PrimitivePtr kPrimSigmoidCrossEntropyWithLogits =
  std::make_shared<Primitive>("SigmoidCrossEntropyWithLogits");
MS_CORE_API inline const PrimitivePtr kPrimSigmoidCrossEntropyWithLogitsGrad =
  std::make_shared<Primitive>("SigmoidCrossEntropyWithLogitsGrad");
MS_CORE_API inline const PrimitivePtr kPrimSparseSoftmaxCrossEntropyWithLogits =
  std::make_shared<Primitive>("SparseSoftmaxCrossEntropyWithLogits");
MS_CORE_API inline const PrimitivePtr kPrimMomentum = std::make_shared<Primitive>("Momentum");
MS_CORE_API inline const PrimitivePtr kPrimApplyMomentum = std::make_shared<Primitive>("ApplyMomentum");
MS_CORE_API inline const PrimitivePtr kPrimApplyFtrl = std::make_shared<Primitive>("ApplyFtrl");
MS_CORE_API inline const PrimitivePtr kPrimLrn = std::make_shared<Primitive>("LRN");
MS_CORE_API inline const PrimitivePtr kPrimLayerNorm = std::make_shared<Primitive>(kLayerNorm);
MS_CORE_API inline const PrimitivePtr kPrimLayerNormGrad = std::make_shared<Primitive>(kLayerNormGrad);
MS_CORE_API inline const PrimitivePtr kPrimLayerNormXBackprop = std::make_shared<Primitive>("LayerNormXBackprop");
MS_CORE_API inline const PrimitivePtr kPrimLayerNormXBackpropV2 = std::make_shared<Primitive>("LayerNormXBackpropV2");
MS_CORE_API inline const PrimitivePtr kPrimLayerNormBetaGammaBackprop =
  std::make_shared<Primitive>("LayerNormBetaGammaBackprop");
MS_CORE_API inline const PrimitivePtr kPrimLayerNormBetaGammaBackpropV2 =
  std::make_shared<Primitive>("LayerNormBetaGammaBackpropV2");
MS_CORE_API inline const PrimitivePtr kPrimLog1p = std::make_shared<Primitive>("Log1p");
MS_CORE_API inline const PrimitivePtr kPrimDropoutGenMask = std::make_shared<Primitive>(kDropoutGenMask);
MS_CORE_API inline const PrimitivePtr kPrimDropoutDoMask = std::make_shared<Primitive>(kDropoutDoMask);
MS_CORE_API inline const PrimitivePtr kPrimDropoutDoMaskV3 = std::make_shared<Primitive>(kDropoutDoMaskV3);
MS_CORE_API inline const PrimitivePtr kPrimDropoutGrad = std::make_shared<Primitive>(kDropoutGrad);
MS_CORE_API inline const PrimitivePtr kPrimDropout = std::make_shared<Primitive>(kDropout);
MS_CORE_API inline const PrimitivePtr kPrimUniformReal = std::make_shared<Primitive>("UniformReal");
MS_CORE_API inline const PrimitivePtr kPrimCudnnUniformReal = std::make_shared<Primitive>("CudnnUniformReal");
MS_CORE_API inline const PrimitivePtr kPrimOneHot = std::make_shared<Primitive>("OneHot");
MS_CORE_API inline const PrimitivePtr kPrimGeLU = std::make_shared<Primitive>(kGeLU);
MS_CORE_API inline const PrimitivePtr kPrimGeLUGrad = std::make_shared<Primitive>(kGeLUGrad);
MS_CORE_API inline const PrimitivePtr kPrimFastGeLU = std::make_shared<Primitive>(kFastGeLU);
MS_CORE_API inline const PrimitivePtr kPrimFastGeLUGrad = std::make_shared<Primitive>(kFastGeLUGrad);
MS_CORE_API inline const PrimitivePtr kPrimRelu = std::make_shared<Primitive>(kReLU);
MS_CORE_API inline const PrimitivePtr kPrimElu = std::make_shared<Primitive>("Elu");
MS_CORE_API inline const PrimitivePtr kPrimEluGrad = std::make_shared<Primitive>("EluGrad");
MS_CORE_API inline const PrimitivePtr kPrimRelu6 = std::make_shared<Primitive>(kReLU6);
MS_CORE_API inline const PrimitivePtr kPrimReluV2 = std::make_shared<Primitive>(kReLUV2);
MS_CORE_API inline const PrimitivePtr kPrimPRelu = std::make_shared<Primitive>("PReLU");
MS_CORE_API inline const PrimitivePtr kPrimSelu = std::make_shared<Primitive>("SeLU");
MS_CORE_API inline const PrimitivePtr kPrimSoftplus = std::make_shared<Primitive>("Softplus");
MS_CORE_API inline const PrimitivePtr kPrimSoftplusGrad = std::make_shared<Primitive>("SoftplusGrad");
MS_CORE_API inline const PrimitivePtr kPrimZeros = std::make_shared<Primitive>("Zeros");
MS_CORE_API inline const PrimitivePtr kPrimZerosLike = std::make_shared<Primitive>(kZerosLike);
MS_CORE_API inline const PrimitivePtr kPrimOnes = std::make_shared<Primitive>(kOnes);
MS_CORE_API inline const PrimitivePtr kPrimOnesLike = std::make_shared<Primitive>(kOnesLike);
MS_CORE_API inline const PrimitivePtr kPrimBpropCut = std::make_shared<Primitive>("bprop_cut");
MS_CORE_API inline const PrimitivePtr kPrimFakeQuantPerLayer = std::make_shared<Primitive>("FakeQuantPerLayer");
MS_CORE_API inline const PrimitivePtr kPrimFakeQuantPerChannel = std::make_shared<Primitive>("FakeQuantPerChannel");
MS_CORE_API inline const PrimitivePtr kPrimFakeLearnedScaleQuantPerLayer =
  std::make_shared<Primitive>("FakeLearnedScaleQuantPerLayer");
MS_CORE_API inline const PrimitivePtr kPrimFakeLearnedScaleQuantPerChannel =
  std::make_shared<Primitive>("FakeLearnedScaleQuantPerChannel");
MS_CORE_API inline const PrimitivePtr kPrimFakeQuantWithMinMaxVars =
  std::make_shared<Primitive>("FakeQuantWithMinMaxVars");
MS_CORE_API inline const PrimitivePtr kPrimApplyRMSProp = std::make_shared<Primitive>("ApplyRMSProp");
MS_CORE_API inline const PrimitivePtr kPrimSparseApplyFtrl = std::make_shared<Primitive>("SparseApplyFtrl");
MS_CORE_API inline const PrimitivePtr kPrimSparseApplyProximalAdagrad =
  std::make_shared<Primitive>("SparseApplyProximalAdagrad");
MS_CORE_API inline const PrimitivePtr kPrimFusedAdam = std::make_shared<Primitive>("FusedAdam");
MS_CORE_API inline const PrimitivePtr kPrimFusedAdamWeightDecay = std::make_shared<Primitive>("FusedAdamWeightDecay");
MS_CORE_API inline const PrimitivePtr kPrimSGD = std::make_shared<Primitive>("SGD");
MS_CORE_API inline const PrimitivePtr kPrimBCEWithLogitsLoss = std::make_shared<Primitive>("BCEWithLogitsLoss");
MS_CORE_API inline const PrimitivePtr kPrimClipByNormNoDivSum = std::make_shared<Primitive>("ClipByNormNoDivSum");
MS_CORE_API inline const PrimitivePtr kPrimTensorMove = std::make_shared<Primitive>("TensorMove");
MS_CORE_API inline const PrimitivePtr kPrimL2Normalize = std::make_shared<Primitive>("L2Normalize");
MS_CORE_API inline const PrimitivePtr kPrimCustomExtractFeatures = std::make_shared<Primitive>("CustomExtractFeatures");
MS_CORE_API inline const PrimitivePtr kLambApplyOptimizerAssign =
  std::make_shared<Primitive>("LambApplyOptimizerAssign");
MS_CORE_API inline const PrimitivePtr kLambApplyWeightAssign = std::make_shared<Primitive>("LambApplyWeightAssign");
MS_CORE_API inline const PrimitivePtr kSoftmaxGradExt = std::make_shared<Primitive>("SoftmaxGradExt");
MS_CORE_API inline const PrimitivePtr kPrimSparseApplyAdadelta = std::make_shared<Primitive>(kSparseApplyAdadelta);
MS_CORE_API inline const PrimitivePtr kSquareSumV1 = std::make_shared<Primitive>("SquareSumV1");
MS_CORE_API inline const PrimitivePtr kFusedMulAdd = std::make_shared<Primitive>("FusedMulAdd");
MS_CORE_API inline const PrimitivePtr kPrimSoftShrink = std::make_shared<Primitive>("SoftShrink");
MS_CORE_API inline const PrimitivePtr kPrimSoftShrinkGrad = std::make_shared<Primitive>("SoftShrinkGrad");
MS_CORE_API inline const PrimitivePtr kPrimHShrink = std::make_shared<Primitive>("HShrink");
MS_CORE_API inline const PrimitivePtr kPrimHShrinkGrad = std::make_shared<Primitive>("HShrinkGrad");
MS_CORE_API inline const PrimitivePtr kPrimHSVToRGB = std::make_shared<Primitive>("HSVToRGB");
MS_CORE_API inline const PrimitivePtr kPrimApplyAdagradDA = std::make_shared<Primitive>("ApplyAdagradDA");
MS_CORE_API inline const PrimitivePtr kPrimApplyAdagradV2 = std::make_shared<Primitive>("ApplyAdagradV2");
MS_CORE_API inline const PrimitivePtr kPrimApplyProximalGradientDescent =
  std::make_shared<Primitive>("ApplyProximalGradientDescent");
MS_CORE_API inline const PrimitivePtr kPrimSparseApplyRMSProp = std::make_shared<Primitive>("SparseApplyRMSProp");
MS_CORE_API inline const PrimitivePtr kPrimApplyKerasMomentum = std::make_shared<Primitive>("ApplyKerasMomentum");
MS_CORE_API inline const PrimitivePtr kPrimLARSUpdate = std::make_shared<Primitive>("LARSUpdate");
MS_CORE_API inline const PrimitivePtr kPrimBoundingBoxDecode = std::make_shared<Primitive>("BoundingBoxDecode");
MS_CORE_API inline const PrimitivePtr kPrimROIAlign = std::make_shared<Primitive>("ROIAlign");
MS_CORE_API inline const PrimitivePtr kPrimApplyAddSign = std::make_shared<Primitive>("ApplyAddSign");
MS_CORE_API inline const PrimitivePtr kPrimApplyAdagrad = std::make_shared<Primitive>("ApplyAdagrad");
MS_CORE_API inline const PrimitivePtr kPrimApplyAdadelta = std::make_shared<Primitive>("ApplyAdadelta");
MS_CORE_API inline const PrimitivePtr kPrimApplyAdamWithAmsgrad = std::make_shared<Primitive>("ApplyAdamWithAmsgrad");
MS_CORE_API inline const PrimitivePtr kPrimGridSampler3D = std::make_shared<Primitive>(kGridSampler3D);
MS_CORE_API inline const PrimitivePtr kPrimGridSampler3DGrad = std::make_shared<Primitive>(kGridSampler3DGrad);
MS_CORE_API inline const PrimitivePtr kPrimBNTrainingUpdate = std::make_shared<Primitive>("BNTrainingUpdate");
MS_CORE_API inline const PrimitivePtr kPrimBNTrainingUpdateGrad = std::make_shared<Primitive>("BNTrainingUpdateGrad");

// Comm ops
MS_CORE_API inline const PrimitivePtr kPrimMirror = std::make_shared<Primitive>("_MirrorOperator");
MS_CORE_API inline const PrimitivePtr kPrimMirrorMiniStep = std::make_shared<Primitive>("_MirrorMiniStepOperator");
MS_CORE_API inline const PrimitivePtr kPrimMiniStepAllGather = std::make_shared<Primitive>("_MiniStepAllGather");
MS_CORE_API inline const PrimitivePtr kPrimMicroStepAllGather = std::make_shared<Primitive>("_MicroStepAllGather");
MS_CORE_API inline const PrimitivePtr kPrimVirtualDiv = std::make_shared<Primitive>("_VirtualDiv");
MS_CORE_API inline const PrimitivePtr kPrimVirtualAdd = std::make_shared<Primitive>("_VirtualAdd");
MS_CORE_API inline const PrimitivePtr kPrimVirtualDataset = std::make_shared<Primitive>("_VirtualDataset");
MS_CORE_API inline const PrimitivePtr kPrimVirtualOutput = std::make_shared<Primitive>("_VirtualOutput");
MS_CORE_API inline const PrimitivePtr kPrimSend = std::make_shared<Primitive>("Send");
MS_CORE_API inline const PrimitivePtr kPrimReceive = std::make_shared<Primitive>("Receive");
MS_CORE_API inline const PrimitivePtr kPrimAllReduce = std::make_shared<Primitive>("AllReduce");
MS_CORE_API inline const PrimitivePtr kPrimNeighborExchange = std::make_shared<Primitive>("NeighborExchange");
MS_CORE_API inline const PrimitivePtr kPrimNeighborExchangeV2 = std::make_shared<Primitive>("NeighborExchangeV2");
MS_CORE_API inline const PrimitivePtr kPrimNeighborExchangeV2Grad =
  std::make_shared<Primitive>("NeighborExchangeV2Grad");
MS_CORE_API inline const PrimitivePtr kPrimAllToAll = std::make_shared<Primitive>("AlltoAll");
MS_CORE_API inline const PrimitivePtr kPrimAllToAllv = std::make_shared<Primitive>("AllToAllv");
MS_CORE_API inline const PrimitivePtr kPrimAllSwap = std::make_shared<Primitive>("_AllSwap");
MS_CORE_API inline const PrimitivePtr kPrimBroadcast = std::make_shared<Primitive>("Broadcast");
MS_CORE_API inline const PrimitivePtr kPrimAllGather = std::make_shared<Primitive>("AllGather");
MS_CORE_API inline const PrimitivePtr kPrimReduceScatter = std::make_shared<Primitive>("ReduceScatter");
MS_CORE_API inline const PrimitivePtr kPrimMemCpyAsync = std::make_shared<Primitive>("memcpy_async");
MS_CORE_API inline const PrimitivePtr kPrimFill = std::make_shared<Primitive>("Fill");
MS_CORE_API inline const PrimitivePtr kPrimFusedPushWeight = std::make_shared<Primitive>("FusedPushWeight");
MS_CORE_API inline const PrimitivePtr kPrimFusedPullWeight = std::make_shared<Primitive>("FusedPullWeight");
MS_CORE_API inline const PrimitivePtr kPrimInitDataSetQueue = std::make_shared<Primitive>("InitDataSetQueue");
MS_CORE_API inline const PrimitivePtr kPrimVirtualAssignAdd = std::make_shared<Primitive>("_VirtualAssignAdd");
MS_CORE_API inline const PrimitivePtr kPrimVirtualAccuGrad = std::make_shared<Primitive>("_VirtualAccuGrad");
MS_CORE_API inline const PrimitivePtr kPrimMirrorMicroStep = std::make_shared<Primitive>("_MirrorMicroStepOperator");
MS_CORE_API inline const PrimitivePtr kPrimApplyProximalAdagrad = std::make_shared<Primitive>("ApplyProximalAdagrad");

// Quant ops
MS_CORE_API inline const PrimitivePtr kPrimBatchNormFold = std::make_shared<Primitive>("BatchNormFold");
MS_CORE_API inline const PrimitivePtr kPrimFakeQuantWithMinMaxVarsPerChannel =
  std::make_shared<Primitive>("FakeQuantWithMinMaxVarsPerChannel");
// Control ops
MS_CORE_API inline const PrimitivePtr kPrimMerge = std::make_shared<Primitive>("Merge");
// RowTensor
MS_CORE_API inline const PrimitivePtr kPrimMakeRowTensor = std::make_shared<Primitive>("MakeRowTensor");
MS_CORE_API inline const PrimitivePtr kPrimRowTensorGetValues = std::make_shared<Primitive>("RowTensorGetValues");
MS_CORE_API inline const PrimitivePtr kPrimRowTensorGetIndices = std::make_shared<Primitive>("RowTensorGetIndices");
MS_CORE_API inline const PrimitivePtr kPrimRowTensorGetDenseShape =
  std::make_shared<Primitive>("RowTensorGetDenseShape");
MS_CORE_API inline const PrimitivePtr kPrimRowTensorAdd = std::make_shared<Primitive>("RowTensorAdd");

// COOTensor
MS_CORE_API inline const PrimitivePtr kPrimMakeCOOTensor = std::make_shared<Primitive>(kMakeCOOTensor);
MS_CORE_API inline const PrimitivePtr kPrimCOOTensorGetValues = std::make_shared<Primitive>(kCOOTensorGetValues);
MS_CORE_API inline const PrimitivePtr kPrimCOOTensorGetIndices = std::make_shared<Primitive>(kCOOTensorGetIndices);
MS_CORE_API inline const PrimitivePtr kPrimCOOTensorGetDenseShape =
  std::make_shared<Primitive>(kCOOTensorGetDenseShapes);

// CSRTensor
MS_CORE_API inline const PrimitivePtr kPrimMakeCSRTensor = std::make_shared<Primitive>(kMakeCSRTensor);
MS_CORE_API inline const PrimitivePtr kPrimCSRTensorGetValues = std::make_shared<Primitive>(kCSRTensorGetValues);
MS_CORE_API inline const PrimitivePtr kPrimCSRTensorGetIndptr = std::make_shared<Primitive>(kCSRTensorGetIndptr);
MS_CORE_API inline const PrimitivePtr kPrimCSRTensorGetIndices = std::make_shared<Primitive>(kCSRTensorGetIndices);
MS_CORE_API inline const PrimitivePtr kPrimCSRTensorGetDenseShape =
  std::make_shared<Primitive>(kCSRTensorGetDenseShape);
MS_CORE_API inline const PrimitivePtr kPrimIsCSRFunc = std::make_shared<Primitive>(kIsCSRFunc);

// Sparse ops
MS_CORE_API inline const PrimitivePtr kPrimSparseTensorDenseMatmul =
  std::make_shared<Primitive>(kSparseTensorDenseMatmul);
MS_CORE_API inline const PrimitivePtr kPrimCOOTensorDenseMatmul = std::make_shared<Primitive>(kCOOTensorDenseMatmul);
MS_CORE_API inline const PrimitivePtr kPrimCSRDenseMul = std::make_shared<Primitive>(kCSRDenseMul);
MS_CORE_API inline const PrimitivePtr kPrimCSRReduceSum = std::make_shared<Primitive>(kCSRReduceSum);
MS_CORE_API inline const PrimitivePtr kPrimCSRMV = std::make_shared<Primitive>(kCSRMV);
MS_CORE_API inline const PrimitivePtr kPrimCSRMul = std::make_shared<Primitive>(kCSRMul);

// TensorList
MS_CORE_API inline const PrimitivePtr kPrimTensorListFromTensor = std::make_shared<Primitive>("TensorListFromTensor");
MS_CORE_API inline const PrimitivePtr kPrimTensorListReserve = std::make_shared<Primitive>("TensorListReserve");
MS_CORE_API inline const PrimitivePtr kPrimTensorListStack = std::make_shared<Primitive>("TensorListStack");
MS_CORE_API inline const PrimitivePtr kPrimTensorListSetItem = std::make_shared<Primitive>("TensorListSetItem");

// Maths
MS_CORE_API inline const PrimitivePtr kPrimCross = std::make_shared<Primitive>(kCross);
MS_CORE_API inline const PrimitivePtr kPrimBesselI0 = std::make_shared<Primitive>("BesselI0");
MS_CORE_API inline const PrimitivePtr kPrimBesselI1 = std::make_shared<Primitive>("BesselI1");
MS_CORE_API inline const PrimitivePtr kPrimGer = std::make_shared<Primitive>("Ger");
MS_CORE_API inline const PrimitivePtr kPrimCeil = std::make_shared<Primitive>("Ceil");
MS_CORE_API inline const PrimitivePtr kPrimLuSolve = std::make_shared<Primitive>("LuSolve");
MS_CORE_API inline const PrimitivePtr kPrimCholeskyInverse = std::make_shared<Primitive>("CholeskyInverse");
MS_CORE_API inline const PrimitivePtr kPrimTensorAdd = std::make_shared<Primitive>("TensorAdd");
MS_CORE_API inline const PrimitivePtr kPrimAdd = std::make_shared<Primitive>(kAdd);
MS_CORE_API inline const PrimitivePtr kPrimAddcdiv = std::make_shared<Primitive>(kAddcdiv);
MS_CORE_API inline const PrimitivePtr kPrimAddcmul = std::make_shared<Primitive>(kAddcmul);
MS_CORE_API inline const PrimitivePtr kPrimMatMul = std::make_shared<Primitive>("MatMul");
MS_CORE_API inline const PrimitivePtr kPrimMatMulV2 = std::make_shared<Primitive>("MatMulV2");
MS_CORE_API inline const PrimitivePtr kPrimMatrixDiag = std::make_shared<Primitive>("MatrixDiag");
MS_CORE_API inline const PrimitivePtr kPrimMatrixDiagPart = std::make_shared<Primitive>("MatrixDiagPart");
MS_CORE_API inline const PrimitivePtr kPrimBatchMatMul = std::make_shared<Primitive>("BatchMatMul");
MS_CORE_API inline const PrimitivePtr kPrimBatchMatMulV2 = std::make_shared<Primitive>("BatchMatMulV2");
MS_CORE_API inline const PrimitivePtr kPrimMaximumGrad = std::make_shared<Primitive>("MaximumGrad");
MS_CORE_API inline const PrimitivePtr kPrimMinimumGrad = std::make_shared<Primitive>("MinimumGrad");
MS_CORE_API inline const PrimitivePtr kPrimReduce = std::make_shared<Primitive>("Reduce");
MS_CORE_API inline const PrimitivePtr kPrimReduceMean = std::make_shared<Primitive>("ReduceMean");
MS_CORE_API inline const PrimitivePtr kPrimReduceSum = std::make_shared<Primitive>("ReduceSum");
MS_CORE_API inline const PrimitivePtr kPrimReduceAll = std::make_shared<Primitive>("ReduceAll");
MS_CORE_API inline const PrimitivePtr kPrimReduceAny = std::make_shared<Primitive>("ReduceAny");
MS_CORE_API inline const PrimitivePtr kPrimReduceMax = std::make_shared<Primitive>("ReduceMax");
MS_CORE_API inline const PrimitivePtr kPrimReduceMin = std::make_shared<Primitive>("ReduceMin");
MS_CORE_API inline const PrimitivePtr kPrimCentralization = std::make_shared<Primitive>("Centralization");
MS_CORE_API inline const PrimitivePtr kPrimNeg = std::make_shared<Primitive>(kNeg);
MS_CORE_API inline const PrimitivePtr kPrimSin = std::make_shared<Primitive>("Sin");
MS_CORE_API inline const PrimitivePtr kPrimCos = std::make_shared<Primitive>(kCos);
MS_CORE_API inline const PrimitivePtr kPrimSub = std::make_shared<Primitive>(kSub);
MS_CORE_API inline const PrimitivePtr kPrimMul = std::make_shared<Primitive>(kMul);
MS_CORE_API inline const PrimitivePtr kPrimMulNoNan = std::make_shared<Primitive>(kMulNoNan);
MS_CORE_API inline const PrimitivePtr kPrimDiv = std::make_shared<Primitive>("Div");
MS_CORE_API inline const PrimitivePtr kPrimMod = std::make_shared<Primitive>("Mod");
MS_CORE_API inline const PrimitivePtr kPrimFloor = std::make_shared<Primitive>("Floor");
MS_CORE_API inline const PrimitivePtr kPrimInvert = std::make_shared<Primitive>("Invert");
MS_CORE_API inline const PrimitivePtr kPrimDivNoNan = std::make_shared<Primitive>("DivNoNan");
MS_CORE_API inline const PrimitivePtr kPrimMinimum = std::make_shared<Primitive>("Minimum");
MS_CORE_API inline const PrimitivePtr kPrimMaximum = std::make_shared<Primitive>("Maximum");
MS_CORE_API inline const PrimitivePtr kPrimSquare = std::make_shared<Primitive>(kSquare);
MS_CORE_API inline const PrimitivePtr kPrimCumSum = std::make_shared<Primitive>("CumSum");
MS_CORE_API inline const PrimitivePtr kPrimCumProd = std::make_shared<Primitive>("CumProd");
MS_CORE_API inline const PrimitivePtr kPrimSubscalar = std::make_shared<Primitive>("Subscalar");
MS_CORE_API inline const PrimitivePtr kPrimInplaceAdd = std::make_shared<Primitive>("InplaceAdd");
MS_CORE_API inline const PrimitivePtr kPrimLpNorm = std::make_shared<Primitive>(kLpNorm);
MS_CORE_API inline const PrimitivePtr kPrimInplaceSub = std::make_shared<Primitive>("InplaceSub");
MS_CORE_API inline const PrimitivePtr kPrimPow = std::make_shared<Primitive>("Pow");
MS_CORE_API inline const PrimitivePtr kPrimPower = std::make_shared<Primitive>("Power");
MS_CORE_API inline const PrimitivePtr kPrimRealDiv = std::make_shared<Primitive>(kRealDiv);
MS_CORE_API inline const PrimitivePtr kPrimFloorDiv = std::make_shared<Primitive>("FloorDiv");
MS_CORE_API inline const PrimitivePtr kPrimTruncateDiv = std::make_shared<Primitive>("TruncateDiv");
MS_CORE_API inline const PrimitivePtr kPrimSqrt = std::make_shared<Primitive>("Sqrt");
MS_CORE_API inline const PrimitivePtr kPrimTruncateMod = std::make_shared<Primitive>("TruncateMod");
MS_CORE_API inline const PrimitivePtr kPrimSqrtGrad = std::make_shared<Primitive>("SqrtGrad");
MS_CORE_API inline const PrimitivePtr kPrimReciprocal = std::make_shared<Primitive>(kReciprocal);
MS_CORE_API inline const PrimitivePtr kPrimReciprocalGrad = std::make_shared<Primitive>("ReciprocalGrad");
MS_CORE_API inline const PrimitivePtr kPrimExpandDims = std::make_shared<Primitive>("ExpandDims");
MS_CORE_API inline const PrimitivePtr kPrimAbs = std::make_shared<Primitive>(kAbs);
MS_CORE_API inline const PrimitivePtr kPrimAbsGrad = std::make_shared<Primitive>("AbsGrad");
MS_CORE_API inline const PrimitivePtr kPrimRint = std::make_shared<Primitive>("Rint");
MS_CORE_API inline const PrimitivePtr kPrimRound = std::make_shared<Primitive>("Round");
MS_CORE_API inline const PrimitivePtr kPrimExp = std::make_shared<Primitive>(kExp);
MS_CORE_API inline const PrimitivePtr kPrimExpm1 = std::make_shared<Primitive>("Expm1");
MS_CORE_API inline const PrimitivePtr kPrimLog = std::make_shared<Primitive>(kLog);
MS_CORE_API inline const PrimitivePtr kPrimRsqrt = std::make_shared<Primitive>("Rsqrt");
MS_CORE_API inline const PrimitivePtr kPrimRsqrtGrad = std::make_shared<Primitive>("RsqrtGrad");
MS_CORE_API inline const PrimitivePtr kPrimLinSpace = std::make_shared<Primitive>("LinSpace");
MS_CORE_API inline const PrimitivePtr kPrimNonMaxSuppression = std::make_shared<Primitive>("NonMaxSuppression");
MS_CORE_API inline const PrimitivePtr kPrimSign = std::make_shared<Primitive>("Sign");
MS_CORE_API inline const PrimitivePtr kPrimACos = std::make_shared<Primitive>(kACos);
MS_CORE_API inline const PrimitivePtr kPrimAsinGrad = std::make_shared<Primitive>("AsinGrad");
MS_CORE_API inline const PrimitivePtr kPrimACosGrad = std::make_shared<Primitive>(kACosGrad);
MS_CORE_API inline const PrimitivePtr kPrimAtanGrad = std::make_shared<Primitive>("AtanGrad");
MS_CORE_API inline const PrimitivePtr kPrimAsinhGrad = std::make_shared<Primitive>("AsinhGrad");
MS_CORE_API inline const PrimitivePtr kPrimAcoshGrad = std::make_shared<Primitive>("AcoshGrad");
MS_CORE_API inline const PrimitivePtr kPrimFloorMod = std::make_shared<Primitive>("FloorMod");
MS_CORE_API inline const PrimitivePtr kPrimCdist = std::make_shared<Primitive>(kCdist);
MS_CORE_API inline const PrimitivePtr kPrimCdistGrad = std::make_shared<Primitive>(kCdistGrad);
MS_CORE_API inline const PrimitivePtr kPrimWhere = std::make_shared<Primitive>("Where");
MS_CORE_API inline const PrimitivePtr kPrimMatrixInverse = std::make_shared<Primitive>(kMatrixInverse);
MS_CORE_API inline const PrimitivePtr kPrimMatrixDeterminant = std::make_shared<Primitive>(kMatrixDeterminant);
MS_CORE_API inline const PrimitivePtr kPrimLogMatrixDeterminant = std::make_shared<Primitive>(kLogMatrixDeterminant);
MS_CORE_API inline const PrimitivePtr kPrimIndexAdd = std::make_shared<Primitive>("IndexAdd");
MS_CORE_API inline const PrimitivePtr kPrimIdentityMath = std::make_shared<Primitive>("Identity", kSideEffectPropagate);
MS_CORE_API inline const PrimitivePtr kPrimInvGrad = std::make_shared<Primitive>("InvGrad");
MS_CORE_API inline const PrimitivePtr kPrimErfinv = std::make_shared<Primitive>("Erfinv");
MS_CORE_API inline const PrimitivePtr kPrimIsNan = std::make_shared<Primitive>("IsNan");
MS_CORE_API inline const PrimitivePtr kPrimIsInf = std::make_shared<Primitive>("IsInf");
MS_CORE_API inline const PrimitivePtr kPrimIsFinite = std::make_shared<Primitive>("IsFinite");
MS_CORE_API inline const PrimitivePtr kPrimIsClose = std::make_shared<Primitive>("IsClose");
MS_CORE_API inline const PrimitivePtr kPrimLerp = std::make_shared<Primitive>("Lerp");
MS_CORE_API inline const PrimitivePtr kPrimSquareSumAll = std::make_shared<Primitive>("SquareSumAll");
MS_CORE_API inline const PrimitivePtr kPrimComplex = std::make_shared<Primitive>("Complex");
MS_CORE_API inline const PrimitivePtr kPrimXdivy = std::make_shared<Primitive>("Xdivy");
MS_CORE_API inline const PrimitivePtr kPrimXlogy = std::make_shared<Primitive>("Xlogy");
MS_CORE_API inline const PrimitivePtr kPrimInv = std::make_shared<Primitive>("Inv");
MS_CORE_API inline const PrimitivePtr kPrimBitwiseOr = std::make_shared<Primitive>("BitwiseOr");
MS_CORE_API inline const PrimitivePtr kPrimBitwiseAnd = std::make_shared<Primitive>("BitwiseAnd");
MS_CORE_API inline const PrimitivePtr kPrimBitwiseXor = std::make_shared<Primitive>("BitwiseXor");
MS_CORE_API inline const PrimitivePtr kPrimEinsum = std::make_shared<Primitive>("Einsum");
MS_CORE_API inline const PrimitivePtr kPrimEinsumGrad = std::make_shared<Primitive>("EinsumGrad");

// Image
MS_CORE_API inline const PrimitivePtr kPrimNonMaxSuppressionV3 = std::make_shared<Primitive>("NonMaxSuppressionV3");

// Statements
MS_CORE_API inline const PrimitivePtr kPrimReturn = std::make_shared<Primitive>("Return");
MS_CORE_API inline const PrimitivePtr kPrimUnroll = std::make_shared<Primitive>("Unroll");
MS_CORE_API inline const PrimitivePtr kPrimSwitch = std::make_shared<Primitive>("Switch");
MS_CORE_API inline const PrimitivePtr kPrimSwitchLayer = std::make_shared<Primitive>("switch_layer");
MS_CORE_API inline const PrimitivePtr kPrimAssign = std::make_shared<Primitive>(kAssign);
MS_CORE_API inline const PrimitivePtr kPrimAssignAdd = std::make_shared<Primitive>(kAssignAdd);
MS_CORE_API inline const PrimitivePtr kPrimAssignSub = std::make_shared<Primitive>(kAssignSub);
MS_CORE_API inline const PrimitivePtr kPrimSelect = std::make_shared<Primitive>(kSelect);
MS_CORE_API inline const PrimitivePtr kPrimCall = std::make_shared<Primitive>("call");

MS_CORE_API inline const PrimitivePtr kPrimMakeTuple = std::make_shared<Primitive>(kMakeTuple);
MS_CORE_API inline const PrimitivePtr kPrimMakeSlice = std::make_shared<Primitive>("make_slice");
MS_CORE_API inline const PrimitivePtr kPrimTupleGetItem = std::make_shared<Primitive>(kTupleGetItem);
MS_CORE_API inline const PrimitivePtr kPrimSliceGetItem = std::make_shared<Primitive>(kSliceGetItem);
MS_CORE_API inline const PrimitivePtr kPrimArrayGetItem = std::make_shared<Primitive>("array_getitem");
MS_CORE_API inline const PrimitivePtr kPrimTupleSetItem = std::make_shared<Primitive>("tuple_setitem");
MS_CORE_API inline const PrimitivePtr kPrimArraySetItem = std::make_shared<Primitive>("array_setitem");
MS_CORE_API inline const PrimitivePtr kPrimGetAttr = std::make_shared<Primitive>("getattr");
MS_CORE_API inline const PrimitivePtr kPrimTupleLen = std::make_shared<Primitive>("tuple_len");
MS_CORE_API inline const PrimitivePtr kPrimArrayLen = std::make_shared<Primitive>("array_len");
MS_CORE_API inline const PrimitivePtr kPrimTileShape = std::make_shared<Primitive>("tile_shape");
MS_CORE_API inline const PrimitivePtr kPrimGenerateShapeIndex = std::make_shared<Primitive>("generate_shape_index");
MS_CORE_API inline const PrimitivePtr kPrimGenerateInverseIndex = std::make_shared<Primitive>("generate_inverse_index");

// Debug ops
MS_CORE_API inline const PrimitivePtr kPrimAssert = std::make_shared<Primitive>("Assert");
#ifndef ENABLE_SECURITY
MS_CORE_API inline const PrimitivePtr kPrimScalarSummary = std::make_shared<Primitive>("ScalarSummary");
MS_CORE_API inline const PrimitivePtr kPrimImageSummary = std::make_shared<Primitive>("ImageSummary");
MS_CORE_API inline const PrimitivePtr kPrimTensorSummary = std::make_shared<Primitive>("TensorSummary");
MS_CORE_API inline const PrimitivePtr kPrimHistogramSummary = std::make_shared<Primitive>("HistogramSummary");
#endif
MS_CORE_API inline const PrimitivePtr kPrimDebug = std::make_shared<Primitive>("Debug");

// Dynamic shape testing
MS_CORE_API inline const PrimitivePtr kPrimGpuConvertToDynamicShape =
  std::make_shared<Primitive>("GpuConvertToDynamicShape");
MS_CORE_API inline const PrimitivePtr kPrimErrorOnDynamicShapeInput =
  std::make_shared<Primitive>("ErrorOnDynamicShapeInput");

// Other miscellaneous
MS_CORE_API inline const PrimitivePtr kPrimDepend = std::make_shared<Primitive>("Depend", kSideEffectPropagate);
MS_CORE_API inline const PrimitivePtr kPrimIOU = std::make_shared<Primitive>("IOU");
MS_CORE_API inline const PrimitivePtr kPrimReformat = std::make_shared<Primitive>("Reformat");
MS_CORE_API inline const PrimitivePtr kPrimLoad = std::make_shared<Primitive>("Load");
MS_CORE_API inline const PrimitivePtr kPrimUpdateState = std::make_shared<Primitive>("UpdateState");
MS_CORE_API inline const PrimitivePtr kPrimPartial = std::make_shared<Primitive>("Partial", kSideEffectPropagate);
MS_CORE_API inline const PrimitivePtr kPrimIdentity = std::make_shared<Primitive>("identity", kSideEffectPropagate);
MS_CORE_API inline const PrimitivePtr kPrimHookBackward = std::make_shared<Primitive>("HookBackward");
MS_CORE_API inline const PrimitivePtr kPrimPrintShapeType = std::make_shared<Primitive>("PrintShapeType");
MS_CORE_API inline const PrimitivePtr kPrimSameTypeShape = std::make_shared<Primitive>("SameTypeShape");
MS_CORE_API inline const PrimitivePtr kPrimPrint = std::make_shared<Primitive>("Print");
MS_CORE_API inline const PrimitivePtr kPrimIs_ = std::make_shared<Primitive>("is_");
MS_CORE_API inline const PrimitivePtr kPrimIsNot = std::make_shared<Primitive>("is_not");
MS_CORE_API inline const PrimitivePtr kPrimInDict = std::make_shared<Primitive>("in_dict");
MS_CORE_API inline const PrimitivePtr kPrimNotInDict = std::make_shared<Primitive>("not_in_dict");
MS_CORE_API inline const PrimitivePtr kPrimIsConsant = std::make_shared<Primitive>("is_constant");
MS_CORE_API inline const PrimitivePtr kPrimEquivFormat = std::make_shared<Primitive>("EquivFormat");
MS_CORE_API inline const PrimitivePtr kPrimLshProjection = std::make_shared<Primitive>("LshProjection");
MS_CORE_API inline const PrimitivePtr kPrimHashtableLookup = std::make_shared<Primitive>("HashtableLookup");
MS_CORE_API inline const PrimitivePtr kPrimCustomPredict = std::make_shared<Primitive>("CustomPredict");
MS_CORE_API inline const PrimitivePtr kPrimPriorBox = std::make_shared<Primitive>("PriorBox");
MS_CORE_API inline const PrimitivePtr kPrimQuantDTypeCast = std::make_shared<Primitive>("QuantDTypeCast");
MS_CORE_API inline const PrimitivePtr kPrimWhile = std::make_shared<Primitive>("While");
MS_CORE_API inline const PrimitivePtr kPrimPull = std::make_shared<Primitive>("Pull");
MS_CORE_API inline const PrimitivePtr kPrimPush = std::make_shared<Primitive>("Push");
MS_CORE_API inline const PrimitivePtr kPrimNPUAllocFloatStatus = std::make_shared<Primitive>("NPUAllocFloatStatus");
MS_CORE_API inline const PrimitivePtr kPrimPyFunc = std::make_shared<Primitive>("PyFunc");

// Structures
MS_CORE_API inline const PrimitivePtr kPrimMakeList = std::make_shared<Primitive>("make_list");
MS_CORE_API inline const PrimitivePtr kPrimMakeKeywordArg = std::make_shared<Primitive>("make_keyword_arg");
MS_CORE_API inline const PrimitivePtr kPrimListGetItem = std::make_shared<Primitive>("list_getitem");
MS_CORE_API inline const PrimitivePtr kPrimListSetItem = std::make_shared<Primitive>("list_setitem");
MS_CORE_API inline const PrimitivePtr kPrimDictGetItem = std::make_shared<Primitive>("dict_getitem");
MS_CORE_API inline const PrimitivePtr kPrimDictSetItem = std::make_shared<Primitive>("dict_setitem");
MS_CORE_API inline const PrimitivePtr kPrimDictGetKeys = std::make_shared<Primitive>("dict_getkeys");
MS_CORE_API inline const PrimitivePtr kPrimDictGetValues = std::make_shared<Primitive>("dict_getvalues");
MS_CORE_API inline const PrimitivePtr kPrimDictItems = std::make_shared<Primitive>("dict_items");
MS_CORE_API inline const PrimitivePtr kPrimListAppend = std::make_shared<Primitive>("list_append");
MS_CORE_API inline const PrimitivePtr kPrimListLen = std::make_shared<Primitive>("list_len");

// Other miscellaneous
MS_CORE_API inline const PrimitivePtr kPrimEnvironCreate = std::make_shared<Primitive>(kEnvironCreate);
MS_CORE_API inline const PrimitivePtr kPrimEnvironSet = std::make_shared<Primitive>(kEnvironSet);
MS_CORE_API inline const PrimitivePtr kPrimEnvironGet = std::make_shared<Primitive>(kEnvironGet);
MS_CORE_API inline const PrimitivePtr kPrimEnvironAdd = std::make_shared<Primitive>(kEnvironAdd);
MS_CORE_API inline const PrimitivePtr kPrimEnvironDestroyAll = std::make_shared<Primitive>(kEnvironDestroyAll);
MS_CORE_API inline const PrimitivePtr kPrimMakeRefKey = std::make_shared<Primitive>("MakeRefKey");
MS_CORE_API inline const PrimitivePtr kPrimGetRefKey = std::make_shared<Primitive>("get_ref_key");
MS_CORE_API inline const PrimitivePtr kPrimMakeRef = std::make_shared<Primitive>("make_ref");
MS_CORE_API inline const PrimitivePtr kPrimGetRefValue = std::make_shared<Primitive>("get_ref_value");

// Python interpreter runner
MS_CORE_API inline const PrimitivePtr kPrimPyInterpret = std::make_shared<Primitive>("PyInterpret");

// Other primitive not used by backend but used in core;
MS_CORE_API inline const PrimitivePtr kPrimStateSetItem = std::make_shared<Primitive>("state_setitem");
MS_CORE_API inline const PrimitivePtr kPrimJ = std::make_shared<Primitive>(kJ, kSideEffectPropagate);
MS_CORE_API inline const PrimitivePtr kPrimShard = std::make_shared<Primitive>("Shard", kSideEffectPropagate);

// Used to build graph which have keyword arguments
MS_CORE_API inline const PrimitivePtr kPrimExtractKeywordArg = std::make_shared<Primitive>("extract_keyword_arg");
MS_CORE_API inline const PrimitivePtr kPrimMakeDict = std::make_shared<Primitive>("make_dict");

// GraphKernel ops
MS_CORE_API inline const PrimitivePtr kPrimInplaceAssign = std::make_shared<Primitive>("InplaceAssign");

// Custom
MS_CORE_API inline const PrimitivePtr kPrimCustom = std::make_shared<Primitive>("Custom");

// Only used in lite
MS_CORE_API inline const PrimitivePtr kPrimLeakyRelu = std::make_shared<Primitive>("LeakyRelu");
MS_CORE_API inline const PrimitivePtr kPrimConstant = std::make_shared<Primitive>("Constant");
MS_CORE_API inline const PrimitivePtr kPrimLocalResponseNormalization =
  std::make_shared<Primitive>("LocalResponseNormalization");
MS_CORE_API inline const PrimitivePtr kPrimFftReal = std::make_shared<Primitive>("FftReal");
MS_CORE_API inline const PrimitivePtr kPrimMfcc = std::make_shared<Primitive>("Mfcc");
MS_CORE_API inline const PrimitivePtr kPrimRfft = std::make_shared<Primitive>("Rfft");
MS_CORE_API inline const PrimitivePtr kPrimFftImag = std::make_shared<Primitive>("FftImag");
MS_CORE_API inline const PrimitivePtr kPrimSkipGram = std::make_shared<Primitive>("SkipGram");
MS_CORE_API inline const PrimitivePtr kPrimConv2DFusion = std::make_shared<Primitive>("Conv2DFusion");
MS_CORE_API inline const PrimitivePtr kPrimConv2dTransposeFusion = std::make_shared<Primitive>("Conv2dTransposeFusion");
MS_CORE_API inline const PrimitivePtr kPrimDepthWiseConv2DFusion = std::make_shared<Primitive>("DepthWiseConv2DFusion");
MS_CORE_API inline const PrimitivePtr kPrimAddFusion = std::make_shared<Primitive>("AddFusion");
MS_CORE_API inline const PrimitivePtr kPrimScaleFusion = std::make_shared<Primitive>("ScaleFusion");
MS_CORE_API inline const PrimitivePtr kPrimSubFusion = std::make_shared<Primitive>("SubFusion");
MS_CORE_API inline const PrimitivePtr kPrimMulFusion = std::make_shared<Primitive>("MulFusion");
MS_CORE_API inline const PrimitivePtr kPrimSigmoid = std::make_shared<Primitive>("Sigmoid");
MS_CORE_API inline const PrimitivePtr kPrimSigmoidGrad = std::make_shared<Primitive>("SigmoidGrad");
MS_CORE_API inline const PrimitivePtr kPrimHSigmoid = std::make_shared<Primitive>("HSigmoid");
MS_CORE_API inline const PrimitivePtr kPrimHSigmoidGrad = std::make_shared<Primitive>("HSigmoidGrad");
MS_CORE_API inline const PrimitivePtr kPrimClip = std::make_shared<Primitive>("Clip");
MS_CORE_API inline const PrimitivePtr kPrimHardTanh = std::make_shared<Primitive>("HardTanh");
MS_CORE_API inline const PrimitivePtr kPrimDepthWiseConv2DTransposeFusion =
  std::make_shared<Primitive>("DepthWiseConv2DTransposeFusion");
MS_CORE_API inline const PrimitivePtr kPrimArgMinFusion = std::make_shared<Primitive>("ArgMinFusion");
MS_CORE_API inline const PrimitivePtr kPrimArgMaxFusion = std::make_shared<Primitive>("ArgMaxFusion");
MS_CORE_API inline const PrimitivePtr kPrimSpaceToDepth = std::make_shared<Primitive>("SpaceToDepth");
MS_CORE_API inline const PrimitivePtr kPrimPadFusion = std::make_shared<Primitive>("PadFusion");
MS_CORE_API inline const PrimitivePtr kPrimPowFusion = std::make_shared<Primitive>("PowFusion");
MS_CORE_API inline const PrimitivePtr kPrimResize = std::make_shared<Primitive>("Resize");
MS_CORE_API inline const PrimitivePtr kPrimArgMinWithValue = std::make_shared<Primitive>("ArgMinWithValue");
MS_CORE_API inline const PrimitivePtr kPrimIf = std::make_shared<Primitive>("If");
MS_CORE_API inline const PrimitivePtr kPrimAvgPoolFusion = std::make_shared<Primitive>("AvgPoolFusion");
MS_CORE_API inline const PrimitivePtr kPrimMaxPoolFusion = std::make_shared<Primitive>("MaxPoolFusion");
MS_CORE_API inline const PrimitivePtr kPrimActivation = std::make_shared<Primitive>("Activation");
MS_CORE_API inline const PrimitivePtr kPrimPReLUFusion = std::make_shared<Primitive>("PReLUFusion");
MS_CORE_API inline const PrimitivePtr kPrimTopKFusion = std::make_shared<Primitive>("TopKFusion");
MS_CORE_API inline const PrimitivePtr kPrimTileFusion = std::make_shared<Primitive>("TileFusion");
MS_CORE_API inline const PrimitivePtr kPrimReduceFusion = std::make_shared<Primitive>("ReduceFusion");
MS_CORE_API inline const PrimitivePtr kPrimLayerNormFusion = std::make_shared<Primitive>("LayerNormFusion");
MS_CORE_API inline const PrimitivePtr kPrimDType = std::make_shared<Primitive>("DType");
MS_CORE_API inline const PrimitivePtr kPrimDivFusion = std::make_shared<Primitive>("DivFusion");
MS_CORE_API inline const PrimitivePtr kPrimErf = std::make_shared<Primitive>("Erf");
MS_CORE_API inline const PrimitivePtr kPrimErfc = std::make_shared<Primitive>("Erfc");
MS_CORE_API inline const PrimitivePtr kPrimSplice = std::make_shared<Primitive>("Splice");
MS_CORE_API inline const PrimitivePtr kPrimAffine = std::make_shared<Primitive>("Affine");
MS_CORE_API inline const PrimitivePtr kPrimEltwise = std::make_shared<Primitive>("Eltwise");
MS_CORE_API inline const PrimitivePtr kPrimMatMulFusion = std::make_shared<Primitive>("MatMulFusion");
MS_CORE_API inline const PrimitivePtr kPrimDynamicQuant = std::make_shared<Primitive>("DynamicQuant");

// Type introspection
MS_CORE_API inline const PrimitivePtr kPrimTypeOf = std::make_shared<Primitive>("typeof");
MS_CORE_API inline const PrimitivePtr kPrimHasType = std::make_shared<Primitive>("hastype");

MS_CORE_API inline const PrimitivePtr kPrimResolve = std::make_shared<Primitive>("resolve");
MS_CORE_API inline const PrimitivePtr kPrimEmbed = std::make_shared<Primitive>("embed");
MS_CORE_API inline const PrimitivePtr kPrimRefToEmbed = std::make_shared<Primitive>("RefToEmbed");
MS_CORE_API inline const PrimitivePtr kPrimCreateInstance = std::make_shared<Primitive>("create_instance");

// Other miscellaneous
MS_CORE_API inline const PrimitivePtr kPrimGetRefOrigin = std::make_shared<Primitive>("get_ref_origin");
MS_CORE_API inline const PrimitivePtr kPrimInsertGradientOf = std::make_shared<Primitive>("InsertGradientOf");
MS_CORE_API inline const PrimitivePtr kPrimCheckBprop = std::make_shared<Primitive>("CheckBprop");
MS_CORE_API inline const PrimitivePtr kPrimMixedPrecisionCast = std::make_shared<Primitive>("mixed_precision_cast");
MS_CORE_API inline const PrimitivePtr kPrimMakeRecord = std::make_shared<Primitive>("make_record");

// Structures
MS_CORE_API inline const PrimitivePtr kPrimListMap = std::make_shared<Primitive>("list_map");
MS_CORE_API inline const PrimitivePtr kPrimListReduce = std::make_shared<Primitive>("list_reduce");
MS_CORE_API inline const PrimitivePtr kPrimTupleReversed = std::make_shared<Primitive>("tuple_reversed");
MS_CORE_API inline const PrimitivePtr kPrimReducedShape = std::make_shared<Primitive>("reduced_shape");
MS_CORE_API inline const PrimitivePtr kPrimTupleDiv = std::make_shared<Primitive>("tuple_div");
MS_CORE_API inline const PrimitivePtr kPrimTupleToArray = std::make_shared<Primitive>("tuple_to_array");
MS_CORE_API inline const PrimitivePtr kPrimShapeMul = std::make_shared<Primitive>("shape_mul");
MS_CORE_API inline const PrimitivePtr kPrimTupleEqual = std::make_shared<Primitive>("tuple_equal");
MS_CORE_API inline const PrimitivePtr kPrimListEqual = std::make_shared<Primitive>("list_equal");
MS_CORE_API inline const PrimitivePtr kPrimMakeRange = std::make_shared<Primitive>("make_range");
MS_CORE_API inline const PrimitivePtr kPrimStopGradient = std::make_shared<Primitive>("stop_gradient");
MS_CORE_API inline const PrimitivePtr kPrimStringEqual = std::make_shared<Primitive>("string_equal");
MS_CORE_API inline const PrimitivePtr kPrimStringConcat = std::make_shared<Primitive>("string_concat");
MS_CORE_API inline const PrimitivePtr kPrimDictLen = std::make_shared<Primitive>("dict_len");
MS_CORE_API inline const PrimitivePtr kPrimFakeBprop = std::make_shared<Primitive>("fake_bprop");
MS_CORE_API inline const PrimitivePtr kPrimBroadcastGradientArgs = std::make_shared<Primitive>("BroadcastGradientArgs");
MS_CORE_API inline const PrimitivePtr kPrimDynamicBroadcastGradientArgs =
  std::make_shared<Primitive>(kDynamicBroadcastGradientArgs);

// Random
MS_CORE_API inline const PrimitivePtr kPrimStandardNormal = std::make_shared<Primitive>("StandardNormal");

// RL Ops
MS_CORE_API inline const PrimitivePtr kPrimTensorArrayStack = std::make_shared<Primitive>("TensorArrayStack");
MS_CORE_API inline const PrimitivePtr kPrimTensorArray = std::make_shared<Primitive>("TensorArray");
MS_CORE_API inline const PrimitivePtr kPrimTensorArrayWrite = std::make_shared<Primitive>("TensorArrayWrite");
MS_CORE_API inline const PrimitivePtr kPrimTensorArrayGather = std::make_shared<Primitive>("TensorArrayGather");

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
