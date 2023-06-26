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

#ifndef MINDSPORE_CORE_BASE_ARRAY_OPS_H_
#define MINDSPORE_CORE_BASE_ARRAY_OPS_H_

#include <memory>
#include "ir/anf.h"
#include "ir/primitive.h"
#include "utils/hash_map.h"

namespace mindspore {
namespace prim {
// Arrays
constexpr auto kTopK = "TopK";
constexpr auto kLeftShift = "LeftShift";
constexpr auto kCountNonZero = "CountNonZero";
constexpr auto kFillDiagonal = "FillDiagonal";
constexpr auto kSegmentMax = "SegmentMax";
constexpr auto kSegmentSum = "SegmentSum";
constexpr auto kSegmentMin = "SegmentMin";
constexpr auto kSegmentMean = "SegmentMean";
constexpr auto kSegmentProd = "SegmentProd";
constexpr auto kDynamicShape = "DynamicShape";
constexpr auto kTensorShape = "TensorShape";
constexpr auto kCheckNumerics = "CheckNumerics";
constexpr auto kStack = "Stack";
constexpr auto kPack = "Pack";
constexpr auto kLogNormalReverse = "LogNormalReverse";
constexpr auto kUnstack = "Unstack";
constexpr auto kUnpack = "Unpack";
constexpr auto kIdentity = "Identity";
constexpr auto kUnravelIndex = "UnravelIndex";
constexpr auto kDynamicBroadcastTo = "DynamicBroadcastTo";
constexpr auto kConcat = "Concat";
constexpr auto kParallelConcat = "ParallelConcat";
constexpr auto kFlattenConcat = "FlattenConcat";
constexpr auto kConjugateTranspose = "ConjugateTranspose";
constexpr auto kTranspose = "Transpose";
constexpr auto kGatherDGrad = "GatherDGrad";
constexpr auto kGatherDGradV2 = "GatherDGradV2";
constexpr auto kCoalesce = "Coalesce";
constexpr auto kStridedSlice = "StridedSlice";
constexpr auto kStridedSliceGrad = "StridedSliceGrad";
constexpr auto kUnsortedSegmentSum = "UnsortedSegmentSum";
constexpr auto kUnsortedSegmentSumD = "UnsortedSegmentSumD";
constexpr auto kUnsortedSegmentProd = "UnsortedSegmentProd";
constexpr auto kTile = "Tile";
constexpr auto kPadding = "Padding";
constexpr auto kMirrorPad = "MirrorPad";
constexpr auto kScatterAddWithAxis = "ScatterAddWithAxis";
constexpr auto kDiag = "Diag";
constexpr auto kDiagPart = "DiagPart";
constexpr auto kMatrixDiagV3 = "MatrixDiagV3";
constexpr auto kMatrixDiagPartV3 = "MatrixDiagPartV3";
constexpr auto kMatrixSetDiagV3 = "MatrixSetDiagV3";
constexpr auto kMatrixBandPart = "MatrixBandPart";
constexpr auto kRealInner = "RealInner";
constexpr auto kSplitV = "SplitV";
constexpr auto kFillV2 = "FillV2";
constexpr auto kFills = "Fills";
constexpr auto kLstsq = "Lstsq";
constexpr auto kLowerBound = "LowerBound";
constexpr auto kUpperBound = "UpperBound";
constexpr auto kCummax = "Cummax";
constexpr auto kMvlgamma = "Mvlgamma";
constexpr auto kMvlgammaGrad = "MvlgammaGrad";
constexpr auto kRightShift = "RightShift";
constexpr auto kLogSpace = "LogSpace";
constexpr auto kTril = "Tril";
constexpr auto kEye = "Eye";
constexpr auto kTriu = "Triu";
constexpr auto kMeshgrid = "Meshgrid";
constexpr auto kAffineGrid = "AffineGrid";
constexpr auto kAffineGridGrad = "AffineGridGrad";
constexpr auto kBroadcastTo = "BroadcastTo";
constexpr auto kBincount = "Bincount";
constexpr auto kReshape = "Reshape";
constexpr auto kNonZero = "NonZero";
constexpr auto kScatterNdMax = "ScatterNdMax";
constexpr auto kScatterNdMin = "ScatterNdMin";
constexpr auto kSlice = "Slice";
constexpr auto kZerosLike = "ZerosLike";
constexpr auto kOnes = "Ones";
constexpr auto kOnesLike = "OnesLike";

// Arrays
GVAR_DEF(PrimitivePtr, kPrimExpand, std::make_shared<Primitive>("Expand"));
GVAR_DEF(PrimitivePtr, kPrimExpandDims, std::make_shared<Primitive>("ExpandDims"));
GVAR_DEF(PrimitivePtr, kPrimMakeRange, std::make_shared<Primitive>("make_range"));
GVAR_DEF(PrimitivePtr, kPrimBroadcast, std::make_shared<Primitive>("Broadcast"));
GVAR_DEF(PrimitivePtr, kPrimZeros, std::make_shared<Primitive>("Zeros"));
GVAR_DEF(PrimitivePtr, kPrimZerosLike, std::make_shared<Primitive>(kZerosLike));
GVAR_DEF(PrimitivePtr, kPrimOnes, std::make_shared<Primitive>(kOnes));
GVAR_DEF(PrimitivePtr, kPrimOnesLike, std::make_shared<Primitive>(kOnesLike));
GVAR_DEF(PrimitivePtr, kPrimFill, std::make_shared<Primitive>("Fill"));
GVAR_DEF(PrimitivePtr, kPrimLeftShift, std::make_shared<Primitive>(kLeftShift));
GVAR_DEF(PrimitivePtr, kPrimFillDiagonal, std::make_shared<Primitive>(kFillDiagonal));
GVAR_DEF(PrimitivePtr, kPrimIdentitys, std::make_shared<Primitive>(kIdentity));
GVAR_DEF(PrimitivePtr, kPrimUnravelIndex, std::make_shared<Primitive>(kUnravelIndex));
GVAR_DEF(PrimitivePtr, kPrimDynamicBroadcastTo, std::make_shared<Primitive>(kDynamicBroadcastTo));
GVAR_DEF(PrimitivePtr, kPrimCummin, std::make_shared<Primitive>("Cummin"));
GVAR_DEF(PrimitivePtr, kPrimBroadcastTo, std::make_shared<Primitive>("BroadcastTo"));
GVAR_DEF(PrimitivePtr, kPrimScalarToArray, std::make_shared<Primitive>("scalar_to_array"));
GVAR_DEF(PrimitivePtr, kPrimLogNormalReverse, std::make_shared<Primitive>("LogNormalReverse"));
GVAR_DEF(PrimitivePtr, kPrimTopK, std::make_shared<Primitive>(kTopK));
GVAR_DEF(PrimitivePtr, kPrimInTopK, std::make_shared<Primitive>("InTopK"));
GVAR_DEF(PrimitivePtr, kPrimInTopKD, std::make_shared<Primitive>("InTopKD"));
GVAR_DEF(PrimitivePtr, kPrimArrayToScalar, std::make_shared<Primitive>("array_to_scalar"));
GVAR_DEF(PrimitivePtr, kPrimBroadcastShape, std::make_shared<Primitive>("broadcast_shape"));
GVAR_DEF(PrimitivePtr, kPrimArrayMap, std::make_shared<Primitive>("array_map"));
GVAR_DEF(PrimitivePtr, kPrimArrayReduce, std::make_shared<Primitive>("array_reduce"));
GVAR_DEF(PrimitivePtr, kPrimCast, std::make_shared<Primitive>("Cast"));
GVAR_DEF(PrimitivePtr, kPrimConcat, std::make_shared<Primitive>(kConcat));
GVAR_DEF(PrimitivePtr, kPrimConcatD, std::make_shared<Primitive>("ConcatD"));
GVAR_DEF(PrimitivePtr, kPrimParallelConcat, std::make_shared<Primitive>(kParallelConcat));
GVAR_DEF(PrimitivePtr, kPrimCountNonZero, std::make_shared<Primitive>("CountNonZero"));
GVAR_DEF(PrimitivePtr, kPrimFlattenConcat, std::make_shared<Primitive>(kFlattenConcat));
GVAR_DEF(PrimitivePtr, kPrimSqueeze, std::make_shared<Primitive>("Squeeze"));
GVAR_DEF(PrimitivePtr, kPrimSqueezeV3, std::make_shared<Primitive>("SqueezeV3"));
GVAR_DEF(PrimitivePtr, kPrimUnsqueeze, std::make_shared<Primitive>("Unsqueeze"));
GVAR_DEF(PrimitivePtr, kPrimConjugateTranspose, std::make_shared<Primitive>(kConjugateTranspose));
GVAR_DEF(PrimitivePtr, kPrimTransposeD, std::make_shared<Primitive>("TransposeD"));
GVAR_DEF(PrimitivePtr, kPrimTranspose, std::make_shared<Primitive>(kTranspose));
GVAR_DEF(PrimitivePtr, kPrimGatherV2, std::make_shared<Primitive>("GatherV2"));
GVAR_DEF(PrimitivePtr, kPrimGatherD, std::make_shared<Primitive>("GatherD"));
GVAR_DEF(PrimitivePtr, kPrimGatherDGrad, std::make_shared<Primitive>(kGatherDGrad));
GVAR_DEF(PrimitivePtr, kPrimGatherDGradV2, std::make_shared<Primitive>(kGatherDGradV2));
GVAR_DEF(PrimitivePtr, kPrimGather, std::make_shared<Primitive>("Gather"));
GVAR_DEF(PrimitivePtr, kPrimGatherNd, std::make_shared<Primitive>("GatherNd"));
GVAR_DEF(PrimitivePtr, kPrimSparseGatherV2, std::make_shared<Primitive>("SparseGatherV2"));
GVAR_DEF(PrimitivePtr, kPrimCoalesce, std::make_shared<Primitive>(kCoalesce));
GVAR_DEF(PrimitivePtr, kPrimShapeCalc, std::make_shared<Primitive>("ShapeCalc"));
GVAR_DEF(PrimitivePtr, kPrimStridedRead, std::make_shared<Primitive>("StridedRead"));
GVAR_DEF(PrimitivePtr, kPrimStridedWrite, std::make_shared<Primitive>("StridedWrite"));
GVAR_DEF(PrimitivePtr, kPrimStridedSlice, std::make_shared<Primitive>(kStridedSlice));
GVAR_DEF(PrimitivePtr, kPrimStridedSliceGrad, std::make_shared<Primitive>(kStridedSliceGrad));
GVAR_DEF(PrimitivePtr, kPrimTensorShape, std::make_shared<Primitive>(kTensorShape));
GVAR_DEF(PrimitivePtr, kPrimDynamicShape, std::make_shared<Primitive>(kDynamicShape));
GVAR_DEF(PrimitivePtr, kPrimCheckNumerics, std::make_shared<Primitive>(kCheckNumerics));
GVAR_DEF(PrimitivePtr, kPrimSize, std::make_shared<Primitive>("Size"));
GVAR_DEF(PrimitivePtr, kPrimArgMax, std::make_shared<Primitive>("Argmax"));
GVAR_DEF(PrimitivePtr, kPrimArgmin, std::make_shared<Primitive>("Argmin"));
GVAR_DEF(PrimitivePtr, kPrimArgMin, std::make_shared<Primitive>("ArgMin"));
GVAR_DEF(PrimitivePtr, kPrimArgminV2, std::make_shared<Primitive>("ArgminV2"));
GVAR_DEF(PrimitivePtr, kPrimPack, std::make_shared<Primitive>("Pack"));
GVAR_DEF(PrimitivePtr, kPrimStack, std::make_shared<Primitive>(kStack));
GVAR_DEF(PrimitivePtr, kPrimUnpack, std::make_shared<Primitive>("Unpack"));
GVAR_DEF(PrimitivePtr, kPrimUnstack, std::make_shared<Primitive>(kUnstack));
GVAR_DEF(PrimitivePtr, kPrimUnsortedSegmentMax, std::make_shared<Primitive>("UnsortedSegmentMax"));
GVAR_DEF(PrimitivePtr, kPrimUnsortedSegmentProd, std::make_shared<Primitive>(kUnsortedSegmentProd));
GVAR_DEF(PrimitivePtr, kPrimUnsortedSegmentSum, std::make_shared<Primitive>(kUnsortedSegmentSum));
GVAR_DEF(PrimitivePtr, kPrimUnsortedSegmentSumD, std::make_shared<Primitive>(kUnsortedSegmentSumD));
GVAR_DEF(PrimitivePtr, kPrimUnsortedSegmentMin, std::make_shared<Primitive>("UnsortedSegmentMin"));
GVAR_DEF(PrimitivePtr, kPrimConcatOffset, std::make_shared<Primitive>("ConcatOffset"));
GVAR_DEF(PrimitivePtr, kPrimConcatOffsetV1, std::make_shared<Primitive>("ConcatOffsetV1"));
GVAR_DEF(PrimitivePtr, kPrimIdentityN, std::make_shared<Primitive>("IdentityN"));
GVAR_DEF(PrimitivePtr, kPrimReshape, std::make_shared<Primitive>("Reshape"));
GVAR_DEF(PrimitivePtr, kPrimSubAndFilter, std::make_shared<Primitive>("SubAndFilter"));
GVAR_DEF(PrimitivePtr, kPrimMapCacheIdx, std::make_shared<Primitive>("MapCacheIdx"));
GVAR_DEF(PrimitivePtr, kPrimUpdateCache, std::make_shared<Primitive>("UpdateCache"));
GVAR_DEF(PrimitivePtr, kPrimComputeAccidentalHits, std::make_shared<Primitive>("ComputeAccidentalHits"));
GVAR_DEF(PrimitivePtr, kPrimCacheSwapTable, std::make_shared<Primitive>("CacheSwapTable"));
GVAR_DEF(PrimitivePtr, kPrimPadAndShift, std::make_shared<Primitive>("PadAndShift"));
GVAR_DEF(PrimitivePtr, kPrimSlice, std::make_shared<Primitive>(kSlice));
GVAR_DEF(PrimitivePtr, kPrimSliceGrad, std::make_shared<Primitive>("SliceGrad"));
GVAR_DEF(PrimitivePtr, kPrimSliceFusion, std::make_shared<Primitive>("SliceFusion"));
GVAR_DEF(PrimitivePtr, kPrimTile, std::make_shared<Primitive>(kTile));
GVAR_DEF(PrimitivePtr, kPrimTileD, std::make_shared<Primitive>("TileD"));
GVAR_DEF(PrimitivePtr, kPrimAddN, std::make_shared<Primitive>("AddN"));
GVAR_DEF(PrimitivePtr, kPrimAccumulateNV2, std::make_shared<Primitive>("AccumulateNV2"));
GVAR_DEF(PrimitivePtr, kPrimTransData, std::make_shared<Primitive>("TransData"));
GVAR_DEF(PrimitivePtr, kPrimTransDataRNN, std::make_shared<Primitive>("TransDataRNN"));
GVAR_DEF(PrimitivePtr, kPrimPad, std::make_shared<Primitive>("Pad"));
GVAR_DEF(PrimitivePtr, kPrimPadD, std::make_shared<Primitive>("PadD"));
GVAR_DEF(PrimitivePtr, kPrimPadding, std::make_shared<Primitive>(kPadding));
GVAR_DEF(PrimitivePtr, kPrimMirrorPad, std::make_shared<Primitive>(kMirrorPad));
GVAR_DEF(PrimitivePtr, kPrimArgMaxWithValue, std::make_shared<Primitive>("ArgMaxWithValue"));
GVAR_DEF(PrimitivePtr, kPrimArgMinWithValue, std::make_shared<Primitive>("ArgMinWithValue"));
GVAR_DEF(PrimitivePtr, kPrimUnique, std::make_shared<Primitive>("Unique"));
GVAR_DEF(PrimitivePtr, kPrimUniqueWithPad, std::make_shared<Primitive>("UniqueWithPad"));
GVAR_DEF(PrimitivePtr, kPrimUniqueGrad, std::make_shared<Primitive>("UniqueGrad"));
GVAR_DEF(PrimitivePtr, kPrimUniqueConsecutive, std::make_shared<Primitive>("UniqueConsecutive"));
GVAR_DEF(PrimitivePtr, kPrimExtractImagePatches, std::make_shared<Primitive>("ExtractImagePatches"));
GVAR_DEF(PrimitivePtr, kPrimDynamicRNN, std::make_shared<Primitive>("DynamicRNN"));
GVAR_DEF(PrimitivePtr, kPrimCudnnGRU, std::make_shared<Primitive>("CudnnGRU"));
GVAR_DEF(PrimitivePtr, kPrimGRUV2, std::make_shared<Primitive>("GRUV2"));
GVAR_DEF(PrimitivePtr, kPrimGRUV2Grad, std::make_shared<Primitive>("GRUV2Grad"));
GVAR_DEF(PrimitivePtr, kPrimLSTMV2, std::make_shared<Primitive>("LSTMV2"));
GVAR_DEF(PrimitivePtr, kPrimDynamicRNNGrad, std::make_shared<Primitive>("DynamicRNNGrad"));
GVAR_DEF(PrimitivePtr, kPrimDynamicGRUV2, std::make_shared<Primitive>("DynamicGRUV2"));
GVAR_DEF(PrimitivePtr, kPrimDynamicGRUV2Grad, std::make_shared<Primitive>("DynamicGRUV2Grad"));
GVAR_DEF(PrimitivePtr, kPrimSearchSorted, std::make_shared<Primitive>("SearchSorted"));
GVAR_DEF(PrimitivePtr, kPrimScatterAdd, std::make_shared<Primitive>("ScatterAdd"));
GVAR_DEF(PrimitivePtr, kPrimScatterSub, std::make_shared<Primitive>("ScatterSub"));
GVAR_DEF(PrimitivePtr, kPrimScatterMul, std::make_shared<Primitive>("ScatterMul"));
GVAR_DEF(PrimitivePtr, kPrimScatterDiv, std::make_shared<Primitive>("ScatterDiv"));
GVAR_DEF(PrimitivePtr, kPrimScatterMax, std::make_shared<Primitive>("ScatterMax"));
GVAR_DEF(PrimitivePtr, kPrimScatterMin, std::make_shared<Primitive>("ScatterMin"));
GVAR_DEF(PrimitivePtr, kPrimScatterNdAdd, std::make_shared<Primitive>("ScatterNdAdd"));
GVAR_DEF(PrimitivePtr, kPrimScatterNdSub, std::make_shared<Primitive>("ScatterNdSub"));
GVAR_DEF(PrimitivePtr, kPrimScatterNdMax, std::make_shared<Primitive>("ScatterNdMax"));
GVAR_DEF(PrimitivePtr, kPrimScatterNdMin, std::make_shared<Primitive>("ScatterNdMin"));
GVAR_DEF(PrimitivePtr, kPrimScatterNdMul, std::make_shared<Primitive>("ScatterNdMul"));
GVAR_DEF(PrimitivePtr, kPrimScatterNdDiv, std::make_shared<Primitive>("ScatterNdDiv"));
GVAR_DEF(PrimitivePtr, kPrimScatterUpdate, std::make_shared<Primitive>("ScatterUpdate"));
GVAR_DEF(PrimitivePtr, kPrimScatterElements, std::make_shared<Primitive>("ScatterElements"));
GVAR_DEF(PrimitivePtr, kPrimScatterAddWithAxis, std::make_shared<Primitive>(kScatterAddWithAxis));
GVAR_DEF(PrimitivePtr, kPrimTensorScatterElements, std::make_shared<Primitive>("TensorScatterElements"));
GVAR_DEF(PrimitivePtr, kPrimTensorScatterUpdate, std::make_shared<Primitive>("TensorScatterUpdate"));
GVAR_DEF(PrimitivePtr, kPrimTensorScatterAdd, std::make_shared<Primitive>("TensorScatterAdd"));
GVAR_DEF(PrimitivePtr, kPrimTensorScatterSub, std::make_shared<Primitive>("TensorScatterSub"));
GVAR_DEF(PrimitivePtr, kPrimTensorScatterMul, std::make_shared<Primitive>("TensorScatterMul"));
GVAR_DEF(PrimitivePtr, kPrimTensorScatterDiv, std::make_shared<Primitive>("TensorScatterDiv"));
GVAR_DEF(PrimitivePtr, kPrimTensorScatterMax, std::make_shared<Primitive>("TensorScatterMax"));
GVAR_DEF(PrimitivePtr, kPrimTensorScatterMin, std::make_shared<Primitive>("TensorScatterMin"));
GVAR_DEF(PrimitivePtr, kPrimTensorCopySlices, std::make_shared<Primitive>("TensorCopySlices"));
GVAR_DEF(PrimitivePtr, kPrimMapUniform, std::make_shared<Primitive>("MapUniform"));
GVAR_DEF(PrimitivePtr, kPrimSplit, std::make_shared<Primitive>("Split"));
GVAR_DEF(PrimitivePtr, kPrimSplitD, std::make_shared<Primitive>("SplitD"));
GVAR_DEF(PrimitivePtr, kPrimSplitV, std::make_shared<Primitive>(kSplitV));
GVAR_DEF(PrimitivePtr, kPrimSplitVD, std::make_shared<Primitive>("SplitVD"));
GVAR_DEF(PrimitivePtr, kPrimSequenceMask, std::make_shared<Primitive>("SequenceMask"));
GVAR_DEF(PrimitivePtr, kPrimRange, std::make_shared<Primitive>("Range"));
GVAR_DEF(PrimitivePtr, kPrimRangeV2, std::make_shared<Primitive>("RangeV2"));
GVAR_DEF(PrimitivePtr, kPrimSpaceToBatchND, std::make_shared<Primitive>("SpaceToBatchND"));
GVAR_DEF(PrimitivePtr, kPrimBatchToSpaceND, std::make_shared<Primitive>("BatchToSpaceND"));
GVAR_DEF(PrimitivePtr, kPrimBatchToSpaceNDV2, std::make_shared<Primitive>("BatchToSpaceNDV2"));
GVAR_DEF(PrimitivePtr, kPrimDepthToSpace, std::make_shared<Primitive>("DepthToSpace"));
GVAR_DEF(PrimitivePtr, kPrimBatchToSpace, std::make_shared<Primitive>("BatchToSpace"));
GVAR_DEF(PrimitivePtr, kPrimFakeQuantParam, std::make_shared<Primitive>("FakeQuantParam"));
GVAR_DEF(PrimitivePtr, kPrimSpaceToBatch, std::make_shared<Primitive>("SpaceToBatch"));
GVAR_DEF(PrimitivePtr, kPrimScatterNd, std::make_shared<Primitive>("ScatterNd"));
GVAR_DEF(PrimitivePtr, kPrimScatterNdUpdate, std::make_shared<Primitive>("ScatterNdUpdate"));
GVAR_DEF(PrimitivePtr, kPrimScatterNonAliasingAdd, std::make_shared<Primitive>("ScatterNonAliasingAdd"));
GVAR_DEF(PrimitivePtr, kPrimConstantOfShape, std::make_shared<Primitive>("ConstantOfShape"));
GVAR_DEF(PrimitivePtr, kPrimSquaredDifference, std::make_shared<Primitive>("SquaredDifference"));
GVAR_DEF(PrimitivePtr, kPrimReverseV2, std::make_shared<Primitive>("ReverseV2"));
GVAR_DEF(PrimitivePtr, kPrimReverseSequence, std::make_shared<Primitive>("ReverseSequence"));
GVAR_DEF(PrimitivePtr, kPrimRank, std::make_shared<Primitive>("Rank"));
GVAR_DEF(PrimitivePtr, kPrimSort, std::make_shared<Primitive>("Sort"));
GVAR_DEF(PrimitivePtr, kPrimMaskedFill, std::make_shared<Primitive>("MaskedFill"));
GVAR_DEF(PrimitivePtr, kPrimMaskedScatter, std::make_shared<Primitive>("MaskedScatter"));
GVAR_DEF(PrimitivePtr, kPrimMaskedSelect, std::make_shared<Primitive>("MaskedSelect"));
GVAR_DEF(PrimitivePtr, kPrimMaskedSelectGrad, std::make_shared<Primitive>("MaskedSelectGrad"));
GVAR_DEF(PrimitivePtr, kPrimDiag, std::make_shared<Primitive>(kDiag));
GVAR_DEF(PrimitivePtr, kPrimDiagD, std::make_shared<Primitive>("DiagD"));
GVAR_DEF(PrimitivePtr, kPrimDiagPart, std::make_shared<Primitive>(kDiagPart));
GVAR_DEF(PrimitivePtr, kPrimDiagPartD, std::make_shared<Primitive>("DiagPartD"));
GVAR_DEF(PrimitivePtr, kPrimMatrixDiagV3, std::make_shared<Primitive>(kMatrixDiagV3));
GVAR_DEF(PrimitivePtr, kPrimMatrixDiagPartV3, std::make_shared<Primitive>(kMatrixDiagPartV3));
GVAR_DEF(PrimitivePtr, kPrimMatrixSetDiagV3, std::make_shared<Primitive>(kMatrixSetDiagV3));
GVAR_DEF(PrimitivePtr, kPrimMatrixBandPart, std::make_shared<Primitive>(kMatrixBandPart));
GVAR_DEF(PrimitivePtr, kPrimNonZero, std::make_shared<Primitive>("NonZero"));
GVAR_DEF(PrimitivePtr, kPrimNonZeroWithValue, std::make_shared<Primitive>("NonZeroWithValue"));
GVAR_DEF(PrimitivePtr, kPrimNonZeroWithValueShape, std::make_shared<Primitive>("NonZeroWithValueShape"));
GVAR_DEF(PrimitivePtr, kPrimNoRepeatNGram, std::make_shared<Primitive>("NoRepeatNGram"));
GVAR_DEF(PrimitivePtr, kPrimRealInner, std::make_shared<Primitive>(kRealInner));
GVAR_DEF(PrimitivePtr, kPrimFillV2, std::make_shared<Primitive>(kFillV2));
GVAR_DEF(PrimitivePtr, kPrimFills, std::make_shared<Primitive>(kFills));
GVAR_DEF(PrimitivePtr, kPrimExtractVolumePatches, std::make_shared<Primitive>("ExtractVolumePatches"));
GVAR_DEF(PrimitivePtr, kPrimLstsq, std::make_shared<Primitive>(kLstsq));
GVAR_DEF(PrimitivePtr, kPrimLowerBound, std::make_shared<Primitive>(kLowerBound));
GVAR_DEF(PrimitivePtr, kPrimUpperBound, std::make_shared<Primitive>(kUpperBound));
GVAR_DEF(PrimitivePtr, kPrimCummax, std::make_shared<Primitive>(kCummax));
GVAR_DEF(PrimitivePtr, kPrimMvlgamma, std::make_shared<Primitive>(kMvlgamma));
GVAR_DEF(PrimitivePtr, kPrimMvlgammaGrad, std::make_shared<Primitive>(kMvlgammaGrad));
GVAR_DEF(PrimitivePtr, kPrimRightShift, std::make_shared<Primitive>(kRightShift));
GVAR_DEF(PrimitivePtr, kPrimLogSpace, std::make_shared<Primitive>(kLogSpace));
GVAR_DEF(PrimitivePtr, kPrimTril, std::make_shared<Primitive>(kTril));
GVAR_DEF(PrimitivePtr, kPrimEye, std::make_shared<Primitive>(kEye));
GVAR_DEF(PrimitivePtr, kPrimTriu, std::make_shared<Primitive>(kTriu));
GVAR_DEF(PrimitivePtr, kPrimMeshgrid, std::make_shared<Primitive>(kMeshgrid));
GVAR_DEF(PrimitivePtr, kPrimSegmentMax, std::make_shared<Primitive>(kSegmentMax));
GVAR_DEF(PrimitivePtr, kPrimSegmentMin, std::make_shared<Primitive>(kSegmentMin));
GVAR_DEF(PrimitivePtr, kPrimSegmentSum, std::make_shared<Primitive>(kSegmentSum));
GVAR_DEF(PrimitivePtr, kPrimAffineGrid, std::make_shared<Primitive>(kAffineGrid));
GVAR_DEF(PrimitivePtr, kPrimAffineGridGrad, std::make_shared<Primitive>(kAffineGridGrad));
GVAR_DEF(PrimitivePtr, kPrimSegmentMean, std::make_shared<Primitive>(kSegmentMean));
GVAR_DEF(PrimitivePtr, kPrimSegmentProd, std::make_shared<Primitive>(kSegmentProd));
GVAR_DEF(PrimitivePtr, kPrimBincount, std::make_shared<Primitive>(kBincount));
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_ARRAY_OPS_H_
