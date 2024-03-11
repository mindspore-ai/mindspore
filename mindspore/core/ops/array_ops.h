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
#include "ops/array_op_name.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "utils/hash_map.h"
#include "ops/auto_generate/gen_ops_primitive.h"

namespace mindspore {
namespace prim {
// Arrays
GVAR_DEF(PrimitivePtr, kPrimExpand, std::make_shared<Primitive>("Expand"));
GVAR_DEF(PrimitivePtr, kPrimMakeRange, std::make_shared<Primitive>("make_range"));
GVAR_DEF(PrimitivePtr, kPrimBroadcast, std::make_shared<Primitive>("Broadcast"));
GVAR_DEF(PrimitivePtr, kPrimZeros, std::make_shared<Primitive>("Zeros"));
GVAR_DEF(PrimitivePtr, kPrimOnes, std::make_shared<Primitive>(kOnesOpName));
GVAR_DEF(PrimitivePtr, kPrimFill, std::make_shared<Primitive>("Fill"));
GVAR_DEF(PrimitivePtr, kPrimLeftShift, std::make_shared<Primitive>(kLeftShiftOpName));
GVAR_DEF(PrimitivePtr, kPrimFillDiagonal, std::make_shared<Primitive>(kFillDiagonalOpName));
GVAR_DEF(PrimitivePtr, kPrimUnravelIndex, std::make_shared<Primitive>(kUnravelIndexOpName));
GVAR_DEF(PrimitivePtr, kPrimDynamicBroadcastTo, std::make_shared<Primitive>(kDynamicBroadcastToOpName));
GVAR_DEF(PrimitivePtr, kPrimScalarToArray, std::make_shared<Primitive>("scalar_to_array"));
GVAR_DEF(PrimitivePtr, kPrimLogNormalReverse, std::make_shared<Primitive>("LogNormalReverse"));
GVAR_DEF(PrimitivePtr, kPrimTopK, std::make_shared<Primitive>(kTopKOpName));
GVAR_DEF(PrimitivePtr, kPrimInTopK, std::make_shared<Primitive>("InTopK"));
GVAR_DEF(PrimitivePtr, kPrimInTopKD, std::make_shared<Primitive>("InTopKD"));
GVAR_DEF(PrimitivePtr, kPrimArrayToScalar, std::make_shared<Primitive>("array_to_scalar"));
GVAR_DEF(PrimitivePtr, kPrimBroadcastShape, std::make_shared<Primitive>("broadcast_shape"));
GVAR_DEF(PrimitivePtr, kPrimArrayMap, std::make_shared<Primitive>("array_map"));
GVAR_DEF(PrimitivePtr, kPrimArrayReduce, std::make_shared<Primitive>("array_reduce"));
GVAR_DEF(PrimitivePtr, kPrimConcatD, std::make_shared<Primitive>("ConcatD"));
GVAR_DEF(PrimitivePtr, kPrimParallelConcat, std::make_shared<Primitive>(kParallelConcatOpName));
GVAR_DEF(PrimitivePtr, kPrimCountNonZero, std::make_shared<Primitive>("CountNonZero"));
GVAR_DEF(PrimitivePtr, kPrimFlattenConcat, std::make_shared<Primitive>(kFlattenConcatOpName));
GVAR_DEF(PrimitivePtr, kPrimSqueeze, std::make_shared<Primitive>("Squeeze"));
GVAR_DEF(PrimitivePtr, kPrimSqueezeV3, std::make_shared<Primitive>("SqueezeV3"));
GVAR_DEF(PrimitivePtr, kPrimUnsqueeze, std::make_shared<Primitive>("Unsqueeze"));
GVAR_DEF(PrimitivePtr, kPrimConjugateTranspose, std::make_shared<Primitive>(kConjugateTransposeOpName));
GVAR_DEF(PrimitivePtr, kPrimTransposeD, std::make_shared<Primitive>("TransposeD"));
GVAR_DEF(PrimitivePtr, kPrimSelectView, std::make_shared<Primitive>("SelectView"));
GVAR_DEF(PrimitivePtr, kPrimSparseGatherV2, std::make_shared<Primitive>("SparseGatherV2"));
GVAR_DEF(PrimitivePtr, kPrimCoalesce, std::make_shared<Primitive>(kCoalesceOpName));
GVAR_DEF(PrimitivePtr, kPrimShapeCalc, std::make_shared<Primitive>("ShapeCalc"));
GVAR_DEF(PrimitivePtr, kPrimStridedRead, std::make_shared<Primitive>("StridedRead"));
GVAR_DEF(PrimitivePtr, kPrimStridedWrite, std::make_shared<Primitive>("StridedWrite"));
GVAR_DEF(PrimitivePtr, kPrimStridedSliceGrad, std::make_shared<Primitive>(kStridedSliceGradOpName));
GVAR_DEF(PrimitivePtr, kPrimDynamicShape, std::make_shared<Primitive>(kDynamicShapeOpName));
GVAR_DEF(PrimitivePtr, kPrimCheckNumerics, std::make_shared<Primitive>(kCheckNumericsOpName));
GVAR_DEF(PrimitivePtr, kPrimSize, std::make_shared<Primitive>("Size"));
GVAR_DEF(PrimitivePtr, kPrimArgMax, std::make_shared<Primitive>("Argmax"));
GVAR_DEF(PrimitivePtr, kPrimArgMin, std::make_shared<Primitive>("ArgMin"));
GVAR_DEF(PrimitivePtr, kPrimArgminV2, std::make_shared<Primitive>("ArgminV2"));
GVAR_DEF(PrimitivePtr, kPrimPack, std::make_shared<Primitive>("Pack"));
GVAR_DEF(PrimitivePtr, kPrimStack, std::make_shared<Primitive>(kStackOpName));
GVAR_DEF(PrimitivePtr, kPrimUnpack, std::make_shared<Primitive>("Unpack"));
GVAR_DEF(PrimitivePtr, kPrimUnstack, std::make_shared<Primitive>(kUnstackOpName));
GVAR_DEF(PrimitivePtr, kPrimUnsortedSegmentMax, std::make_shared<Primitive>("UnsortedSegmentMax"));
GVAR_DEF(PrimitivePtr, kPrimUnsortedSegmentProd, std::make_shared<Primitive>(kUnsortedSegmentProdOpName));
GVAR_DEF(PrimitivePtr, kPrimUnsortedSegmentSumD, std::make_shared<Primitive>(kUnsortedSegmentSumDOpName));
GVAR_DEF(PrimitivePtr, kPrimUnsortedSegmentMin, std::make_shared<Primitive>("UnsortedSegmentMin"));
GVAR_DEF(PrimitivePtr, kPrimConcatOffset, std::make_shared<Primitive>("ConcatOffset"));
GVAR_DEF(PrimitivePtr, kPrimConcatOffsetV1, std::make_shared<Primitive>("ConcatOffsetV1"));
GVAR_DEF(PrimitivePtr, kPrimIdentityN, std::make_shared<Primitive>("IdentityN"));
GVAR_DEF(PrimitivePtr, kPrimSubAndFilter, std::make_shared<Primitive>("SubAndFilter"));
GVAR_DEF(PrimitivePtr, kPrimMapCacheIdx, std::make_shared<Primitive>("MapCacheIdx"));
GVAR_DEF(PrimitivePtr, kPrimUpdateCache, std::make_shared<Primitive>("UpdateCache"));
GVAR_DEF(PrimitivePtr, kPrimComputeAccidentalHits, std::make_shared<Primitive>("ComputeAccidentalHits"));
GVAR_DEF(PrimitivePtr, kPrimCacheSwapTable, std::make_shared<Primitive>("CacheSwapTable"));
GVAR_DEF(PrimitivePtr, kPrimPadAndShift, std::make_shared<Primitive>("PadAndShift"));
GVAR_DEF(PrimitivePtr, kPrimSlice, std::make_shared<Primitive>(kSliceOpName));
GVAR_DEF(PrimitivePtr, kPrimSliceGrad, std::make_shared<Primitive>("SliceGrad"));
GVAR_DEF(PrimitivePtr, kPrimSliceFusion, std::make_shared<Primitive>("SliceFusion"));
GVAR_DEF(PrimitivePtr, kPrimTileD, std::make_shared<Primitive>("TileD"));
GVAR_DEF(PrimitivePtr, kPrimAccumulateNV2, std::make_shared<Primitive>("AccumulateNV2"));
GVAR_DEF(PrimitivePtr, kPrimTransData, std::make_shared<Primitive>("TransData"));
GVAR_DEF(PrimitivePtr, kPrimTransDataRNN, std::make_shared<Primitive>("TransDataRNN"));
GVAR_DEF(PrimitivePtr, kPrimPad, std::make_shared<Primitive>("Pad"));
// GVAR_DEF(PrimitivePtr, kPrimPadV3, std::make_shared<Primitive>("PadV3"));
GVAR_DEF(PrimitivePtr, kPrimPadD, std::make_shared<Primitive>("PadD"));
GVAR_DEF(PrimitivePtr, kPrimPadding, std::make_shared<Primitive>(kPaddingOpName));
GVAR_DEF(PrimitivePtr, kPrimMirrorPad, std::make_shared<Primitive>(kMirrorPadOpName));
GVAR_DEF(PrimitivePtr, kPrimUnique, std::make_shared<Primitive>("Unique"));
GVAR_DEF(PrimitivePtr, kPrimUniqueWithPad, std::make_shared<Primitive>("UniqueWithPad"));
GVAR_DEF(PrimitivePtr, kPrimUniqueGrad, std::make_shared<Primitive>("UniqueGrad"));
GVAR_DEF(PrimitivePtr, kPrimUniqueConsecutive, std::make_shared<Primitive>("UniqueConsecutive"));
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
GVAR_DEF(PrimitivePtr, kPrimScatterAddWithAxis, std::make_shared<Primitive>(kScatterAddWithAxisOpName));
GVAR_DEF(PrimitivePtr, kPrimTensorScatterElements, std::make_shared<Primitive>("TensorScatterElements"));
GVAR_DEF(PrimitivePtr, kPrimTensorScatterUpdate, std::make_shared<Primitive>("TensorScatterUpdate"));
GVAR_DEF(PrimitivePtr, kPrimTensorScatterAdd, std::make_shared<Primitive>("TensorScatterAdd"));
GVAR_DEF(PrimitivePtr, kPrimTensorScatterSub, std::make_shared<Primitive>("TensorScatterSub"));
GVAR_DEF(PrimitivePtr, kPrimTensorScatterMul, std::make_shared<Primitive>("TensorScatterMul"));
GVAR_DEF(PrimitivePtr, kPrimTensorScatterDiv, std::make_shared<Primitive>("TensorScatterDiv"));
GVAR_DEF(PrimitivePtr, kPrimTensorScatterMax, std::make_shared<Primitive>("TensorScatterMax"));
GVAR_DEF(PrimitivePtr, kPrimTensorScatterMin, std::make_shared<Primitive>("TensorScatterMin"));
GVAR_DEF(PrimitivePtr, kPrimMapUniform, std::make_shared<Primitive>("MapUniform"));
GVAR_DEF(PrimitivePtr, kPrimSplitD, std::make_shared<Primitive>("SplitD"));
GVAR_DEF(PrimitivePtr, kPrimSplitV, std::make_shared<Primitive>(kSplitVOpName));
GVAR_DEF(PrimitivePtr, kPrimSplitVD, std::make_shared<Primitive>("SplitVD"));
GVAR_DEF(PrimitivePtr, kPrimSequenceMask, std::make_shared<Primitive>("SequenceMask"));
GVAR_DEF(PrimitivePtr, kPrimRangeV2, std::make_shared<Primitive>("RangeV2"));
GVAR_DEF(PrimitivePtr, kPrimSpaceToBatchND, std::make_shared<Primitive>("SpaceToBatchND"));
GVAR_DEF(PrimitivePtr, kPrimBatchToSpaceND, std::make_shared<Primitive>("BatchToSpaceND"));
GVAR_DEF(PrimitivePtr, kPrimBatchToSpaceNDV2, std::make_shared<Primitive>("BatchToSpaceNDV2"));
GVAR_DEF(PrimitivePtr, kPrimDepthToSpace, std::make_shared<Primitive>("DepthToSpace"));
GVAR_DEF(PrimitivePtr, kPrimBatchToSpace, std::make_shared<Primitive>("BatchToSpace"));
GVAR_DEF(PrimitivePtr, kPrimFakeQuantParam, std::make_shared<Primitive>("FakeQuantParam"));
GVAR_DEF(PrimitivePtr, kPrimSpaceToBatch, std::make_shared<Primitive>("SpaceToBatch"));
GVAR_DEF(PrimitivePtr, kPrimScatterNdUpdate, std::make_shared<Primitive>("ScatterNdUpdate"));
GVAR_DEF(PrimitivePtr, kPrimScatterNonAliasingAdd, std::make_shared<Primitive>("ScatterNonAliasingAdd"));
GVAR_DEF(PrimitivePtr, kPrimConstantOfShape, std::make_shared<Primitive>("ConstantOfShape"));
GVAR_DEF(PrimitivePtr, kPrimSquaredDifference, std::make_shared<Primitive>("SquaredDifference"));
GVAR_DEF(PrimitivePtr, kPrimReverseSequence, std::make_shared<Primitive>("ReverseSequence"));
GVAR_DEF(PrimitivePtr, kPrimSort, std::make_shared<Primitive>("Sort"));
GVAR_DEF(PrimitivePtr, kPrimMaskedScatter, std::make_shared<Primitive>("MaskedScatter"));
GVAR_DEF(PrimitivePtr, kPrimMaskedSelect, std::make_shared<Primitive>("MaskedSelect"));
GVAR_DEF(PrimitivePtr, kPrimMaskedSelectGrad, std::make_shared<Primitive>("MaskedSelectGrad"));
GVAR_DEF(PrimitivePtr, kPrimDiagD, std::make_shared<Primitive>("DiagD"));
GVAR_DEF(PrimitivePtr, kPrimDiagPart, std::make_shared<Primitive>(kDiagPartOpName));
GVAR_DEF(PrimitivePtr, kPrimDiagPartD, std::make_shared<Primitive>("DiagPartD"));
GVAR_DEF(PrimitivePtr, kPrimMatrixDiagV3, std::make_shared<Primitive>(kMatrixDiagV3OpName));
GVAR_DEF(PrimitivePtr, kPrimMatrixDiagPartV3, std::make_shared<Primitive>(kMatrixDiagPartV3OpName));
GVAR_DEF(PrimitivePtr, kPrimMatrixSetDiagV3, std::make_shared<Primitive>(kMatrixSetDiagV3OpName));
GVAR_DEF(PrimitivePtr, kPrimMatrixBandPart, std::make_shared<Primitive>(kMatrixBandPartOpName));
GVAR_DEF(PrimitivePtr, kPrimNonZeroWithValue, std::make_shared<Primitive>("NonZeroWithValue"));
GVAR_DEF(PrimitivePtr, kPrimNonZeroWithValueShape, std::make_shared<Primitive>("NonZeroWithValueShape"));
GVAR_DEF(PrimitivePtr, kPrimNoRepeatNGram, std::make_shared<Primitive>("NoRepeatNGram"));
GVAR_DEF(PrimitivePtr, kPrimGenerateEodMask, std::make_shared<Primitive>("GenerateEodMask"));
GVAR_DEF(PrimitivePtr, kPrimRealInner, std::make_shared<Primitive>(kRealInnerOpName));
GVAR_DEF(PrimitivePtr, kPrimFillV2, std::make_shared<Primitive>(kFillV2OpName));
GVAR_DEF(PrimitivePtr, kPrimFills, std::make_shared<Primitive>(kFillsOpName));
GVAR_DEF(PrimitivePtr, kPrimExtractVolumePatches, std::make_shared<Primitive>("ExtractVolumePatches"));
GVAR_DEF(PrimitivePtr, kPrimLstsq, std::make_shared<Primitive>(kLstsqOpName));
GVAR_DEF(PrimitivePtr, kPrimLowerBound, std::make_shared<Primitive>(kLowerBoundOpName));
GVAR_DEF(PrimitivePtr, kPrimUpperBound, std::make_shared<Primitive>(kUpperBoundOpName));
GVAR_DEF(PrimitivePtr, kPrimMvlgamma, std::make_shared<Primitive>(kMvlgammaOpName));
GVAR_DEF(PrimitivePtr, kPrimMvlgammaGrad, std::make_shared<Primitive>(kMvlgammaGradOpName));
GVAR_DEF(PrimitivePtr, kPrimLogSpace, std::make_shared<Primitive>(kLogSpaceOpName));
GVAR_DEF(PrimitivePtr, kPrimTril, std::make_shared<Primitive>(kTrilOpName));
GVAR_DEF(PrimitivePtr, kPrimTriu, std::make_shared<Primitive>(kTriuOpName));
GVAR_DEF(PrimitivePtr, kPrimMeshgrid, std::make_shared<Primitive>(kMeshgridOpName));
GVAR_DEF(PrimitivePtr, kPrimSegmentMax, std::make_shared<Primitive>(kSegmentMaxOpName));
GVAR_DEF(PrimitivePtr, kPrimSegmentMin, std::make_shared<Primitive>(kSegmentMinOpName));
GVAR_DEF(PrimitivePtr, kPrimSegmentSum, std::make_shared<Primitive>(kSegmentSumOpName));
GVAR_DEF(PrimitivePtr, kPrimAffineGrid, std::make_shared<Primitive>(kAffineGridOpName));
GVAR_DEF(PrimitivePtr, kPrimAffineGridGrad, std::make_shared<Primitive>(kAffineGridGradOpName));
GVAR_DEF(PrimitivePtr, kPrimSegmentMean, std::make_shared<Primitive>(kSegmentMeanOpName));
GVAR_DEF(PrimitivePtr, kPrimSegmentProd, std::make_shared<Primitive>(kSegmentProdOpName));
GVAR_DEF(PrimitivePtr, kPrimBincount, std::make_shared<Primitive>(kBincountOpName));
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_ARRAY_OPS_H_
