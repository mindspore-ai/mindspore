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

#ifndef MINDSPORE_CORE_BASE_ARRAY_OP_NAME_H_
#define MINDSPORE_CORE_BASE_ARRAY_OP_NAME_H_

namespace mindspore {
// Arrays
constexpr auto kTopKOpName = "TopK";
constexpr auto kLeftShiftOpName = "LeftShift";
constexpr auto kCountNonZeroOpName = "CountNonZero";
constexpr auto kFillDiagonalOpName = "FillDiagonal";
constexpr auto kSegmentMaxOpName = "SegmentMax";
constexpr auto kSegmentSumOpName = "SegmentSum";
constexpr auto kSegmentMinOpName = "SegmentMin";
constexpr auto kSegmentMeanOpName = "SegmentMean";
constexpr auto kSegmentProdOpName = "SegmentProd";
constexpr auto kShapeOpName = "Shape";
constexpr auto kDynamicShapeOpName = "DynamicShape";
constexpr auto kTensorShapeOpName = "TensorShape";
constexpr auto kCheckNumericsOpName = "CheckNumerics";
constexpr auto kStackOpName = "Stack";
constexpr auto kPackOpName = "Pack";
constexpr auto kLogNormalReverseOpName = "LogNormalReverse";
constexpr auto kUnstackOpName = "Unstack";
constexpr auto kUnpackOpName = "Unpack";
constexpr auto kIdentityOpName = "Identity";
constexpr auto kUnravelIndexOpName = "UnravelIndex";
constexpr auto kDynamicBroadcastToOpName = "DynamicBroadcastTo";
constexpr auto kConcatOpName = "Concat";
constexpr auto kParallelConcatOpName = "ParallelConcat";
constexpr auto kFlattenConcatOpName = "FlattenConcat";
constexpr auto kConjugateTransposeOpName = "ConjugateTranspose";
constexpr auto kTransposeOpName = "Transpose";
constexpr auto kGatherDGradOpName = "GatherDGrad";
constexpr auto kGatherDGradV2OpName = "GatherDGradV2";
constexpr auto kCoalesceOpName = "Coalesce";
constexpr auto kStridedSliceOpName = "StridedSlice";
constexpr auto kStridedSliceGradOpName = "StridedSliceGrad";
constexpr auto kUnsortedSegmentSumOpName = "UnsortedSegmentSum";
constexpr auto kUnsortedSegmentSumDOpName = "UnsortedSegmentSumD";
constexpr auto kUnsortedSegmentProdOpName = "UnsortedSegmentProd";
constexpr auto kTileOpName = "Tile";
constexpr auto kPaddingOpName = "Padding";
constexpr auto kMirrorPadOpName = "MirrorPad";
constexpr auto kScatterAddWithAxisOpName = "ScatterAddWithAxis";
constexpr auto kDiagOpName = "Diag";
constexpr auto kDiagPartOpName = "DiagPart";
constexpr auto kMatrixDiagV3OpName = "MatrixDiagV3";
constexpr auto kMatrixDiagPartV3OpName = "MatrixDiagPartV3";
constexpr auto kMatrixSetDiagV3OpName = "MatrixSetDiagV3";
constexpr auto kMatrixBandPartOpName = "MatrixBandPart";
constexpr auto kRealInnerOpName = "RealInner";
constexpr auto kSplitVOpName = "SplitV";
constexpr auto kFillV2OpName = "FillV2";
constexpr auto kFillsOpName = "Fills";
constexpr auto kLstsqOpName = "Lstsq";
constexpr auto kLowerBoundOpName = "LowerBound";
constexpr auto kUpperBoundOpName = "UpperBound";
constexpr auto kCummaxOpName = "Cummax";
constexpr auto kMvlgammaOpName = "Mvlgamma";
constexpr auto kMvlgammaGradOpName = "MvlgammaGrad";
constexpr auto kRightShiftOpName = "RightShift";
constexpr auto kLogSpaceOpName = "LogSpace";
constexpr auto kTrilOpName = "Tril";
constexpr auto kEyeOpName = "Eye";
constexpr auto kTriuOpName = "Triu";
constexpr auto kMeshgridOpName = "Meshgrid";
constexpr auto kAffineGridOpName = "AffineGrid";
constexpr auto kAffineGridGradOpName = "AffineGridGrad";
constexpr auto kBroadcastToOpName = "BroadcastTo";
constexpr auto kBincountOpName = "Bincount";
constexpr auto kReshapeOpName = "Reshape";
constexpr auto kNonZeroOpName = "NonZero";
constexpr auto kScatterNdMaxOpName = "ScatterNdMax";
constexpr auto kScatterNdMinOpName = "ScatterNdMin";
constexpr auto kSliceOpName = "Slice";
constexpr auto kZerosLikeOpName = "ZerosLike";
constexpr auto kOnesOpName = "Ones";
constexpr auto kOnesLikeOpName = "OnesLike";
constexpr auto kAccumulateNV2OpName = "AccumulateNV2";
constexpr auto kAddNOpName = "AddN";
constexpr auto kArgmaxOpName = "Argmax";
constexpr auto kArgMinDOpName = "ArgMinD";
constexpr auto kArgminOpName = "Argmin";
constexpr auto kArgMinOpName = "ArgMin";
constexpr auto kArgminV2OpName = "ArgminV2";
constexpr auto kArgMinWithValueOpName = "ArgMinWithValue";
constexpr auto kArgMaxWithValueOpName = "ArgMaxWithValue";
constexpr auto kBatchToSpaceNDDOpName = "BatchToSpaceNDD";
constexpr auto kBatchToSpaceNDOpName = "BatchToSpaceND";
constexpr auto kBatchToSpaceNDV2OpName = "BatchToSpaceNDV2";
constexpr auto kBatchToSpaceOpName = "BatchToSpace";
constexpr auto kBatchToSpaceDOpName = "BatchToSpaceD";
constexpr auto kBroadcastToDOpName = "BroadcastToD";
constexpr auto kCacheSwapTableOpName = "CacheSwapTable";
constexpr auto kCastOpName = "Cast";
constexpr auto kComputeAccidentalHitsOpName = "ComputeAccidentalHits";
constexpr auto kConcatDOpName = "ConcatD";
constexpr auto kConcatOffsetOpName = "ConcatOffset";
constexpr auto kDepthToSpaceOpName = "DepthToSpace";
constexpr auto kDiagPartDOpName = "DiagPartD";
constexpr auto kDiagDOpName = "DiagD";
constexpr auto kDynamicGRUV2OpName = "DynamicGRUV2";
constexpr auto kDynamicRNNOpName = "DynamicRNN";
constexpr auto kExpandOpName = "Expand";
constexpr auto kExpandDOpName = "ExpandD";
constexpr auto kExpandDimsOpName = "ExpandDims";
constexpr auto kExtractImagePatchesOpName = "ExtractImagePatches";
constexpr auto kFillOpName = "Fill";
constexpr auto kFillDOpName = "FillD";
constexpr auto kGatherDOpName = "GatherD";
constexpr auto kGatherNdOpName = "GatherNd";
constexpr auto kGatherV2OpName = "GatherV2";
constexpr auto kGatherV2DOpName = "GatherV2D";
constexpr auto kIdentityNOpName = "IdentityN";
constexpr auto kInTopKOpName = "InTopK";
constexpr auto kInTopKDOpName = "InTopKD";
constexpr auto kMaskedFillOpName = "MaskedFill";
constexpr auto kMaskedSelectOpName = "MaskedSelect";
constexpr auto kMaskedSelectGradOpName = "MaskedSelectGrad";
constexpr auto kMaskedScatterOpName = "MaskedScatter";
constexpr auto kPadAndShiftOpName = "PadAndShift";
constexpr auto kPadOpName = "Pad";
constexpr auto kPadDOpName = "PadD";
constexpr auto kShapeCalcOpName = "ShapeCalc";
constexpr auto kRangeOpName = "Range";
constexpr auto kRangeDOpName = "RangeD";
constexpr auto kReverseV2DOpName = "ReverseV2D";
constexpr auto kScatterAddOpName = "ScatterAdd";
constexpr auto kScatterNdOpName = "ScatterNd";
constexpr auto kScatterNdDOpName = "ScatterNdD";
constexpr auto kScatterNdUpdateOpName = "ScatterNdUpdate";
constexpr auto kScatterUpdateOpName = "ScatterUpdate";
constexpr auto kSliceGradOpName = "SliceGrad";
constexpr auto kSortOpName = "Sort";
constexpr auto kSpaceToBatchOpName = "SpaceToBatch";
constexpr auto kSpaceToBatchDOpName = "SpaceToBatchD";
constexpr auto kSpaceToBatchNDOpName = "SpaceToBatchND";
constexpr auto kSpaceToBatchNDDOpName = "SpaceToBatchNDD";
constexpr auto kSparseGatherV2OpName = "SparseGatherV2";
constexpr auto kSplitOpName = "Split";
constexpr auto kSplitDOpName = "SplitD";
constexpr auto kSplitVDOpName = "SplitVD";
constexpr auto kSqueezeOpName = "Squeeze";
constexpr auto kStridedReadOpName = "StridedRead";
constexpr auto kStridedWriteOpName = "StridedWrite";
constexpr auto kSubAndFilterOpName = "SubAndFilter";
constexpr auto kTensorCopySlicesOpName = "TensorCopySlices";
constexpr auto kTensorScatterUpdateOpName = "TensorScatterUpdate";
constexpr auto kTileDOpName = "TileD";
constexpr auto kTransDataOpName = "TransData";
constexpr auto kTransDataRNNOpName = "TransDataRNN";
constexpr auto kTransposeDOpName = "TransposeD";
constexpr auto kUniqueConsecutiveOpName = "UniqueConsecutive";
constexpr auto kUniqueOpName = "Unique";
constexpr auto kUniqueWithPadOpName = "UniqueWithPad";
constexpr auto kUnsortedSegmentMaxOpName = "UnsortedSegmentMax";
constexpr auto kUnsortedSegmentMaxDOpName = "UnsortedSegmentMaxD";
constexpr auto kUnsortedSegmentMinOpName = "UnsortedSegmentMin";
constexpr auto kUnsortedSegmentMinDOpName = "UnsortedSegmentMinD";
constexpr auto kUpdateCacheOpName = "UpdateCache";
constexpr auto kBroadcastOpName = "Broadcast";
constexpr auto kCopyWithScileOpName = "CopyWithSlice";
}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_ARRAY_OP_NAME_H_
