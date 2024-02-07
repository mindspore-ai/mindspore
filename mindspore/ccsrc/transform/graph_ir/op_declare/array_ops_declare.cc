/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "transform/graph_ir/op_declare/array_ops_declare.h"
#include <string>
#include <vector>
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/other_ops.h"
#include "mindspore/core/ops/structure_ops.h"

namespace mindspore::transform {
// const
INPUT_MAP(Const) = EMPTY_INPUT_MAP;
ATTR_MAP(Const) = {{"value", ATTR_DESC(value, AnyTraits<ValueAny>())}};
OUTPUT_MAP(Const) = {{0, OUTPUT_DESC(y)}};

// Constant
INPUT_MAP(Constant) = EMPTY_INPUT_MAP;
ATTR_MAP(Constant) = {{"value", ATTR_DESC(value, AnyTraits<ValueAny>())}};
OUTPUT_MAP(Constant) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Constant, kNameConst, ADPT_DESC(Constant, Const))

// ScalarSummary
INPUT_MAP(Summary) = {{2, INPUT_DESC(x)}};
ATTR_MAP(Summary) = EMPTY_ATTR_MAP;
REG_ADPT_DESC(Debug, prim::kPrimDebug->name(), ADPT_DESC(Summary))

// OutfeedEnqueueOpV2
DYN_INPUT_MAP(OutfeedEnqueueOpV2) = {{2, DYN_INPUT_DESC(x)}};
INPUT_MAP(OutfeedEnqueueOpV2) = {{1, INPUT_DESC(tensor_name)}};
ATTR_MAP(OutfeedEnqueueOpV2) = {{"channel_name", ATTR_DESC(channel_name, AnyTraits<std::string>())}};
OUTPUT_MAP(OutfeedEnqueueOpV2) = EMPTY_OUTPUT_MAP;
REG_ADPT_DESC(TensorSummary, "TensorSummary", ADPT_DESC(OutfeedEnqueueOpV2))
REG_ADPT_DESC(ScalarSummary, "ScalarSummary", ADPT_DESC(OutfeedEnqueueOpV2))
REG_ADPT_DESC(ImageSummary, "ImageSummary", ADPT_DESC(OutfeedEnqueueOpV2))
REG_ADPT_DESC(HistogramSummary, "HistogramSummary", ADPT_DESC(OutfeedEnqueueOpV2))
REG_ADPT_DESC(TensorDump, kNameTensorDump, ADPT_DESC(OutfeedEnqueueOpV2))
REG_ADPT_DESC(Print, kNamePrint, ADPT_DESC(OutfeedEnqueueOpV2))

// Data
INPUT_MAP(Data) = EMPTY_INPUT_MAP;
ATTR_MAP(Data) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Data) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Data, kNameParam, ADPT_DESC(Data))

// Shape
INPUT_MAP(Shape) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Shape) = {{"dtype", ATTR_DESC(dtype, AnyTraits<GEType>(), AnyTraits<int64_t>())}};
OUTPUT_MAP(Shape) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Shape, kNameShape, ADPT_DESC(Shape))

// TensorShape
REG_ADPT_DESC(TensorShape, kNameTensorShape, ADPT_DESC(Shape))

// DynamicShape
REG_ADPT_DESC(DynamicShape, kNameDynamicShape, ADPT_DESC(Shape))

// GetShape
INPUT_MAP(GetShape) = EMPTY_INPUT_MAP;
DYN_INPUT_MAP(GetShape) = {{1, DYN_INPUT_DESC(x)}};
ATTR_MAP(GetShape) = EMPTY_ATTR_MAP;
OUTPUT_MAP(GetShape) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(GetShape, kNameGetShape, ADPT_DESC(GetShape));

// Reshape
INPUT_MAP(Reshape) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(shape)}};
ATTR_MAP(Reshape) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Reshape) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Reshape, kNameReshape, ADPT_DESC(Reshape))
REG_ADPT_DESC(FlattenGrad, kNameFlattenGrad, ADPT_DESC(Reshape))

// TransShape
INPUT_MAP(TransShape) = {{1, INPUT_DESC(x)}};
INPUT_ATTR_MAP(TransShape) = {{2, ATTR_DESC(outShape, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())}};
ATTR_MAP(TransShape) = EMPTY_ATTR_MAP;
OUTPUT_MAP(TransShape) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(TransShape, kNameTransShape, ADPT_DESC(TransShape))

// MirrorPad
INPUT_MAP(MirrorPad) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(paddings)}};
ATTR_MAP(MirrorPad) = {{"mode", ATTR_DESC(mode, AnyTraits<std::string>())}};
OUTPUT_MAP(MirrorPad) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MirrorPad, kNameMirrorPad, ADPT_DESC(MirrorPad))

// MirrorPadGrad
INPUT_MAP(MirrorPadGrad) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(paddings)}};
ATTR_MAP(MirrorPadGrad) = {{"mode", ATTR_DESC(mode, AnyTraits<std::string>())}};
OUTPUT_MAP(MirrorPadGrad) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MirrorPadGrad, kNameMirrorPadGrad, ADPT_DESC(MirrorPadGrad))

// Expand
INPUT_MAP(Expand) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(shape)}};
ATTR_MAP(Expand) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Expand) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Expand, "Expand", ADPT_DESC(Expand))

// ExpandDims
INPUT_MAP(ExpandDims) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(axis)}};
ATTR_INPUT_MAP(ExpandDims) = {{"axis", "axis"}};
ATTR_MAP(ExpandDims) = EMPTY_ATTR_MAP;
OUTPUT_MAP(ExpandDims) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ExpandDims, kNameExpandDims, ADPT_DESC(ExpandDims))

// Squeeze
INPUT_MAP(Squeeze) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Squeeze) = {{"axis", ATTR_DESC(axis, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())}};
OUTPUT_MAP(Squeeze) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Squeeze, prim::kPrimSqueeze->name(), ADPT_DESC(Squeeze))

INPUT_MAP(SqueezeV3) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(axes)}};
ATTR_MAP(SqueezeV3) = EMPTY_ATTR_MAP;
OUTPUT_MAP(SqueezeV3) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(SqueezeV3, prim::kPrimSqueezeV3->name(), ADPT_DESC(SqueezeV3))

// ReverseSequence
INPUT_MAP(ReverseSequence) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(seq_lengths)}};
ATTR_MAP(ReverseSequence) = {{"seq_dim", ATTR_DESC(seq_dim, AnyTraits<int64_t>())},
                             {"batch_dim", ATTR_DESC(batch_dim, AnyTraits<int64_t>())}};
OUTPUT_MAP(ReverseSequence) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ReverseSequence, kNameReverseSequence, ADPT_DESC(ReverseSequence))

// EditDistance
INPUT_MAP(EditDistance) = {{1, INPUT_DESC(hypothesis_indices)}, {2, INPUT_DESC(hypothesis_values)},
                           {3, INPUT_DESC(hypothesis_shape)},   {4, INPUT_DESC(truth_indices)},
                           {5, INPUT_DESC(truth_values)},       {6, INPUT_DESC(truth_shape)}};
ATTR_MAP(EditDistance) = {{"normalize", ATTR_DESC(normalize, AnyTraits<bool>())}};
OUTPUT_MAP(EditDistance) = {{0, OUTPUT_DESC(output)}};
REG_ADPT_DESC(EditDistance, kNameEditDistance, ADPT_DESC(EditDistance))

// NonZeroWithValue
INPUT_MAP(NonZeroWithValue) = {{1, INPUT_DESC(x)}};
ATTR_MAP(NonZeroWithValue) = {{"transpose", ATTR_DESC(transpose, AnyTraits<bool>())}};
OUTPUT_MAP(NonZeroWithValue) = {{0, OUTPUT_DESC(value)}, {1, OUTPUT_DESC(index)}, {2, OUTPUT_DESC(count)}};
REG_ADPT_DESC(NonZeroWithValue, kNameNonZeroWithValue, ADPT_DESC(NonZeroWithValue))

// NonZeroWithValueShape
INPUT_MAP(NonZeroWithValueShape) = {{1, INPUT_DESC(value)}, {2, INPUT_DESC(index)}, {3, INPUT_DESC(count)}};
ATTR_MAP(NonZeroWithValueShape) = EMPTY_ATTR_MAP;
OUTPUT_MAP(NonZeroWithValueShape) = {{0, OUTPUT_DESC(out_value)}, {1, OUTPUT_DESC(out_index)}};
REG_ADPT_DESC(NonZeroWithValueShape, kNameNonZeroWithValueShape, ADPT_DESC(NonZeroWithValueShape))

// Unsqueeze
INPUT_MAP(Unsqueeze) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Unsqueeze) = {{"axis", ATTR_DESC(axes, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())}};
OUTPUT_MAP(Unsqueeze) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Unsqueeze, kNameUnsqueeze, ADPT_DESC(Unsqueeze))

// Identity
INPUT_MAP(Identity) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Identity) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Identity) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(IdentityLoad, kNameLoad, ADPT_DESC(Identity))
REG_ADPT_DESC(IdentityListGetItem, kNameListGetItem, ADPT_DESC(Identity))
REG_ADPT_DESC(IdentityIdentity, kNameIdentity, ADPT_DESC(Identity))

// IdentityN
INPUT_MAP(IdentityN) = EMPTY_INPUT_MAP;
DYN_INPUT_MAP(IdentityN) = {{1, DYN_INPUT_DESC(x)}};
ATTR_MAP(IdentityN) = EMPTY_ATTR_MAP;
DYN_OUTPUT_MAP(IdentityN) = {{0, DYN_OUTPUT_DESC(y)}};
REG_ADPT_DESC(IdentityNMakeTuple, kNameMakeTuple, ADPT_DESC(IdentityN))
REG_ADPT_DESC(IdentityNMakeList, kNameMakeList, ADPT_DESC(IdentityN))
REG_ADPT_DESC(IdentityNDepend, kNameDepend, ADPT_DESC(IdentityN))
REG_ADPT_DESC(IdentityNReturn, kNameReturn, ADPT_DESC(IdentityN))
REG_ADPT_DESC(IdentityN, kNameIdentityN, ADPT_DESC(IdentityN))
// TupleGetItem's output may be a tuple when input is a nested tuple
REG_ADPT_DESC(IdentityNTupleGetItem, kNameTupleGetItem, ADPT_DESC(IdentityN))

// SelectV2
INPUT_MAP(SelectV2) = {{1, INPUT_DESC(condition)}, {2, INPUT_DESC(then)}, {3, INPUT_DESC(else)}};
ATTR_MAP(SelectV2) = EMPTY_ATTR_MAP;
OUTPUT_MAP(SelectV2) = {{0, OUTPUT_DESC(result)}};
REG_ADPT_DESC(SelectV2, kNameSelectV2, ADPT_DESC(SelectV2))

// Where
INPUT_MAP(Where) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Where) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Where) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Where, kNameWhere, ADPT_DESC(Where))

// Unique
INPUT_MAP(Unique) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Unique) = {{"out_idx", ATTR_DESC(out_idx, AnyTraits<GEType>())}};
OUTPUT_MAP(Unique) = {{0, OUTPUT_DESC(y)}, {1, OUTPUT_DESC(idx)}};
REG_ADPT_DESC(Unique, kNameUnique, ADPT_DESC(Unique))

// BroadcastGradientArgs
INPUT_MAP(BroadcastGradientArgs) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(BroadcastGradientArgs) = EMPTY_ATTR_MAP;
OUTPUT_MAP(BroadcastGradientArgs) = {{0, OUTPUT_DESC(y1)}, {1, OUTPUT_DESC(y2)}};
REG_ADPT_DESC(BroadcastGradientArgs, kNameDynamicBroadcastGradientArgs, ADPT_DESC(BroadcastGradientArgs))

// QueueData
INPUT_MAP(QueueData) = EMPTY_INPUT_MAP;
OUTPUT_MAP(QueueData) = {{0, OUTPUT_DESC(y)}};
ATTR_MAP(QueueData) = {{"index", ATTR_DESC(index, AnyTraits<int64_t>())},
                       {"queue_name", ATTR_DESC(queue_name, AnyTraits<string>())},
                       {"output_types", ATTR_DESC(output_types, AnyTraits<std::vector<GEType>>())},
                       {"output_shapes", ATTR_DESC(output_shapes, AnyTraits<std::vector<std::vector<int64_t>>>())}};
REG_ADPT_DESC(QueueData, prim::kPrimQueueData->name(), ADPT_DESC(QueueData))

// Size
INPUT_MAP(Size) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Size) = {{"dtype", ATTR_DESC(dtype, AnyTraits<int64_t>())}};
OUTPUT_MAP(Size) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Size, kNameSize, ADPT_DESC(Size))

// Meshgrid
INPUT_MAP(Meshgrid) = EMPTY_INPUT_MAP;
DYN_INPUT_MAP(Meshgrid) = {{1, DYN_INPUT_DESC(x)}};
ATTR_MAP(Meshgrid) = {{"indexing", ATTR_DESC(indexing, AnyTraits<std::string>())}};
DYN_OUTPUT_MAP(Meshgrid) = {{0, DYN_OUTPUT_DESC(y)}};
REG_ADPT_DESC(Meshgrid, prim::kPrimMeshgrid->name(), ADPT_DESC(Meshgrid))

// SliceGrad
CUST_INPUT_MAP(SliceGrad) = {{1, INPUT_DESC(dy)}, {2, INPUT_DESC(x)}, {3, INPUT_DESC(begin)}, {4, INPUT_DESC(size)}};
CUST_ATTR_MAP(SliceGrad) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(SliceGrad) = {{0, OUTPUT_DESC(dx)}};
REG_ADPT_DESC(SliceGrad, prim::kPrimSliceGrad->name(), CUST_ADPT_DESC(SliceGrad))

// MaskedSelectGrad
CUST_INPUT_MAP(MaskedSelectGrad) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(mask)}, {3, INPUT_DESC(grad)}};
CUST_ATTR_MAP(MaskedSelectGrad) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(MaskedSelectGrad) = {{0, OUTPUT_DESC(dx)}};
REG_ADPT_DESC(MaskedSelectGrad, prim::kPrimMaskedSelectGrad->name(), CUST_ADPT_DESC(MaskedSelectGrad))

// GradDGradV2
CUST_INPUT_MAP(GatherDGradV2) = {{1, INPUT_DESC(x)}, {3, INPUT_DESC(index)}, {4, INPUT_DESC(grad)}};
CUST_INPUT_ATTR_MAP(GatherDGradV2) = {{2, ATTR_DESC(dim, AnyTraits<int64_t>())}};
CUST_ATTR_MAP(GatherDGradV2) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(GatherDGradV2) = {{0, OUTPUT_DESC(output)}};
REG_ADPT_DESC(GatherDGradV2, prim::kPrimGatherDGradV2->name(), CUST_ADPT_DESC(GatherDGradV2))

// AsStrided
INPUT_MAP(AsStrided) = {
  {1, INPUT_DESC(x)}, {2, INPUT_DESC(size)}, {3, INPUT_DESC(stride)}, {4, INPUT_DESC(storage_offset)}};
ATTR_INPUT_MAP(AsStrided) = {{"size", "size"}, {"stride", "stride"}, {"storage_offset", "storage_offset"}};
ATTR_MAP(AsStrided) = EMPTY_ATTR_MAP;
OUTPUT_MAP(AsStrided) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(AsStrided, kNameAsStrided, ADPT_DESC(AsStrided))

// ViewCopy
INPUT_MAP(ViewCopy) = {
  {1, INPUT_DESC(dst)}, {2, INPUT_DESC(dst_size)}, {3, INPUT_DESC(dst_stride)}, {4, INPUT_DESC(dst_storage_offset)},
  {5, INPUT_DESC(src)}, {6, INPUT_DESC(src_size)}, {7, INPUT_DESC(src_stride)}, {8, INPUT_DESC(src_storage_offset)}};
ATTR_INPUT_MAP(ViewCopy) = {
  {"dst_size", "dst_size"}, {"dst_stride", "dst_stride"}, {"dst_storage_offset", "dst_storage_offset"},
  {"src_size", "src_size"}, {"src_stride", "src_stride"}, {"src_storage_offset", "src_storage_offset"}};
ATTR_MAP(ViewCopy) = EMPTY_ATTR_MAP;
OUTPUT_MAP(ViewCopy) = {{0, OUTPUT_DESC(dst)}};
REG_ADPT_DESC(ViewCopy, kNameViewCopy, ADPT_DESC(ViewCopy))

// CheckNumerics
INPUT_MAP(CheckNumerics) = {{1, INPUT_DESC(x)}};
ATTR_MAP(CheckNumerics) = {{"message", ATTR_DESC(message, AnyTraits<std::string>())}};
OUTPUT_MAP(CheckNumerics) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(CheckNumerics, prim::kPrimCheckNumerics->name(), ADPT_DESC(CheckNumerics));

// HammingWindow
CUST_INPUT_MAP(HammingWindow) = {{1, INPUT_DESC(length)}};
CUST_ATTR_MAP(HammingWindow) = {{"periodic", ATTR_DESC(periodic, AnyTraits<bool>())},
                                {"alpha", ATTR_DESC(alpha, AnyTraits<float>())},
                                {"beta", ATTR_DESC(beta, AnyTraits<float>())},
                                {"dtype", ATTR_DESC(dtype, AnyTraits<int64_t>())}};
CUST_OUTPUT_MAP(HammingWindow) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(HammingWindow, prim::kPrimHammingWindow->name(), CUST_ADPT_DESC(HammingWindow));

// LowerBound
INPUT_MAP(LowerBound) = {{1, INPUT_DESC(sorted_x)}, {2, INPUT_DESC(values)}};
ATTR_MAP(LowerBound) = EMPTY_ATTR_MAP;
OUTPUT_MAP(LowerBound) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(LowerBound, prim::kPrimLowerBound->name(), ADPT_DESC(LowerBound));

// ListDiff
INPUT_MAP(ListDiff) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(y)}};
ATTR_MAP(ListDiff) = {{"out_idx", ATTR_DESC(out_idx, AnyTraits<GEType>())}};
OUTPUT_MAP(ListDiff) = {{0, OUTPUT_DESC(out)}, {1, OUTPUT_DESC(idx)}};
REG_ADPT_DESC(ListDiff, kNameListDiff, ADPT_DESC(ListDiff));

// IndexFill
CUST_INPUT_MAP(IndexFill) = {
  {1, INPUT_DESC(x)}, {2, INPUT_DESC(dim)}, {3, INPUT_DESC(indices)}, {4, INPUT_DESC(value)}};
CUST_ATTR_MAP(IndexFill) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(IndexFill) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(IndexFill, kNameIndexFill, CUST_ADPT_DESC(IndexFill));

// Mvlgamma
CUST_INPUT_MAP(Mvlgamma) = {{1, INPUT_DESC(x)}};
CUST_ATTR_MAP(Mvlgamma) = {{"p", ATTR_DESC(p, AnyTraits<int64_t>())}};
CUST_OUTPUT_MAP(Mvlgamma) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Mvlgamma, prim::kPrimMvlgamma->name(), CUST_ADPT_DESC(Mvlgamma));

// MvlgammaGrad
CUST_INPUT_MAP(MvlgammaGrad) = {{1, INPUT_DESC(y_grad)}, {2, INPUT_DESC(x)}};
CUST_ATTR_MAP(MvlgammaGrad) = {{"p", ATTR_DESC(p, AnyTraits<int64_t>())}};
CUST_OUTPUT_MAP(MvlgammaGrad) = {{0, OUTPUT_DESC(x_grad)}};
REG_ADPT_DESC(MvlgammaGrad, prim::kPrimMvlgammaGrad->name(), CUST_ADPT_DESC(MvlgammaGrad));

// LogSpace
CUST_INPUT_MAP(LogSpace) = {{1, INPUT_DESC(start)}, {2, INPUT_DESC(end)}};
CUST_ATTR_MAP(LogSpace) = {{"steps", ATTR_DESC(steps, AnyTraits<int64_t>())},
                           {"base", ATTR_DESC(base, AnyTraits<int64_t>())}};
CUST_OUTPUT_MAP(LogSpace) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(LogSpace, prim::kPrimLogSpace->name(), CUST_ADPT_DESC(LogSpace));

// UniqueConsecutive
INPUT_MAP(UniqueConsecutive) = {{1, INPUT_DESC(x)}};
ATTR_MAP(UniqueConsecutive) = {{"return_idx", ATTR_DESC(return_idx, AnyTraits<bool>())},
                               {"return_counts", ATTR_DESC(return_counts, AnyTraits<bool>())},
                               {"axis", ATTR_DESC(axis, AnyTraits<int64_t>())}};
OUTPUT_MAP(UniqueConsecutive) = {{0, OUTPUT_DESC(y)}, {1, OUTPUT_DESC(idx)}, {2, OUTPUT_DESC(count)}};
REG_ADPT_DESC(UniqueConsecutive, prim::kPrimUniqueConsecutive->name(), ADPT_DESC(UniqueConsecutive));

// UniqueWithPad
INPUT_MAP(UniqueWithPad) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(pad_num)}};
ATTR_MAP(UniqueWithPad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(UniqueWithPad) = {{0, OUTPUT_DESC(y)}, {1, OUTPUT_DESC(idx)}};
REG_ADPT_DESC(UniqueWithPad, prim::kPrimUniqueWithPad->name(), ADPT_DESC(UniqueWithPad));

// UpperBound
INPUT_MAP(UpperBound) = {{1, INPUT_DESC(sorted_x)}, {2, INPUT_DESC(values)}};
ATTR_MAP(UpperBound) = EMPTY_ATTR_MAP;
OUTPUT_MAP(UpperBound) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(UpperBound, prim::kPrimUpperBound->name(), ADPT_DESC(UpperBound));

// UnravelIndex
INPUT_MAP(UnravelIndex) = {{1, INPUT_DESC(indices)}, {2, INPUT_DESC(dims)}};
ATTR_MAP(UnravelIndex) = EMPTY_ATTR_MAP;
OUTPUT_MAP(UnravelIndex) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(UnravelIndex, prim::kPrimUnravelIndex->name(), ADPT_DESC(UnravelIndex));

// NoRepeatNGram
CUST_INPUT_MAP(NoRepeatNGram) = {{1, INPUT_DESC(state_seq)}, {2, INPUT_DESC(log_probs)}};
CUST_ATTR_MAP(NoRepeatNGram) = {{"ngram_size", ATTR_DESC(ngram_size, AnyTraits<int64_t>())}};
CUST_OUTPUT_MAP(NoRepeatNGram) = {{0, OUTPUT_DESC(out)}};
REG_ADPT_DESC(NoRepeatNGram, prim::kPrimNoRepeatNGram->name(), CUST_ADPT_DESC(NoRepeatNGram));

// GenerateEodMask
CUST_INPUT_MAP(GenerateEodMask) = {{1, INPUT_DESC(inputs_ids)}};
CUST_ATTR_MAP(GenerateEodMask) = {{"n_pos", ATTR_DESC(n_pos, AnyTraits<int64_t>())},
                                  {"eod_token_id", ATTR_DESC(eod_token_id, AnyTraits<int64_t>())},
                                  {"n_step", ATTR_DESC(n_step, AnyTraits<std::vector<int64_t>>())},
                                  {"n_error_mode", ATTR_DESC(n_error_mode, AnyTraits<std::string>())}};
CUST_OUTPUT_MAP(GenerateEodMask) = {{0, OUTPUT_DESC(position_ids)}};
REG_ADPT_DESC(GenerateEodMask, prim::kPrimGenerateEodMask->name(), CUST_ADPT_DESC(GenerateEodMask));

// NonZero
INPUT_MAP(NonZero) = {{1, INPUT_DESC(x)}};
ATTR_MAP(NonZero) = {{"transpose", ATTR_DESC(transpose, AnyTraits<bool>())}};
OUTPUT_MAP(NonZero) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(NonZeroV2, kNameNonZeroV2, ADPT_DESC(NonZero))
REG_ADPT_DESC(NonZero, kNameNonZero, ADPT_DESC(NonZero))

// Coalesce
CUST_INPUT_MAP(Coalesce) = {{1, INPUT_DESC(x_indices)}, {2, INPUT_DESC(x_values)}, {3, INPUT_DESC(x_shape)}};
CUST_ATTR_MAP(Coalesce) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(Coalesce) = {{0, OUTPUT_DESC(y_indices)}, {1, OUTPUT_DESC(y_values)}, {2, OUTPUT_DESC(y_shape)}};
REG_ADPT_DESC(Coalesce, prim::kPrimCoalesce->name(), CUST_ADPT_DESC(Coalesce))

// Padding
CUST_INPUT_MAP(Padding) = {{1, INPUT_DESC(x)}};
CUST_ATTR_MAP(Padding) = {{"pad_dim_size", ATTR_DESC(pad_dim_size, AnyTraits<int64_t>())}};
CUST_OUTPUT_MAP(Padding) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Padding, prim::kPrimPadding->name(), CUST_ADPT_DESC(Padding));

// MatrixBandPart
INPUT_MAP(MatrixBandPart) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(num_lower)}, {3, INPUT_DESC(num_upper)}};
ATTR_MAP(MatrixBandPart) = EMPTY_ATTR_MAP;
OUTPUT_MAP(MatrixBandPart) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MatrixBandPart, prim::kPrimMatrixBandPart->name(), ADPT_DESC(MatrixBandPart));
}  // namespace mindspore::transform
