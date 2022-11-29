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

#include <vector>
#include "transform/graph_ir/op_declare/selection_ops_declare.h"

namespace mindspore::transform {
// CumsumD
INPUT_MAP(CumsumD) = {{1, INPUT_DESC(x)}};
INPUT_ATTR_MAP(CumsumD) = {{2, ATTR_DESC(axis, AnyTraits<int64_t>())}};
ATTR_MAP(CumsumD) = {{"exclusive", ATTR_DESC(exclusive, AnyTraits<bool>())},
                     {"reverse", ATTR_DESC(reverse, AnyTraits<bool>())}};
OUTPUT_MAP(CumsumD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(CumsumD, kNameCumSum, ADPT_DESC(CumsumD))

// CumprodD
INPUT_MAP(CumprodD) = {{1, INPUT_DESC(x)}};
INPUT_ATTR_MAP(CumprodD) = {{2, ATTR_DESC(axis, AnyTraits<int64_t>())}};
ATTR_MAP(CumprodD) = {{"exclusive", ATTR_DESC(exclusive, AnyTraits<bool>())},
                      {"reverse", ATTR_DESC(reverse, AnyTraits<bool>())}};
OUTPUT_MAP(CumprodD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(CumprodD, kNameCumProd, ADPT_DESC(CumprodD))

INPUT_MAP(Tile) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(multiples)}};
ATTR_INPUT_MAP(Tile) = {{"multiples", 2}};
ATTR_MAP(Tile) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Tile) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Tile, kNameTile, ADPT_DESC(Tile))

INPUT_MAP(Slice) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(offsets)}, {3, INPUT_DESC(size)}};
ATTR_MAP(Slice) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Slice) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Slice, kNameSlice, ADPT_DESC(Slice))

// TopK
INPUT_MAP(TopK) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(k)}};
ATTR_MAP(TopK) = {{"sorted", ATTR_DESC(sorted, AnyTraits<bool>())}};
OUTPUT_MAP(TopK) = {{0, OUTPUT_DESC(values)}, {1, OUTPUT_DESC(indices)}};
REG_ADPT_DESC(TopK, kNameTopK, ADPT_DESC(TopK))

// InTopK
INPUT_MAP(InTopKD) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(InTopKD) = {{"k", ATTR_DESC(k, AnyTraits<int64_t>())}};
OUTPUT_MAP(InTopKD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(InTopKD, kNameInTopKD, ADPT_DESC(InTopKD))

// OneHot
INPUT_MAP(OneHot) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(depth)}, {3, INPUT_DESC(on_value)}, {4, INPUT_DESC(off_value)}};
ATTR_INPUT_MAP(OneHot) = {{"depth", 2}};
ATTR_MAP(OneHot) = {{"axis", ATTR_DESC(axis, AnyTraits<int64_t>())}};
OUTPUT_MAP(OneHot) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(OneHot, prim::kPrimOneHot->name(), ADPT_DESC(OneHot))

// GatherV2
INPUT_MAP(GatherV2) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(indices)}, {3, INPUT_DESC(axis)}};
ATTR_INPUT_MAP(GatherV2) = {{"axis", 3}};
ATTR_MAP(GatherV2) = EMPTY_ATTR_MAP;
OUTPUT_MAP(GatherV2) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(GatherV2, prim::kPrimGatherV2->name(), ADPT_DESC(GatherV2))
REG_ADPT_DESC(Gather, prim::kPrimGather->name(), ADPT_DESC(GatherV2))

// ScatterNd
INPUT_MAP(ScatterNd) = {{1, INPUT_DESC(indices)}, {2, INPUT_DESC(x)}, {3, INPUT_DESC(shape)}};
ATTR_INPUT_MAP(ScatterNd) = {{"shape", 3}};
ATTR_MAP(ScatterNd) = EMPTY_ATTR_MAP;
OUTPUT_MAP(ScatterNd) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ScatterNd, kNameScatterNd, ADPT_DESC(ScatterNd))

// ScatterNonAliasingAdd
INPUT_MAP(ScatterNonAliasingAdd) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(indices)}, {3, INPUT_DESC(updates)}};
ATTR_MAP(ScatterNonAliasingAdd) = EMPTY_ATTR_MAP;
OUTPUT_MAP(ScatterNonAliasingAdd) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ScatterNonAliasingAdd, kNameScatterNonAliasingAdd, ADPT_DESC(ScatterNonAliasingAdd))

// GatherNd
INPUT_MAP(GatherNd) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(indices)}};
ATTR_MAP(GatherNd) = EMPTY_ATTR_MAP;
OUTPUT_MAP(GatherNd) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(GatherNd, kNameGatherNd, ADPT_DESC(GatherNd))

// GatherD
INPUT_MAP(GatherD) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(dim)}, {3, INPUT_DESC(index)}};
ATTR_MAP(GatherD) = EMPTY_ATTR_MAP;
OUTPUT_MAP(GatherD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(GatherD, kNameGatherD, ADPT_DESC(GatherD))

// Range
INPUT_MAP(RangeD) = {{1, INPUT_DESC(x)}};
ATTR_MAP(RangeD) = {{"start", ATTR_DESC(start, AnyTraits<float>())},
                    {"limit", ATTR_DESC(limit, AnyTraits<float>())},
                    {"delta", ATTR_DESC(delta, AnyTraits<float>())}};
OUTPUT_MAP(RangeD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(RangeD, kNameRange, ADPT_DESC(RangeD))

// RangeV2
INPUT_MAP(Range) = {{1, INPUT_DESC(start)}, {2, INPUT_DESC(limit)}, {3, INPUT_DESC(delta)}};
ATTR_MAP(Range) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Range) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(RangeV2, kNameRangeV2, ADPT_DESC(Range))

// InplaceAdd
INPUT_MAP(InplaceAdd) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(indices)}, {3, INPUT_DESC(v)}};
ATTR_MAP(InplaceAdd) = EMPTY_ATTR_MAP;
ATTR_INPUT_MAP(InplaceAdd) = {{"indices", 2}};
OUTPUT_MAP(InplaceAdd) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(InplaceAdd, kNameInplaceAddD, ADPT_DESC(InplaceAdd))

// InplaceSub
INPUT_MAP(InplaceSub) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(indices)}, {3, INPUT_DESC(v)}};
ATTR_MAP(InplaceSub) = EMPTY_ATTR_MAP;
ATTR_INPUT_MAP(InplaceSub) = {{"indices", 2}};
OUTPUT_MAP(InplaceSub) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(InplaceSub, kNameInplaceSubD, ADPT_DESC(InplaceSub))

// InplaceUpdate
INPUT_MAP(InplaceUpdate) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(indices)}, {3, INPUT_DESC(v)}};
ATTR_MAP(InplaceUpdate) = EMPTY_ATTR_MAP;
ATTR_INPUT_MAP(InplaceUpdate) = {{"indices", 2}};
OUTPUT_MAP(InplaceUpdate) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(InplaceUpdate, kNameInplaceUpdateD, ADPT_DESC(InplaceUpdate))

// Select
INPUT_MAP(Select) = {{1, INPUT_DESC(condition)}, {2, INPUT_DESC(x1)}, {3, INPUT_DESC(x2)}};
ATTR_MAP(Select) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Select) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Select, prim::kPrimSelect->name(), ADPT_DESC(Select))

// StridedSliceGrad
INPUT_MAP(StridedSliceGrad) = {
  {1, INPUT_DESC(dy)}, {2, INPUT_DESC(shape)}, {3, INPUT_DESC(begin)}, {4, INPUT_DESC(end)}, {5, INPUT_DESC(strides)}};
ATTR_MAP(StridedSliceGrad) = {{"begin_mask", ATTR_DESC(begin_mask, AnyTraits<int64_t>())},
                              {"end_mask", ATTR_DESC(end_mask, AnyTraits<int64_t>())},
                              {"ellipsis_mask", ATTR_DESC(ellipsis_mask, AnyTraits<int64_t>())},
                              {"new_axis_mask", ATTR_DESC(new_axis_mask, AnyTraits<int64_t>())},
                              {"shrink_axis_mask", ATTR_DESC(shrink_axis_mask, AnyTraits<int64_t>())}};
OUTPUT_MAP(StridedSliceGrad) = {{0, OUTPUT_DESC(output)}};
REG_ADPT_DESC(StridedSliceGrad, kNameStridedSliceGrad, ADPT_DESC(StridedSliceGrad))

// StridedSlice
INPUT_MAP(StridedSlice) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(begin)}, {3, INPUT_DESC(end)}, {4, INPUT_DESC(strides)}};
ATTR_INPUT_MAP(StridedSlice) = {{"begin", 2}, {"end", 3}, {"strides", 4}};
ATTR_MAP(StridedSlice) = {{"begin_mask", ATTR_DESC(begin_mask, AnyTraits<int64_t>())},
                          {"end_mask", ATTR_DESC(end_mask, AnyTraits<int64_t>())},
                          {"ellipsis_mask", ATTR_DESC(ellipsis_mask, AnyTraits<int64_t>())},
                          {"new_axis_mask", ATTR_DESC(new_axis_mask, AnyTraits<int64_t>())},
                          {"shrink_axis_mask", ATTR_DESC(shrink_axis_mask, AnyTraits<int64_t>())}};
OUTPUT_MAP(StridedSlice) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(StridedSlice, kNameStridedSlice, ADPT_DESC(StridedSlice))

// StridedSliceV2
INPUT_MAP(StridedSliceV2) = {
  {1, INPUT_DESC(x)}, {2, INPUT_DESC(begin)}, {3, INPUT_DESC(end)}, {4, INPUT_DESC(axes)}, {5, INPUT_DESC(strides)}};
ATTR_MAP(StridedSliceV2) = {{"begin_mask", ATTR_DESC(begin_mask, AnyTraits<int64_t>())},
                            {"end_mask", ATTR_DESC(end_mask, AnyTraits<int64_t>())},
                            {"ellipsis_mask", ATTR_DESC(ellipsis_mask, AnyTraits<int64_t>())},
                            {"new_axis_mask", ATTR_DESC(new_axis_mask, AnyTraits<int64_t>())},
                            {"shrink_axis_mask", ATTR_DESC(shrink_axis_mask, AnyTraits<int64_t>())}};
OUTPUT_MAP(StridedSliceV2) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(StridedSliceV2, kNameStridedSliceV2, ADPT_DESC(StridedSliceV2))

// UnsortedSegmentSum
INPUT_MAP(UnsortedSegmentSum) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(segment_ids)}, {3, INPUT_DESC(num_segments)}};
ATTR_INPUT_MAP(UnsortedSegmentSum) = {{"num_segments", 3}};
ATTR_MAP(UnsortedSegmentSum) = EMPTY_ATTR_MAP;
OUTPUT_MAP(UnsortedSegmentSum) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(UnsortedSegmentSumD, prim::kPrimUnsortedSegmentSumD->name(), ADPT_DESC(UnsortedSegmentSum))
REG_ADPT_DESC(UnsortedSegmentSum, prim::kPrimUnsortedSegmentSum->name(), ADPT_DESC(UnsortedSegmentSum))

// UnsortedSegmentProd
INPUT_MAP(UnsortedSegmentProd) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(segment_ids)}, {3, INPUT_DESC(num_segments)}};
ATTR_INPUT_MAP(UnsortedSegmentProd) = {{"num_segments", 3}};
ATTR_MAP(UnsortedSegmentProd) = EMPTY_ATTR_MAP;
OUTPUT_MAP(UnsortedSegmentProd) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(UnsortedSegmentProd, kNameUnsortedSegmentProdD, ADPT_DESC(UnsortedSegmentProd))

// UnsortedSegmentMaxD
INPUT_MAP(UnsortedSegmentMaxD) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(segment_ids)}};
INPUT_ATTR_MAP(UnsortedSegmentMaxD) = {{3, ATTR_DESC(num_segments, AnyTraits<int64_t>())}};
ATTR_MAP(UnsortedSegmentMaxD) = EMPTY_ATTR_MAP;
OUTPUT_MAP(UnsortedSegmentMaxD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(UnsortedSegmentMaxD, kNameUnsortedSegmentMaxD, ADPT_DESC(UnsortedSegmentMaxD))

// UnsortedSegmentMin
INPUT_MAP(UnsortedSegmentMin) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(segment_ids)}, {3, INPUT_DESC(num_segments)}};
ATTR_MAP(UnsortedSegmentMin) = EMPTY_ATTR_MAP;
OUTPUT_MAP(UnsortedSegmentMin) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(UnsortedSegmentMin, prim::kPrimUnsortedSegmentMin->name(), ADPT_DESC(UnsortedSegmentMin))

// ReverseV2
INPUT_MAP(ReverseV2D) = {{1, INPUT_DESC(x)}};
ATTR_MAP(ReverseV2D) = {{"axis", ATTR_DESC(axis, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())}};
OUTPUT_MAP(ReverseV2D) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ReverseV2D, kNameReverseV2, ADPT_DESC(ReverseV2D))

// MaskedSelect
INPUT_MAP(MaskedSelect) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(mask)}};
ATTR_MAP(MaskedSelect) = EMPTY_ATTR_MAP;
OUTPUT_MAP(MaskedSelect) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MaskedSelect, kNameMaskedSelect, ADPT_DESC(MaskedSelect))

// MaskedFill
INPUT_MAP(MaskedFill) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(mask)}, {3, INPUT_DESC(value)}};
ATTR_MAP(MaskedFill) = EMPTY_ATTR_MAP;
OUTPUT_MAP(MaskedFill) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MaskedFill, prim::kPrimMaskedFill->name(), ADPT_DESC(MaskedFill))
}  // namespace mindspore::transform
