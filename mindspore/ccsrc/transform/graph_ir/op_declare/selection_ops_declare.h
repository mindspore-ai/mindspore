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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_SELECTION_OPS_DECLARE_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_SELECTION_OPS_DECLARE_H_

#include "utils/hash_map.h"
#include "transform/graph_ir/op_declare/op_declare_macro.h"
#include "ops/selection_ops.h"

namespace mindspore::transform {
DECLARE_OP_ADAPTER(Slice)
DECLARE_OP_USE_OUTPUT(Slice)

DECLARE_OP_ADAPTER(ScatterNd)
DECLARE_OP_USE_OUTPUT(ScatterNd)

DECLARE_OP_ADAPTER(ScatterNonAliasingAdd)
DECLARE_OP_USE_OUTPUT(ScatterNonAliasingAdd)

DECLARE_OP_ADAPTER(GatherNd)
DECLARE_OP_USE_OUTPUT(GatherNd)

DECLARE_OP_ADAPTER(GatherD)
DECLARE_OP_USE_OUTPUT(GatherD)

DECLARE_OP_ADAPTER(TopK)
DECLARE_OP_USE_OUTPUT(TopK)

DECLARE_OP_ADAPTER(TopKV2)
DECLARE_OP_USE_OUTPUT(TopKV2)

DECLARE_OP_ADAPTER(InTopKD)
DECLARE_OP_USE_OUTPUT(InTopKD)

DECLARE_OP_ADAPTER(Select)
DECLARE_OP_USE_OUTPUT(Select)

DECLARE_OP_ADAPTER(StridedSliceGrad)
DECLARE_OP_USE_OUTPUT(StridedSliceGrad)

DECLARE_OP_ADAPTER(StridedSlice)
DECLARE_OP_USE_OUTPUT(StridedSlice)

DECLARE_OP_ADAPTER(StridedSliceV2)
DECLARE_OP_USE_OUTPUT(StridedSliceV2)

DECLARE_OP_ADAPTER(UnsortedSegmentSum)
DECLARE_OP_USE_OUTPUT(UnsortedSegmentSum)

DECLARE_OP_ADAPTER(UnsortedSegmentProd)
DECLARE_OP_USE_OUTPUT(UnsortedSegmentProd)

DECLARE_OP_ADAPTER(UnsortedSegmentMax)
DECLARE_OP_USE_OUTPUT(UnsortedSegmentMax)

DECLARE_OP_ADAPTER(UnsortedSegmentMin)
DECLARE_OP_USE_OUTPUT(UnsortedSegmentMin)

DECLARE_OP_ADAPTER(CumprodD)
DECLARE_OP_USE_INPUT_ATTR(CumprodD)
DECLARE_OP_USE_OUTPUT(CumprodD)

DECLARE_OP_ADAPTER(Tile)
DECLARE_OP_USE_OUTPUT(Tile)

DECLARE_OP_ADAPTER(TileD)
DECLARE_OP_USE_INPUT_ATTR(TileD)
DECLARE_OP_USE_OUTPUT(TileD)

DECLARE_OP_ADAPTER(OneHot)
DECLARE_OP_USE_OUTPUT(OneHot)

DECLARE_OP_ADAPTER(GatherV2)
DECLARE_OP_USE_OUTPUT(GatherV2)

DECLARE_OP_ADAPTER(RangeD)
DECLARE_OP_USE_OUTPUT(RangeD)

DECLARE_OP_ADAPTER(Range)
DECLARE_OP_USE_OUTPUT(Range)

DECLARE_OP_ADAPTER(InplaceAddD)
DECLARE_OP_USE_OUTPUT(InplaceAddD)

DECLARE_OP_ADAPTER(InplaceSubD)
DECLARE_OP_USE_OUTPUT(InplaceSubD)

DECLARE_OP_ADAPTER(InplaceUpdateD)
DECLARE_OP_USE_OUTPUT(InplaceUpdateD)

DECLARE_OP_ADAPTER(InplaceAdd)
DECLARE_OP_USE_OUTPUT(InplaceAdd)

DECLARE_OP_ADAPTER(InplaceSub)
DECLARE_OP_USE_OUTPUT(InplaceSub)

DECLARE_OP_ADAPTER(InplaceUpdate)
DECLARE_OP_USE_OUTPUT(InplaceUpdate)

DECLARE_OP_ADAPTER(CumsumD)
DECLARE_OP_USE_INPUT_ATTR(CumsumD)
DECLARE_OP_USE_OUTPUT(CumsumD)

DECLARE_OP_ADAPTER(GatherV2)
DECLARE_OP_USE_OUTPUT(GatherV2)

DECLARE_OP_ADAPTER(ReverseV2D)
DECLARE_OP_USE_OUTPUT(ReverseV2D)

DECLARE_OP_ADAPTER(ReverseV2)
DECLARE_OP_USE_OUTPUT(ReverseV2)

DECLARE_OP_ADAPTER(MaskedSelect)
DECLARE_OP_USE_OUTPUT(MaskedSelect)

DECLARE_OP_ADAPTER(MaskedFill)
DECLARE_OP_USE_OUTPUT(MaskedFill)

DECLARE_OP_ADAPTER(Cummin)
DECLARE_OP_USE_OUTPUT(Cummin)

DECLARE_OP_ADAPTER(Cumsum)
DECLARE_OP_USE_OUTPUT(Cumsum)

DECLARE_OP_ADAPTER(StridedRead)
DECLARE_OP_USE_OUTPUT(StridedRead)

DECLARE_OP_ADAPTER(StridedWrite)
DECLARE_OP_USE_OUTPUT(StridedWrite)

DECLARE_OP_ADAPTER(InplaceIndexAdd)
DECLARE_OP_USE_OUTPUT(InplaceIndexAdd)
}  // namespace mindspore::transform
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_SELECTION_OPS_DECLARE_H_
