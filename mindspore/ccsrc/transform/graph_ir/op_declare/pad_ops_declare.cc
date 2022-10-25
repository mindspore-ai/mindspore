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

#include "transform/graph_ir/op_declare/pad_ops_declare.h"
#include <vector>
#include <string>

namespace mindspore::transform {
// PadD
INPUT_MAP(PadD) = {{1, INPUT_DESC(x)}};
ATTR_MAP(PadD) = {{"paddings", ATTR_DESC(paddings, AnyTraits<std::vector<std::vector<int64_t>>>())}};
OUTPUT_MAP(PadD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(PadD, kNamePadD, ADPT_DESC(PadD))

// Pad
INPUT_MAP(Pad) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(paddings)}};
ATTR_MAP(Pad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Pad) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Pad, kNamePadV1, ADPT_DESC(Pad))

// BroadcastToD
INPUT_MAP(BroadcastToD) = {{1, INPUT_DESC(x)}};
ATTR_MAP(BroadcastToD) = {{"shape", ATTR_DESC(shape, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())}};
OUTPUT_MAP(BroadcastToD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(BroadcastToD, kNameBroadcastTo, ADPT_DESC(BroadcastToD))

// DynamicBroadcastTo
INPUT_MAP(BroadcastTo) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(shape)}};
OUTPUT_MAP(BroadcastTo) = {{0, OUTPUT_DESC(y)}};
ATTR_MAP(BroadcastTo) = EMPTY_ATTR_MAP;
REG_ADPT_DESC(BroadcastTo, kDynamicBroadcastToOpName, ADPT_DESC(BroadcastTo))

// Diag
INPUT_MAP(Diag) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Diag) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Diag) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Diag, kNameDiag, ADPT_DESC(Diag))

// FillD
INPUT_MAP(FillD) = {{1, INPUT_DESC(value)}};
ATTR_MAP(FillD) = {{"dims", ATTR_DESC(dims, AnyTraits<std::vector<int64_t>>())}};
OUTPUT_MAP(FillD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(FillD, kNameFillD, ADPT_DESC(FillD))

// Fill
INPUT_MAP(Fill) = {{1, INPUT_DESC(dims)}, {2, INPUT_DESC(value)}};
ATTR_MAP(Fill) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Fill) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Fill, kNameFillV1, ADPT_DESC(Fill))

// PadV3
INPUT_MAP(PadV3) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(paddings)}, {3, INPUT_DESC(constant_values)}};
ATTR_MAP(PadV3) = {{"mode", ATTR_DESC(mode, AnyTraits<std::string>())},
                   {"pad_contiguous", ATTR_DESC(paddings_contiguous, AnyTraits<bool>())}};
OUTPUT_MAP(PadV3) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(PadV3, kNamePadV3, ADPT_DESC(PadV3))

// PadV2
INPUT_MAP(PadV2) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(paddings)}, {3, INPUT_DESC(constant_values)}};
ATTR_MAP(PadV2) = EMPTY_ATTR_MAP;
OUTPUT_MAP(PadV2) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(PadV2, kNamePadV2, ADPT_DESC(PadV2))
}  // namespace mindspore::transform
