/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

namespace mindspore::transform {
// PadD
INPUT_MAP(PadD) = {{1, INPUT_DESC(x)}};
ATTR_MAP(PadD) = {{"paddings", ATTR_DESC(paddings, AnyTraits<std::vector<std::vector<int64_t>>>())}};
OUTPUT_MAP(PadD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(PadD, kNamePadD, ADPT_DESC(PadD))

// Diag
INPUT_MAP(Diag) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Diag) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Diag) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Diag, kNameDiag, ADPT_DESC(Diag))
}  // namespace mindspore::transform
