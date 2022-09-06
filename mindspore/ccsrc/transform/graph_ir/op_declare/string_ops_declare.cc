/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "transform/graph_ir/op_declare/string_ops_declare.h"
#include <vector>

namespace mindspore::transform {
INPUT_MAP(StringUpper) = {{1, INPUT_DESC(input)}};
ATTR_MAP(StringUpper) = {{"encoding", ATTR_DESC(encoding, AnyTraits<std::string>())}};
OUTPUT_MAP(StringUpper) = {{0, OUTPUT_DESC(output)}};
REG_ADPT_DESC(StringUpper, kNameStringUpper, ADPT_DESC(StringUpper))

INPUT_MAP(StringLength) = {{1, INPUT_DESC(x)}};
ATTR_MAP(StringLength) = {{"unit", ATTR_DESC(unit, AnyTraits<std::string>())}};
OUTPUT_MAP(StringLength) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(StringLength, kNameStringLength, ADPT_DESC(StringLength))

INPUT_MAP(DecodeBase64) = {{1, INPUT_DESC(x)}};
ATTR_MAP(DecodeBase64) = EMPTY_ATTR_MAP;
OUTPUT_MAP(DecodeBase64) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(DecodeBase64, kNameDecodeBase64, ADPT_DESC(DecodeBase64))
}  // namespace mindspore::transform
