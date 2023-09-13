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

#include "transform/graph_ir/op_declare/logging_ops_declare.h"
#include <string>

namespace mindspore::transform {
// PrintV2
INPUT_MAP(PrintV2) = {{1, INPUT_DESC(x)}};
ATTR_MAP(PrintV2) = {{"output_stream", ATTR_DESC(output_stream, AnyTraits<std::string>())}};
REG_ADPT_DESC(PrintV2, kNamePrint, ADPT_DESC(PrintV2))

INPUT_MAP(Assert) = {{1, INPUT_DESC(input_condition)}};
DYN_INPUT_MAP(Assert) = {{2, DYN_INPUT_DESC(input_data)}};
ATTR_MAP(Assert) = {{"summarize", ATTR_DESC(summarize, AnyTraits<int64_t>())}};
REG_ADPT_DESC(Assert, kNameAssert, ADPT_DESC(Assert))
}  // namespace mindspore::transform
