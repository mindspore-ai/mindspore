/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "transform/graph_ir/op_declare/nn_quantize_declare.h"
#include <string>

namespace mindspore::transform {

// TransQuantParamV2
INPUT_MAP(TransQuantParamV2) = {{1, INPUT_DESC(scale)}, {2, INPUT_DESC(offset)}};
ATTR_MAP(TransQuantParamV2) = EMPTY_ATTR_MAP;
OUTPUT_MAP(TransQuantParamV2) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(TransQuantParamV2, kNameTransQuantParamV2, ADPT_DESC(TransQuantParamV2))

}  // namespace mindspore::transform
