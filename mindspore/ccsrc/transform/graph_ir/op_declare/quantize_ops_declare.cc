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

#include "transform/graph_ir/op_declare/quantize_ops_declare.h"

namespace mindspore::transform {
// AscendQuant
INPUT_MAP(AscendQuant) = {{1, INPUT_DESC(x)}};
ATTR_MAP(AscendQuant) = {{"scale", ATTR_DESC(scale, AnyTraits<float>())},
                         {"offset", ATTR_DESC(offset, AnyTraits<float>())},
                         {"sqrt_mode", ATTR_DESC(sqrt_mode, AnyTraits<bool>())},
                         {"round_mode", ATTR_DESC(round_mode, AnyTraits<std::string>())}};
OUTPUT_MAP(AscendQuant) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(AscendQuant, kNameAscendQuant, ADPT_DESC(AscendQuant))

// AscendDequant
INPUT_MAP(AscendDequant) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(deq_scale)}};
ATTR_MAP(AscendDequant) = {{"sqrt_mode", ATTR_DESC(sqrt_mode, AnyTraits<bool>())},
                           {"relu_flag", ATTR_DESC(relu_flag, AnyTraits<bool>())},
                           {"dtype", ATTR_DESC(dtype, AnyTraits<GEType>())}};
OUTPUT_MAP(AscendDequant) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(AscendDequant, kNameAscendDequant, ADPT_DESC(AscendDequant))
}  // namespace mindspore::transform
