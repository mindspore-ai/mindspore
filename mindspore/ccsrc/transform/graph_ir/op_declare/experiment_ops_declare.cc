/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "transform/graph_ir/op_declare/experiment_ops_declare.h"
#include <vector>
#include <string>
namespace mindspore::transform {

// GeGluV2
INPUT_MAP(GeGluV2) = {{1, INPUT_DESC(x)}};
ATTR_MAP(GeGluV2) = {{"dim", ATTR_DESC(dim, AnyTraits<int64_t>())},
                     {"approximate", ATTR_DESC(approximate, AnyTraits<int64_t>())},
                     {"activate_left", ATTR_DESC(activate_left, AnyTraits<bool>())}};
OUTPUT_MAP(GeGluV2) = {{0, OUTPUT_DESC(y)}, {1, OUTPUT_DESC(gelu)}};
REG_ADPT_DESC(GeGluV2, "GeGluV2", ADPT_DESC(GeGluV2))
}  // namespace mindspore::transform
