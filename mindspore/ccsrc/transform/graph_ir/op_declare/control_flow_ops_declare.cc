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

#include "transform/graph_ir/op_declare/control_flow_ops_declare.h"

namespace mindspore::transform {
// Merge
INPUT_MAP(Merge) = EMPTY_INPUT_MAP;
DYN_INPUT_MAP(Merge) = {{1, DYN_INPUT_DESC(x)}};
ATTR_MAP(Merge) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Merge) = {{0, OUTPUT_DESC(y)}, {1, OUTPUT_DESC(value_index)}};
REG_ADPT_DESC(Merge, kNameMerge, ADPT_DESC(Merge))

// Switch
INPUT_MAP(Switch) = {{1, INPUT_DESC(data)}, {2, INPUT_DESC(pred)}};
OUTPUT_MAP(Switch) = {{0, OUTPUT_DESC(output_false)}, {1, OUTPUT_DESC(output_true)}};
ATTR_MAP(Switch) = EMPTY_ATTR_MAP;
REG_ADPT_DESC(Switch, kNameGeSwitch, ADPT_DESC(Switch))
}  // namespace mindspore::transform
