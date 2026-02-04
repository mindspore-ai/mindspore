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

#include "plugin/ascend/res_manager/op_adapter/op_declare/state_ops_declare.h"

namespace mindspore::device::ascend {
// Variable
INPUT_MAP(Variable) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Variable) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Variable) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Variable, "Variable", ADPT_DESC(Variable))
}  // namespace mindspore::device::ascend
