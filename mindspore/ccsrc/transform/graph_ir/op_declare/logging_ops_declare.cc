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

namespace mindspore::transform {
#ifdef ENABLE_GE
// Print
INPUT_MAP(Print) = EMPTY_INPUT_MAP;
DYN_INPUT_MAP(Print) = {{1, DYN_INPUT_DESC(x)}};
ATTR_MAP(Print) = EMPTY_ATTR_MAP;
REG_ADPT_DESC(Print, kNamePrint, ADPT_DESC(Print))
#endif
}  // namespace mindspore::transform
