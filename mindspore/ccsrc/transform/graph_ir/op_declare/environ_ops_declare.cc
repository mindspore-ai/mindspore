/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#include "transform/graph_ir/op_declare/environ_ops_declare.h"
#include <string>
#include <vector>
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/other_ops.h"
#include "mindspore/core/ops/structure_ops.h"

namespace mindspore::transform {
// EnvironCreate
CUST_INPUT_MAP(EnvironCreate) = EMPTY_INPUT_MAP;
CUST_ATTR_MAP(EnvironCreate) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(EnvironCreate) = {{0, OUTPUT_DESC(handle)}};
REG_ADPT_DESC(EnvironCreate, kNameEnvironCreate, CUST_ADPT_DESC(EnvironCreate))

// EnvironDestroyAll
CUST_INPUT_MAP(EnvironDestroyAll) = EMPTY_INPUT_MAP;
CUST_ATTR_MAP(EnvironDestroyAll) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(EnvironDestroyAll) = {{0, OUTPUT_DESC(result)}};
REG_ADPT_DESC(EnvironDestroyAll, kNameEnvironDestroyAll, CUST_ADPT_DESC(EnvironDestroyAll))

// EnvironGet
CUST_INPUT_MAP(EnvironGet) = {{1, INPUT_DESC(env)}, {2, INPUT_DESC(key)}, {3, INPUT_DESC(default)}};
CUST_ATTR_MAP(EnvironGet) = {{"value_type", ATTR_DESC(value_type, AnyTraits<int32_t>())}};
CUST_OUTPUT_MAP(EnvironGet) = {{0, OUTPUT_DESC(value)}};
REG_ADPT_DESC(EnvironGet, kNameEnvironGet, CUST_ADPT_DESC(EnvironGet))

// EnvironSet
CUST_INPUT_MAP(EnvironSet) = {{1, INPUT_DESC(env)}, {2, INPUT_DESC(key)}, {3, INPUT_DESC(value)}};
CUST_ATTR_MAP(EnvironSet) = {{"value_type", ATTR_DESC(value_type, AnyTraits<int32_t>())}};
CUST_OUTPUT_MAP(EnvironSet) = {{0, OUTPUT_DESC(env)}};
REG_ADPT_DESC(EnvironSet, kNameEnvironSet, CUST_ADPT_DESC(EnvironSet))
}  // namespace mindspore::transform
