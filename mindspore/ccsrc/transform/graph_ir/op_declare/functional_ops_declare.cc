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

#include "transform/graph_ir/op_declare/functional_ops_declare.h"

namespace mindspore::transform {
// Case
INPUT_MAP(Case) = {{1, INPUT_DESC(branch_index)}};
DYN_INPUT_MAP(Case) = {{2, DYN_INPUT_DESC(input)}};
ATTR_MAP(Case) = EMPTY_ATTR_MAP;
DYN_OUTPUT_MAP(Case) = {{0, DYN_OUTPUT_DESC(output)}};
DYN_SUBGRAPH_MAP(Case) = {{0, DYN_SUBGRAPH_DESC(branches)}};
REG_ADPT_DESC(Case, kNameCase, ADPT_DESC(Case));

// While
DYN_INPUT_MAP(While) = {{1, DYN_INPUT_DESC(input)}};
ATTR_MAP(While) = {{"parallel_iterations", ATTR_DESC(parallel_iterations, AnyTraits<int32_t>())}};
DYN_OUTPUT_MAP(While) = {{0, DYN_OUTPUT_DESC(output)}};
SUBGRAPH_MAP(While) = {{0, SUBGRAPH_DESC(cond)}, {1, SUBGRAPH_DESC(body)}};
REG_ADPT_DESC(While, kNameWhile, ADPT_DESC(While));
}  // namespace mindspore::transform
