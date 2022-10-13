/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_FUNCTIONAL_OPS_DECLARE_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_FUNCTIONAL_OPS_DECLARE_H_

#include "utils/hash_map.h"
#include "transform/graph_ir/op_declare/op_declare_macro.h"
#include "ops/functional_ops.h"

namespace mindspore::transform {
DECLARE_OP_ADAPTER(Case)
DECLARE_OP_USE_DYN_INPUT(Case)
DECLARE_OP_USE_DYN_SUBGRAPH(Case)
DECLARE_OP_USE_DYN_OUTPUT(Case)

DECLARE_OP_TYPE(While)
DECLARE_OP_ATTR(While)
DECLARE_OP_USE_DYN_INPUT(While)
DECLARE_OP_USE_SUBGRAPH(While)
DECLARE_OP_USE_DYN_OUTPUT(While)

DECLARE_OP_ADAPTER(PartitionedCall)
DECLARE_OP_USE_DYN_INPUT(PartitionedCall)
DECLARE_OP_USE_SUBGRAPH(PartitionedCall)
DECLARE_OP_USE_DYN_OUTPUT(PartitionedCall)

DECLARE_OP_ADAPTER(If)
DECLARE_OP_USE_DYN_INPUT(If)
DECLARE_OP_USE_SUBGRAPH(If)
DECLARE_OP_USE_DYN_OUTPUT(If)
}  // namespace mindspore::transform
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_FUNCTIONAL_OPS_DECLARE_H_
