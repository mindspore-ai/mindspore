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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_REDUCE_OPS_DECLARE_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_REDUCE_OPS_DECLARE_H_

#include "utils/hash_map.h"
#include "transform/graph_ir/op_declare/op_declare_macro.h"
#include "ops/reduce_ops.h"

namespace mindspore::transform {
DECLARE_OP_ADAPTER(ReduceMean)
DECLARE_OP_USE_OUTPUT(ReduceMean)

DECLARE_OP_ADAPTER(ReduceMinD)
DECLARE_OP_USE_INPUT_ATTR(ReduceMinD)
DECLARE_OP_USE_OUTPUT(ReduceMinD)

DECLARE_OP_ADAPTER(ReduceMaxD)
DECLARE_OP_USE_INPUT_ATTR(ReduceMaxD)
DECLARE_OP_USE_OUTPUT(ReduceMaxD)

DECLARE_OP_ADAPTER(ReduceMax)
DECLARE_OP_USE_OUTPUT(ReduceMax)

DECLARE_OP_ADAPTER(ReduceAllD)
DECLARE_OP_USE_INPUT_ATTR(ReduceAllD)
DECLARE_OP_USE_OUTPUT(ReduceAllD)

DECLARE_OP_ADAPTER(BNTrainingReduce)
DECLARE_OP_USE_OUTPUT(BNTrainingReduce)

DECLARE_OP_ADAPTER(BNTrainingReduceGrad)
DECLARE_OP_USE_OUTPUT(BNTrainingReduceGrad)

DECLARE_OP_ADAPTER(BNTrainingUpdate)
DECLARE_OP_USE_OUTPUT(BNTrainingUpdate)

DECLARE_OP_ADAPTER(BNTrainingUpdateGrad)
DECLARE_OP_USE_OUTPUT(BNTrainingUpdateGrad)

DECLARE_OP_ADAPTER(ReduceSumD)
DECLARE_OP_USE_INPUT_ATTR(ReduceSumD)
DECLARE_OP_USE_OUTPUT(ReduceSumD)

DECLARE_OP_ADAPTER(ReduceSum)
DECLARE_OP_USE_OUTPUT(ReduceSum)

DECLARE_OP_ADAPTER(ReduceAnyD)
DECLARE_OP_USE_INPUT_ATTR(ReduceAnyD)
DECLARE_OP_USE_OUTPUT(ReduceAnyD)

DECLARE_OP_ADAPTER(ReduceProdD)
DECLARE_OP_USE_INPUT_ATTR(ReduceProdD)
DECLARE_OP_USE_OUTPUT(ReduceProdD)
}  // namespace mindspore::transform
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_REDUCE_OPS_DECLARE_H_
