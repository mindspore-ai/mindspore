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

#ifndef MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_CUST_ELEWISE_CALCULATION_OPS_H_
#define MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_CUST_ELEWISE_CALCULATION_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"
#include "deprecated_graph_ir/custom_op_proto/op_proto_macro.h"

/* clang-format off */

namespace ge {
REG_CUST_OP(ArgMax)
  .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT16,
                        DT_UINT32, DT_UINT64, DT_UINT8}))
  .INPUT(dimension, TensorType({DT_INT32, DT_INT64}))
  .ATTR(dtype, Type, DT_INT64)
  .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
  .CUST_OP_END_FACTORY_REG(ArgMax)
}  // namespace ge
#endif  // MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_CUST_ELEWISE_CALCULATION_OPS_H_