/**
 * Copyright (c) 2023 Huawei Technologies Co., Ltd.  All rights reserved.
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

#ifndef MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_CUST_ENVIRON_OPS_H_
#define MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_CUST_ENVIRON_OPS_H_

#include "transform/graph_ir/custom_op_proto/op_proto_macro.h"

/* clang-format off */

namespace ge {
REG_CUST_OP(EnvironCreate)
  .OUTPUT(handle, TensorType({DT_INT64}))
  .CUST_OP_END_FACTORY_REG(EnvironCreate)

REG_CUST_OP(EnvironDestroyAll)
  .OUTPUT(result, TensorType({DT_BOOL}))
  .CUST_OP_END_FACTORY_REG(EnvironDestroyAll)

REG_CUST_OP(EnvironSet)
  .REQUIRED_ATTR(value_type, Int)
  .INPUT(env, TensorType({DT_INT64}))
  .INPUT(key, TensorType({DT_BOOL}))
  .INPUT(value, TensorType({DT_BOOL, DT_INT16, DT_INT32, DT_INT64,
                            DT_UINT8, DT_UINT32, DT_UINT64, DT_FLOAT16, DT_FLOAT}))
  .OUTPUT(env, TensorType({DT_BOOL}))
  .CUST_OP_END_FACTORY_REG(EnvironSet)

REG_CUST_OP(EnvironGet)
  .REQUIRED_ATTR(value_type, Int)
  .INPUT(env, TensorType({DT_INT64}))
  .INPUT(key, TensorType({DT_BOOL}))
  .INPUT(default, TensorType({DT_BOOL, DT_INT16, DT_INT32, DT_INT64,
                              DT_UINT8, DT_UINT32, DT_UINT64, DT_FLOAT16, DT_FLOAT}))
  .OUTPUT(value, TensorType({DT_BOOL, DT_INT16, DT_INT32, DT_INT64,
                             DT_UINT8, DT_UINT32, DT_UINT64, DT_FLOAT16, DT_FLOAT}))
  .CUST_OP_END_FACTORY_REG(EnvironGet)
}  // namespace ge
#endif  // MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_CUST_ENVIRON_OPS_H_
