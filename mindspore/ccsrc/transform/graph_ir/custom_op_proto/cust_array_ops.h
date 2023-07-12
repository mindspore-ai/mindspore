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

#ifndef MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_CUST_ARRAY_OPS_H_
#define MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_CUST_ARRAY_OPS_H_

#include "transform/graph_ir/custom_op_proto/op_proto_macro.h"

/* clang-format off */

namespace ge {
REG_OP(Meshgrid)
  .DYNAMIC_INPUT(x, TensorType({DT_INT8, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
                                DT_UINT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BOOL}))
  .DYNAMIC_OUTPUT(y, TensorType({DT_INT8, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
                                 DT_UINT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BOOL}))
  .ATTR(indexing, String, "")
  .OP_END_FACTORY_REG(Meshgrid)

REG_CUST_OP(SliceGrad)
  .INPUT(dy, TensorType({DT_BOOL, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64,
                         DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
  .INPUT(x, TensorType({DT_BOOL, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64,
                        DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
  .INPUT(begin, TensorType({DT_INT32, DT_INT32}))
  .INPUT(size, TensorType({DT_INT32, DT_INT32}))
  .OUTPUT(dx, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
  .CUST_OP_END_FACTORY_REG(SliceGrad)

REG_CUST_OP(MaskedSelectGrad)
  .INPUT(x, TensorType({DT_BOOL, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
  .INPUT(mask, TensorType({DT_BOOL}))
  .INPUT(grad, TensorType({DT_BOOL, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
  .OUTPUT(dx, TensorType({DT_BOOL, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
  .CUST_OP_END_FACTORY_REG(MaskedSelectGrad)

REG_CUST_OP(GatherDGradV2)
  .REQUIRED_ATTR(dim, Int)
  .INPUT(x, TensorType({DT_BOOL, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64,
                        DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
  .INPUT(index, TensorType({DT_INT32, DT_INT64}))
  .INPUT(grad, TensorType({DT_BOOL, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64,
                           DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
  .OUTPUT(output, TensorType({DT_BOOL, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64,
                              DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
  .CUST_OP_END_FACTORY_REG(GatherDGradV2)
}  // namespace ge
#endif  // MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_CUST_ARRAY_OPS_H_
