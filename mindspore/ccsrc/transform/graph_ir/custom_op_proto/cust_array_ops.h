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
REG_CUST_OP(Meshgrid)
  .DYNAMIC_INPUT(x, TensorType({DT_INT8, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
                                DT_UINT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BOOL}))
  .DYNAMIC_OUTPUT(y, TensorType({DT_INT8, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
                                 DT_UINT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BOOL}))
  .ATTR(indexing, String, "")
  .CUST_OP_END_FACTORY_REG(Meshgrid)

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

REG_CUST_OP(AffineGridGrad)
  .INPUT(y_grad, TensorType({DT_FLOAT, DT_FLOAT16}))
  .INPUT(x_size, TensorType({DT_INT32, DT_INT64}))
  .OUTPUT(x_grad, TensorType({DT_FLOAT, DT_FLOAT16}))
  .REQUIRED_ATTR(align_corners, Bool)
  .CUST_OP_END_FACTORY_REG(AffineGridGrad)
REG_CUST_OP(HammingWindow)
  .INPUT(length, TensorType({DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8}))
  .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
  .REQUIRED_ATTR(periodic, Bool)
  .REQUIRED_ATTR(alpha, Float)
  .REQUIRED_ATTR(beta, Float)
  .REQUIRED_ATTR(dtype, Int)
  .CUST_OP_END_FACTORY_REG(HammingWindow)

REG_CUST_OP(IndexFill)
  .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT16,
                        DT_UINT32, DT_UINT64, DT_UINT8}))
  .INPUT(dim, TensorType({DT_INT32}))
  .INPUT(indices, TensorType({DT_INT32}))
  .INPUT(value, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT16,
                            DT_UINT32, DT_UINT64, DT_UINT8}))
  .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT16,
                         DT_UINT32, DT_UINT64, DT_UINT8}))
  .CUST_OP_END_FACTORY_REG(IndexFill)

REG_CUST_OP(Mvlgamma)
  .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT}))
  .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT}))
  .REQUIRED_ATTR(p, Int)
  .CUST_OP_END_FACTORY_REG(Mvlgamma)

REG_CUST_OP(MvlgammaGrad)
  .INPUT(y_grad, TensorType({DT_DOUBLE, DT_FLOAT}))
  .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT}))
  .OUTPUT(x_grad, TensorType({DT_DOUBLE, DT_FLOAT}))
  .REQUIRED_ATTR(p, Int)
  .CUST_OP_END_FACTORY_REG(MvlgammaGrad)

REG_CUST_OP(LogSpace)
  .INPUT(start, TensorType({DT_FLOAT, DT_FLOAT16}))
  .INPUT(end, TensorType({DT_FLOAT, DT_FLOAT16}))
  .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
  .REQUIRED_ATTR(steps, Int)
  .REQUIRED_ATTR(base, Int)
  .CUST_OP_END_FACTORY_REG(LogSpace)

REG_CUST_OP(Expand)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT8, DT_BOOL}))
    .INPUT(shape, TensorType({DT_INT16, DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT8, DT_BOOL}))
    .CUST_OP_END_FACTORY_REG(Expand)
}  // namespace ge
#endif  // MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_CUST_ARRAY_OPS_H_
