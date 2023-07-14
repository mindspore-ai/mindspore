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

#ifndef MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_CUST_NN_OPS_H_
#define MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_CUST_NN_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"
#include "transform/graph_ir/custom_op_proto/op_proto_macro.h"

/* clang-format off */

namespace ge {
REG_CUST_OP(AdaptiveAvgPool3dGrad)
  .INPUT(input_grad, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT8}))
  .INPUT(orig_input_shape, TensorType({DT_INT32}))
  .OUTPUT(output_grad, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT8}))
  .CUST_OP_END_FACTORY_REG(AdaptiveAvgPool3dGrad)

REG_CUST_OP(AdaptiveMaxPool2dGrad)
  .INPUT(y_grad, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
  .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
  .INPUT(argmax, TensorType({DT_INT32, DT_INT64}))
  .OUTPUT(x_grad, TensorType({DT_FLOAT, DT_FLOAT16}))
  .CUST_OP_END_FACTORY_REG(AdaptiveMaxPool2dGrad)

REG_CUST_OP(AdaptiveAvgPool2D)
  .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
  .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
  .REQUIRED_ATTR(output_size, ListInt)
  .CUST_OP_END_FACTORY_REG(AdaptiveAvgPool2D)

REG_CUST_OP(AdaptiveAvgPool3d)
  .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT8}))
  .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT8}))
  .REQUIRED_ATTR(output_size, ListInt)
  .CUST_OP_END_FACTORY_REG(AdaptiveAvgPool3d)

REG_CUST_OP(AdaptiveAvgPool2DGrad)
  .INPUT(input_grad, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
  .INPUT(orig_input_shape, TensorType({DT_INT64}))
  .OUTPUT(output_grad, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
  .CUST_OP_END_FACTORY_REG(AdaptiveAvgPool2DGrad)

REG_CUST_OP(AdaptiveMaxPool3d)
  .INPUT(x, TensorType::RealNumberType())
  .INPUT(output_size, TensorType({DT_INT32}))
  .OUTPUT(y, TensorType::RealNumberType())
  .OUTPUT(argmax, TensorType({DT_INT32}))
  .CUST_OP_END_FACTORY_REG(AdaptiveMaxPool3d)

REG_CUST_OP(AdaptiveMaxPool3dGrad)
  .INPUT(input_grad, TensorType::RealNumberType())
  .INPUT(x, TensorType::RealNumberType())
  .INPUT(argmax, TensorType({DT_INT32}))
  .OUTPUT(output_grad, TensorType::RealNumberType())
  .CUST_OP_END_FACTORY_REG(AdaptiveMaxPool3dGrad)

REG_CUST_OP(MultiMarginLossGrad)
  .INPUT(y_grad, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
  .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
  .INPUT(target, TensorType({DT_INT64}))
  .INPUT(weight, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
  .OUTPUT(x_grad, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
  .REQUIRED_ATTR(p, Int)
  .REQUIRED_ATTR(margin, Float)
  .REQUIRED_ATTR(reduction, String)
  .CUST_OP_END_FACTORY_REG(MultiMarginLossGrad)

REG_CUST_OP(MaxPool3DGradWithArgmax)
  .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT16,
                        DT_UINT32, DT_UINT64, DT_UINT8}))
  .INPUT(grads, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT16,
                            DT_UINT32, DT_UINT64, DT_UINT8}))
  .INPUT(argmax, TensorType({DT_INT32, DT_INT64}))
  .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT16,
                         DT_UINT32, DT_UINT64, DT_UINT8}))
  .REQUIRED_ATTR(ksize, ListInt)
  .REQUIRED_ATTR(strides, ListInt)
  .REQUIRED_ATTR(pads, ListInt)
  .REQUIRED_ATTR(dilation, ListInt)
  .REQUIRED_ATTR(ceil_mode, Bool)
  .REQUIRED_ATTR(data_format, String)
  .REQUIRED_ATTR(argmax_type, String)
  .CUST_OP_END_FACTORY_REG(MaxPool3DGradWithArgmax)

REG_CUST_OP(MultiMarginLoss)
  .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
  .INPUT(target, TensorType({DT_INT64}))
  .INPUT(weight, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
  .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
  .REQUIRED_ATTR(p, Int)
  .REQUIRED_ATTR(margin, Float)
  .REQUIRED_ATTR(reduction, String)
  .CUST_OP_END_FACTORY_REG(MultiMarginLoss)
}  // namespace ge
#endif  // MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_CUST_NN_OPS_H_

