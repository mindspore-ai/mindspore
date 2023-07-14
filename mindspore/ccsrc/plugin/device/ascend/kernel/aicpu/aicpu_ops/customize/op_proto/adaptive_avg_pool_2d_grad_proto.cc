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

#include "custom_op_proto/cust_nn_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"

namespace ge {
CUST_IMPLEMT_INFERFUNC(AdaptiveAvgPool2DGrad, AdaptiveAvgPool2dGradInferShape) {
  std::vector<std::string> input_infer_depends = {"orig_input_shape"};
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  op_desc->SetOpInferDepends(input_infer_depends);
  DataType input_dtype = op.GetInputDescByName("input_grad").GetDataType();
  Shape output_shape;
  Tensor orig_input_shape_tensor;
  if (op.GetInputConstData("orig_input_shape", orig_input_shape_tensor) != GRAPH_SUCCESS) {
    auto output_desc = op.GetOutputDescByName("output_grad");
    output_desc.SetDataType(input_dtype);
    output_desc.SetShape(Shape(ge::UNKNOWN_RANK));
    return op.UpdateOutputDesc("output_grad", output_desc);
  }
  MakeShapeFromShapeTensor(orig_input_shape_tensor, output_shape, op);
  TensorDesc output_grad = op.GetOutputDescByName("output_grad");
  output_grad.SetShape(output_shape);
  output_grad.SetDataType(input_dtype);
  return op.UpdateOutputDesc("output_grad", output_grad);
}

CUST_INFER_FUNC_REG(AdaptiveAvgPool2DGrad, AdaptiveAvgPool2dGradInferShape);
}  // namespace ge