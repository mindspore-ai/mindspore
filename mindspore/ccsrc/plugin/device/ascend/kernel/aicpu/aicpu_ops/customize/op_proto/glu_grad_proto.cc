/**
 * Copyright (c) 2022-2022 Huawei Technologies Co., Ltd.  All rights reserved.
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

#include "inc/glu_grad_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"

namespace ge {
// ------------------GluGrad Begin--------------------
IMPLEMT_COMMON_INFERFUNC(GluGradInferShape) {
  TensorDesc x = op.GetInputDescByName("x");
  Shape x_shape = x.GetShape();
  std::vector<int64_t> dim_vector = x_shape.GetDims();
  Shape output_shape(dim_vector);
  TensorDesc out = op.GetOutputDescByName("y");
  out.SetShape(output_shape);
  op.UpdateOutputDesc("y", out);
  out.SetDataType(op.GetInputDescByName("x").GetDataType());
  if (op.UpdateOutputDesc("y", out) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_IMPLEMT_VERIFIER(GluGrad, GluGradVerify) {
  DataType input_type_grad = op.GetInputDescByName("grads").GetDataType();
  DataType input_type_x = op.GetInputDescByName("x").GetDataType();
  if (input_type_grad != input_type_x) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(GluGrad, GluGradInferShape);
CUST_VERIFY_FUNC_REG(GluGrad, GluGradVerify);
// ------------------GluGrad END--------------------
}  // namespace ge