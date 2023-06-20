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

#include "inc/trace_grad_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"

namespace ge {
// ----------------TraceGrad Begin------------------------
IMPLEMT_COMMON_INFERFUNC(TraceGradInferShape) {
  Shape shape = op.GetInputDescByName("y_grad").GetShape();
  DataType input_dtype = op.GetInputDescByName("y_grad").GetDataType();
  Tensor tensor_input;
  std::vector<int64_t> dim_vector;
  if (op.GetInputConstData("x_shape", tensor_input) == GRAPH_SUCCESS) {
    uint8_t *input_shape = tensor_input.GetData();
    for (int64_t i = 0; i < 2; i++) {
      dim_vector.push_back(shape.GetDim(*(input_shape + i)));
    }
  }
  Shape output_shape(dim_vector);
  TensorDesc td = op.GetOutputDescByName("x_grad");
  td.SetShape(output_shape);
  td.SetDataType(input_dtype);
  td.SetFormat(FORMAT_ND);
  (void)op.UpdateOutputDesc("x_grad", td);
  return GRAPH_SUCCESS;
}
CUST_IMPLEMT_VERIFIER(TraceGrad, TraceGradVerify) {
  DataType x_shape_dtype = op.GetInputDescByName("x_shape").GetDataType();
  if ((x_shape_dtype != DT_INT32) || (x_shape_dtype != DT_INT64)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(TraceGrad, TraceGradInferShape);
CUST_VERIFY_FUNC_REG(TraceGrad, TraceGradVerify);
// ---------------TraceGrad END-------------------------------
}  // namespace ge