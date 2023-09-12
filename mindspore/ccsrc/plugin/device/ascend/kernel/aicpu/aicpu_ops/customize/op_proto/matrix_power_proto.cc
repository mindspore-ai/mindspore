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

#include "inc/matrix_power_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"

namespace ge {
// ----------------------MatrixPower------------------------
CUST_IMPLEMT_VERIFIER(MatrixPower, MatrixPowerVerify) {
  DataType x_type = op.GetInputDescByName("x").GetDataType();
  Shape x_shape = op.GetInputDescByName("x").GetShape();
  int64_t n;
  if (op.GetAttr("n", n) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "failed to get attribute [n] value.");
    return GRAPH_FAILED;
  }
  if (x_shape.GetDimNum() != 3) {
    OP_LOGE(TbeGetName(op).c_str(), "x should be a 3-D tensor, but got x is a %lu-D tensor.", x_shape.GetDimNum());
    return GRAPH_FAILED;
  }
  if (x_shape.GetDim(1) != x_shape.GetDim(2)) {
    OP_LOGE(TbeGetName(op).c_str(), "sizes of dim1 and dim2 of x should be equal, but got %lu and %lu.",
            x_shape.GetDim(1), x_shape.GetDim(2));
    return GRAPH_FAILED;
  }
  if (x_type != DT_FLOAT && x_type != DT_FLOAT16) {
    OP_LOGE(TbeGetName(op).c_str(), "the data type of x should be DT_FLOAT or DT_FLOAT16.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(MatrixPowerInferShape) {
  DataType x_type = op.GetInputDescByName("x").GetDataType();
  Shape x_shape = op.GetInputDescByName("x").GetShape();
  TensorDesc y_desc = op.GetOutputDescByName("y");
  y_desc.SetDataType(x_type);
  y_desc.SetShape(x_shape);
  (void)op.UpdateOutputDesc("y", y_desc);
  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(MatrixPower, MatrixPowerInferShape);
CUST_VERIFY_FUNC_REG(MatrixPower, MatrixPowerVerify);
// ----------------------MatrixPower END------------------------
}  // namespace ge