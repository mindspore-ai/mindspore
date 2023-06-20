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

#include "inc/cholesky_solve_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
namespace ge {
// ----------------------CholeskySlove Starts----------------------
CUST_IMPLEMT_VERIFIER(CholeskySolve, CholeskySolveVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_IMPLEMT_INFERFUNC(CholeskySolve, CholeskySolveInfer) {
  DataType x1_type = op.GetInputDescByName("x1").GetDataType();
  Shape x1_shape = op.GetInputDescByName("x1").GetShape();
  Shape x2_shape = op.GetInputDescByName("x2").GetShape();
  if (x1_shape.GetDimNum() != 2 && x1_shape.GetDimNum() != 3) {
    // OP_LOGE(TbeGetName(op), "Op CholeskySolve's inputs must be rank 2 or 3.");
    return GRAPH_FAILED;
  }
  if (x1_shape.GetDimNum() != x2_shape.GetDimNum()) {
    // OP_LOGE(TbeGetName(op), "Op CholeskySolve's two inputs cannot match.");
    return GRAPH_FAILED;
  }
  size_t rank = x1_shape.GetDimNum();
  if (rank == 2) {
    if (x1_shape.GetDim(0) != x2_shape.GetDim(0)) {
      // OP_LOGE(TbeGetName(op), "Op CholeskySolve's two inputs cannot match.");
      return GRAPH_FAILED;
    }
    if (x2_shape.GetDim(0) != x2_shape.GetDim(1)) {
      // OP_LOGE(TbeGetName(op), "Op CholeskySolve's second input should be batch square.");
      return GRAPH_FAILED;
    }
  } else {
    if (x1_shape.GetDim(0) != x2_shape.GetDim(0) || x1_shape.GetDim(1) != x2_shape.GetDim(1)) {
      // OP_LOGE(TbeGetName(op), "Op CholeskySolve's two inputs cannot match.");
      return GRAPH_FAILED;
    }
    if (x2_shape.GetDim(1) != x2_shape.GetDim(2)) {
      // OP_LOGE(TbeGetName(op), "Op CholeskySolve's second input should be batch square.");
      return GRAPH_FAILED;
    }
  }
  TensorDesc y_desc = op.GetOutputDescByName("y");
  y_desc.SetShape(x1_shape);
  y_desc.SetDataType(x1_type);
  op.UpdateOutputDesc("y", y_desc);
  return GRAPH_SUCCESS;
}

// Registered inferfunction
CUST_INFER_FUNC_REG(CholeskySolve, CholeskySolveInfer);
// Registered verify function
CUST_VERIFY_FUNC_REG(CholeskySolve, CholeskySolveVerify);
// ----------------------CholeskySolve End----------------------
}  // namespace ge