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

#include "inc/lu_solve_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"

namespace ge {
// -----------------------LuSolve---------------------------------
IMPLEMT_COMMON_INFERFUNC(LuSolveInferShape) {
  Shape b_shape = op.GetInputDescByName("x").GetShape();
  Shape lu_shape = op.GetInputDescByName("lu_data").GetShape();
  size_t b_dims = b_shape.GetDimNum();
  size_t lu_dims = lu_shape.GetDimNum();
  DataType input_dtype = op.GetInputDescByName("x").GetDataType();
  Format input_format = op.GetInputDescByName("x").GetFormat();
  std::vector<int64_t> dim_vector;
  if (b_dims >= lu_dims) {
    Shape output_shape = b_shape;
    TensorDesc td = op.GetOutputDescByName("y");
    td.SetShape(output_shape);
    td.SetDataType(input_dtype);
    td.SetFormat(input_format);
    (void)op.UpdateOutputDesc("y", td);
    return GRAPH_SUCCESS;
  } else {
    for (size_t i = 0; i <= lu_dims - b_dims - 1; i++) {
      dim_vector.push_back(lu_shape.GetDim(i));
    }
    for (size_t i = 0; i <= b_dims - 1; i++) {
      dim_vector.push_back(b_shape.GetDim(i));
    }
    Shape output_shape(dim_vector);
    TensorDesc td = op.GetOutputDescByName("y");
    td.SetShape(output_shape);
    td.SetDataType(input_dtype);
    td.SetFormat(input_format);
    (void)op.UpdateOutputDesc("y", td);
    return GRAPH_SUCCESS;
  }
}

CUST_IMPLEMT_VERIFIER(LuSolve, LuSolveVerify) {
  DataType input_type_x = op.GetInputDescByName("x").GetDataType();
  DataType input_type_y = op.GetInputDescByName("lu_data").GetDataType();
  if (input_type_x != input_type_y) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(LuSolve, LuSolveInferShape);
CUST_VERIFY_FUNC_REG(LuSolve, LuSolveVerify);
// -----------------------LuSolve END---------------------------------
}  // namespace ge