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

#include "inc/sparse_addmm.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"
namespace ge {
// -------------------------------SparseAddmm Begin------------------------------- //
CUST_IMPLEMT_INFERFUNC(SparseAddmm, SparseAddmmInfer) {
  int64_t unused_dim = 0;
  Shape unused_shape;
  GeShape x1_shape;
  Shape x2_shape;
  auto x1_indices_tensor = op.get_input_desc_x1_indices();
  auto x1_values_tensor = op.get_input_desc_x1_values();
  auto x1_shape_tensor = op.get_input_desc_x1_shape();
  auto x2_tensor = op.get_input_desc_x2();
  std::string err_msg;
  if (WithRank(x1_indices_tensor, 2, unused_shape, op) != GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(0, DebugString(x1_indices_tensor.GetShape().GetDims()), "2D");
    err_msg = string("failed to call WithRank function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(x1_values_tensor, 1, unused_shape, op) != GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(1, DebugString(x1_values_tensor.GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (MakeShapeFromShapeTensor(op, "x1_shape", x1_shape) != GRAPH_SUCCESS) {
    err_msg = ConcatString(
      "failed to call MakeShapeFromShapeTensor function to make shape from "
      "input[x1_shape]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRankShape(x1_shape, 2, op) != GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(2, DebugString(x1_shape.GetDims()), "2D");
    err_msg = string("failed to call WithRank function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(x2_tensor, 2, x2_shape, op) != GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(3, DebugString(x2_tensor.GetShape().GetDims()), "2D");
    err_msg = string("failed to call WithRank function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  int64_t output_right_shape = x2_shape.GetDim(1);
  int64_t output_left_shape = x1_shape.GetDim(0);
  int64_t inner_left_shape = x1_shape.GetDim(1);
  int64_t inner_right_shape = x2_shape.GetDim(0);
  graphStatus status = Merge(inner_left_shape, inner_right_shape, unused_dim);
  if (status != GRAPH_SUCCESS) {
    err_msg = ConcatString("failed to call Merge function to merge ", inner_left_shape, " and ", inner_right_shape);
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  Shape output_shape;
  status = Matrix(output_left_shape, output_right_shape, output_shape);
  if (status != GRAPH_SUCCESS) {
    err_msg =
      ConcatString("failed to call Matrix function to create matrix ", output_left_shape, " and ", output_right_shape);
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  TensorDesc y_tensor = op.GetOutputDescByName("y");
  y_tensor.SetDataType(x1_values_tensor.GetDataType());
  y_tensor.SetShape(output_shape);
  return op.UpdateOutputDesc("y", y_tensor);
}

CUST_INFER_FUNC_REG(SparseAddmm, SparseAddmmInfer);

// -------------------------------SparseAddmm End------------------------------- //
}  // namespace ge