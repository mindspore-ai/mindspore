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

#include "inc/csr_sparse_matrix_to_dense_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"

namespace ge {
CUST_IMPLEMT_VERIFIER(CSRSparseMatrixToDense, CSRSparseMatrixToDenseVerify) {
  string err_msg;
  DataType value_type_x = op.GetInputDescByName("x_values").GetDataType();
  DataType shape_type_x = op.GetInputDescByName("x_dense_shape").GetDataType();
  DataType batch_type_x = op.GetInputDescByName("x_batch_pointers").GetDataType();
  DataType row_type_x = op.GetInputDescByName("x_row_pointers").GetDataType();
  DataType col_type_x = op.GetInputDescByName("x_col_indices").GetDataType();
  if (shape_type_x != batch_type_x || batch_type_x != row_type_x || row_type_x != col_type_x) {
    err_msg = "Data type of batch | shape | row | col is not the same!";
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  bool validValueType = (value_type_x != DT_FLOAT) && (value_type_x != DT_DOUBLE) && (value_type_x != DT_COMPLEX64) &&
                        (value_type_x != DT_COMPLEX128);
  bool validShapeType = (shape_type_x != DT_INT32) && (shape_type_x != DT_INT64);
  bool validRowType = (row_type_x != DT_INT32) && (row_type_x != DT_INT64);
  bool validColType = (col_type_x != DT_INT32) && (col_type_x != DT_INT64);
  if (validValueType || validShapeType || validShapeType || validRowType || validColType) {
    err_msg = "Data type of some input is wrong!";
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
CUST_VERIFY_FUNC_REG(CSRSparseMatrixToDense, CSRSparseMatrixToDenseVerify);

CUST_IMPLEMT_INFERFUNC(CSRSparseMatrixToDense, CSRSparseMatrixToDenseInfer) {
  std::string err_msg;
  Shape x_dense_shape_shape = op.GetInputDescByName("x_dense_shape").GetShape();
  DataType value_type_x = op.GetInputDescByName("x_values").GetDataType();
  const int rank_x = x_dense_shape_shape.GetDim(0);
  if ((rank_x != 2 && rank_x != 3)) {
    err_msg = "Dense rank of input should be 2 or 3, please check!";
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  TensorDesc y_desc = op.GetOutputDescByName("y");
  y_desc.SetDataType(value_type_x);
  std::vector<int64_t> newDims;
  if (rank_x == 2) {
    newDims.push_back(ge::UNKNOWN_DIM);
    newDims.push_back(ge::UNKNOWN_DIM);
    y_desc.SetShape(Shape(newDims));
  } else {
    newDims.push_back(ge::UNKNOWN_DIM);
    newDims.push_back(ge::UNKNOWN_DIM);
    newDims.push_back(ge::UNKNOWN_DIM);
    y_desc.SetShape(Shape(newDims));
  }
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    err_msg = "CSRSparseMatrixToDense failed to update output y.";
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
CUST_INFER_FUNC_REG(CSRSparseMatrixToDense, CSRSparseMatrixToDenseInfer);
}  // namespace ge