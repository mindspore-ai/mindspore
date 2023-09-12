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

#include "inc/sparse_matrix_transpose_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"

namespace ge {
CUST_IMPLEMT_INFERFUNC(SparseMatrixTranspose, SparseMatrixTransposeInfer) {
  std::string err_msg;
  TensorDesc x_dense_shape_desc = op.GetInputDescByName("x_dense_shape");
  Format input_format = x_dense_shape_desc.GetFormat();
  TensorDesc x_batch_pointers_desc = op.GetInputDescByName("x_batch_pointers");
  TensorDesc x_row_pointers_desc = op.GetInputDescByName("x_row_pointers");
  TensorDesc x_col_indices_desc = op.GetInputDescByName("x_col_indices");
  TensorDesc x_values_desc = op.GetInputDescByName("x_values");
  TensorDesc y_dense_shape_desc = op.GetOutputDescByName("y_dense_shape");
  TensorDesc y_batch_pointers_desc = op.GetOutputDescByName("y_batch_pointers");
  TensorDesc y_row_pointers_desc = op.GetOutputDescByName("y_row_pointers");
  TensorDesc y_values_desc = op.GetOutputDescByName("y_values");
  TensorDesc y_col_indices_desc = op.GetOutputDescByName("y_col_indices");
  y_dense_shape_desc.SetShape(x_dense_shape_desc.GetShape());
  y_dense_shape_desc.SetDataType(x_dense_shape_desc.GetDataType());
  y_dense_shape_desc.SetFormat(input_format);
  y_batch_pointers_desc.SetShape(x_batch_pointers_desc.GetShape());
  y_batch_pointers_desc.SetDataType(x_batch_pointers_desc.GetDataType());
  y_batch_pointers_desc.SetFormat(input_format);
  y_row_pointers_desc.SetShape(Shape({ge::UNKNOWN_DIM}));
  y_row_pointers_desc.SetDataType(x_row_pointers_desc.GetDataType());
  y_row_pointers_desc.SetFormat(input_format);
  y_col_indices_desc.SetShape(x_col_indices_desc.GetShape());
  y_col_indices_desc.SetDataType(x_col_indices_desc.GetDataType());
  y_col_indices_desc.SetFormat(input_format);
  y_values_desc.SetShape(x_values_desc.GetShape());
  y_values_desc.SetDataType(x_values_desc.GetDataType());
  y_values_desc.SetFormat(input_format);
  if (op.UpdateOutputDesc("y_dense_shape", y_dense_shape_desc) != GRAPH_SUCCESS) {
    err_msg = "fail to update output y_dense_shape";
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (op.UpdateOutputDesc("y_batch_pointers", y_batch_pointers_desc) != GRAPH_SUCCESS) {
    err_msg = "fail to update output y_batch_pointers";
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (op.UpdateOutputDesc("y_row_pointers", y_row_pointers_desc) != GRAPH_SUCCESS) {
    err_msg = "fail to update output y_row_pointers_desc";
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (op.UpdateOutputDesc("y_col_indices", y_col_indices_desc) != GRAPH_SUCCESS) {
    err_msg = "fail to update output y_col_indices_desc";
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (op.UpdateOutputDesc("y_values", y_values_desc) != GRAPH_SUCCESS) {
    err_msg = "fail to update output y_values_desc";
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_IMPLEMT_VERIFIER(SparseMatrixTranspose, SparseMatrixTransposeVerify) {
  string err_msg;
  bool conjugate = false;
  op.GetAttr("conjugate", conjugate);
  DataType shape_type_x = op.GetInputDescByName("x_dense_shape").GetDataType();
  DataType batch_type_x = op.GetInputDescByName("x_batch_pointers").GetDataType();
  DataType row_type_x = op.GetInputDescByName("x_row_pointers").GetDataType();
  DataType col_type_x = op.GetInputDescByName("x_col_indices").GetDataType();
  DataType value_type_x = op.GetInputDescByName("x_values").GetDataType();
  bool validValueType = (value_type_x != DT_FLOAT) && (value_type_x != DT_DOUBLE) && (value_type_x != DT_COMPLEX64) &&
                        (value_type_x != DT_COMPLEX128) && (value_type_x != DT_INT8) && (value_type_x != DT_UINT8) &&
                        (value_type_x != DT_INT16) && (value_type_x != DT_UINT16) && (value_type_x != DT_INT32) &&
                        (value_type_x != DT_UINT32) && (value_type_x != DT_INT64) && (value_type_x != DT_UINT64);
  bool validShapeType = (shape_type_x != DT_INT32) && (shape_type_x != DT_INT64);
  bool validBatchType = (batch_type_x != DT_INT32) && (batch_type_x != DT_INT64);
  bool validRowType = (row_type_x != DT_INT32) && (row_type_x != DT_INT64);
  bool validColType = (col_type_x != DT_INT32) && (col_type_x != DT_INT64);
  if (validValueType || validShapeType || validBatchType || validRowType || validColType) {
    err_msg = "datatype is wrong!";
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  Shape x_dense_shape = op.GetInputDescByName("x_dense_shape").GetShape();
  const int rank_x = x_dense_shape.GetDim(0);
  if (rank_x != 2 && rank_x != 3) {
    err_msg = "rank of input must be 2 or 3, please check!";
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  Shape x_col_indices = op.GetInputDescByName("x_col_indices").GetShape();
  const int rank_x_col_indices = x_col_indices.GetDim(0);
  Shape x_values = op.GetInputDescByName("x_values").GetShape();
  const int rank_x_values = x_values.GetDim(0);
  if (rank_x_values != rank_x_col_indices) {
    err_msg = "rank of x_values and x_col_indices  must be same, please check!";
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_INFER_FUNC_REG(SparseMatrixTranspose, SparseMatrixTransposeInfer);
CUST_VERIFY_FUNC_REG(SparseMatrixTranspose, SparseMatrixTransposeVerify);
}  // namespace ge