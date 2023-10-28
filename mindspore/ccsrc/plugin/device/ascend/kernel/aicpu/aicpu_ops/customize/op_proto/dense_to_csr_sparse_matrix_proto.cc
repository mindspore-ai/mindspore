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

#include "inc/dense_to_csr_sparse_matrix_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"

namespace ge {
CUST_IMPLEMT_VERIFIER(DenseToCSRSparseMatrix, DenseToCSRSparseMatrixVerify) {
  string err_msg;
  DataType dense_type = op.GetInputDescByName("dense_input").GetDataType();
  DataType indices_type = op.GetInputDescByName("indices").GetDataType();
  bool validValueType = (dense_type != DT_FLOAT) && (dense_type != DT_DOUBLE) && (dense_type != DT_COMPLEX64) &&
                        (dense_type != DT_COMPLEX128);
  bool validIdxType = (indices_type != DT_INT32) && (indices_type != DT_INT64);
  if (validValueType || validIdxType) {
    err_msg = "Data type of some input is wrong!";
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
CUST_VERIFY_FUNC_REG(DenseToCSRSparseMatrix, DenseToCSRSparseMatrixVerify);

CUST_IMPLEMT_INFERFUNC(DenseToCSRSparseMatrix, DenseToCSRSparseMatrixInfer) {
  std::string err_msg;
  Shape dense_input_shape = op.GetInputDescByName("dense_input").GetShape();
  DataType dense_input_type = op.GetInputDescByName("dense_input").GetDataType();
  Shape indices_shape = op.GetInputDescByName("indices").GetShape();
  DataType indices_type = op.GetInputDescByName("indices").GetDataType();
  const int64_t dense_input_rank = static_cast<int64_t>(dense_input_shape.GetDimNum());
  const int64_t indices_rank = static_cast<int64_t>(indices_shape.GetDimNum());
  if ((dense_input_rank != 2 && dense_input_rank != 3)) {
    err_msg = "Rank of dense input should be 2 or 3, please check!";
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if ((indices_rank != 2)) {
    err_msg = "Indices must be a matrix, please check!";
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if ((dense_input_rank != indices_shape.GetDim(1))) {
    err_msg = "Indices.shape[1] must be equal to the rank of dense input, please check!";
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  TensorDesc y_dense_shape_desc = op.GetOutputDescByName("y_dense_shape");
  TensorDesc y_batch_pointers_desc = op.GetOutputDescByName("y_batch_pointers");
  TensorDesc y_row_pointers_desc = op.GetOutputDescByName("y_row_pointers");
  TensorDesc y_col_indices_desc = op.GetOutputDescByName("y_col_indices");
  TensorDesc y_values_desc = op.GetOutputDescByName("y_values");
  y_dense_shape_desc.SetDataType(indices_type);
  y_batch_pointers_desc.SetDataType(indices_type);
  y_row_pointers_desc.SetDataType(indices_type);
  y_col_indices_desc.SetDataType(indices_type);
  y_values_desc.SetDataType(dense_input_type);
  const int64_t total_nnz = indices_shape.GetDim(0);
  const int64_t batch_size = (dense_input_rank == 2) ? 1 : dense_input_shape.GetDim(0);
  const int64_t num_rows = (dense_input_rank == 2) ? dense_input_shape.GetDim(0) : dense_input_shape.GetDim(1);
  std::vector<int64_t> newDims;
  newDims.push_back(dense_input_rank);
  y_dense_shape_desc.SetShape(Shape(newDims));
  newDims.clear();
  newDims.push_back((batch_size + 1));
  y_batch_pointers_desc.SetShape(Shape(newDims));
  newDims.clear();
  newDims.push_back((batch_size * (num_rows + 1)));
  y_row_pointers_desc.SetShape(Shape(newDims));
  newDims.clear();
  newDims.push_back(total_nnz);
  y_col_indices_desc.SetShape(Shape(newDims));
  newDims.clear();
  newDims.push_back(total_nnz);
  y_values_desc.SetShape(Shape(newDims));
  if (op.UpdateOutputDesc("y_dense_shape", y_dense_shape_desc) != GRAPH_SUCCESS) {
    err_msg = "DenseToCSRSparseMatrix failed to update output y_dense_shape.";
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (op.UpdateOutputDesc("y_batch_pointers", y_batch_pointers_desc) != GRAPH_SUCCESS) {
    err_msg = "DenseToCSRSparseMatrix failed to update output y_dense_shape.";
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (op.UpdateOutputDesc("y_row_pointers", y_row_pointers_desc) != GRAPH_SUCCESS) {
    err_msg = "DenseToCSRSparseMatrix failed to update output y_dense_shape.";
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (op.UpdateOutputDesc("y_col_indices", y_col_indices_desc) != GRAPH_SUCCESS) {
    err_msg = "DenseToCSRSparseMatrix failed to update output y_dense_shape.";
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (op.UpdateOutputDesc("y_values", y_values_desc) != GRAPH_SUCCESS) {
    err_msg = "DenseToCSRSparseMatrix failed to update output y_dense_shape.";
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
CUST_INFER_FUNC_REG(DenseToCSRSparseMatrix, DenseToCSRSparseMatrixInfer);
}  // namespace ge