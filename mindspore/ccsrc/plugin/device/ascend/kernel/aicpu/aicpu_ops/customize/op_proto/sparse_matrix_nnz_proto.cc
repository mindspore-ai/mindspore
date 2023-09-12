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

#include "inc/sparse_matrix_nnz_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"

namespace ge {
CUST_IMPLEMT_INFERFUNC(SparseMatrixNNZ, SparseMatrixNNZInfer) {
  std::string err_msg;
  TensorDesc x_batch_pointers_desc = op.GetInputDescByName("x_batch_pointers");
  TensorDesc y = op.GetOutputDescByName("y");
  int32_t batch_size = static_cast<int32_t>(x_batch_pointers_desc.GetShape().GetDim(0)) - 1;
  y.SetShape(Shape({batch_size}));
  y.SetDataType(DT_INT32);

  if (op.UpdateOutputDesc("y", y) != GRAPH_SUCCESS) {
    err_msg = "fail to update y shape";
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_INFER_FUNC_REG(SparseMatrixNNZ, SparseMatrixNNZInfer);

// verify of SparseMatrixNNZ
CUST_IMPLEMT_VERIFIER(SparseMatrixNNZ, SparseMatrixNNZVerify) {
  string err_msg;
  Shape x_dense_shape = op.GetInputDescByName("x_dense_shape").GetShape();
  // check input matrix rank
  const int rank_x = x_dense_shape.GetDim(0);
  if (rank_x != 2 && rank_x != 3) {
    err_msg = "rank of input must be 2 or 3, please check!";
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  // check data type of input
  DataType value_type_x = op.GetInputDescByName("x_values").GetDataType();
  DataType shape_type_x = op.GetInputDescByName("x_dense_shape").GetDataType();
  DataType batch_type_x = op.GetInputDescByName("x_batch_pointers").GetDataType();
  DataType row_type_x = op.GetInputDescByName("x_row_pointers").GetDataType();
  DataType col_type_x = op.GetInputDescByName("x_col_indices").GetDataType();

  if (shape_type_x != batch_type_x || batch_type_x != row_type_x || row_type_x != col_type_x) {
    err_msg = "datatype of batch | shape | row | col is not same!";
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  bool validValueType = (value_type_x != DT_INT8) && (value_type_x != DT_UINT8) && (value_type_x != DT_INT16) &&
                        (value_type_x != DT_UINT16) && (value_type_x != DT_INT32) && (value_type_x != DT_INT64) &&
                        (value_type_x != DT_BOOL) && (value_type_x != DT_FLOAT16) && (value_type_x != DT_FLOAT) &&
                        (value_type_x != DT_DOUBLE) && (value_type_x != DT_COMPLEX64) &&
                        (value_type_x != DT_COMPLEX128);
  bool validShapeType = (shape_type_x != DT_INT32) && (shape_type_x != DT_INT64);
  bool validBatchType = (batch_type_x != DT_INT32) && (batch_type_x != DT_INT64);
  bool validRowType = (row_type_x != DT_INT32) && (row_type_x != DT_INT64);
  bool validColType = (col_type_x != DT_INT32) && (col_type_x != DT_INT64);
  if (validValueType || validShapeType || validBatchType || validRowType || validColType) {
    err_msg = "datatype is wrong!";
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_VERIFY_FUNC_REG(SparseMatrixNNZ, SparseMatrixNNZVerify);
}  // namespace ge