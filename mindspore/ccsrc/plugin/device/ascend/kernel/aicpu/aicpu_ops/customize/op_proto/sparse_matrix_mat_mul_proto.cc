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

#include "inc/sparse_matrix_mat_mul_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"

namespace ge {
CUST_IMPLEMT_INFERFUNC(SparseMatrixMatMul, SparseMatrixMatMulInfer) {
  // first we check whether the sparsematrix A inputted will be transpose
  // if it's, then we could infer the output shape well
  // otherwise, we temporarily use some way to get the A's row and col elegently
  string err_msg;
  string debug_msg;
  // get attrs of A, B and output Y to check if A, B and Y need be transposed
  bool transpose_a = false, transpose_b = false, adjoint_a = false, adjoint_b = false, transpose_output = false;
  op.GetAttr("transpose_x1", transpose_a);
  op.GetAttr("transpose_x2", transpose_b);
  op.GetAttr("adjoint_x1", adjoint_a);
  op.GetAttr("adjoint_x2", adjoint_b);
  op.GetAttr("transpose_output", transpose_output);

  bool transpose_A = transpose_a || adjoint_a;
  bool transpose_B = transpose_b || adjoint_b;
  // row and col of B
  Shape shape_b = op.GetInputDescByName("x2_dense").GetShape();
  int rank = shape_b.GetDimNum();
  int64_t row_b = rank == 2 ? shape_b.GetDim(0) : shape_b.GetDim(1);
  int64_t col_b = rank == 2 ? shape_b.GetDim(1) : shape_b.GetDim(2);

  int64_t row_a;
  if (!transpose_A) {
    // in this case, we could computer the output shape
    // row of A
    int batch = op.GetInputDescByName("x1_batch_pointers").GetShape().GetDim(0) - 1;
    row_a = op.GetInputDescByName("x1_row_pointers").GetShape().GetDim(0) / batch - 1;
  } else {
    // row_a is the col of the origin dense since the transpose_A
    Tensor dense_shape_a;
    if (op.GetInputConstData("x1_dense_shape", dense_shape_a) != GRAPH_SUCCESS) {
      OP_LOGE(TbeGetName(op).c_str(), "failed to get tensor from x1_dense_shape in infer shape stage");
      return GRAPH_FAILED;
    }
    DataType shape_dt = op.GetInputDescByName("x1_dense_shape").GetDataType();
    switch (shape_dt) {
      case DT_INT32:
        row_a = reinterpret_cast<int32_t *>(dense_shape_a.GetData())[rank - 2];
        break;
      case DT_INT64:
        row_a = reinterpret_cast<int64_t *>(dense_shape_a.GetData())[rank - 2];
        break;
      default:
        return GRAPH_FAILED;
    }
  }
  // batch,row, col of output 'y'
  // check whether B need be transposed
  int row_y = row_a;
  int col_y = !transpose_B ? col_b : row_b;
  // further more, we need to update the col and row of y according to the attr of output
  if (transpose_output) {
    int temp = col_y;
    col_y = row_y;
    row_y = temp;
  }
  Shape shape_y = shape_b;
  int row_dim = rank == 2 ? 0 : 1;
  int col_dim = row_dim + 1;
  shape_y.SetDim(row_dim, row_y);
  shape_y.SetDim(col_dim, col_y);

  TensorDesc y_dense_desc = op.GetOutputDescByName("y_dense");
  y_dense_desc.SetShape(shape_y);
  y_dense_desc.SetDataType(op.GetInputDescByName("x2_dense").GetDataType());
  y_dense_desc.SetFormat(op.GetInputDescByName("x2_dense").GetFormat());

  // update output desc
  if (op.UpdateOutputDesc("y_dense", y_dense_desc) != GRAPH_SUCCESS) {
    err_msg = "fail to update output y_dense_shape";
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_INFER_FUNC_REG(SparseMatrixMatMul, SparseMatrixMatMulInfer);

CUST_IMPLEMT_VERIFIER(SparseMatrixMatMul, SparseMatrixMatMulVerify) {
  string err_msg;
  // check1: only one can be true between adjoint and transpose
  bool transpose_a = false, transpose_b = false, adjoint_a = false, adjoint_b = false;
  op.GetAttr("transpose_x1", transpose_a);
  op.GetAttr("transpose_x2", transpose_b);
  op.GetAttr("adjoint_x1", adjoint_a);
  op.GetAttr("adjoint_x2", adjoint_b);

  if (adjoint_a && transpose_a) {
    err_msg = "Only one of adjoint_a and transpose_a may be true.";
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (adjoint_b && transpose_b) {
    err_msg = "Only one of adjoint_b and transpose_b may be true.";
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  // check2: datatype
  DataType value_type_x1 = op.GetInputDescByName("x1_values").GetDataType();
  DataType col_type_x1 = op.GetInputDescByName("x1_col_indices").GetDataType();
  DataType row_type_x1 = op.GetInputDescByName("x1_row_pointers").GetDataType();
  DataType batch_type_x1 = op.GetInputDescByName("x1_batch_pointers").GetDataType();
  DataType shape_type_x1 = op.GetInputDescByName("x1_dense_shape").GetDataType();
  DataType value_type_x2 = op.GetInputDescByName("x2_dense").GetDataType();

  bool validValueType = (value_type_x1 != DT_FLOAT) && (value_type_x1 != DT_DOUBLE) &&
                        (value_type_x1 != DT_COMPLEX64) && (value_type_x1 != DT_COMPLEX128);
  bool validShapeType = (shape_type_x1 != DT_INT32) && (shape_type_x1 != DT_INT64);
  bool validRowType = (row_type_x1 != DT_INT32) && (row_type_x1 != DT_INT64);
  bool validColType = (col_type_x1 != DT_INT32) && (col_type_x1 != DT_INT64);

  if (value_type_x1 != value_type_x2) {
    err_msg = "datatype of two inputs is different!";
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  if (shape_type_x1 != batch_type_x1 || batch_type_x1 != row_type_x1 || row_type_x1 != col_type_x1) {
    err_msg = "datatype of batch | shape | row | col is not same!";
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  if (validValueType || validShapeType || validShapeType || validRowType || validColType) {
    err_msg = "datatype is wrong!";
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  // check3 rank
  const int rank_x1 = op.GetInputDescByName("x1_dense_shape").GetShape().GetDim(0);
  const int rank_x2 = op.GetInputDescByName("x2_dense").GetShape().GetDimNum();
  if ((rank_x1 != 2 && rank_x1 != 3) || (rank_x2 != 2 && rank_x2 != 3)) {
    err_msg = "both rank of two input must be 2 or 3, please check!";
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (rank_x1 != rank_x2) {
    err_msg = "rank of two input must equal, please check!";
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  // check4 batch size
  const int batch_x1 = op.GetInputDescByName("x1_batch_pointers").GetShape().GetDim(0) - 1;
  const int batch_x2 = rank_x2 == 2 ? 1 : op.GetInputDescByName("x2_dense").GetShape().GetDim(0);
  if (batch_x1 != batch_x2) {
    err_msg = "batch size is different! please checck!";
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

CUST_VERIFY_FUNC_REG(SparseMatrixMatMul, SparseMatrixMatMulVerify);
}  // namespace ge