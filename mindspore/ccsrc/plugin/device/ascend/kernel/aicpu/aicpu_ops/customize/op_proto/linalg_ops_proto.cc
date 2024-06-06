/**
 * Copyright (c) 2023 Huawei Technologies Co., Ltd.  All rights reserved.
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

#include "op_proto/inc/linalg_ops.h"
#include "custom_op_proto/cust_linalg_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/linalg_ops_shape_fns.h"
#include "utils/common_shape_fns.h"
#include "op_proto/inc/linalg_ops.h"

namespace ge {
// ----------------MatrixSolve-------------------
IMPLEMT_INFERFUNC(MatrixSolve, MatrixSolveInfer) {
  auto matrix_tensor = op.get_input_desc_matrix();
  auto rhs_tensor = op.get_input_desc_rhs();
  Shape result;
  if (MatrixSolve(matrix_tensor, rhs_tensor, true, result, op) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Op MatrixSolve Call MatrixSolve Infer Shape fns Failed.");
    return GRAPH_FAILED;
  }
  DataType type = op.GetInputDescByName("matrix").GetDataType();

  TensorDesc y_desc = op.GetOutputDescByName("y");
  y_desc.SetShape(Shape(result));
  y_desc.SetDataType(type);
  op.UpdateOutputDesc("y", y_desc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MatrixSolve, MatrixSolveInfer);
// ----------------MatrixSolve End-------------------

// ----------------MatrixDeterminant-------------------
IMPLEMT_INFERFUNC(MatrixDeterminant, MatrixDeterminantInfer) {
  auto tensor = op.get_input_desc_x();
  Shape s;
  if (WithRankAtLeast(tensor, 2, s, op) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "The rank of x must be at least 2.");
    return GRAPH_FAILED;
  }

  int64_t existing = static_cast<int64_t>(s.GetDimNum());
  int64_t dim1 = s.GetDim(existing - 1);
  int64_t dim2 = s.GetDim(existing - 2);
  int64_t unused_dim = 0;

  if (Merge(dim1, dim2, unused_dim) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Merge two dimension failed.");
    return GRAPH_FAILED;
  }

  Shape result;
  if (SubShape(s, 0, -2, 1, result, op) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Op MatrixDeterminant Get SubShape Failed.");
    return GRAPH_FAILED;
  }

  DataType type = op.GetInputDescByName("x").GetDataType();

  TensorDesc y_desc = op.GetOutputDescByName("y");
  y_desc.SetShape(Shape(result));
  y_desc.SetDataType(type);
  op.UpdateOutputDesc("y", y_desc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MatrixDeterminant, MatrixDeterminantInfer);
// ----------------MatrixDeterminant End-------------------

// ----------------MatrixTriangularSolve-------------------
IMPLEMT_INFERFUNC(MatrixTriangularSolve, MatrixTriangularSolveInfer) {
  auto matrix_tensor = op.get_input_desc_matrix();
  auto rhs_tensor = op.get_input_desc_rhs();
  Shape result;
  if (MatrixSolve(matrix_tensor, rhs_tensor, true, result, op) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Op MatrixTriangularSolve Call MatrixSolve Infer Shape fns Failed.");
    return GRAPH_FAILED;
  }

  DataType type = op.GetInputDescByName("matrix").GetDataType();

  TensorDesc y_desc = op.GetOutputDescByName("y");
  y_desc.SetShape(Shape(result));
  y_desc.SetDataType(type);
  op.UpdateOutputDesc("y", y_desc);
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(MatrixTriangularSolve, MatrixTriangularSolveInfer);
// ----------------MatrixTriangularSolve END-------------------

// ---------------Geqrf-------------------
CUST_IMPLEMT_INFERFUNC(Geqrf, GeqrfInfer) {
  auto tensor = op.get_input_desc_x();
  Shape input;
  if (WithRank(tensor, 2, input, op) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  int dim_num = static_cast<int>(input.GetDimNum());
  int m = input.GetDim(dim_num - 2);
  int n = input.GetDim(dim_num - 1);
  Shape r_shape;
  Shape tau_shape;
  int p = m > n ? n : m;
  Matrix(m, n, r_shape);
  Vector(p, tau_shape);

  DataType type = op.GetInputDescByName("x").GetDataType();
  TensorDesc r_desc = op.GetOutputDescByName("y");
  r_desc.SetShape(Shape(r_shape));
  r_desc.SetDataType(type);
  if (op.UpdateOutputDesc("y", r_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Update y desc failed.");
    return GRAPH_FAILED;
  }

  TensorDesc tau_desc = op.GetOutputDescByName("tau");
  tau_desc.SetShape(Shape(tau_shape));
  tau_desc.SetDataType(type);
  if (op.UpdateOutputDesc("tau", tau_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Update tau desc failed.");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

CUST_INFER_FUNC_REG(Geqrf, GeqrfInfer);

CUST_IMPLEMT_VERIFIER(Geqrf, GeqrfVerify) {
  DataType type = op.GetInputDescByName("x").GetDataType();
  if (type != DT_FLOAT16 && type != DT_FLOAT && type != DT_DOUBLE && type != DT_COMPLEX64 && type != DT_COMPLEX128) {
    OP_LOGE(TbeGetName(op).c_str(), "Expect a floating point or complex tensor as input.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_VERIFY_FUNC_REG(Geqrf, GeqrfVerify);
// ---------------Geqrf End---------------

// ---------------LuUnpack---------------
CUST_IMPLEMT_INFERFUNC(LuUnpack, LuUnpackInferShape) {
  Shape LU_data;
  if (WithRankAtLeast(op.GetInputDesc(0), 2, LU_data, op) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "LU_data rank must be at least 2.");
    return GRAPH_FAILED;
  }

  int64_t existing = static_cast<int64_t>(LU_data.GetDimNum());
  int64_t dim1 = LU_data.GetDim(existing - 2);
  int64_t dim2 = LU_data.GetDim(existing - 1);

  Shape batch_shape;
  if (SubShape(LU_data, 0, -2, 1, batch_shape, op) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Op LuUnpack Get SubShape Failed.");
    return GRAPH_FAILED;
  }

  Shape L_shape;
  vector<int64_t> L_dims;
  L_dims.reserve(2);
  if (dim1 >= dim2) {
    L_dims.push_back(dim1);
    L_dims.push_back(dim2);
  } else {
    L_dims.push_back(dim1);
    L_dims.push_back(dim1);
  }
  Shape L_sec_shape(L_dims);
  if (Concatenate(batch_shape, L_sec_shape, L_shape) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Concatenate L_shape failed!");
    return GRAPH_FAILED;
  }

  Shape U_shape;
  vector<int64_t> U_dims;
  U_dims.reserve(2);
  if (dim1 >= dim2) {
    U_dims.push_back(dim2);
    U_dims.push_back(dim2);
  } else {
    U_dims.push_back(dim1);
    U_dims.push_back(dim2);
  }
  Shape U_sec_shape(U_dims);
  if (Concatenate(batch_shape, U_sec_shape, U_shape) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Concatenate U_shape failed!");
    return GRAPH_FAILED;
  }

  Shape pivots_shape;
  vector<int64_t> pivots_dims;
  pivots_dims.reserve(2);
  pivots_dims.push_back(dim1);
  pivots_dims.push_back(dim1);
  Shape pivots_sec_shape(pivots_dims);
  if (Concatenate(batch_shape, pivots_sec_shape, pivots_shape) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Concatenate pivots_shape failed!");
    return GRAPH_FAILED;
  }

  TensorDesc L_desc = op.GetOutputDescByName("L");
  L_desc.SetShape(Shape(L_shape));
  DataType L_type = op.GetInputDescByName("LU_data").GetDataType();
  L_desc.SetDataType(L_type);
  if (L_desc.GetDataType() != L_type) {
    OP_LOGE(TbeGetName(op).c_str(), "the type of L must be the same as the type of LU_data.");
    return GRAPH_FAILED;
  }
  if (op.UpdateOutputDesc("L", L_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "fail to update output L.");
    return GRAPH_FAILED;
  }
  TensorDesc U_desc = op.GetOutputDescByName("U");
  U_desc.SetShape(Shape(U_shape));
  DataType U_type = op.GetInputDescByName("LU_data").GetDataType();
  U_desc.SetDataType(U_type);
  if (U_desc.GetDataType() != U_type) {
    OP_LOGE(TbeGetName(op).c_str(), "the type of U must be the same as the type of LU_data.");
    return GRAPH_FAILED;
  }
  if (op.UpdateOutputDesc("U", U_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "fail to update output U.");
    return GRAPH_FAILED;
  }
  TensorDesc pivots_desc = op.GetOutputDescByName("pivots");
  pivots_desc.SetShape(Shape(pivots_shape));
  DataType pivots_type = op.GetInputDescByName("LU_data").GetDataType();
  pivots_desc.SetDataType(pivots_type);
  if (pivots_desc.GetDataType() != pivots_type) {
    OP_LOGE(TbeGetName(op).c_str(), "the type of pivots must be the same as the type of LU_data.");
    return GRAPH_FAILED;
  }
  if (op.UpdateOutputDesc("pivots", pivots_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "fail to update output pivots.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_IMPLEMT_VERIFIER(LuUnpack, LuUnpackVerify) {
  DataType LU_data_type = op.GetInputDescByName("LU_data").GetDataType();
  DataType LU_pivots_type = op.GetInputDescByName("LU_pivots").GetDataType();
  if (LU_data_type != DT_FLOAT16 && LU_data_type != DT_FLOAT && LU_data_type != DT_DOUBLE && LU_data_type != DT_INT8 &&
      LU_data_type != DT_UINT8 && LU_data_type != DT_INT16 && LU_data_type != DT_INT32 && LU_data_type != DT_INT64) {
    std::string err_msg;
    err_msg = ConcatString("Op LuUnpack first input LU_data_type's data type should be of the follows: ",
                           "DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE,",
                           "but this type is ", LU_data_type, ".");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (LU_pivots_type != DT_INT8 && LU_pivots_type != DT_UINT8 && LU_pivots_type != DT_INT16 &&
      LU_pivots_type != DT_INT32 && LU_pivots_type != DT_INT64) {
    std::string err_msg;
    err_msg =
      ConcatString("Op LuUnpack first input LU_data_type's data type should be of the follows: ",
                   "DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64,", "but this type is ", LU_pivots_type, ".");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_INFER_FUNC_REG(LuUnpack, LuUnpackInferShape);
CUST_VERIFY_FUNC_REG(LuUnpack, LuUnpackVerify);
// ---------------LuUnpack END---------------

// ---------------LuUnpackGrad---------------
IMPLEMT_COMMON_INFERFUNC(LuUnpackGradInferShape) {
  TensorDesc L_grad;
  TensorDesc U_grad;
  TensorDesc LU_data;
  if (op.TryGetInputDesc("LU_data", LU_data) == GRAPH_FAILED) {
    OP_LOGE(TbeGetName(op).c_str(), "LU_data can not be null.");
    return GRAPH_FAILED;
  }
  TensorDesc L_data_grad = op.GetOutputDescByName("L_data_grad");
  L_data_grad.SetDataType(op.GetInputDescByName("LU_data").GetDataType());
  L_data_grad.SetShape(op.GetInputDescByName("LU_data").GetShape());
  if (op.UpdateOutputDesc("L_data_grad", L_data_grad) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "fail to update output L_data_grad.");
    return GRAPH_FAILED;
  }
  TensorDesc U_data_grad = op.GetOutputDescByName("U_data_grad");
  U_data_grad.SetDataType(op.GetInputDescByName("LU_data").GetDataType());
  U_data_grad.SetShape(op.GetInputDescByName("LU_data").GetShape());
  if (op.UpdateOutputDesc("U_data_grad", U_data_grad) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "fail to update output U_data_grad.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
CUST_IMPLEMT_VERIFIER(LuUnpackGrad, LuUnpackGradVerify) {
  DataType LU_data_type = op.GetInputDescByName("LU_data").GetDataType();
  Shape LU_data_shape = op.GetInputDescByName("LU_data").GetShape();
  TensorDesc L_grad;
  TensorDesc U_grad;
  if (op.TryGetInputDesc("L_grad", L_grad) == GRAPH_SUCCESS) {
    DataType L_grad_type = op.GetInputDescByName("L_grad").GetDataType();
    Shape L_data_shape = op.GetInputDescByName("L_grad").GetShape();
    auto L_data_dim1 = L_data_shape.GetDim(-2);
    auto L_data_dim2 = L_data_shape.GetDim(-1);
    auto LU_data_dim1 = LU_data_shape.GetDim(-2);
    auto LU_data_dim2 = LU_data_shape.GetDim(-1);
    int64_t LU_data_min = std::min(LU_data_dim1, LU_data_dim2);
    if (LU_data_dim1 != L_data_dim1) {
      OP_LOGE(TbeGetName(op).c_str(), "L_grad's data dim[-2] and LU_data's dim[-2] should be same.");
      return GRAPH_FAILED;
    }
    if (LU_data_min != L_data_dim2) {
      OP_LOGE(TbeGetName(op).c_str(), "L_grad's data dim[-1] and LU_data's minimum dim should be same.");
      return GRAPH_FAILED;
    }
    if (LU_data_type != L_grad_type) {
      OP_LOGE(TbeGetName(op).c_str(), "L_grad's data type and LU_data's type should be same.");
      return GRAPH_FAILED;
    }
  }
  if (op.TryGetInputDesc("U_grad", U_grad) == GRAPH_SUCCESS) {
    DataType U_grad_type = op.GetInputDescByName("U_grad").GetDataType();
    Shape U_data_shape = op.GetInputDescByName("U_grad").GetShape();
    auto U_data_dim1 = U_data_shape.GetDim(-2);
    auto U_data_dim2 = U_data_shape.GetDim(-1);
    auto LU_data_dim1 = LU_data_shape.GetDim(-2);
    auto LU_data_dim2 = LU_data_shape.GetDim(-1);
    int64_t LU_data_min = std::min(LU_data_dim1, LU_data_dim2);
    if (U_data_dim2 != LU_data_dim2) {
      OP_LOGE(TbeGetName(op).c_str(), "U_grad's data dim[-1] and LU_data's dim[-1] should be same.");
      return GRAPH_FAILED;
    }
    if (LU_data_min != U_data_dim1) {
      OP_LOGE(TbeGetName(op).c_str(), "U_grad's data dim[-2] and LU_data's minimum dim should be same.");
      return GRAPH_FAILED;
    }
    if (LU_data_type != U_grad_type) {
      OP_LOGE(TbeGetName(op).c_str(), "U_grad's data type and LU_data's type should be same.");
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}
CUST_COMMON_INFER_FUNC_REG(LuUnpackGrad, LuUnpackGradInferShape);
CUST_VERIFY_FUNC_REG(LuUnpackGrad, LuUnpackGradVerify);
// ---------------LuUnpackGrad End---------------

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

// -----------------------Qr---------------------------------
IMPLEMT_INFERFUNC(Qr, QrInfer) {
  auto tensor = op.get_input_desc_x();
  Shape input;
  if (WithRankAtLeast(tensor, 2, input, op) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  Shape batch_shape;
  if (SubShape(input, 0, -2, 1, batch_shape, op) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  int dim_num = static_cast<int>(input.GetDimNum());
  int m = input.GetDim(dim_num - 2);
  int n = input.GetDim(dim_num - 1);
  Shape q_shape;
  Shape r_shape;
  auto full_matrices = op.get_attr_full_matrices();

  if (full_matrices) {
    // [...,M,M]; [...,M,N], if full_matrices is true
    Shape m_m_shape;
    Shape m_n_shape;
    Matrix(m, m, m_m_shape);
    Matrix(m, n, m_n_shape);

    Concatenate(batch_shape, m_m_shape, q_shape);
    Concatenate(batch_shape, m_n_shape, r_shape);
  } else {
    // [...,M,P]; [...,P,N], if full_matrices is false
    int p = m > n ? n : m;
    Shape m_p_shape;
    Shape p_n_shape;
    Matrix(m, p, m_p_shape);
    Matrix(p, n, p_n_shape);

    Concatenate(batch_shape, m_p_shape, q_shape);
    Concatenate(batch_shape, p_n_shape, r_shape);
  }

  DataType type = op.GetInputDescByName("x").GetDataType();
  TensorDesc q_desc = op.GetOutputDescByName("q");
  q_desc.SetShape(Shape(q_shape));
  q_desc.SetDataType(type);
  if (op.UpdateOutputDesc("q", q_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Update q desc failed.");
    return GRAPH_FAILED;
  }

  TensorDesc r_desc = op.GetOutputDescByName("r");
  r_desc.SetShape(Shape(r_shape));
  r_desc.SetDataType(type);
  if (op.UpdateOutputDesc("r", r_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Update r desc failed.");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Qr, QrInfer);
// -----------------------Qr END---------------------------------

// -----------------------CholeskyGrad---------------------------------
IMPLEMT_INFERFUNC(CholeskyGrad, CholeskyGradInfer) {
  auto x_desc = op.GetInputDesc(0);

  Shape y_shape;
  if (MakeBatchSquareMatrix(x_desc, y_shape, op) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(),
            "Op CholeskyGrad first input x tensor make batch square matrix "
            "failed.");
    return GRAPH_FAILED;
  }

  DataType type = x_desc.GetDataType();
  auto y_desc = op.GetOutputDesc(0);
  y_desc.SetShape(y_shape);
  y_desc.SetDataType(type);
  op.UpdateOutputDesc("y", y_desc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(CholeskyGrad, CholeskyGradInfer);
// -----------------------CholeskyGrad END---------------------------------

// -----------------------LinearSumAssignment---------------------------------
CUST_IMPLEMT_INFERFUNC(LinearSumAssignment, LinearSumAssignmentInfer) {
  TensorDesc cost_matrix_tensor = op.get_input_desc_cost_matrix();
  Shape cost_matrix_shape = cost_matrix_tensor.GetShape();

  auto row_ind_desc = op.GetOutputDesc(0);
  auto col_ind_desc = op.GetOutputDesc(1);
  Shape row_ind_shape, col_ind_shape;
  if (!RankKnown(cost_matrix_shape)) {
    row_ind_shape = Shape(ge::UNKNOWN_SHAPE);
    col_ind_shape = Shape(ge::UNKNOWN_SHAPE);
  } else {
    constexpr int64_t kNumber2 = 2;
    if (cost_matrix_shape.GetDimNum() != kNumber2) {
      OP_LOGE(TbeGetName(op).c_str(), "cost_matrix dim num should be 2. But got [%lu].", cost_matrix_shape.GetDimNum());
      return GRAPH_FAILED;
    }
    int64_t row_num = cost_matrix_shape.GetDim(0);
    int64_t col_num = cost_matrix_shape.GetDim(1);
    int64_t out_dim = std::min(row_num, col_num);
    std::vector<int64_t> shape_vec{out_dim};
    row_ind_shape = Shape(shape_vec);
    col_ind_shape = Shape(shape_vec);
  }
  row_ind_desc.SetShape(row_ind_shape);
  col_ind_desc.SetShape(col_ind_shape);

  TensorDesc dimension_limit_tensor = op.get_input_desc_dimension_limit();
  TensorDesc maximize_tensor = op.get_input_desc_maximize();

  DataType cost_matrix_type = cost_matrix_tensor.GetDataType();
  DataType dimension_limit_type = dimension_limit_tensor.GetDataType();
  DataType maximize_type = maximize_tensor.GetDataType();
  std::vector<DataType> valid_dtypes{DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BOOL,   DT_INT16,  DT_INT32,
                                     DT_INT64,   DT_INT8,  DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8};
  auto iter = std::find(valid_dtypes.begin(), valid_dtypes.end(), cost_matrix_type);
  if (iter == valid_dtypes.end()) {
    std::string err_msg;
    err_msg = ConcatString("Op LinearSumAssignment first input cost_matrix's data type should be of the follows: ",
                           "DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BOOL, DT_INT16, DT_INT32, DT_INT64, DT_INT8, ",
                           "DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8, but this type is ", cost_matrix_type, ".");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  if (dimension_limit_type != DT_INT64) {
    std::string err_msg;
    err_msg = ConcatString("Op LinearSumAssignment second input dimension_limit's data type should be of the follows: ",
                           "DT_INT64,", "but this type is ", dimension_limit_type, ".");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  if (maximize_type != DT_BOOL) {
    std::string err_msg;
    err_msg =
      ConcatString("Op LinearSumAssignment third input maximize's data type should be of the follows: ", "DT_BOOL,",
                   "but this type is ", maximize_type, ".");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  row_ind_desc.SetDataType(DT_INT64);
  col_ind_desc.SetDataType(DT_INT64);
  op.UpdateOutputDesc("row_ind", row_ind_desc);
  op.UpdateOutputDesc("col_ind", col_ind_desc);
  return GRAPH_SUCCESS;
}

CUST_INFER_FUNC_REG(LinearSumAssignment, LinearSumAssignmentInfer);
// -----------------------LinearSumAssignment END---------------------------------

// -----------------------SolveTriangular---------------------------------
CUST_IMPLEMT_INFERFUNC(SolveTriangular, SolveTriangularInfer) {
  TensorDesc x_desc = op.GetOutputDescByName("x");
  // infer type
  DataType b_type = op.GetInputDescByName("b").GetDataType();
  DataType x_type;
  static const std::vector<DataType> type_to_float32 = {DT_INT16,  DT_INT32,  DT_INT8, DT_BOOL,
                                                        DT_UINT16, DT_UINT32, DT_UINT8};
  static const std::vector<DataType> type_to_float64 = {DT_INT64, DT_UINT64};
  bool is_type_to_float32 = std::any_of(type_to_float32.begin(), type_to_float32.end(),
                                        [&b_type](const DataType &dtype) { return b_type == dtype; });
  bool is_type_to_float64 = std::any_of(type_to_float64.begin(), type_to_float64.end(),
                                        [&b_type](const DataType &dtype) { return b_type == dtype; });
  if (is_type_to_float32)
    x_type = DT_FLOAT;
  else if (is_type_to_float64)
    x_type = DT_DOUBLE;
  else
    x_type = b_type;
  x_desc.SetDataType(x_type);

  // infer shape
  Shape a_shape = op.GetInputDescByName("a").GetShape();
  Shape b_shape = op.GetInputDescByName("b").GetShape();
  x_desc.SetShape(b_shape);
  if (op.UpdateOutputDesc("x", x_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Failed to update output desc.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_INFER_FUNC_REG(SolveTriangular, SolveTriangularInfer);
// -----------------------SolveTriangular END---------------------------------

// -----------------------SolveTriangularGrad---------------------------------
CUST_IMPLEMT_INFERFUNC(SolveTriangularGrad, SolveTriangularGradInfer) {
  TensorDesc da_desc = op.GetOutputDescByName("da");
  TensorDesc db_desc = op.GetOutputDescByName("db");
  // infer type
  DataType a_type = op.GetInputDescByName("a").GetDataType();
  DataType grad_type;
  static const std::vector<DataType> type_to_float32 = {DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_FLOAT16};
  bool is_type_to_float32 = std::any_of(type_to_float32.begin(), type_to_float32.end(),
                                        [&a_type](const DataType &dtype) { return a_type == dtype; });
  if (is_type_to_float32)
    grad_type = DT_FLOAT;
  else
    grad_type = a_type;
  da_desc.SetDataType(grad_type);
  db_desc.SetDataType(grad_type);

  // infer shape
  Shape a_shape = op.GetInputDescByName("a").GetShape();
  da_desc.SetShape(a_shape);
  if (op.UpdateOutputDesc("da", da_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Failed to update output desc.");
    return GRAPH_FAILED;
  }
  Shape b_shape = op.GetInputDescByName("b").GetShape();
  db_desc.SetShape(b_shape);
  if (op.UpdateOutputDesc("db", db_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Failed to update output desc.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_INFER_FUNC_REG(SolveTriangularGrad, SolveTriangularGradInfer);
// -----------------------SolveTriangularGrad END---------------------------------

// -----------------------LstsqV2---------------------------------

bool LstsqBroadCast(const std::vector<int64_t> &a_batch_dims, const std::vector<int64_t> &b_batch_dims,
                    std::vector<int64_t> &broadcast_batch_dims) {
  for (size_t i = 0; i < a_batch_dims.size(); i++) {
    if (a_batch_dims[i] == b_batch_dims[i]) {
      broadcast_batch_dims.emplace_back(a_batch_dims[i]);
    } else {
      int64_t max_dim = a_batch_dims[i] > b_batch_dims[i] ? a_batch_dims[i] : b_batch_dims[i];
      int64_t min_dim = a_batch_dims[i] < b_batch_dims[i] ? a_batch_dims[i] : b_batch_dims[i];
      if (min_dim == 1 || min_dim == -1)
        broadcast_batch_dims.emplace_back(max_dim);
      else
        return false;
    }
  }
  return true;
}

void LstsqGetDriver(Operator &op, int64_t &driver_value) {
  Tensor driver_data;
  constexpr int64_t Driver_GELSY = 1;
  bool is_unknown_driver{true};
  if (op.GetInputConstData("driver", driver_data) == GRAPH_SUCCESS) {
    is_unknown_driver = false;
  }
  OP_LOGD(TbeGetName(op), "lstsqv2 driver is unknown[%s].", is_unknown_driver ? "true" : "false");
  driver_value = Driver_GELSY;
  if (!is_unknown_driver) {
    DataType dtype = op.GetInputDescByName("driver").GetDataType();
    std::vector<int64_t> const_vec;
    if (!GetConstValue(op, driver_data, dtype, const_vec)) {
      is_unknown_driver = true;
      OP_LOGW(TbeGetName(op), "Get lstsqv2 driver value failed.");
    } else {
      driver_value = const_vec[0];
    }
  }
}

void LstsqHandleOutShape(bool calculate, std::vector<int64_t> &out_dims, std::vector<int64_t> &broadcast_batch_dims) {
  if (calculate) {
    out_dims = std::vector<int64_t>(broadcast_batch_dims);
  } else {
    out_dims.emplace_back(0);
  }
}

graphStatus SetOutputDesc(Operator &op, TensorDesc &solution_desc, TensorDesc &residuals_desc, TensorDesc &rank_desc,
                          TensorDesc &singular_values_desc) {
  if (op.UpdateOutputDesc("solution", solution_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Failed to update solution desc.");
    return GRAPH_FAILED;
  }
  if (op.UpdateOutputDesc("residuals", residuals_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Failed to update residuals desc.");
    return GRAPH_FAILED;
  }
  if (op.UpdateOutputDesc("rank", rank_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Failed to update rank desc.");
    return GRAPH_FAILED;
  }
  if (op.UpdateOutputDesc("singular_values", singular_values_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Failed to update singularValue desc.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_IMPLEMT_INFERFUNC(LstsqV2, LstsqV2Infer) {
  TensorDesc solution_desc = op.GetOutputDescByName("solution");
  TensorDesc residuals_desc = op.GetOutputDescByName("residuals");
  TensorDesc rank_desc = op.GetOutputDescByName("rank");
  TensorDesc singular_values_desc = op.GetOutputDescByName("singular_values");

  // infer type
  DataType a_type = op.GetInputDescByName("a").GetDataType();
  DataType solution_type = a_type;
  DataType residuals_type = a_type;
  DataType rank_type = DT_INT64;
  DataType singular_values_type = a_type;
  if (a_type == DT_COMPLEX64) {
    residuals_type = DT_FLOAT;
    singular_values_type = DT_FLOAT;
  } else if (a_type == DT_COMPLEX128) {
    residuals_type = DT_DOUBLE;
    singular_values_type = DT_DOUBLE;
  }
  solution_desc.SetDataType(solution_type);
  residuals_desc.SetDataType(residuals_type);
  rank_desc.SetDataType(rank_type);
  singular_values_desc.SetDataType(singular_values_type);

  // infer shape
  Shape a_shape = op.GetInputDescByName("a").GetShape();
  Shape b_shape = op.GetInputDescByName("b").GetShape();

  if (IsUnknownRankShape(a_shape) || IsUnknownRankShape(b_shape)) {
    solution_desc.SetShape(Shape({UNKNOWN_RANK}));
    residuals_desc.SetShape(Shape({UNKNOWN_RANK}));
    rank_desc.SetShape(Shape({UNKNOWN_RANK}));
    solution_desc.SetShape(Shape({UNKNOWN_RANK}));
  } else {
    std::vector<int64_t> a_dims = a_shape.GetDims();
    std::vector<int64_t> b_dims = b_shape.GetDims();

    size_t a_rank = a_shape.GetDimNum();
    size_t b_rank = b_shape.GetDimNum();

    constexpr size_t mat_size = 2;
    constexpr size_t vec_size = 1;
    constexpr int64_t Driver_GELS = 0;
    constexpr int64_t Driver_GELSY = 1;
    constexpr int64_t Driver_GELSD = 2;
    constexpr int64_t Driver_GELSS = 3;
    const size_t expected_b_dim = (b_rank == a_rank - 1) ? vec_size : mat_size;
    std::vector<int64_t> a_batch_dims(a_dims.begin(), a_dims.end() - mat_size);
    std::vector<int64_t> b_batch_dims(b_dims.begin(), b_dims.end() - expected_b_dim);
    std::vector<int64_t> broadcast_batch_dims;
    if (LstsqBroadCast(a_batch_dims, b_batch_dims, broadcast_batch_dims) == false) {
      OP_LOGE(TbeGetName(op).c_str(),
              "Batch Dimensions of a and b should can be broadcast with the minimum dimension set to 1.");
      return GRAPH_FAILED;
    }
    int64_t driver_value;
    LstsqGetDriver(op, driver_value);

    int64_t m = a_dims[a_rank - 2];
    int64_t n = a_dims[a_rank - 1];
    int64_t k = b_rank == a_rank ? b_dims[b_rank - 1] : 1;
    std::vector<int64_t> solution_dims;
    std::vector<int64_t> residual_dims;
    std::vector<int64_t> rank_dims;
    std::vector<int64_t> singular_values_dims;
    solution_dims.emplace_back(n);

    bool calculate_res = (m == -1 || n == -1 || m > n) && driver_value != Driver_GELSY;
    bool calculate_rank = driver_value != Driver_GELS;
    bool calculate_singular_values = driver_value == Driver_GELSD || driver_value == Driver_GELSS;
    LstsqHandleOutShape(calculate_res, residual_dims, broadcast_batch_dims);
    LstsqHandleOutShape(calculate_rank, rank_dims, a_batch_dims);
    if (calculate_res) residual_dims.emplace_back(k);
    if (calculate_singular_values) {
      singular_values_dims = std::vector<int64_t>(a_batch_dims);
      singular_values_dims.emplace_back(m < n ? m : n);
    } else {
      singular_values_dims.emplace_back(0);
    }
    if (expected_b_dim == 2) {
      solution_dims.emplace_back(k);
    }
    solution_desc.SetShape(Shape(solution_dims));
    residuals_desc.SetShape(Shape(residual_dims));
    rank_desc.SetShape(Shape(rank_dims));
    singular_values_desc.SetShape(Shape(singular_values_dims));
  }
  return SetOutputDesc(op, solution_desc, residuals_desc, rank_desc, singular_values_desc);
}

CUST_INFER_FUNC_REG(LstsqV2, LstsqV2Infer);
// -----------------------LstsqV2 END---------------------------------

// -----------------------LstsqV2Grad---------------------------------
CUST_IMPLEMT_INFERFUNC(LstsqV2Grad, LstsqV2GradInfer) {
  TensorDesc ga_desc = op.GetOutputDescByName("ga");
  TensorDesc gb_desc = op.GetOutputDescByName("gb");

  // infer type
  DataType a_type = op.GetInputDescByName("a").GetDataType();
  DataType ga_type = a_type;
  DataType gb_type = a_type;
  ga_desc.SetDataType(ga_type);
  gb_desc.SetDataType(gb_type);

  // infer shape
  Shape a_shape = op.GetInputDescByName("a").GetShape();
  Shape b_shape = op.GetInputDescByName("b").GetShape();

  ga_desc.SetShape(a_shape);
  if (op.UpdateOutputDesc("ga", ga_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Failed to update ga desc.");
    return GRAPH_FAILED;
  }
  gb_desc.SetShape(b_shape);
  if (op.UpdateOutputDesc("gb", gb_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Failed to update gb desc.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_INFER_FUNC_REG(LstsqV2Grad, LstsqV2GradInfer);
// -----------------------LstsqV2Grad END---------------------------------
}  // namespace ge