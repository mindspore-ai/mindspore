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

#include "inc/ops/linalg_ops.h"
#include "custom_op_proto/cust_linalg_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/linalg_ops_shape_fns.h"
#include "utils/common_shape_fns.h"

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

  int64_t existing = s.GetDimNum();
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

  int dim_num = input.GetDimNum();
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

  int64_t existing = LU_data.GetDimNum();
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

  int dim_num = input.GetDimNum();
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
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto x_desc = op_desc->MutableInputDesc(0);

  GeShape y_shape;
  if (MakeBatchSquareMatrix(x_desc, y_shape, op) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(),
            "Op CholeskyGrad first input x tensor make batch square matrix "
            "failed.");
    return GRAPH_FAILED;
  }

  DataType type = x_desc->GetDataType();
  auto y_desc = op_desc->MutableOutputDesc(0);
  y_desc->SetShape(y_shape);
  y_desc->SetDataType(type);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(CholeskyGrad, CholeskyGradInfer);
// -----------------------CholeskyGrad END---------------------------------
}  // namespace ge