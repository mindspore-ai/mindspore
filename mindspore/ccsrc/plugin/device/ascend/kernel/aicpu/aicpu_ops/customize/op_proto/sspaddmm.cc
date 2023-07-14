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

#include "inc/sspaddmm_op.h"
#include <set>
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"
namespace ge {
// -------------------------------Sspaddmm Begin------------------------------- //
CUST_IMPLEMT_INFERFUNC(Sspaddmm, SspaddmmInfer) {
  // refer SparseTensorDenseMatMul
  int64_t unused_dim1{0};
  Shape unused_shape1, unused_shape2, mat2_shape;
  std::string err_msg;

  auto mat1_indices_tensor = op.get_input_desc_mat1_indices();
  auto mat1_values_tensor = op.GetInputDesc(4);
  auto mat1_shape_tensor = op.get_input_desc_mat1_shape();
  auto input_indices_tensor = op.get_input_desc_input_indices();
  auto input_values_tensor = op.get_input_desc_input_values();
  auto input_shape_tensor = op.get_input_desc_input_shape();
  auto mat2_tensor = op.get_input_desc_mat2();
  TensorDesc y_indices_desc = op.GetOutputDescByName("output_indices");
  TensorDesc y_values_desc = op.GetOutputDescByName("output_values");
  TensorDesc y_shape_desc = op.GetOutputDescByName("output_shape");
  // get mat1_indices const value
  vector<int64_t> mat1_indices_value;
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto mat1_indices_idx = static_cast<uint32_t>(op_info->GetInputIndexByName("mat1_indices"));
  const GeTensor *mat1_indices_getensor = OpDescUtils::GetInputConstData(op, mat1_indices_idx);
  if (mat1_indices_getensor != nullptr) {
    auto const_desc = op_info->MutableInputDesc("mat1_indices");
    auto const_dtype = const_desc->GetDataType();
    if (!GetConstValue(op, mat1_indices_getensor, const_dtype, mat1_indices_value)) {
      OP_LOGW("Sspaddmm", "Get mat1_indices const from const tensor failed");
      return GRAPH_FAILED;
    }
  } else {
    OP_LOGW("Sspaddmm", "Get mat1_indices nullptr");
    return GRAPH_FAILED;
  }
  auto mat1_indices_desc = mat1_indices_getensor->GetTensorDesc();
  if (mat1_indices_desc.GetShape().GetDimNum() != 2) {
    err_msg = string("mat1 indices is not martix");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  // get mat1 shape
  vector<int64_t> mat1_shape_value;
  auto mat1_shape_idx = static_cast<uint32_t>(op_info->GetInputIndexByName("mat1_shape"));
  const GeTensor *mat1_shape_getensor = OpDescUtils::GetInputConstData(op, mat1_shape_idx);
  if (mat1_shape_getensor != nullptr) {
    auto const_desc = op_info->MutableInputDesc("mat1_shape");
    auto const_dtype = const_desc->GetDataType();
    if (!GetConstValue(op, mat1_shape_getensor, const_dtype, mat1_shape_value)) {
      OP_LOGW("Sspaddmm", "Get mat1_shape const from const tensor failed");
      return GRAPH_FAILED;
    }
  } else {
    OP_LOGW("Sspaddmm", "Get mat1_shape nullptr");
    return GRAPH_FAILED;
  }
  auto mat1_shape_desc = mat1_shape_getensor->GetTensorDesc();
  std::cout << mat1_shape_desc.GetShape().GetDimNum() << std::endl;
  if (mat1_shape_desc.GetShape().GetDimNum() != 1) {
    err_msg = string("mat1 shape should be 1D");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  // get input shape
  vector<int64_t> input_shape_value;
  auto input_shape_idx = static_cast<uint32_t>(op_info->GetInputIndexByName("input_shape"));
  const GeTensor *input_shape_getensor = OpDescUtils::GetInputConstData(op, input_shape_idx);
  if (input_shape_getensor != nullptr) {
    auto const_desc = op_info->MutableInputDesc("input_shape");
    auto const_dtype = const_desc->GetDataType();
    if (!GetConstValue(op, input_shape_getensor, const_dtype, input_shape_value)) {
      OP_LOGW("Sspaddmm", "Get input_shape const from const tensor failed");
      return GRAPH_FAILED;
    }
  } else {
    OP_LOGW("Sspaddmm", "Get input_shape nullptr");
    return GRAPH_FAILED;
  }
  auto input_shape_desc = input_shape_getensor->GetTensorDesc();
  if (input_shape_desc.GetShape().GetDimNum() != 1) {
    err_msg = string("input shape is not 1D");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  // number check
  if (mat1_indices_desc.GetShape().GetDim(1) != mat1_values_tensor.GetShape().GetDim(0)) {
    err_msg = ConcatString("mat1 indices dim[1] not equal values size");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  // check dimension
  if (WithRank(mat1_values_tensor, 1, unused_shape1, op) != GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(1, DebugString(mat1_values_tensor.GetShape().GetDims()), "1D");
    err_msg = string("MAT1 Values failed to call WithRank function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(input_indices_tensor, 2, unused_shape2, op) != GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(0, DebugString(input_indices_tensor.GetShape().GetDims()), "2D");
    err_msg = string("input indices failed to call WithRank function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(input_values_tensor, 1, unused_shape1, op) != GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(1, DebugString(input_values_tensor.GetShape().GetDims()), "1D");
    err_msg = string("input values failed to call WithRank function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (input_indices_tensor.GetShape().GetDim(1) != input_values_tensor.GetShape().GetDim(0)) {
    err_msg = ConcatString("input indices dim[1] not equal values size");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(mat2_tensor, 2, mat2_shape, op) != GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(3, DebugString(mat2_tensor.GetShape().GetDims()), "2D");
    err_msg = string("mat2 tensor failed to call WithRank function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  // check shape
  int64_t mat2_right_shape = mat2_shape.GetDim(1);
  int64_t mat2_left_shape = mat2_shape.GetDim(0);
  int64_t mat1_right_shape = mat1_shape_value[1];
  int64_t mat1_left_shape = mat1_shape_value[0];
  int64_t input_right_shape = input_shape_value[1];
  int64_t input_left_shape = input_shape_value[0];
  int64_t mat1_indices_right_shape = mat1_indices_desc.GetShape().GetDim(1);
  // for mat1 @mat2
  graphStatus status = Merge(mat1_right_shape, mat2_left_shape, unused_dim1);
  if (status != GRAPH_SUCCESS) {
    err_msg =
      ConcatString("failed to call Merge function to merge mat1 and mat2 ", mat1_right_shape, " and ", mat2_left_shape);
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  // for input + mat1
  status = Merge(mat1_left_shape, input_left_shape, unused_dim1);
  if (status != GRAPH_SUCCESS) {
    err_msg =
      ConcatString("failed to call Merge function to merge input and mat1", mat1_left_shape, " and ", input_left_shape);
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  status = Merge(mat2_right_shape, input_right_shape, unused_dim1);
  if (status != GRAPH_SUCCESS) {
    err_msg = ConcatString("failed to call Merge function to merge input and mat2", mat2_right_shape, " and ",
                           input_right_shape);
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  // refer SparseConcat, Set for output
  std::set<int64_t> mat1_unique_row_set;
  for (int i = 0; i < mat1_indices_right_shape; i++) {
    mat1_unique_row_set.insert(mat1_indices_value[i]);
  }

  int64_t y_indices_right = mat1_unique_row_set.size() * mat2_right_shape + input_indices_tensor.GetShape().GetDim(1);

  // output shape and indices to be set as zero, we will memecpy it at
  // calculating
  Shape y_indices_shape({2, y_indices_right});
  y_indices_desc.SetDataType(DT_INT64);
  y_indices_desc.SetShape(y_indices_shape);
  if (op.UpdateOutputDesc("output_indices", y_indices_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("update output[output_indices] desc failed."));
    return GRAPH_FAILED;
  }

  auto mat1_type = mat1_values_tensor.GetDataType();
  auto input_type = input_values_tensor.GetDataType();
  auto mat2_type = mat2_tensor.GetDataType();
  if (mat1_type != input_type || mat1_type != mat2_type || input_type != mat2_type) {
    err_msg = ConcatString("dtype of input, mat1, mat2 should be same");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  y_values_desc.SetDataType(input_type);
  Shape y_values_shape({y_indices_right});
  y_values_desc.SetShape(y_values_shape);
  if (op.UpdateOutputDesc("output_values", y_values_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("update output[output_values] desc failed."));
    return GRAPH_FAILED;
  }

  y_shape_desc.SetDataType(DT_INT64);
  Shape output_shape_shape({2});
  y_shape_desc.SetShape(output_shape_shape);
  if (op.UpdateOutputDesc("output_shape", y_shape_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("update output[output_shape] desc failed."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_INFER_FUNC_REG(Sspaddmm, SspaddmmInfer);
// -------------------------------Sspaddmm End------------------------------- //
}  // namespace ge