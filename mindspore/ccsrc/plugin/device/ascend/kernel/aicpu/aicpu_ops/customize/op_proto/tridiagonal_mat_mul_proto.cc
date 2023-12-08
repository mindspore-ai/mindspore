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

#include "inc/tridiagonal_mat_mul_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"

namespace ge {
CUST_IMPLEMT_INFERFUNC(TridiagonalMatMul, TridiagonalMatMulInfer) {
  std::string error_msg;
  TensorDesc superdiag_desc = op.GetInputDesc(0);
  TensorDesc maindiag_desc = op.GetInputDesc(1);
  TensorDesc subdiag_desc = op.GetInputDesc(2);
  Shape rhs;
  TensorDesc rhs_desc = op.GetInputDesc(3);
  auto rhs_shape = rhs_desc.GetShape().GetDims();
  if (WithRankAtLeast(rhs_desc, 2, rhs, op) != GRAPH_SUCCESS) {
    error_msg =
      ConcatString("failed to call WithRankatleast function, ", "the rank of input[rhs] must at least be 2, but get ",
                   rhs_desc.GetShape().GetDimNum(), ".");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op).c_str(), error_msg);
    return GRAPH_FAILED;
  }
  int64_t existing_superdiag = static_cast<int64_t>(superdiag_desc.GetShape().GetDimNum());
  int64_t existing_maindiag = static_cast<int64_t>(maindiag_desc.GetShape().GetDimNum());
  int64_t existing_subdiag = static_cast<int64_t>(subdiag_desc.GetShape().GetDimNum());
  int64_t existing_rhs = static_cast<int64_t>(rhs_desc.GetShape().GetDimNum());
  int64_t m_superdiag = superdiag_desc.GetShape().GetDim(existing_superdiag - 1);
  int64_t m_maindiag = maindiag_desc.GetShape().GetDim(existing_maindiag - 1);
  int64_t m_subdiag = subdiag_desc.GetShape().GetDim(existing_subdiag - 1);
  int64_t m_rhs = rhs_desc.GetShape().GetDim(existing_rhs - 2);
  if (!(m_maindiag == m_superdiag && m_subdiag == m_maindiag && m_rhs == m_subdiag)) {
    error_msg = ConcatString("the length of each iuput must be the same with the row of rhs.");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), error_msg);
    return GRAPH_FAILED;
  }
  TensorDesc y_desc = op.GetOutputDescByName("y");
  y_desc.SetShape(Shape(rhs_shape));
  DataType type = op.GetInputDescByName("superdiag").GetDataType();
  y_desc.SetDataType(type);
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), std::string("fail to update output[y] desc."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
CUST_INFER_FUNC_REG(TridiagonalMatMul, TridiagonalMatMulInfer);
}  // namespace ge
