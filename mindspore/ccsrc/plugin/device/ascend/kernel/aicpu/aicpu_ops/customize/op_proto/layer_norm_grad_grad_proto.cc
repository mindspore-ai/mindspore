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

#include "inc/layer_norm_grad_grad_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"

namespace ge {
namespace {
#define Check2InputShapeSame(op1_name, op2_name)                                                    \
  do {                                                                                              \
    std::vector<int64_t> op1_name##_dims = op.GetInputDescByName(#op1_name).GetShape().GetDims();   \
    std::vector<int64_t> op2_name##_dims = op.GetInputDescByName(#op2_name).GetShape().GetDims();   \
    if (!(op1_name##_dims == op2_name##_dims)) {                                                    \
      string msg = ConcatString("The shape of ", #op1_name, "and", #op2_name, " must be the same"); \
      std::string err_msg = OtherErrMsg(msg);                                                       \
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);                         \
      return GRAPH_FAILED;                                                                          \
    }                                                                                               \
  } while (0)

#define Check3InputShapeSame(op1_name, op2_name, op3_name)                                                     \
  do {                                                                                                         \
    std::vector<int64_t> op1_name##_dims = op.GetInputDescByName(#op1_name).GetShape().GetDims();              \
    std::vector<int64_t> op2_name##_dims = op.GetInputDescByName(#op2_name).GetShape().GetDims();              \
    std::vector<int64_t> op3_name##_dims = op.GetInputDescByName(#op3_name).GetShape().GetDims();              \
    if (!(op1_name##_dims == op2_name##_dims && op1_name##_dims == op3_name##_dims)) {                         \
      string msg = ConcatString("The shape of ", #op1_name, #op2_name, "and", #op3_name, " must be the same"); \
      std::string err_msg = OtherErrMsg(msg);                                                                  \
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);                                    \
      return GRAPH_FAILED;                                                                                     \
    }                                                                                                          \
  } while (0)
}  // namespace

IMPLEMT_COMMON_INFERFUNC(LayerNormGradGradInferShape) {
  std::vector<std::string> vec_input;
  vec_input.push_back("x");
  vec_input.push_back("dy");
  vec_input.push_back("variance");
  vec_input.push_back("mean");
  vec_input.push_back("gamma");
  vec_input.push_back("d_dx");
  vec_input.push_back("d_dg");
  vec_input.push_back("d_db");

  if (!CheckInputDtypeSame(op, vec_input)) {
    OP_LOGE(TbeGetName(op).c_str(), "Input dtypes are not the same.");
    return GRAPH_FAILED;
  }

  Check3InputShapeSame(x, d_dx, dy);
  Check3InputShapeSame(gamma, d_dg, d_db);
  Check2InputShapeSame(variance, mean);

  if (op.GetInputDescByName("gamma").GetShape().GetDims().size() < 1) {
    OP_LOGE(TbeGetName(op).c_str(), "normalized shape to be at least 1-dimensional");
    return GRAPH_FAILED;
  }

  TensorDesc dy_desc = op.GetInputDescByName("dy");
  TensorDesc gamma_desc = op.GetInputDescByName("gamma");
  TensorDesc x_desc = op.GetInputDescByName("x");

  TensorDesc sopd_x_desc = op.GetOutputDescByName("sopd_x");
  TensorDesc sopd_dy_desc = op.GetOutputDescByName("sopd_dy");
  TensorDesc sopd_gamma_desc = op.GetOutputDescByName("sopd_gamma");

  sopd_x_desc.SetFormat(x_desc.GetFormat());
  sopd_x_desc.SetShape(x_desc.GetShape());
  sopd_x_desc.SetDataType(x_desc.GetDataType());
  if (op.UpdateOutputDesc("sopd_x", sopd_x_desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  sopd_dy_desc.SetFormat(dy_desc.GetFormat());
  sopd_dy_desc.SetDataType(dy_desc.GetDataType());
  sopd_dy_desc.SetShape(dy_desc.GetShape());
  if (op.UpdateOutputDesc("sopd_dy", sopd_dy_desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  sopd_gamma_desc.SetFormat(gamma_desc.GetFormat());
  sopd_gamma_desc.SetDataType(gamma_desc.GetDataType());
  sopd_gamma_desc.SetShape(gamma_desc.GetShape());
  if (op.UpdateOutputDesc("sopd_gamma", sopd_gamma_desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(LayerNormGradGrad, LayerNormGradGradInferShape);
}  // namespace ge