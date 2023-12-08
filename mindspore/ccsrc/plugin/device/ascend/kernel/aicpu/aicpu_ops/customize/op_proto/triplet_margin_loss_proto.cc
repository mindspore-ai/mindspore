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

#include "inc/triplet_margin_loss_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"

namespace ge {
// ----------------TripletMarginLoss Begin-------------------
CUST_IMPLEMT_VERIFIER(TripletMarginLoss, TripletMarginLossVerify) {
  Shape shape_x = op.GetInputDescByName("x").GetShape();
  Shape shape_xp = op.GetInputDescByName("positive").GetShape();
  Shape shape_xn = op.GetInputDescByName("negative").GetShape();
  DataType x_dtype = op.GetInputDescByName("x").GetDataType();
  DataType xp_dtype = op.GetInputDescByName("positive").GetDataType();
  DataType xn_dtype = op.GetInputDescByName("negative").GetDataType();
  if ((x_dtype != xp_dtype || xp_dtype != xn_dtype)) {
    string err_msg1 = ConcatString("dtype of input x  positive, negative must be the same.");
    std::string err_msg = OtherErrMsg(err_msg1);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }
  if ((shape_x.GetDimNum() > 8) || (shape_xp.GetDimNum() > 8) || (shape_xn.GetDimNum() > 8)) {
    string err_msg1 = ConcatString(
      "dimensions of input x or positive, negative must be smaller "
      "than 8, shape_x: ",
      shape_x.GetDimNum(), ", shape_positive: ", shape_xp.GetDimNum(), ", shape_negative.GetDimNum()",
      shape_xn.GetDimNum());
    std::string err_msg = OtherErrMsg(err_msg1);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }
  if ((shape_x.GetDimNum() <= 1) && (shape_xp.GetDimNum() <= 1) && (shape_xn.GetDimNum() <= 1)) {
    string err_msg1 = ConcatString(
      "At least one of dimensions of input x or positive, negative must be bigger"
      "than 1, rank_x:",
      shape_x.GetDimNum(), ", rank_positive:", shape_xp.GetDimNum(), ", rank_negative", shape_xn.GetDimNum());
    std::string err_msg = OtherErrMsg(err_msg1);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }
  std::string reduction;
  op.GetAttr("reduction", reduction);
  if ((reduction != "mean") && (reduction != "sum") && (reduction != "none")) {
    OP_LOGE(TbeGetName(op).c_str(), "The val of reduction is invalid.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(TripletMarginLossInferShape) {
  auto shape_x = op.GetInputDescByName("x").GetShape().GetDims();
  auto shape_xp = op.GetInputDescByName("positive").GetShape().GetDims();
  auto shape_xn = op.GetInputDescByName("negative").GetShape().GetDims();
  std::vector<int64_t> y_shape;
  int32_t dims = std::max(std::max(shape_x.size(), shape_xp.size()), shape_xn.size());
  std::reverse(shape_x.begin(), shape_x.end());
  std::reverse(shape_xp.begin(), shape_xp.end());
  std::reverse(shape_xn.begin(), shape_xn.end());
  shape_x.resize(dims, 1);
  shape_xp.resize(dims, 1);
  shape_xn.resize(dims, 1);
  std::reverse(shape_x.begin(), shape_x.end());
  std::reverse(shape_xp.begin(), shape_xp.end());
  std::reverse(shape_xn.begin(), shape_xn.end());
  for (int32_t i = 0; i < dims; i++) {
    y_shape.push_back(static_cast<int64_t>(std::max(std::max(shape_x[i], shape_xp[i]), shape_xn[i])));
    if ((shape_x[i] != y_shape[i] && shape_x[i] != 1) || (shape_xp[i] != y_shape[i] && shape_xp[i] != 1) ||
        (shape_xn[i] != y_shape[i] && shape_xn[i] != 1)) {
      std::string err_msg = OtherErrMsg("Inputs' shape can't broadcast");
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
      return GRAPH_FAILED;
    }
  }
  std::string reduction;
  (void)op.GetAttr("reduction", reduction);
  TensorDesc tensordesc_output = op.GetOutputDescByName("y");
  y_shape.erase(y_shape.begin() + 1);
  if ((reduction == "mean") || (reduction == "sum")) {
    Shape scalar_shape;
    Scalar(scalar_shape);
    tensordesc_output.SetShape(scalar_shape);
  }
  if (reduction == "none") {
    tensordesc_output.SetShape(Shape(y_shape));
  }
  if (op.GetInputDescByName("x").GetDataType() == DT_FLOAT16) {
    tensordesc_output.SetDataType(DT_FLOAT16);
  } else {
    tensordesc_output.SetDataType(DT_FLOAT);
  };
  tensordesc_output.SetFormat(FORMAT_ND);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(TripletMarginLoss, TripletMarginLossInferShape);
CUST_VERIFY_FUNC_REG(TripletMarginLoss, TripletMarginLossVerify);
// ----------------TripletMarginLoss END---------------------
}  // namespace ge
