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

#include "inc/multilabel_margin_loss_grad_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"

namespace ge {
CUST_IMPLEMT_VERIFIER(MultilabelMarginLossGrad, MultilabelMarginLossGradVerify) { return GRAPH_SUCCESS; }

CUST_VERIFY_FUNC_REG(MultilabelMarginLossGrad, MultilabelMarginLossGradVerify);

IMPLEMT_COMMON_INFERFUNC(MultilabelMarginLossGradInferShape) {
  Shape shape_x = op.GetInputDescByName("x").GetShape();
  Shape shape_target = op.GetInputDescByName("target").GetShape();
  Shape shape_is_target = op.GetInputDescByName("is_target").GetShape();
  DataType x_dtype = op.GetInputDescByName("x").GetDataType();
  DataType y_grad_dtype = op.GetInputDescByName("y_grad").GetDataType();
  DataType target_dtype = op.GetInputDescByName("target").GetDataType();
  DataType is_target_dtype = op.GetInputDescByName("is_target").GetDataType();
  size_t dims = shape_x.GetDims().size();
  if (y_grad_dtype != x_dtype) {
    string err_msg1 = ConcatString("Dtype of input x must be the same as y_grad.");
    std::string err_msg = OtherErrMsg(err_msg1);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (x_dtype != DT_FLOAT && x_dtype != DT_FLOAT16) {
    string err_msg1 = ConcatString("Dtype of input x must be float or float16.");
    std::string err_msg = OtherErrMsg(err_msg1);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (target_dtype != DT_INT32 || is_target_dtype != DT_INT32) {
    string err_msg1 = ConcatString("Dtype of input target and is_target must be int32.");
    std::string err_msg = OtherErrMsg(err_msg1);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if ((shape_x.GetDimNum() != 2) && (shape_x.GetDimNum() != 1)) {
    string err_msg2 = ConcatString("Rank of x must be 1 or 2, shape_x.GetDimNum():", shape_x.GetDimNum());
    std::string err_msg = OtherErrMsg(err_msg2);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (shape_x.GetDimNum() != shape_target.GetDimNum()) {
    string err_msg2 = ConcatString("Rank of target must be the same as x, shape_x.GetDimNum():", shape_x.GetDimNum(),
                                   ", shape_target.GetDimNum():", shape_target.GetDimNum());
    std::string err_msg = OtherErrMsg(err_msg2);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  for (size_t i = 0; i < dims; i++) {
    if (shape_x.GetDim(i) != shape_target.GetDim(i) || shape_target.GetDim(i) != shape_is_target.GetDim(i)) {
      string err_msg2 = ConcatString("Shape of x, target, is_target must be the same.");
      std::string err_msg = OtherErrMsg(err_msg2);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }
  std::string reduction = "mean";
  op.GetAttr("reduction", reduction);
  if ((reduction != "mean") && (reduction != "sum") && (reduction != "none")) {
    string expected_reduction_list = ConcatString("mean, sum, none");
    std::string err_msg = GetInputFormatNotSupportErrMsg("reduction", expected_reduction_list, reduction);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  TensorDesc tensordesc_output = op.GetOutputDescByName("x_grad");
  Shape x_grad_shape = Shape(shape_x);
  tensordesc_output.SetShape(x_grad_shape);
  TensorDesc input_desc = op.GetInputDescByName("x");
  tensordesc_output.SetDataType(input_desc.GetDataType());
  op.UpdateOutputDesc("x_grad", tensordesc_output);
  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(MultilabelMarginLossGrad, MultilabelMarginLossGradInferShape);
}  // namespace ge