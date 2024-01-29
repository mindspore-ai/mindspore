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

#include "inc/minimum_grad_grad.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"
namespace ge {
// ----------------MinimumGradGrad Begin-------------------
CUST_IMPLEMT_VERIFIER(MinimumGradGrad, MinimumGradGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "grad_y1", "grad_y2") || !CheckTwoInputDtypeSame(op, "x1", "x2") ||
      !CheckTwoInputDtypeSame(op, "x1", "grad_y1")) {
    return GRAPH_FAILED;
  }
  ge::Shape shape_grad_y1 = op.GetInputDescByName("grad_y1").GetShape();
  ge::Shape shape_grad_y2 = op.GetInputDescByName("grad_y2").GetShape();
  ge::Shape shape_x1 = op.GetInputDescByName("x1").GetShape();
  ge::Shape shape_x2 = op.GetInputDescByName("x2").GetShape();
  std::vector<int64_t> dim_grad_y1 = shape_grad_y1.GetDims();
  std::vector<int64_t> dim_grad_y2 = shape_grad_y2.GetDims();
  std::vector<int64_t> dims_x1 = shape_x1.GetDims();
  std::vector<int64_t> dims_x2 = shape_x2.GetDims();
  if ((dim_grad_y1 != dims_x1)) {
    std::string err_msg1 = "The size of inputs grad_y1 and x1 are not the same.";
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg1);
    return GRAPH_FAILED;
  }
  if ((dim_grad_y2 != dims_x2)) {
    std::string err_msg2 = "The size of input grad_y2 and x2 are not the same.";
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg2);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
CUST_VERIFY_FUNC_REG(MinimumGradGrad, MinimumGradGradVerify);

IMPLEMT_COMMON_INFERFUNC(MinimumGradGradInferShape) {
  bool is_dynamic_output = true;
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "grad_y1", "grad_y2", "spod_grads", is_dynamic_output)) {
    return GRAPH_FAILED;
  }

  auto grad_y1_dtype = op.GetInputDesc("grad_y1").GetDataType();
  std::vector<int64_t> x_dims = {1};
  auto spod_x1_desc = op.GetOutputDesc("spod_x1");
  spod_x1_desc.SetShape(Shape(x_dims));
  spod_x1_desc.SetDataType(grad_y1_dtype);

  auto spod_x2_desc = op.GetOutputDesc("spod_x2");
  spod_x2_desc.SetShape(Shape(x_dims));
  spod_x2_desc.SetDataType(grad_y1_dtype);
  op.UpdateOutputDesc("spod_x1", spod_x1_desc);
  op.UpdateOutputDesc("spod_x2", spod_x2_desc);
  return GRAPH_SUCCESS;
}
CUST_COMMON_INFER_FUNC_REG(MinimumGradGrad, MinimumGradGradInferShape);
// ----------------MinimumGradGrad END-----------
}  // namespace ge