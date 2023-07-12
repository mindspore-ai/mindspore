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

#include "inc/mvlgamma_grad_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"

namespace ge {
// ----------------MvlgammaGrad Begin-------------------
CUST_IMPLEMT_INFERFUNC(MvlgammaGrad, MvlgammaGradInferShape) {
  const char *op_name = "MvlgammaGrad";
  OP_LOGD(op_name, "MvlgammaGradInferShape begin.");
  TensorDesc tensordesc_input = op.GetInputDescByName("y_grad");
  Shape input_shape = tensordesc_input.GetShape();
  std::vector<int64_t> dims_input = input_shape.GetDims();
  DataType input_dtype = tensordesc_input.GetDataType();

  TensorDesc tensordesc_output1 = op.GetOutputDescByName("x_grad");
  tensordesc_output1.SetDataType(input_dtype);
  tensordesc_output1.SetShape(ge::Shape(dims_input));

  (void)op.UpdateOutputDesc("x_grad", tensordesc_output1);
  OP_LOGD(op_name, "MvlgammaGradInferShape end.");
  return GRAPH_SUCCESS;
}

CUST_IMPLEMT_VERIFIER(MvlgammaGrad, MvlgammaGradVerify) { return GRAPH_SUCCESS; }

CUST_INFER_FUNC_REG(MvlgammaGrad, MvlgammaGradInferShape);
CUST_VERIFY_FUNC_REG(MvlgammaGrad, MvlgammaGradVerify);
// ----------------MvlgammaGrad END---------------------
}  // namespace ge