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

#include "inc/adaptive_max_pool3_d_grad_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"

namespace ge {
CUST_IMPLEMT_INFERFUNC(AdaptiveMaxPool3dGrad, AdaptiveMaxPool3dGradInferShape) {
  TensorDesc output_grad = op.GetOutputDescByName("output_grad");
  TensorDesc input = op.GetInputDescByName("x");
  DataType input_dtype = input.GetDataType();
  Shape input_shape = input.GetShape();
  output_grad.SetShape(input_shape);
  output_grad.SetDataType(input_dtype);
  if (op.UpdateOutputDesc("output_grad", output_grad) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_IMPLEMT_VERIFIER(AdaptiveMaxPool3dGrad, AdaptiveMaxPool3dGradVerify) { return GRAPH_SUCCESS; }

CUST_INFER_FUNC_REG(AdaptiveMaxPool3dGrad, AdaptiveMaxPool3dGradInferShape);
CUST_VERIFY_FUNC_REG(AdaptiveMaxPool3dGrad, AdaptiveMaxPool3dGradVerify);
}  // namespace ge