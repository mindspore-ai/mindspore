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

#include "inc/pdist_grad_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"

namespace ge {
// --------------------PdistGrad----------------------
IMPLEMT_COMMON_INFERFUNC(PdistGradInferShape) {
  TensorDesc output_desc = op.GetOutputDescByName("x_grad");
  DataType input_dtype = op.GetInputDescByName("x").GetDataType();
  Format input_format = op.GetInputDescByName("x").GetFormat();

  ge::Shape grad_shape = op.GetInputDescByName("y_grad").GetShape();
  ge::Shape pdist_shape = op.GetInputDescByName("pdist").GetShape();
  ge::Shape input_shape = op.GetInputDescByName("x").GetShape();

  std::vector<int64_t> grad_dim = grad_shape.GetDims();
  std::vector<int64_t> pdist_dim = pdist_shape.GetDims();
  std::vector<int64_t> input_dim = input_shape.GetDims();
  if (input_dim.size() != 2) {
    OP_LOGE(TbeGetName(op).c_str(), "The shape of input x must be 2.");
    return GRAPH_FAILED;
  }
  if ((pdist_dim.size() != 1) || (grad_dim.size() != 1)) {
    OP_LOGE(TbeGetName(op).c_str(), "The shape of pdist and grad must both be 1.");
    return GRAPH_FAILED;
  }
  ge::Shape output_shape = ge::Shape(input_dim);
  output_desc.SetShape(output_shape);
  output_desc.SetDataType(input_dtype);
  output_desc.SetFormat(input_format);
  op.UpdateOutputDesc("x_grad", output_desc);
  return GRAPH_SUCCESS;
}

CUST_IMPLEMT_VERIFIER(PdistGrad, PdistGradVerify) {
  DataType input_dtype = op.GetInputDescByName("x").GetDataType();
  DataType grad_dtype = op.GetInputDescByName("y_grad").GetDataType();
  DataType pdist_dtype = op.GetInputDescByName("pdist").GetDataType();
  if ((grad_dtype != input_dtype) || (pdist_dtype != input_dtype)) {
    OP_LOGE(TbeGetName(op).c_str(), "The three input datatype must be the same.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(PdistGrad, PdistGradInferShape);
CUST_VERIFY_FUNC_REG(PdistGrad, PdistGradVerify);
// --------------------PdistGrad END----------------------
}  // namespace ge