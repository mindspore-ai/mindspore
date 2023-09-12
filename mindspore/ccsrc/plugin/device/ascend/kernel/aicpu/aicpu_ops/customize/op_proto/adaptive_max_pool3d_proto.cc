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

#include "inc/adaptive_max_pool3d_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"

namespace ge {
CUST_IMPLEMT_INFERFUNC(AdaptiveMaxPool3d, AdaptiveMaxPool3dInferShape) {
  TensorDesc input = op.GetInputDesc(0);
  TensorDesc output_size = op.GetInputDesc(1);
  TensorDesc output = op.GetOutputDesc(0);
  TensorDesc argmax = op.GetOutputDesc(1);

  const size_t input_num_dims = input.GetShape().GetDimNum();
  const std::vector<int64_t> output_size_shape = output_size.GetShape().GetDims();
  if ((input_num_dims == 4 || input_num_dims == 5) == false) {
    OP_LOGE(TbeGetName(op), "Input dimensions must be equal to 4 or 5.");
    return GRAPH_FAILED;
  }
  if (output_size_shape.size() != 1) {
    OP_LOGE(TbeGetName(op), "output_size dim should be equal to 1.");
    return GRAPH_FAILED;
  }
  if (output_size_shape[0] != 3) {
    OP_LOGE(TbeGetName(op), "output_size shape[0] should be equal to 3.");
    return GRAPH_FAILED;
  }

  DataType input_dtype = input.GetDataType();
  Shape output_shape(UNKNOWN_SHAPE);
  output.SetDataType(input_dtype);
  output.SetShape(output_shape);
  argmax.SetDataType(DT_INT32);
  argmax.SetShape(output_shape);
  (void)op.UpdateOutputDesc("y", output);
  (void)op.UpdateOutputDesc("argmax", argmax);
  return GRAPH_SUCCESS;
}

CUST_IMPLEMT_VERIFIER(AdaptiveMaxPool3d, AdaptiveMaxPool3dVerify) { return GRAPH_SUCCESS; }

CUST_INFER_FUNC_REG(AdaptiveMaxPool3d, AdaptiveMaxPool3dInferShape);
CUST_VERIFY_FUNC_REG(AdaptiveMaxPool3d, AdaptiveMaxPool3dVerify);
}  // namespace ge