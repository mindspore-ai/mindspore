/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.  All rights reserved.
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

#include "custom_op_proto/cust_other_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"

namespace ge {
// -----------------------GenerateEodMaskV2 Starts----------------------
IMPLEMT_COMMON_INFERFUNC(GenerateEodMaskV2InferShape) {
  Shape x_shape = op.GetInputDescByName("inputs_ids").GetShape();
  DataType x_dtype = op.GetInputDescByName("inputs_ids").GetDataType();

  TensorDesc output_ids = op.GetOutputDescByName("output_ids");
  output_ids.SetDataType(x_dtype);
  output_ids.SetShape(ge::Shape(x_shape.GetDims()));
  if (op.UpdateOutputDesc("output_ids", output_ids) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Failed to update output desc.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(GenerateEodMaskV2, GenerateEodMaskV2InferShape);
}  // namespace ge
