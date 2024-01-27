/**
 * Copyright (c) 2023 Huawei Technologies Co., Ltd.  All rights reserved.
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

#include "op_proto/inc/bitwise_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"

namespace ge {
// -----------------------------LeftShift-----------------------------
IMPLEMT_VERIFIER(LeftShift, LeftShiftVerify) {
  if (!CheckTwoInputDtypeSame(op, "x", "y")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(LeftShift, LeftShiftInferShape) {
  Shape data_shape = op.GetInputDescByName("x").GetShape();
  DataType input_dtype = op.GetInputDescByName("x").GetDataType();
  TensorDesc td = op.GetOutputDescByName("z");
  td.SetShape(ge::Shape(data_shape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("z", td);
  return BROADCAST_INFER("x", "y", "z")(op);
}

INFER_FUNC_REG(LeftShift, LeftShiftInferShape);
VERIFY_FUNC_REG(LeftShift, LeftShiftVerify);
// -----------------------------LeftShift END-----------------------------
}  // namespace ge
