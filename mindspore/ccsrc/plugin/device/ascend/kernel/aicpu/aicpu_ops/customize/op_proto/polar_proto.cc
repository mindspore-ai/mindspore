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

#include "transform/graph_ir/custom_op_proto/cust_math_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"

namespace ge {
// -----------------------------Polar-----------------------------
CUST_IMPLEMT_INFERFUNC(Polar, PolarInfer) {
  if (!CheckTwoInputDtypeSame(op, "abs", "angle")) {
    return GRAPH_FAILED;
  }
  TensorDesc abs_desc = op.GetInputDescByName("abs");
  TensorDesc angle_desc = op.GetInputDescByName("angle");
  DataType abs_type = abs_desc.GetDataType();
  Shape abs_shape = abs_desc.GetShape();
  Shape angle_shape = angle_desc.GetShape();
  if (abs_type != DT_FLOAT && abs_type != DT_DOUBLE) {
    OP_LOGE(TbeGetName(op).c_str(), "the data type of 'abs' should be DT_FLOAT or DT_DOUBLE.");
    return GRAPH_FAILED;
  }
  if (abs_shape.GetDims() != angle_shape.GetDims()) {
    OP_LOGE(TbeGetName(op).c_str(), "the shape of 'abs' and 'angle' should be the same.");
    return GRAPH_FAILED;
  }

  DataType input_dtype = abs_desc.GetDataType();
  TensorDesc td = op.GetOutputDescByName("y");
  td.SetShape(ge::Shape(abs_shape));
  if (input_dtype == DT_FLOAT) {
    td.SetDataType(DT_COMPLEX64);
  } else {
    td.SetDataType(DT_COMPLEX128);
  }

  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

CUST_INFER_FUNC_REG(Polar, PolarInfer);
// -----------------------------Polar END-----------------------------
}  // namespace ge
