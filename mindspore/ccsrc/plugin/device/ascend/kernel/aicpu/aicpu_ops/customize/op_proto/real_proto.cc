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
// -----------------------------Real-----------------------------
CUST_IMPLEMT_INFERFUNC(Real, RealInfer) {
  TensorDesc input_desc = op.GetInputDescByName("input");
  DataType input_type = input_desc.GetDataType();
  Shape input_shape = input_desc.GetShape();
  if (input_type != DT_COMPLEX64 && input_type != DT_COMPLEX128) {
    OP_LOGE(TbeGetName(op).c_str(), "the data type of 'input' should be DT_COMPLEX64 or DT_COMPLEX128.");
    return GRAPH_FAILED;
  }

  DataType input_dtype = input_desc.GetDataType();
  TensorDesc td = op.GetOutputDescByName("output");
  td.SetShape(ge::Shape(input_shape));
  if (input_dtype == DT_COMPLEX64) {
    td.SetDataType(DT_FLOAT);
  } else {
    td.SetDataType(DT_DOUBLE);
  }

  (void)op.UpdateOutputDesc("output", td);
  return GRAPH_SUCCESS;
}

CUST_INFER_FUNC_REG(Real, RealInfer);
// -----------------------------Real END-----------------------------
}  // namespace ge
