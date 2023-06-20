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

#include "inc/index_fill.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"
namespace ge {
// ----------------IndexFill-------------------
// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(IndexFillInferShape) {
  TensorDesc v_output_desc = op.GetOutputDescByName("y");

  DataType input_dtype = op.GetInputDescByName("x").GetDataType();
  Format input_format = op.GetInputDescByName("x").GetFormat();
  // shape of output y is the same as input x
  ge::Shape shape_input = op.GetInputDescByName("x").GetShape();

  v_output_desc.SetShape(shape_input);
  v_output_desc.SetDataType(input_dtype);
  v_output_desc.SetFormat(input_format);

  if (op.UpdateOutputDesc("y", v_output_desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Registered inferfunction
CUST_COMMON_INFER_FUNC_REG(IndexFill, IndexFillInferShape);
// ----------------IndexFill END-------------------
}  // namespace ge