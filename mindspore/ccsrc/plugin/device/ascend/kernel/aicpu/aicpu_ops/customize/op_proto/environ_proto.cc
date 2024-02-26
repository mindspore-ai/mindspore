/**
 * Copyright (c) 2022-2023 Huawei Technologies Co., Ltd.  All rights reserved.
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

#include "custom_op_proto/cust_environ_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"

namespace ge {
// ----------------EnvironCreate InferShape------------------------
IMPLEMT_COMMON_INFERFUNC(EnvironCreateInferShape) {
  TensorDesc output_desc = op.GetOutputDescByName("handle");
  output_desc.SetShape(ge::Shape({1}));
  output_desc.SetDataType(DT_INT64);
  if (op.UpdateOutputDesc("handle", output_desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
CUST_COMMON_INFER_FUNC_REG(EnvironCreate, EnvironCreateInferShape);

// ----------------EnvironDestroyAll InferShape------------------------
IMPLEMT_COMMON_INFERFUNC(EnvironDestroyAllInferShape) {
  TensorDesc output_desc = op.GetOutputDescByName("result");
  output_desc.SetShape(ge::Shape({1}));
  output_desc.SetDataType(DT_BOOL);
  if (op.UpdateOutputDesc("result", output_desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
CUST_COMMON_INFER_FUNC_REG(EnvironDestroyAll, EnvironDestroyAllInferShape);

// ----------------EnvironSet InferShape------------------------
IMPLEMT_COMMON_INFERFUNC(EnvironSetInferShape) { return GRAPH_SUCCESS; }
CUST_COMMON_INFER_FUNC_REG(EnvironSet, EnvironSetInferShape);

// ----------------EnvironGet InferShape------------------------
IMPLEMT_COMMON_INFERFUNC(EnvironGetInferShape) {
  auto default_desc = op.GetInputDescByName("default");
  TensorDesc output_desc = op.GetOutputDescByName("value");
  DataType input_dtype = default_desc.GetDataType();
  ge::Shape input_shape = default_desc.GetShape();
  // shape of output  is the same as input default
  output_desc.SetShape(input_shape);
  output_desc.SetDataType(input_dtype);
  if (op.UpdateOutputDesc("value", output_desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}
CUST_COMMON_INFER_FUNC_REG(EnvironGet, EnvironGetInferShape);
}  // namespace ge