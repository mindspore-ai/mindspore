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

#include "inc/hamming_window_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"

namespace ge {
// ----------------HammingWindow Begin---------------------
IMPLEMT_COMMON_INFERFUNC(HammingWindowInferShape) {
  std::vector<int64_t> input_dim = op.GetInputDesc(0).GetShape().GetDims();
  if (input_dim.size() != 1) {
    OP_LOGE(TbeGetName(op).c_str(), "Tensor length input must be 1D.");
    return GRAPH_FAILED;
  }

  Tensor length_tensor;
  int64_t length_data;
  if (op.GetInputConstData("length", length_tensor) == GRAPH_SUCCESS) {
    uint8_t *length = length_tensor.GetData();
    length_data = static_cast<int64_t>(*length);
  } else {
    length_data = UNKNOWN_DIM;
  }
  std::vector<int64_t> output_dim;
  if (length_data != UNKNOWN_DIM && length_data < 0) {
    OP_LOGE(TbeGetName(op).c_str(), "Non-negative window length required, got [%ld].", length_data);
    return GRAPH_FAILED;
  }
  if (length_data != 0) {
    output_dim.push_back(length_data);
  }
  ge::Shape output_shape = ge::Shape(output_dim);

  Operator::OpInt dtype;
  if (op.GetAttr("dtype", dtype) != GRAPH_SUCCESS) {
    dtype = 0;
  }
  DataType output_dtype = static_cast<DataType>(dtype);

  TensorDesc output_desc = op.GetOutputDescByName("y");
  output_desc.SetShape(output_shape);
  output_desc.SetDataType(output_dtype);
  op.UpdateOutputDesc("y", output_desc);

  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(HammingWindow, HammingWindowInferShape);
// ----------------HammingWindow End---------------------
}  // namespace ge