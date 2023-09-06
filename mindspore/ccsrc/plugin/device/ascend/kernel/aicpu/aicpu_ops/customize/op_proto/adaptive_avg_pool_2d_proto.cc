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

#include "custom_op_proto/cust_nn_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"

namespace ge {
// ---------------AdaptiveAvgPool2D-------------------
CUST_IMPLEMT_INFERFUNC(AdaptiveAvgPool2D, AdaptiveAvgPool2dInferShape) {
  OP_LOGI(TbeGetName(op).c_str(), " AdaptiveAvgPool2d inferShape begin!");
  const size_t DIM_SIZE2 = 2;
  auto input_tensor_desc = op.GetInputDescByName("x");
  auto shape = input_tensor_desc.GetShape();
  // get output_size
  std::vector<int64_t> ouput_size_list;
  if (GRAPH_SUCCESS != op.GetAttr("output_size", ouput_size_list)) {
    OP_LOGE(TbeGetName(op).c_str(), "GetOpAttr ouput_size_list failed!");
    return GRAPH_FAILED;
  }
  // check output size
  if (ouput_size_list.size() != DIM_SIZE2) {
    OP_LOGE(TbeGetName(op).c_str(), "length of output_size must be 2");
    return GRAPH_FAILED;
  }
  std::vector<int64_t> dims_input = shape.GetDims();
  // set output shape
  std::vector<int64_t> dim_vector;
  for (size_t i = 0; i < dims_input.size(); i++) {
    int64_t dims = dims_input[i];
    dim_vector.push_back(dims);
  }
  size_t index0 = dims_input.size() - 2;
  size_t index1 = dims_input.size() - 1;
  dim_vector[index0] = ouput_size_list[0];
  dim_vector[index1] = ouput_size_list[1];
  TensorDesc td = op.GetOutputDescByName("y");
  DataType input_dtype = input_tensor_desc.GetDataType();
  Shape output_shape(dim_vector);
  td.SetShape(output_shape);
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

CUST_IMPLEMT_VERIFIER(AdaptiveAvgPool2D, AdaptiveAvgPool2dVerify) { return GRAPH_SUCCESS; }

CUST_INFER_FUNC_REG(AdaptiveAvgPool2D, AdaptiveAvgPool2dInferShape);
CUST_VERIFY_FUNC_REG(AdaptiveAvgPool2D, AdaptiveAvgPool2dVerify);
// ---------------AdaptiveAvgPool2D End---------------
}  // namespace ge