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

#include "inc/generate_eod_mask.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"

namespace ge {
// -----------------------GenerateEodMask Starts----------------------
IMPLEMT_COMMON_INFERFUNC(GenerateEodMaskInferShape) {
  Shape x_shape = op.GetInputDescByName("inputs_ids").GetShape();
  DataType x_dtype = op.GetInputDescByName("inputs_ids").GetDataType();
  int64_t eod_token_id;
  int64_t n_pos;
  int64_t n_step;

  if (op.GetAttr("eod_token_id", eod_token_id) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "For Eig, get attr compute_v failed!");
    return GRAPH_FAILED;
  }
  if (op.GetAttr("n_step", n_step) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "For Eig, get attr compute_v failed!");
    return GRAPH_FAILED;
  }

  if (op.GetAttr("n_pos", n_pos) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "For Eig, get attr compute_v failed!");
    return GRAPH_FAILED;
  }
  vector<int64_t> x_shape_list = x_shape.GetDims();
  TensorDesc position_ids = op.GetOutputDescByName("position_ids");
  position_ids.SetDataType(x_dtype);
  std::vector<int64_t> position_ids_shape_list = {};
  position_ids_shape_list.assign(x_shape_list.begin(), x_shape_list.end());

  Shape position_ids_shape(position_ids_shape_list);
  position_ids.SetShape(position_ids_shape);
  if (op.UpdateOutputDesc("position_ids", position_ids) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Failed to update output desc.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(GenerateEodMask, GenerateEodMaskInferShape);
}  // namespace ge
