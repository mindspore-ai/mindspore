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

#include "inc/logspace.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"
namespace ge {
// --------------------------LogSpace---------------------
static bool CheckSteps(const Operator &op, const string &attr_num_steps) {
  int64_t steps = 0;
  int64_t steps_ori = 100;
  if (ge::GRAPH_SUCCESS != op.GetAttr(attr_num_steps.c_str(), steps)) {
    steps = steps_ori;
  }
  if (steps < 0) {
    return false;
  }
  return true;
}

CUST_IMPLEMT_VERIFIER(LogSpace, LogSpaceVerify) {
  AscendString opName;
  op.GetName(opName);
  if (op.GetInputDescByName("start").GetShape().GetDims().size() != 1) {
    OP_LOGE(opName.GetString(), "Input start size must be 1.");
    return GRAPH_FAILED;
  }
  if (op.GetInputDescByName("end").GetShape().GetDims().size() != 1) {
    OP_LOGE(opName.GetString(), "Input  end size must be 1.");
    return GRAPH_FAILED;
  }
  DataType input_type_start = op.GetInputDescByName("start").GetDataType();
  DataType input_type_end = op.GetInputDescByName("end").GetDataType();
  if (input_type_start != input_type_end) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(LogSpaceInferShape) {
  AscendString opName1;
  op.GetName(opName1);
  TensorDesc v_output_desc = op.GetOutputDescByName("y");
  int64_t steps;
  int64_t num_rows = 1;
  op.GetAttr("steps", steps);
  if (!CheckSteps(op, "steps")) {
    OP_LOGE(opName1.GetString(), "the attr 'steps' should be greater than or equal to 0.");
    return GRAPH_FAILED;
  }
  std::vector<int64_t> dim_vec;
  dim_vec.push_back(num_rows);
  dim_vec.push_back(steps);
  v_output_desc.SetShape(ge::Shape(dim_vec));
  int64_t dtype = 1;
  if (op.GetAttr("dtype", dtype) != GRAPH_SUCCESS) {
    v_output_desc.SetDataType(DT_FLOAT16);
  } else {
    if (dtype == 1) {
      v_output_desc.SetDataType(DT_FLOAT16);
    }
    if (dtype == 0) {
      v_output_desc.SetDataType(DT_FLOAT);
    }
  }
  (void)op.UpdateOutputDesc("y", v_output_desc);
  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(LogSpace, LogSpaceInferShape);
// Registered verify function
CUST_VERIFY_FUNC_REG(LogSpace, LogSpaceVerify);
// --------------------------LogSpace END---------------------
}  // namespace ge