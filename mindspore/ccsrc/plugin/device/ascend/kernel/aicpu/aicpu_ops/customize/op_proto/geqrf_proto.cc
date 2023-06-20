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

#include "inc/geqrf_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"

namespace ge {
// ---------------Geqrf-------------------
CUST_IMPLEMT_INFERFUNC(Geqrf, GeqrfInfer) {
  auto tensor = op.get_input_desc_x();
  Shape input;
  if (WithRank(tensor, 2, input, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  int dim_num = input.GetDimNum();
  int m = input.GetDim(dim_num - 2);
  int n = input.GetDim(dim_num - 1);
  Shape r_shape;
  Shape tau_shape;
  int p = m > n ? n : m;
  Matrix(m, n, r_shape);
  Vector(p, tau_shape);

  DataType type = op.GetInputDescByName("x").GetDataType();
  TensorDesc r_desc = op.GetOutputDescByName("r");
  r_desc.SetShape(Shape(r_shape));
  r_desc.SetDataType(type);
  if (op.UpdateOutputDesc("r", r_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Update r desc failed.");
    return GRAPH_FAILED;
  }

  TensorDesc tau_desc = op.GetOutputDescByName("tau");
  tau_desc.SetShape(Shape(tau_shape));
  tau_desc.SetDataType(type);
  if (op.UpdateOutputDesc("tau", tau_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Update tau desc failed.");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

CUST_INFER_FUNC_REG(Geqrf, GeqrfInfer);

CUST_IMPLEMT_VERIFIER(Geqrf, GeqrfVerify) {
  DataType type = op.GetInputDescByName("x").GetDataType();
  if (type != DT_FLOAT16 && type != DT_FLOAT && type != DT_DOUBLE && type != DT_COMPLEX64 && type != DT_COMPLEX128) {
    OP_LOGE(TbeGetName(op).c_str(), "Expect a floating point or complex tensor as input.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_VERIFY_FUNC_REG(Geqrf, GeqrfVerify);
// ---------------Geqrf End---------------
}  // namespace ge