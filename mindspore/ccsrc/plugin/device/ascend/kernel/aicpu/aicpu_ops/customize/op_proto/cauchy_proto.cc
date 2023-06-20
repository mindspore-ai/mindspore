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

#include "inc/cauchy_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"

namespace ge {
// ---------Cauchy-------------------
CUST_IMPLEMT_VERIFIER(Cauchy, CauchyVerify) {
  OP_LOGI(TbeGetName(op).c_str(), "Enter CauchyVerify.");
  return GRAPH_SUCCESS;
}
CUST_VERIFY_FUNC_REG(Cauchy, CauchyVerify);

IMPLEMT_COMMON_INFERFUNC(CauchyInferShape) {
  TensorDesc output_desc = op.GetOutputDescByName("y");
  std::vector<int64_t> shape_size{};
  if (op.GetAttr("size", shape_size) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Get attr size failed");
    return GRAPH_FAILED;
  }
  output_desc.SetShape(ge::Shape(shape_size));
  (void)op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(Cauchy, CauchyInferShape);
// ---------Cauchy End-------------------
}  // namespace ge