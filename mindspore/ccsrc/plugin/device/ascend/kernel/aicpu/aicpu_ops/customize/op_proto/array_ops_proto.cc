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

#include "custom_op_proto/cust_array_ops.h"
#include "inc/ops/array_ops.h"
#include "register/op_impl_registry.h"
#include "utils/common_shape_fns.h"
#include "utils/util.h"

namespace ge {
// ---------------SliceGrad-------------------
CUST_IMPLEMT_INFERFUNC(SliceGrad, SliceGradInfer) {
  TensorDesc x_desc = op.GetInputDescByName("x");
  if (op.UpdateOutputDesc("dx", x_desc) != GRAPH_SUCCESS) {
    OP_LOGE("SliceGrad", "Update output desc failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_INFER_FUNC_REG(SliceGrad, SliceGradInfer);
// ---------------SliceGrad End---------------
}  // namespace ge