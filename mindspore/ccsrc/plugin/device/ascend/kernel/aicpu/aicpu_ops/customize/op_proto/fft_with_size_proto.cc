/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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

#include "inc/fft_with_size_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"

namespace ge {
IMPLEMT_COMMON_INFERFUNC(FFTWithSizeInferShape) {
  TensorDesc out_desc = op.GetOutputDescByName("x");
  out_desc.SetDataType(op.GetInputDescByName("x").GetDataType());
  if (op.UpdateOutputDesc("x", out_desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(FFTWithSize, FFTWithSizeInferShape);
}  // namespace ge