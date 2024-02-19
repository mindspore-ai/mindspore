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

#include "op_proto/inc/selection_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"
namespace ge {
// ----------------Cumprod-------------------
IMPLEMT_COMMON_INFERFUNC(CumprodInferShape) {
  TensorDesc desc = op.GetInputDescByName("x");
  return op.UpdateOutputDesc("y", desc);
}

COMMON_INFER_FUNC_REG(Cumprod, CumprodInferShape);
// ----------------Cumprod END-------------------
}  // namespace ge