/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "mindspore/ccsrc/frontend/operator/meta_dsl/func_op/any_ext.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "mindspore/ops/ops_utils/op_constants.h"

namespace mindspore::prim {
BeginFunction(AnyExt, input, dim, keepdim) {
  auto axis = Tuple(dim);
  auto out = Call(Prim(ReduceAny), input, axis, keepdim);
  Return(out);
}
EndFunction(AnyExt)
}  // namespace mindspore::prim
