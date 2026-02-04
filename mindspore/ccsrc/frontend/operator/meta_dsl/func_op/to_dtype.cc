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

#include "mindspore/ccsrc/frontend/operator/meta_dsl/func_op/to_dtype.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"

namespace mindspore::prim {
BeginFunction(ToDtype, x, dtype, non_blocking, copy) {
  const auto &dtype_abs = dtype->abstract();
  MS_EXCEPTION_IF_NULL(dtype_abs);
  auto dtype_value = dtype_abs->BuildValue();
  if (dtype_value->isa<Int64Imm>()) {
    MS_LOG(DEBUG) << "Insert cast for primitive " << prim().get() << " " << prim()->ToString();
    Return(Call(Prim(Cast), x, dtype));
  } else {
    MS_LOG(DEBUG) << "No need insert cast for primitive " << prim().get() << " " << prim()->ToString();
    Return(x);
  }
}
EndFunction(ToDtype)
}  // namespace mindspore::prim
