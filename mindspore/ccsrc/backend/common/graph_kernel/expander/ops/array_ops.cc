/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "backend/common/graph_kernel/expander/base/ir_builder.h"

namespace mindspore::graphkernel::expander {
REG_EXPANDER_FUNC("ZerosLike").SetBody(BODYFUNC(ib) {
  const auto &input_x = ib->input(kIndex0);
  auto shape = ib->Value(input_x->GetShape());
  auto const_zero = ib->Tensor(0, input_x->GetDtype());
  auto result = ib->BroadcastTo(const_zero, shape);
  return {result};
});
}  // namespace mindspore::graphkernel::expander
