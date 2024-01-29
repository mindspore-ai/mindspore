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
REG_EXPANDER_FUNC("Tanh").SetBody(BODYFUNC(ib) {
  auto result = ib->Tanh(ib->input(kIndex0));
  return {result};
});
REG_EXPANDER_FUNC("Sinh").SetBody(BODYFUNC(ib) {
  auto result = ib->Sinh(ib->input(kIndex0));
  return {result};
});
REG_EXPANDER_FUNC("Cosh").SetBody(BODYFUNC(ib) {
  auto result = ib->Cosh(ib->input(kIndex0));
  return {result};
});
}  // namespace mindspore::graphkernel::expander
