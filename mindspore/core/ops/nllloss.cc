/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "ops/nllloss.h"

#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
void NLLLoss::Init(const Reduction &reduction) { set_reduction(reduction); }

void NLLLoss::set_reduction(const Reduction &reduction) {
  int64_t reduce = reduction;
  (void)AddAttr(kReduction, MakeValue(reduce));
}

Reduction NLLLoss::get_reduction() const {
  auto value_ptr = GetAttr(kReduction);
  return Reduction(GetValue<int64_t>(value_ptr));
}

REGISTER_PRIMITIVE_C(kNameNLLLoss, NLLLoss)
}  // namespace ops
}  // namespace mindspore
