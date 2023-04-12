/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "frontend/expander/bprop/bprop_irbuilder.h"
#include "frontend/expander/bprop/grad_ops/common_utils.h"
#include "include/common/utils/utils.h"

namespace mindspore::expander::bprop {
REG_BPROP_BUILDERS_BEGIN(GradSequenceOps)
REG_BPROP_BUILDER("make_range").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInputs();
  auto id_type = ib->GetDtypeId(ib->GetInput(kIndex0));
  if (id_type == TypeId::kNumberTypeInt32) {
    if (x.size() == 1) {
      return {ib->Value(0)};
    } else if (x.size() == 2) {
      return {ib->Value(0), ib->Value(0)};
    } else {
      return {ib->Value(0), ib->Value(0), ib->Value(0)};
    }
  } else {
    if (x.size() == 1) {
      return {ib->Value<int64_t>(0)};
    } else if (x.size() == 2) {
      return {ib->Value<int64_t>(0), ib->Value<int64_t>(0)};
    } else {
      return {ib->Value<int64_t>(0), ib->Value<int64_t>(0), ib->Value<int64_t>(0)};
    }
  }
});
REG_BPROP_BUILDERS_END
}  // namespace mindspore::expander::bprop
