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

#include "extendrt/delegate/ascend_native/ops/ascend_native_composite.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/common.h"
#include "mindapi/ir/value.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(AscendNativeComposite, BaseOperator);
void AscendNativeComposite::Init(int64_t group) { this->set_group(group); }

void AscendNativeComposite::set_group(int64_t group) { (void)this->AddAttr(kGroup, api::MakeValue(group)); }

int64_t AscendNativeComposite::get_group() const {
  auto value_ptr = this->GetAttr(kGroup);
  return GetValue<int64_t>(value_ptr);
}

REGISTER_PRIMITIVE_C(kNameAscendNativeComposite, AscendNativeComposite);
}  // namespace ops
}  // namespace mindspore
