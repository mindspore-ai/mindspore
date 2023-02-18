/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "ops/lsh_projection.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(LshProjection, BaseOperator);
void LshProjection::Init(const LshProjectionType &type) { set_type(type); }

void LshProjection::set_type(const LshProjectionType &type) {
  int64_t swi = static_cast<int64_t>(type);
  (void)AddAttr(kType, api::MakeValue(swi));
}

LshProjectionType LshProjection::get_type() const { return LshProjectionType(GetValue<int64_t>(GetAttr(kType))); }

REGISTER_PRIMITIVE_C(kNameLshProjection, LshProjection);
}  // namespace ops
}  // namespace mindspore
