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
#include "ops/reduce.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void Reduce::set_keep_dims(const bool keep_dims) { (void)this->AddAttr(kKeepDims, api::MakeValue(keep_dims)); }

bool Reduce::get_keep_dims() const { return GetValue<bool>(GetAttr(kKeepDims)); }

void Reduce::set_skip_mode(const bool skip_mode) { (void)this->AddAttr(kSkipMode, api::MakeValue(skip_mode)); }

bool Reduce::get_skip_mode() const { return GetValue<bool>(GetAttr(kSkipMode)); }

void Reduce::Init(const bool keep_dims, const bool skip_mode) {
  this->set_keep_dims(keep_dims);
  this->set_skip_mode(skip_mode);
}

MIND_API_OPERATOR_IMPL(Reduce, BaseOperator);
REGISTER_PRIMITIVE_C(kNameReduce, Reduce);
}  // namespace ops
}  // namespace mindspore
