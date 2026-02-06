/**
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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

#include "infer/cxx_api/mul_fusion.h"

#include "mindapi/ir/value.h"
#include "mindapi/helper.h"
#include "primitive/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(MulFusion, BaseOperator);
void MulFusion::set_activation_type(const ActivationType &activation_type) {
  int64_t swi = activation_type;
  (void)this->AddAttr(kActivationType, api::MakeValue(swi));
}
ActivationType MulFusion::get_activation_type() const {
  auto value_ptr = GetAttr(kActivationType);
  if (value_ptr == nullptr) {
    return NO_ACTIVATION;
  }
  return ActivationType(GetValue<int64_t>(value_ptr));
}
void MulFusion::Init(const ActivationType &activation_type) { this->set_activation_type(activation_type); }
REGISTER_PRIMITIVE_C(kNameMulFusion, MulFusion);
}  // namespace ops
}  // namespace mindspore
