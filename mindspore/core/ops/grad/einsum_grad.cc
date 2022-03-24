/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "ops/grad/einsum_grad.h"
#include <algorithm>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_BASE_IMPL(EinsumGrad, PrimitiveC, BaseOperator);
void EinsumGrad::Init(const std::string equation) { this->set_equation(equation); }

void EinsumGrad::set_equation(const std::string equation) { (void)this->AddAttr(kEquation, api::MakeValue(equation)); }

std::string EinsumGrad::get_equation() const {
  auto value_ptr = this->GetAttr(kEquation);
  return GetValue<std::string>(value_ptr);
}
// REGISTER_PRIMITIVE_EVAL_IMPL(EinsumGrad, prim::kPrimEinsumGrad, EinsumGradInfer, nullptr, true);
REGISTER_PRIMITIVE_C(kNameEinsumGrad, EinsumGrad);
}  // namespace ops
}  // namespace mindspore
