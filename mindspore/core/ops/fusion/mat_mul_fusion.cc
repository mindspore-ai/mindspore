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

#include "ops/fusion/mat_mul_fusion.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "ops/base_operator.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(MatMulFusion, BaseOperator);
void MatMulFusion::Init(bool transpose_a, bool transpose_b, const ActivationType &activation_type) {
  (void)AddAttr(kTransposeA, api::MakeValue(transpose_a));
  (void)AddAttr(kTransposeB, api::MakeValue(transpose_b));
  set_activation_type(activation_type);
}

void MatMulFusion::set_activation_type(const ActivationType activation_type) {
  int64_t act = activation_type;
  (void)this->AddAttr(kActivationType, api::MakeValue(act));
}

ActivationType MatMulFusion::get_activation_type() const {
  auto value_ptr = GetAttr(kActivationType);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return ActivationType(GetValue<int64_t>(value_ptr));
}

void MatMulFusion::set_transpose_a(bool transpose_a) { (void)AddAttr(kTransposeA, api::MakeValue(transpose_a)); }

void MatMulFusion::set_transpose_b(bool transpose_b) { (void)AddAttr(kTransposeB, api::MakeValue(transpose_b)); }

bool MatMulFusion::get_transpose_a() const {
  auto value_ptr = GetAttr(kTransposeA);
  return GetValue<bool>(value_ptr);
}

bool MatMulFusion::get_transpose_b() const {
  auto value_ptr = GetAttr(kTransposeB);
  return GetValue<bool>(value_ptr);
}

REGISTER_PRIMITIVE_C(kNameMatMulFusion, MatMulFusion);
}  // namespace ops
}  // namespace mindspore
