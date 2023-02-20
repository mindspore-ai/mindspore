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

#include "ops/affine.h"

#include <vector>

#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(Affine, BaseOperator);
void Affine::Init(const std::vector<int64_t> &contexts, int64_t output_dim, bool transpose_a, bool transpose_b) {
  this->set_context(contexts);
  this->set_output_dim(output_dim);
  this->set_transpose_a(transpose_a);
  this->set_transpose_b(transpose_b);
}

void Affine::set_context(const std::vector<int64_t> &context) {
  (void)this->AddAttr(kAffineContext, api::MakeValue(context));
}

void Affine::set_output_dim(int64_t output_dim) { (void)this->AddAttr(kAffineOutputDim, api::MakeValue(output_dim)); }

void Affine::set_transpose_a(bool transpose_a) { (void)AddAttr(kTransposeA, api::MakeValue(transpose_a)); }

void Affine::set_transpose_b(bool transpose_b) { (void)AddAttr(kTransposeB, api::MakeValue(transpose_b)); }

void Affine::set_activation_type(const ActivationType &activation_type) {
  (void)this->AddAttr(kActivationType, api::MakeValue(static_cast<int64_t>(activation_type)));
}

bool Affine::get_transpose_a() const {
  auto value_ptr = GetAttr(kTransposeA);
  return GetValue<bool>(value_ptr);
}

bool Affine::get_transpose_b() const {
  auto value_ptr = GetAttr(kTransposeB);
  return GetValue<bool>(value_ptr);
}

std::vector<int64_t> Affine::get_context() const {
  auto value_ptr = GetAttr(kAffineContext);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

int64_t Affine::get_output_dim() const {
  auto value_ptr = GetAttr(kAffineOutputDim);
  return GetValue<int64_t>(value_ptr);
}

ActivationType Affine::get_activation_type() const {
  auto value_ptr = GetAttr(kActivationType);
  return ActivationType(GetValue<int64_t>(value_ptr));
}
REGISTER_PRIMITIVE_C(kNameAffine, Affine);
}  // namespace ops
}  // namespace mindspore
