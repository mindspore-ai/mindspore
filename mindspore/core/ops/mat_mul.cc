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

#include <set>
#include <map>
#include <string>
#include "ops/mat_mul.h"

namespace mindspore {
namespace ops {
void MatMul::Init(bool transpose_a, bool transpose_b) {
  set_transpose_a(transpose_a);
  set_transpose_b(transpose_b);
}

void MatMul::set_transpose_a(bool transpose_a) { (void)AddAttr(kTransposeA, MakeValue(transpose_a)); }

void MatMul::set_transpose_b(bool transpose_b) { (void)AddAttr(kTransposeB, MakeValue(transpose_b)); }

bool MatMul::get_transpose_a() const {
  auto value_ptr = GetAttr(kTransposeA);
  return GetValue<bool>(value_ptr);
}

bool MatMul::get_transpose_b() const {
  auto value_ptr = GetAttr(kTransposeB);
  return GetValue<bool>(value_ptr);
}

REGISTER_PRIMITIVE_C(kNameMatMul, MatMul);
}  // namespace ops
}  // namespace mindspore
