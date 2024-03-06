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

#include "ops/fusion/matmul_biasadd_relu_fusion.h"
#include "mindapi/src/helper.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
void MatMulBiasAddReluFusion::Init(bool transpose_a, bool transpose_b) {
  set_transpose_a(transpose_a);
  set_transpose_b(transpose_b);
}
void MatMulBiasAddReluFusion::set_transpose_a(bool transpose_a) {
  (void)AddAttr(kTransposeA, api::MakeValue(transpose_a));
}

void MatMulBiasAddReluFusion::set_transpose_b(bool transpose_b) {
  (void)AddAttr(kTransposeB, api::MakeValue(transpose_b));
}

bool MatMulBiasAddReluFusion::get_transpose_a() const {
  auto value_ptr = GetAttr(kTransposeA);
  return GetValue<bool>(value_ptr);
}

bool MatMulBiasAddReluFusion::get_transpose_b() const {
  auto value_ptr = GetAttr(kTransposeB);
  return GetValue<bool>(value_ptr);
}
MIND_API_OPERATOR_IMPL(MatMulBiasAddReluFusion, BaseOperator);
REGISTER_PRIMITIVE_C(kNameMatMulBiasAddReluFusion, MatMulBiasAddReluFusion);
}  // namespace ops
}  // namespace mindspore
