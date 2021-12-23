/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "src/ops/populate/populate_register.h"
#include "nnacl/matmul_parameter.h"
using mindspore::schema::PrimitiveType_MatMulFusion;

namespace mindspore {
namespace lite {
OpParameter *PopulateMatMulParameter(const void *prim) {
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_MatMulFusion();
  MS_CHECK_TRUE_RET(value != nullptr, nullptr);

  auto *param = reinterpret_cast<MatMulParameter *>(malloc(sizeof(MatMulParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc MatMulParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(MatMulParameter));
  param->op_parameter_.type_ = primitive->value_type();
  param->b_transpose_ = value->transpose_b();
  param->a_transpose_ = value->transpose_a();
  param->has_bias_ = false;
  param->act_type_ = static_cast<ActType>(value->activation_type());

  return reinterpret_cast<OpParameter *>(param);
}
REG_POPULATE(PrimitiveType_MatMulFusion, PopulateMatMulParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
