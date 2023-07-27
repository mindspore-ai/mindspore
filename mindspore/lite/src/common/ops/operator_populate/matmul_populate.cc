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
#include "src/common/ops/operator_populate/operator_populate_register.h"
#include "src/common/ops/operator_populate/utils.h"
#include "nnacl/matmul_parameter.h"
#include "ops/mat_mul.h"
#include "ops/fusion/mat_mul_fusion.h"
#include "ops/op_name.h"
using mindspore::ops::kActivationType;
using mindspore::ops::kNameMatMul;
using mindspore::ops::kNameMatMulFusion;
using mindspore::ops::kTransposeA;
using mindspore::ops::kTransposeB;
using mindspore::schema::PrimitiveType_MatMulFusion;
namespace mindspore {
namespace lite {
OpParameter *PopulateMatMulOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<MatMulParameter *>(PopulateOpParameter<MatMulParameter>(base_operator));
  if (param == nullptr) {
    MS_LOG(ERROR) << "Make MatMulParameter ptr failed";
    return nullptr;
  }

  param->b_transpose_ = GetAttrWithDefault(base_operator, kTransposeB, false);
  param->a_transpose_ = GetAttrWithDefault(base_operator, kTransposeA, false);
  param->has_bias_ = false;
  param->act_type_ =
    static_cast<ActType>(GetAttrWithDefault<int64_t>(base_operator, kActivationType, ActType::ActType_No));
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameMatMul, PrimitiveType_MatMulFusion, PopulateMatMulOpParameter)
REG_OPERATOR_POPULATE(kNameMatMulFusion, PrimitiveType_MatMulFusion, PopulateMatMulOpParameter)
}  // namespace lite
}  // namespace mindspore
