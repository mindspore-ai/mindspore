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

  auto attr_b_transpose = base_operator->GetPrim()->GetAttr(kTransposeB);
  if (attr_b_transpose == nullptr) {
    MS_LOG(ERROR) << "The attr(" << kTransposeB << ") of operator(" << base_operator->name() << ") not exist";
    free(param);
    return nullptr;
  }
  param->b_transpose_ = GetValue<bool>(attr_b_transpose);

  auto attr_a_transpose = base_operator->GetPrim()->GetAttr(kTransposeA);
  if (attr_a_transpose == nullptr) {
    MS_LOG(ERROR) << "The attr(" << kTransposeA << ") of operator(" << base_operator->name() << ") not exist";
    free(param);
    return nullptr;
  }
  param->a_transpose_ = GetValue<bool>(attr_a_transpose);
  param->has_bias_ = false;

  auto attr_act_type = base_operator->GetPrim()->GetAttr(kActivationType);
  if (attr_act_type != nullptr) {
    param->act_type_ = static_cast<ActType>(GetValue<int64_t>(attr_act_type));
  }
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameMatMul, PrimitiveType_MatMulFusion, PopulateMatMulOpParameter)
REG_OPERATOR_POPULATE(kNameMatMulFusion, PrimitiveType_MatMulFusion, PopulateMatMulOpParameter)
}  // namespace lite
}  // namespace mindspore
