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
#include "nnacl/l2_norm_parameter.h"
using mindspore::schema::PrimitiveType_L2NormalizeFusion;

namespace mindspore {
namespace lite {
OpParameter *PopulateL2NormParameter(const void *prim) {
  auto primitive = static_cast<const schema::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto value = primitive->value_as_L2NormalizeFusion();
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr";
    return nullptr;
  }

  auto *param = reinterpret_cast<L2NormParameter *>(malloc(sizeof(L2NormParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc L2NormParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(L2NormParameter));

  param->op_parameter_.type_ = primitive->value_type();
  auto axis_vec = value->axis();
  if (axis_vec == nullptr) {
    MS_LOG(ERROR) << "axis_vec is nullptr";
    free(param);
    return nullptr;
  }
  param->axis_num_ = axis_vec->size();

  MS_ASSERT(axis_vec->size() < 8);
  for (size_t i = 0; i < axis_vec->size(); i++) {
    param->axis_[i] = static_cast<int>(axis_vec->Get(i));
  }
  if (value->epsilon() < 1e-6) {
    param->epsilon_ = 1e-6;
  } else {
    param->epsilon_ = value->epsilon();
  }
  if (value->activation_type() == static_cast<int>(schema::ActivationType_RELU)) {
    param->act_type_ = ActType_Relu;
  } else if (value->activation_type() == static_cast<int>(schema::ActivationType_RELU6)) {
    param->act_type_ = ActType_Relu6;
  } else {
    param->act_type_ = ActType_No;
  }
  return reinterpret_cast<OpParameter *>(param);
}
REG_POPULATE(PrimitiveType_L2NormalizeFusion, PopulateL2NormParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
