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
#include <cstdint>
#include "src/ops/populate/populate_register.h"
#include "nnacl/l2_norm_parameter.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateL2NormParameter(const void *prim) {
  L2NormParameter *l2_norm_parameter = reinterpret_cast<L2NormParameter *>(malloc(sizeof(L2NormParameter)));
  if (l2_norm_parameter == nullptr) {
    MS_LOG(ERROR) << "malloc L2NormParameter failed.";
    return nullptr;
  }
  memset(l2_norm_parameter, 0, sizeof(L2NormParameter));

  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_L2NormalizeFusion();
  l2_norm_parameter->op_parameter_.type_ = primitive->value_type();

  auto axis_vec = value->axis();
  l2_norm_parameter->axis_num_ = axis_vec->size();

  MS_ASSERT(axis_vec->size() < 8);
  for (size_t i = 0; i < axis_vec->size(); i++) {
    l2_norm_parameter->axis_[i] = static_cast<int>(axis_vec->Get(i));
  }
  if (value->epsilon() < 1e-6) {
    l2_norm_parameter->epsilon_ = 1e-6;
  } else {
    l2_norm_parameter->epsilon_ = value->epsilon();
  }
  if (value->activation_type() == static_cast<int>(schema::ActivationType_RELU)) {
    l2_norm_parameter->act_type_ = ActType_Relu;
  } else if (value->activation_type() == static_cast<int>(schema::ActivationType_RELU6)) {
    l2_norm_parameter->act_type_ = ActType_Relu6;
  } else {
    l2_norm_parameter->act_type_ = ActType_No;
  }
  return reinterpret_cast<OpParameter *>(l2_norm_parameter);
}
Registry L2NormParameterRegistry(schema::PrimitiveType_L2NormalizeFusion, PopulateL2NormParameter, SCHEMA_CUR);

}  // namespace lite
}  // namespace mindspore
