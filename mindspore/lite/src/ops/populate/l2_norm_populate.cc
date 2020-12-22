/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "src/ops/l2_norm.h"
#include <cstdint>
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/l2_norm_parameter.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateL2NormParameter(const mindspore::lite::PrimitiveC *primitive) {
  L2NormParameter *l2_norm_parameter = reinterpret_cast<L2NormParameter *>(malloc(sizeof(L2NormParameter)));
  if (l2_norm_parameter == nullptr) {
    MS_LOG(ERROR) << "malloc L2NormParameter failed.";
    return nullptr;
  }
  memset(l2_norm_parameter, 0, sizeof(L2NormParameter));
  l2_norm_parameter->op_parameter_.type_ = primitive->Type();
  auto param = reinterpret_cast<mindspore::lite::L2Norm *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  MS_ASSERT(param);
  auto axis_vec = param->GetAxis();
  l2_norm_parameter->axis_num_ = axis_vec.size();
  if (axis_vec.size() > SIZE_MAX / sizeof(int)) {
    MS_LOG(ERROR) << "axis_vec size too big";
    free(l2_norm_parameter);
    return nullptr;
  }
  MS_ASSERT(axis_vec.size() < 8);
  for (size_t i = 0; i < axis_vec.size(); i++) {
    l2_norm_parameter->axis_[i] = axis_vec[i];
  }
  if (param->GetEpsilon() < 1e-6) {
    l2_norm_parameter->epsilon_ = 1e-6;
  } else {
    l2_norm_parameter->epsilon_ = param->GetEpsilon();
  }
  if (param->GetActivationType() == static_cast<int>(schema::ActivationType_RELU)) {
    l2_norm_parameter->act_type_ = ActType_Relu;
  } else if (param->GetActivationType() == static_cast<int>(schema::ActivationType_RELU6)) {
    l2_norm_parameter->act_type_ = ActType_Relu6;
  } else {
    l2_norm_parameter->act_type_ = ActType_No;
  }
  return reinterpret_cast<OpParameter *>(l2_norm_parameter);
}
Registry L2NormParameterRegistry(schema::PrimitiveType_L2Norm, PopulateL2NormParameter);

}  // namespace lite
}  // namespace mindspore
