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

#include "src/ops/activation.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/fp32/activation_fp32.h"

namespace mindspore {
namespace lite {
OpParameter *PopulateActivationParameter(const mindspore::lite::PrimitiveC *primitive) {
  ActivationParameter *act_param = reinterpret_cast<ActivationParameter *>(malloc(sizeof(ActivationParameter)));
  if (act_param == nullptr) {
    MS_LOG(ERROR) << "malloc ActivationParameter failed.";
    return nullptr;
  }
  memset(act_param, 0, sizeof(ActivationParameter));
  act_param->op_parameter_.type_ = primitive->Type();
  auto activation =
    reinterpret_cast<mindspore::lite::Activation *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  act_param->type_ = static_cast<int>(activation->GetType());
  act_param->alpha_ = activation->GetAlpha();
  act_param->min_val_ = activation->GetMinVal();
  act_param->max_val_ = activation->GetMaxVal();
  return reinterpret_cast<OpParameter *>(act_param);
}
Registry ActivationParameterRegistry(schema::PrimitiveType_Activation, PopulateActivationParameter);
}  // namespace lite
}  // namespace mindspore
