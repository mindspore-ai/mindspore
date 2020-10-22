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

#include "src/ops/scale.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/scale.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateScaleParameter(const mindspore::lite::PrimitiveC *primitive) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "input primitive is nullptr";
    return nullptr;
  }
  ScaleParameter *scale_param = reinterpret_cast<ScaleParameter *>(malloc(sizeof(ScaleParameter)));
  if (scale_param == nullptr) {
    MS_LOG(ERROR) << "malloc ScaleParameter failed.";
    return nullptr;
  }
  memset(scale_param, 0, sizeof(ScaleParameter));
  scale_param->op_parameter_.type_ = primitive->Type();
  auto param = reinterpret_cast<mindspore::lite::Scale *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  scale_param->axis_ = param->GetAxis();
  scale_param->activation_type_ = param->GetActivationType();
  return reinterpret_cast<OpParameter *>(scale_param);
}
Registry ScaleParameterRegistry(schema::PrimitiveType_Scale, PopulateScaleParameter);

}  // namespace lite
}  // namespace mindspore
