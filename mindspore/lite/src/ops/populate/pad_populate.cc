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
#include "nnacl/pad_parameter.h"

namespace mindspore {
namespace lite {

OpParameter *PopulatePadParameter(const void *prim) {
  PadParameter *pad_param = reinterpret_cast<PadParameter *>(malloc(sizeof(PadParameter)));
  if (pad_param == nullptr) {
    MS_LOG(ERROR) << "malloc PadParameter failed.";
    return nullptr;
  }
  memset(pad_param, 0, sizeof(PadParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_PadFusion();
  pad_param->op_parameter_.type_ = primitive->value_type();
  pad_param->pad_mode_ = value->padding_mode();
  pad_param->constant_value_ = value->constant_value();
  return reinterpret_cast<OpParameter *>(pad_param);
}
Registry PadParameterRegistry(schema::PrimitiveType_PadFusion, PopulatePadParameter, SCHEMA_CUR);

}  // namespace lite
}  // namespace mindspore
