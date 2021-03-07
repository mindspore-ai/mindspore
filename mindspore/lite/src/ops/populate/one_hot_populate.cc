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
#include "nnacl/fp32/one_hot_fp32.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateOneHotParameter(const void *prim) {
  OneHotParameter *one_hot_param = reinterpret_cast<OneHotParameter *>(malloc(sizeof(OneHotParameter)));
  if (one_hot_param == nullptr) {
    MS_LOG(ERROR) << "malloc OneHotParameter failed.";
    return nullptr;
  }
  memset(one_hot_param, 0, sizeof(OneHotParameter));

  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_OneHot();
  one_hot_param->op_parameter_.type_ = primitive->value_type();
  one_hot_param->axis_ = value->axis();
  return reinterpret_cast<OpParameter *>(one_hot_param);
}
Registry OneHotParameterRegistry(schema::PrimitiveType_OneHot, PopulateOneHotParameter, SCHEMA_CUR);

}  // namespace lite
}  // namespace mindspore
