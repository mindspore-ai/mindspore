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

#include "src/ops/unstack.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/unstack_parameter.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateUnstackParameter(const mindspore::lite::PrimitiveC *primitive) {
  UnstackParameter *unstack_param = reinterpret_cast<UnstackParameter *>(malloc(sizeof(UnstackParameter)));
  if (unstack_param == nullptr) {
    MS_LOG(ERROR) << "malloc UnstackParameter failed.";
    return nullptr;
  }
  memset(unstack_param, 0, sizeof(UnstackParameter));
  auto param = reinterpret_cast<mindspore::lite::Unstack *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  unstack_param->op_parameter_.type_ = primitive->Type();
  unstack_param->axis_ = param->GetAxis();
  return reinterpret_cast<OpParameter *>(unstack_param);
}
Registry UnstackParameterRegistry(schema::PrimitiveType_Unstack, PopulateUnstackParameter);
}  // namespace lite
}  // namespace mindspore
