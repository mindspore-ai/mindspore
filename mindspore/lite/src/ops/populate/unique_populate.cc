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

#include "src/ops/unique.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/fp32/unique_fp32.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateUniqueParameter(const mindspore::lite::PrimitiveC *primitive) {
  UniqueParameter *unique_param = reinterpret_cast<UniqueParameter *>(malloc(sizeof(UniqueParameter)));
  if (unique_param == nullptr) {
    MS_LOG(ERROR) << "malloc UniqueParameter failed.";
    return nullptr;
  }
  memset(unique_param, 0, sizeof(UniqueParameter));
  unique_param->op_parameter_.type_ = primitive->Type();
  return reinterpret_cast<OpParameter *>(unique_param);
}

Registry UniqueParameterRegistry(schema::PrimitiveType_Unique, PopulateUniqueParameter);

}  // namespace lite
}  // namespace mindspore
