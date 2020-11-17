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

#include "src/ops/one_hot.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/fp32/one_hot_fp32.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateOneHotParameter(const mindspore::lite::PrimitiveC *primitive) {
  OneHotParameter *one_hot_param = reinterpret_cast<OneHotParameter *>(malloc(sizeof(OneHotParameter)));
  if (one_hot_param == nullptr) {
    MS_LOG(ERROR) << "malloc OneHotParameter failed.";
    return nullptr;
  }
  memset(one_hot_param, 0, sizeof(OneHotParameter));
  one_hot_param->op_parameter_.type_ = primitive->Type();
  auto param = reinterpret_cast<mindspore::lite::OneHot *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  if (param == nullptr) {
    free(one_hot_param);
    MS_LOG(ERROR) << "get OneHot param nullptr.";
    return nullptr;
  }
  one_hot_param->axis_ = param->GetAxis();
  return reinterpret_cast<OpParameter *>(one_hot_param);
}
Registry OneHotParameterRegistry(schema::PrimitiveType_OneHot, PopulateOneHotParameter);

}  // namespace lite
}  // namespace mindspore
