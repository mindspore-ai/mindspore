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

#include "src/ops/clip.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/clip.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateClipParameter(const mindspore::lite::PrimitiveC *primitive) {
  ClipParameter *act_param = reinterpret_cast<ClipParameter *>(malloc(sizeof(ClipParameter)));
  if (act_param == nullptr) {
    MS_LOG(ERROR) << "malloc ClipParameter failed.";
    return nullptr;
  }
  memset(act_param, 0, sizeof(ClipParameter));
  act_param->op_parameter_.type_ = primitive->Type();
  auto activation = reinterpret_cast<mindspore::lite::Clip *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  act_param->min_val_ = activation->GetMin();
  act_param->max_val_ = activation->GetMax();
  return reinterpret_cast<OpParameter *>(act_param);
}

Registry ClipParameterRegistry(schema::PrimitiveType_Clip, PopulateClipParameter);

}  // namespace lite
}  // namespace mindspore
