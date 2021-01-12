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

#include "src/ops/fill.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/fill_parameter.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateFillParameter(const mindspore::lite::PrimitiveC *primitive) {
  const auto param = reinterpret_cast<mindspore::lite::Fill *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  FillParameter *fill_param = reinterpret_cast<FillParameter *>(malloc(sizeof(FillParameter)));
  if (fill_param == nullptr) {
    MS_LOG(ERROR) << "malloc FillParameter failed.";
    return nullptr;
  }
  memset(fill_param, 0, sizeof(FillParameter));
  fill_param->op_parameter_.type_ = primitive->Type();
  auto flatDims = param->GetDims();
  fill_param->num_dims_ = flatDims.size();
  int i = 0;
  for (auto iter = flatDims.begin(); iter != flatDims.end(); iter++) {
    fill_param->dims_[i++] = *iter;
  }
  return reinterpret_cast<OpParameter *>(fill_param);
}

Registry FillParameterRegistry(schema::PrimitiveType_Fill, PopulateFillParameter);

}  // namespace lite
}  // namespace mindspore
