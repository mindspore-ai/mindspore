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

#include "src/ops/reverse.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/fp32/reverse_fp32.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateReverseParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto reverse_attr =
    reinterpret_cast<mindspore::lite::Reverse *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  ReverseParameter *reverse_param = reinterpret_cast<ReverseParameter *>(malloc(sizeof(ReverseParameter)));
  if (reverse_param == nullptr) {
    MS_LOG(ERROR) << "malloc ReverseParameter failed.";
    return nullptr;
  }
  memset(reverse_param, 0, sizeof(ReverseParameter));
  reverse_param->op_parameter_.type_ = primitive->Type();
  auto flatAxis = reverse_attr->GetAxis();
  reverse_param->num_axis_ = flatAxis.size();
  int i = 0;
  for (auto iter = flatAxis.begin(); iter != flatAxis.end(); iter++) {
    reverse_param->axis_[i++] = *iter;
  }
  return reinterpret_cast<OpParameter *>(reverse_param);
}

Registry ReverseParameterRegistry(schema::PrimitiveType_Reverse, PopulateReverseParameter);

}  // namespace lite
}  // namespace mindspore
