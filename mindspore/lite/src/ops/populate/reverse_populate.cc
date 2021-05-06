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
#include "nnacl/fp32/reverse_fp32.h"
using mindspore::schema::PrimitiveType_ReverseV2;

namespace mindspore {
namespace lite {
OpParameter *PopulateReverseParameter(const void *prim) {
  auto primitive = static_cast<const schema::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto value = primitive->value_as_ReverseV2();
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr";
    return nullptr;
  }

  auto *param = reinterpret_cast<ReverseParameter *>(malloc(sizeof(ReverseParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc ReverseParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(ReverseParameter));

  param->op_parameter_.type_ = primitive->value_type();
  auto flatAxis = value->axis();
  if (flatAxis == nullptr) {
    MS_LOG(ERROR) << "flatAxis is nullptr";
    free(param);
    return nullptr;
  }
  param->num_axis_ = flatAxis->size();
  int i = 0;
  for (auto flatAxi : *flatAxis) {
    param->axis_[i++] = static_cast<int>(flatAxi);
  }
  return reinterpret_cast<OpParameter *>(param);
}

REG_POPULATE(PrimitiveType_ReverseV2, PopulateReverseParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
