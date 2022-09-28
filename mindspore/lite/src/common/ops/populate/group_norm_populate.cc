/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include "src/common/ops/populate/populate_register.h"
#include "nnacl/group_norm_parameter.h"
using mindspore::schema::PrimitiveType_GroupNormFusion;

namespace mindspore {
namespace lite {
OpParameter *PopulateIGroupNormParameter(const void *prim) {
  auto primitive = static_cast<const schema::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto value = primitive->value_as_GroupNormFusion();
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr";
    return nullptr;
  }

  auto *param = reinterpret_cast<GroupNormParameter *>(malloc(sizeof(GroupNormParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc GroupNormParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(GroupNormParameter));

  param->op_parameter_.type_ = primitive->value_type();
  param->epsilon_ = value->epsilon();
  param->num_groups_ = value->num_groups();
  if (param->num_groups_ < C1NUM) {
    MS_LOG(ERROR) << "GroupNormParameter num_groups cannot less than 1.";
    free(param);
    return nullptr;
  }
  param->affine_ = value->affine();
  return reinterpret_cast<OpParameter *>(param);
}

REG_POPULATE(PrimitiveType_GroupNormFusion, PopulateIGroupNormParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
