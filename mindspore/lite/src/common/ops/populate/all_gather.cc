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

#include "schema/ops_generated.h"
#include "schema/model_generated.h"
#include "nnacl/all_gather_parameter.h"
#include "src/common/ops/populate/populate_register.h"

using mindspore::schema::PrimitiveType_AllGather;
namespace mindspore {
namespace lite {
OpParameter *PopulateAllGatherParameter(const void *prim) {
  auto *primitive = static_cast<const schema::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto value = primitive->value_as_AllGather();
  if (value == nullptr) {
    MS_LOG(ERROR) << "cast all_gather_primitive to value failed";
    return nullptr;
  }
  auto group = value->group();
  if (group == nullptr) {
    MS_LOG(ERROR) << "attr group must be not a nullptr.";
    return nullptr;
  }
  if (group->size() >= DEFAULT_GROUP_NAME_LEN) {
    MS_LOG(ERROR) << "group name size error: " << value->group()->size() << ", which is larger than 100.";
    return nullptr;
  }

  auto *param = static_cast<AllGatherParameter *>(malloc(sizeof(AllGatherParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "Malloc AllGatherParameter failed.";
    return nullptr;
  }
  (void)memset(param, 0, sizeof(AllGatherParameter));
  (void)memcpy(param->group_, value->group()->c_str(), value->group()->size());
  param->rank_size_ = value->rank_size();
  param->op_parameter_.type_ = primitive->value_type();

  return reinterpret_cast<OpParameter *>(param);
}
REG_POPULATE(PrimitiveType_AllGather, PopulateAllGatherParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
