/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "src/common/ops/operator_populate/operator_populate_register.h"
#include "nnacl/all_gather_parameter.h"
#include "ops/all_gather.h"
using mindspore::ops::kNameAllGather;
using mindspore::schema::PrimitiveType_AllGather;
namespace mindspore {
namespace lite {
OpParameter *PopulateAllGatherOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<AllGatherParameter *>(PopulateOpParameter<AllGatherParameter>(base_operator));
  if (param == nullptr) {
    MS_LOG(ERROR) << "new AllGatherParameter failed.";
    return nullptr;
  }

  auto op = dynamic_cast<ops::AllGather *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "base_operator cast to AllGather failed";
    free(param);
    return nullptr;
  }

  auto group = op->get_group();
  if (group.size() >= DEFAULT_GROUP_NAME_LEN) {
    MS_LOG(ERROR) << "group name size error: " << group.size() << ", which is larger than 100.";
    free(param);
    return nullptr;
  }
  (void)memcpy(param->group_, group.c_str(), group.size());
  param->rank_size_ = op->get_rank_size();
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameAllGather, PrimitiveType_AllGather, PopulateAllGatherOpParameter)
}  // namespace lite
}  // namespace mindspore
