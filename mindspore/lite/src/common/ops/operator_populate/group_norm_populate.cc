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
#include "nnacl/group_norm_parameter.h"
#include "ops/fusion/groupnorm_fusion.h"
using mindspore::ops::kNameGroupNormFusion;
using mindspore::schema::PrimitiveType_GroupNormFusion;
namespace mindspore {
namespace lite {
OpParameter *PopulateGroupNormOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<GroupNormParameter *>(PopulateOpParameter<GroupNormParameter>(base_operator));
  if (param == nullptr) {
    MS_LOG(ERROR) << "new GroupNormParameter failed.";
    return nullptr;
  }

  auto op = dynamic_cast<ops::GroupNormFusion *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "base_operator cast to GroupNormFusion failed";
    free(param);
    return nullptr;
  }

  param->affine_ = op->get_affine();
  param->epsilon_ = op->get_epsilon();
  param->num_groups_ = static_cast<int>(op->get_num_groups());
  if (param->num_groups_ < C1NUM) {
    MS_LOG(ERROR) << "GroupNormParameter num_groups cannot less than 1.";
    free(param);
    return nullptr;
  }
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameGroupNormFusion, PrimitiveType_GroupNormFusion, PopulateGroupNormOpParameter)
}  // namespace lite
}  // namespace mindspore
