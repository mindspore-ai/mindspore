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
#include "nnacl/reduce_scatter_parameter.h"
#include "ops/reduce_scatter.h"
using mindspore::ops::kNameReduceScatter;
using mindspore::schema::PrimitiveType_ReduceScatter;

namespace mindspore {
namespace lite {
OpParameter *PopulateReduceScatterOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<ReduceScatterParameter *>(PopulateOpParameter<ReduceScatterParameter>());
  if (param == nullptr) {
    MS_LOG(ERROR) << "new ReduceScatterParameter failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::ReduceScatter *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "operator is not ReduceScatter.";
    return nullptr;
  }

  auto group = op->get_group();
  if (group.size() >= DEFAULT_GROUP_NAME_LEN) {
    MS_LOG(ERROR) << "group name size error: " << group.size() << ", which is larger than 100.";
    return nullptr;
  }
  (void)memcpy(param->group_, group.c_str(), group.size());
  param->rank_size_ = op->get_rank_size();
  param->mode_ = static_cast<int>(op->get_mode());

  return reinterpret_cast<OpParameter *>(param);
}
REG_OPERATOR_POPULATE(kNameReduceScatter, PrimitiveType_ReduceScatter, PopulateReduceScatterOpParameter)
}  // namespace lite
}  // namespace mindspore
