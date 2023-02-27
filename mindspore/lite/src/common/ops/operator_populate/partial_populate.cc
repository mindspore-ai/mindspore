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
#include "nnacl/partial_fusion_parameter.h"
#include "ops/fusion/partial_fusion.h"
using mindspore::ops::kNamePartialFusion;
using mindspore::schema::PrimitiveType_PartialFusion;

namespace mindspore {
namespace lite {
OpParameter *PopulatePartialOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<PartialParameter *>(PopulateOpParameter<PartialParameter>());
  if (param == nullptr) {
    MS_LOG(ERROR) << "new PartialParameter failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::PartialFusion *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "operator is not PartialFusion.";
    return nullptr;
  }

  param->sub_graph_index_ = static_cast<int>(op->get_sub_graph_index());
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNamePartialFusion, PrimitiveType_PartialFusion, PopulatePartialOpParameter)
}  // namespace lite
}  // namespace mindspore
