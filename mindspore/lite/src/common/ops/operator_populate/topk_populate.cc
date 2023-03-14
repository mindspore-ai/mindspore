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
#include "nnacl/fp32/topk_fp32.h"
#include "ops/fusion/topk_fusion.h"

using mindspore::ops::kNameTopKFusion;
using mindspore::schema::PrimitiveType_TopKFusion;

namespace mindspore {
namespace lite {
OpParameter *PopulateTopKOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<TopkParameter *>(PopulateOpParameter<TopkParameter>());
  if (param == nullptr) {
    MS_LOG(ERROR) << "new TopkParameter failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::TopKFusion *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "operator is not TopKFusion.";
    return nullptr;
  }
  auto axis = op->get_axis();
  CHECK_LESS_RETURN_RET(INT32_MAX, axis, nullptr, param);
  param->axis_ = static_cast<int>(axis);
  param->sorted_ = op->get_sorted();
  return reinterpret_cast<OpParameter *>(param);
}
REG_OPERATOR_POPULATE(kNameTopKFusion, PrimitiveType_TopKFusion, PopulateTopKOpParameter)
}  // namespace lite
}  // namespace mindspore
