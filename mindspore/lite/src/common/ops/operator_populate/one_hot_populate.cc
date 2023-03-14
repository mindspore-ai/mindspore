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
#include "nnacl/fp32/one_hot_fp32.h"
#include "ops/one_hot.h"
using mindspore::ops::kNameOneHot;
using mindspore::schema::PrimitiveType_OneHot;

namespace mindspore {
namespace lite {
OpParameter *PopulateOneHotOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<OneHotParameter *>(PopulateOpParameter<OneHotParameter>());
  if (param == nullptr) {
    MS_LOG(ERROR) << "new OneHotParameter failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::OneHot *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "operator is not PadFusion.";
    return nullptr;
  }

  param->axis_ = static_cast<int>(op->get_axis());
  if (param->axis_ < -C1NUM) {
    MS_LOG(ERROR) << "OneHotParameter axis cannot less than -1.";
    free(param);
    return nullptr;
  }
  return reinterpret_cast<OpParameter *>(param);
}
REG_OPERATOR_POPULATE(kNameOneHot, PrimitiveType_OneHot, PopulateOneHotOpParameter)
}  // namespace lite
}  // namespace mindspore
