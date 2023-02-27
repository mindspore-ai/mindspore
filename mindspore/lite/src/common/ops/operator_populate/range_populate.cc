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
#include "nnacl/fp32/range_fp32.h"
#include "ops/range.h"
using mindspore::ops::kNameRange;
using mindspore::schema::PrimitiveType_Range;

namespace mindspore {
namespace lite {
OpParameter *PopulateRangeOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<RangeParameter *>(PopulateOpParameter<RangeParameter>());
  if (param == nullptr) {
    MS_LOG(ERROR) << "new RangeParameter failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::Range *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "operator is not Range.";
    return nullptr;
  }

  param->start_ = static_cast<int>(op->get_start());
  param->limit_ = static_cast<int>(op->get_limit());
  param->delta_ = static_cast<int>(op->get_delta());
  param->dType_ = static_cast<int>(op->get_d_type());
  return reinterpret_cast<OpParameter *>(param);
}
REG_OPERATOR_POPULATE(kNameRange, PrimitiveType_Range, PopulateRangeOpParameter)
}  // namespace lite
}  // namespace mindspore
