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
#include "nnacl/int8/quant_dtype_cast_int8.h"
#include "ops/quant_dtype_cast.h"
using mindspore::ops::kNameQuantDTypeCast;
using mindspore::schema::PrimitiveType_QuantDTypeCast;

namespace mindspore {
namespace lite {
OpParameter *PopulateQuantDTypeCastOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<QuantDTypeCastParameter *>(PopulateOpParameter<QuantDTypeCastParameter>());
  if (param == nullptr) {
    MS_LOG(ERROR) << "new QuantDTypeCastParameter failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::QuantDTypeCast *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "operator is not QuantDTypeCast.";
    return nullptr;
  }

  param->srcT = static_cast<int>(op->get_src_t());
  param->dstT = static_cast<int>(op->get_dst_t());
  param->axis = static_cast<int>(op->get_axis());
  return reinterpret_cast<OpParameter *>(param);
}
REG_OPERATOR_POPULATE(kNameQuantDTypeCast, PrimitiveType_QuantDTypeCast, PopulateQuantDTypeCastOpParameter);
}  // namespace lite
}  // namespace mindspore
