/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "src/ops/range.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/fp32/range_fp32.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateRangeParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto range_attr = reinterpret_cast<mindspore::lite::Range *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  RangeParameter *range_param = reinterpret_cast<RangeParameter *>(malloc(sizeof(RangeParameter)));
  if (range_param == nullptr) {
    MS_LOG(ERROR) << "malloc RangeParameter failed.";
    return nullptr;
  }
  memset(range_param, 0, sizeof(RangeParameter));
  range_param->op_parameter_.type_ = primitive->Type();
  range_param->start_ = range_attr->GetStart();
  range_param->limit_ = range_attr->GetLimit();
  range_param->delta_ = range_attr->GetDelta();
  range_param->dType_ = range_attr->GetDType();
  return reinterpret_cast<OpParameter *>(range_param);
}
Registry RangeParameterRegistry(schema::PrimitiveType_Range, PopulateRangeParameter);

}  // namespace lite
}  // namespace mindspore
