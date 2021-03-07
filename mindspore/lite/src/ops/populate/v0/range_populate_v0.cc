/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "schema/model_v0_generated.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/fp32/range_fp32.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateRangeParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  auto range_prim = primitive->value_as_Range();

  RangeParameter *range_param = reinterpret_cast<RangeParameter *>(malloc(sizeof(RangeParameter)));
  if (range_param == nullptr) {
    MS_LOG(ERROR) << "malloc RangeParameter failed.";
    return nullptr;
  }
  memset(range_param, 0, sizeof(RangeParameter));
  range_param->op_parameter_.type_ = schema::PrimitiveType_Range;
  range_param->start_ = range_prim->start();
  range_param->limit_ = range_prim->limit();
  range_param->delta_ = range_prim->delta();
  range_param->dType_ = range_prim->dType();
  return reinterpret_cast<OpParameter *>(range_param);
}
}  // namespace

Registry g_rangeV0ParameterRegistry(schema::v0::PrimitiveType_Range, PopulateRangeParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
