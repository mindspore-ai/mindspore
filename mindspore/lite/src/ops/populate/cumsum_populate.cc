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
#include "src/ops/populate/populate_register.h"
#include "nnacl/cumsum_parameter.h"
using mindspore::schema::PrimitiveType_CumSum;

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateCumSumParameter(const void *prim) {
  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto cumsum_prim = primitive->value_as_CumSum();
  CumSumParameter *cumsum_param = reinterpret_cast<CumSumParameter *>(malloc(sizeof(CumSumParameter)));
  if (cumsum_param == nullptr) {
    MS_LOG(ERROR) << "malloc CumsumParameter failed.";
    return nullptr;
  }
  memset(cumsum_param, 0, sizeof(CumSumParameter));
  cumsum_param->op_parameter_.type_ = primitive->value_type();
  cumsum_param->exclusive_ = cumsum_prim->exclusive();
  cumsum_param->reverse_ = cumsum_prim->reverse();
  return reinterpret_cast<OpParameter *>(cumsum_param);
}
}  // namespace

REG_POPULATE(PrimitiveType_CumSum, PopulateCumSumParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
