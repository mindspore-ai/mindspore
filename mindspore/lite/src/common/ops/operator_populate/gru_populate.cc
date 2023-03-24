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
#include "nnacl/fp32/gru_fp32.h"
#include "ops/gru.h"
using mindspore::ops::kBidirectional;
using mindspore::ops::kNameGRU;
using mindspore::schema::PrimitiveType_GRU;

namespace mindspore {
namespace lite {
OpParameter *PopulateGruOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<GruParameter *>(PopulateOpParameter<GruParameter>(base_operator));
  if (param == nullptr) {
    MS_LOG(ERROR) << "new GruParameter failed.";
    return nullptr;
  }

  auto attr_bidirectional = base_operator->GetPrim()->GetAttr(kBidirectional);
  if (attr_bidirectional == nullptr) {
    MS_LOG(ERROR) << "The attr(" << kBidirectional << ") of operator(" << base_operator->name() << ") not exist";
    free(param);
    return nullptr;
  }
  param->bidirectional_ = GetValue<bool>(attr_bidirectional);
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameGRU, PrimitiveType_GRU, PopulateGruOpParameter)
}  // namespace lite
}  // namespace mindspore
