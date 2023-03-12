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
#include "nnacl/fp32/exp_fp32.h"
#include "ops/exp.h"
using mindspore::ops::kBase;
using mindspore::ops::kNameExp;
using mindspore::ops::kScale;
using mindspore::ops::kShift;
using mindspore::schema::PrimitiveType_ExpFusion;

namespace mindspore {
namespace lite {
OpParameter *PopulateExpOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<ExpParameter *>(PopulateOpParameter<ExpParameter>(base_operator));
  if (param == nullptr) {
    MS_LOG(ERROR) << "new ExpParameter failed.";
    return nullptr;
  }

  auto attr_base = base_operator->GetPrim()->GetAttr(kBase);
  if (attr_base == nullptr) {
    MS_LOG(ERROR) << "The attr(" << kBase << ") of operator(" << base_operator->name() << ") not exist";
    free(param);
    return nullptr;
  }
  param->base_ = GetValue<float>(attr_base);

  auto attr_scale = base_operator->GetPrim()->GetAttr(kScale);
  if (attr_scale == nullptr) {
    MS_LOG(ERROR) << "The attr(" << kScale << ") of operator(" << base_operator->name() << ") not exist";
    free(param);
    return nullptr;
  }
  param->scale_ = GetValue<float>(attr_scale);

  auto attr_shift = base_operator->GetPrim()->GetAttr(kShift);
  if (attr_shift == nullptr) {
    MS_LOG(ERROR) << "The attr(" << kShift << ") of operator(" << base_operator->name() << ") not exist";
    free(param);
    return nullptr;
  }
  param->shift_ = GetValue<float>(attr_shift);
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameExp, PrimitiveType_ExpFusion, PopulateExpOpParameter)
}  // namespace lite
}  // namespace mindspore
