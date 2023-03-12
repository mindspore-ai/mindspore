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
#include "nnacl/arg_min_max_parameter.h"
#include "ops/arg_min.h"
using mindspore::ops::kAxis;
using mindspore::ops::kKeepDims;
using mindspore::ops::kNameArgMin;
using mindspore::ops::kOutMaxValue;
using mindspore::ops::kTopK;
using mindspore::schema::PrimitiveType_ArgMinFusion;

namespace mindspore {
namespace lite {
OpParameter *PopulateArgMinOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<ArgMinMaxParameter *>(PopulateOpParameter<ArgMinMaxParameter>(base_operator));
  if (param == nullptr) {
    MS_LOG(ERROR) << "new ArgMinMaxParameter failed.";
    return nullptr;
  }

  auto attr_axis = base_operator->GetPrim()->GetAttr(kAxis);
  if (attr_axis == nullptr) {
    MS_LOG(ERROR) << "The attr(" << kAxis << ") of operator(" << base_operator->name() << ") not exist";
    free(param);
    return nullptr;
  }
  auto axis = GetValue<int64_t>(attr_axis);
  CHECK_LESS_RETURN_RET(INT32_MAX, axis, nullptr, param);
  param->axis_ = axis;

  auto attr_topk = base_operator->GetPrim()->GetAttr(kTopK);
  if (attr_topk == nullptr) {
    MS_LOG(ERROR) << "The attr(" << kTopK << ") of operator(" << base_operator->name() << ") not exist";
    free(param);
    return nullptr;
  }
  param->topk_ = GetValue<int32_t>(attr_topk);

  auto attr_out_value = base_operator->GetPrim()->GetAttr(kOutMaxValue);
  if (attr_out_value == nullptr) {
    MS_LOG(ERROR) << "The attr(" << kOutMaxValue << ") of operator(" << base_operator->name() << ") not exist";
    free(param);
    return nullptr;
  }
  param->out_value_ = GetValue<bool>(attr_out_value);

  auto attr_keep_dims = base_operator->GetPrim()->GetAttr(kKeepDims);
  if (attr_keep_dims == nullptr) {
    MS_LOG(ERROR) << "The attr(" << kKeepDims << ") of operator(" << base_operator->name() << ") not exist";
    free(param);
    return nullptr;
  }
  param->keep_dims_ = GetValue<bool>(attr_keep_dims);

  param->get_max_ = false;
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameArgMin, PrimitiveType_ArgMinFusion, PopulateArgMinOpParameter)
}  // namespace lite
}  // namespace mindspore
