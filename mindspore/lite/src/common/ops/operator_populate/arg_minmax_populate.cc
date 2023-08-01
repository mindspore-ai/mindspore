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
#include "src/common/ops/operator_populate/utils.h"
#include "mindspore/core/ops/array_ops.h"
#include "nnacl/arg_min_max_parameter.h"
#include "ops/arg_max.h"
#include "ops/fusion/arg_max_fusion.h"
#include "ops/arg_min.h"
#include "ops/fusion/arg_min_fusion.h"
using mindspore::ops::kAxis;
using mindspore::ops::kKeepDims;
using mindspore::ops::kNameArgMax;
using mindspore::ops::kNameArgMaxFusion;
using mindspore::ops::kNameArgMin;
using mindspore::ops::kNameArgMinFusion;
using mindspore::ops::kOutMaxValue;
using mindspore::ops::kTopK;
using mindspore::schema::PrimitiveType_ArgMaxFusion;
using mindspore::schema::PrimitiveType_ArgMinFusion;

namespace mindspore {
namespace lite {
OpParameter *PopulateArgMinMaxOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<ArgMinMaxParameter *>(PopulateOpParameter<ArgMinMaxParameter>(base_operator));
  if (param == nullptr) {
    MS_LOG(ERROR) << "new ArgMinMaxParameter failed.";
    return nullptr;
  }

  auto axis = GetAttrWithDefault<int64_t>(base_operator, kAxis, 0);
  CHECK_LESS_RETURN_RET(INT32_MAX, axis, nullptr, param);
  param->axis_ = static_cast<int32_t>(axis);
  auto topk = GetAttrWithDefault<int64_t>(base_operator, kTopK, 1);
  CHECK_LESS_RETURN_RET(INT32_MAX, topk, nullptr, param);
  param->topk_ = static_cast<int32_t>(topk);
  param->out_value_ = GetAttrWithDefault<bool>(base_operator, kOutMaxValue, false);
  param->keep_dims_ = GetAttrWithDefault<bool>(base_operator, kKeepDims, false);
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameArgMax, PrimitiveType_ArgMaxFusion, PopulateArgMinMaxOpParameter)
REG_OPERATOR_POPULATE(kNameArgMaxFusion, PrimitiveType_ArgMaxFusion, PopulateArgMinMaxOpParameter)
REG_OPERATOR_POPULATE(kNameArgMin, PrimitiveType_ArgMinFusion, PopulateArgMinMaxOpParameter)
REG_OPERATOR_POPULATE(kNameArgMinFusion, PrimitiveType_ArgMinFusion, PopulateArgMinMaxOpParameter)
}  // namespace lite
}  // namespace mindspore
