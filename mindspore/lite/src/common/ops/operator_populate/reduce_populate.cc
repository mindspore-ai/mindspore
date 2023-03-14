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
#include "nnacl/reduce_parameter.h"
#include "ops/reduce.h"
#include "ops/fusion/reduce_fusion.h"
#include "ops/op_name.h"
using mindspore::ops::kNameReduce;
using mindspore::ops::kNameReduceFusion;
using mindspore::schema::PrimitiveType_ReduceFusion;
namespace mindspore {
namespace lite {
OpParameter *PopulateReduceOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<ReduceParameter *>(PopulateOpParameter<ReduceParameter>(base_operator));
  if (param == nullptr) {
    MS_LOG(ERROR) << "Make OpParameter ptr failed";
    return nullptr;
  }

  auto keep_dims = base_operator->GetPrim()->GetAttr(mindspore::ops::kKeepDims);
  if (keep_dims == nullptr) {
    MS_LOG(ERROR) << "The attr(" << mindspore::ops::kKeepDims << ") of operator(" << base_operator->name()
                  << ") not exist";
    free(param);
    return nullptr;
  }
  param->keep_dims_ = GetValue<bool>(keep_dims);

  auto reduce_to_end = base_operator->GetPrim()->GetAttr(mindspore::ops::kReduceToEnd);
  if (reduce_to_end != nullptr) {
    param->reduce_to_end_ = GetValue<bool>(reduce_to_end);
  }

  auto coeff = base_operator->GetPrim()->GetAttr(mindspore::ops::kCoeff);
  if (coeff != nullptr) {
    param->coeff = GetValue<float>(coeff);
  }

  auto mode = base_operator->GetPrim()->GetAttr(mindspore::ops::kMode);
  if (mode != nullptr) {
    param->mode_ = GetValue<int64_t>(mode);
  }

  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameReduce, PrimitiveType_ReduceFusion, PopulateReduceOpParameter)
REG_OPERATOR_POPULATE(kNameReduceFusion, PrimitiveType_ReduceFusion, PopulateReduceOpParameter)
}  // namespace lite
}  // namespace mindspore
