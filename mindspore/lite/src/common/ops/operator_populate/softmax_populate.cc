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
#include "nnacl/softmax_parameter.h"
#include "ops/softmax.h"
using mindspore::ops::kAxis;
using mindspore::ops::kNameSoftmax;
using mindspore::schema::PrimitiveType_Softmax;

namespace mindspore {
namespace lite {
OpParameter *PopulateSoftmaxOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<SoftmaxParameter *>(PopulateOpParameter<SoftmaxParameter>());
  if (param == nullptr) {
    MS_LOG(ERROR) << "new SoftmaxParameter failed.";
    return nullptr;
  }
  ValuePtr attr = base_operator->GetPrim()->GetAttr(kAxis);
  if (attr == nullptr) {
    MS_LOG(ERROR) << "The attr(" << kAxis << ") of operator(" << base_operator->name() << ") not exist";
    free(param);
    return nullptr;
  }
  auto flat_axis = GetValue<std::vector<int64_t>>(attr);
  if (flat_axis.size() != 1) {
    MS_LOG(ERROR) << "axis number invalid!number: " << flat_axis.size();
    free(param);
    return nullptr;
  }
  param->axis_ = flat_axis.data()[0];
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameSoftmax, PrimitiveType_Softmax, PopulateSoftmaxOpParameter)
}  // namespace lite
}  // namespace mindspore
