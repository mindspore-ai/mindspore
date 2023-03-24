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
#include "nnacl/l2_norm_parameter.h"
#include "ops/l2_normalize.h"
#include "ops/fusion/l2_normalize_fusion.h"
using mindspore::ops::kActivationType;
using mindspore::ops::kAxis;
using mindspore::ops::kEpsilon;
using mindspore::ops::kNameL2Normalize;
using mindspore::ops::kNameL2NormalizeFusion;
using mindspore::schema::PrimitiveType_L2NormalizeFusion;

namespace mindspore {
namespace lite {
OpParameter *PopulateL2NormOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<L2NormParameter *>(PopulateOpParameter<L2NormParameter>(base_operator));
  if (param == nullptr) {
    MS_LOG(ERROR) << "new L2NormParameter failed.";
    return nullptr;
  }

  auto attr_axis = base_operator->GetPrim()->GetAttr(kAxis);
  if (attr_axis == nullptr) {
    MS_LOG(ERROR) << "The attr(" << kAxis << ") of operator(" << base_operator->name() << ") not exist";
    free(param);
    return nullptr;
  }
  auto axis_vec = GetValue<std::vector<int64_t>>(attr_axis);
  if (axis_vec.size() > MAX_SHAPE_SIZE) {
    MS_LOG(ERROR) << "axis size " << axis_vec.size() << " is invalid!";
    free(param);
    return nullptr;
  }
  param->axis_num_ = axis_vec.size();

  for (size_t i = 0; i < axis_vec.size(); i++) {
    param->axis_[i] = static_cast<int>(axis_vec[i]);
  }

  auto attr_epsilon = base_operator->GetPrim()->GetAttr(kEpsilon);
  if (attr_epsilon == nullptr) {
    MS_LOG(ERROR) << "The attr(" << kEpsilon << ") of operator(" << base_operator->name() << ") not exist";
    free(param);
    return nullptr;
  }
  auto epsilon = GetValue<float>(attr_epsilon);
  if (epsilon < 1e-6) {
    param->epsilon_ = 1e-6;
  } else {
    param->epsilon_ = epsilon;
  }

  auto attr_act_type = base_operator->GetPrim()->GetAttr(kActivationType);
  if (attr_epsilon != nullptr) {
    auto act_type = static_cast<ActType>(GetValue<int64_t>(attr_act_type));
    if (act_type == ActType_Relu || act_type == ActType_Relu6) {
      param->act_type_ = act_type;
    } else {
      param->act_type_ = ActType_No;
    }
  }
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameL2Normalize, PrimitiveType_L2NormalizeFusion, PopulateL2NormOpParameter)
REG_OPERATOR_POPULATE(kNameL2NormalizeFusion, PrimitiveType_L2NormalizeFusion, PopulateL2NormOpParameter)
}  // namespace lite
}  // namespace mindspore
