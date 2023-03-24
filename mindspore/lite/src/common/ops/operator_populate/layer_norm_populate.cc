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
#include "nnacl/layer_norm_parameter.h"
#include "ops/layer_norm.h"
#include "ops/fusion/layer_norm_fusion.h"
using mindspore::ops::kBeginNormAxis;
using mindspore::ops::kBeginParamsAxis;
using mindspore::ops::kElementwiseAffine;
using mindspore::ops::kEpsilon;
using mindspore::ops::kNameLayerNorm;
using mindspore::ops::kNameLayerNormFusion;
using mindspore::schema::PrimitiveType_LayerNormFusion;

namespace mindspore {
namespace lite {
OpParameter *PopulateLayerNormOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<LayerNormParameter *>(PopulateOpParameter<LayerNormParameter>(base_operator));
  if (param == nullptr) {
    MS_LOG(ERROR) << "new LayerNormParameter failed.";
    return nullptr;
  }

  auto attr_begin_norm_axis = base_operator->GetPrim()->GetAttr(kBeginNormAxis);
  if (attr_begin_norm_axis != nullptr) {
    param->begin_norm_axis_ = GetValue<int64_t>(attr_begin_norm_axis);
  }

  auto attr_begin_params_axis = base_operator->GetPrim()->GetAttr(kBeginParamsAxis);
  if (attr_begin_params_axis != nullptr) {
    param->begin_params_axis_ = GetValue<int64_t>(attr_begin_params_axis);
  }

  auto attr_elementwise_affine = base_operator->GetPrim()->GetAttr(kElementwiseAffine);
  if (attr_elementwise_affine == nullptr) {
    MS_LOG(ERROR) << "The attr(" << kElementwiseAffine << ") of operator(" << base_operator->name() << ") not exist";
    free(param);
    return nullptr;
  }
  param->elementwise_affine_ = GetValue<bool>(attr_elementwise_affine);

  auto attr_epsilon = base_operator->GetPrim()->GetAttr(kEpsilon);
  if (attr_epsilon != nullptr) {
    param->epsilon_ = GetValue<float>(attr_epsilon);
  }
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameLayerNorm, PrimitiveType_LayerNormFusion, PopulateLayerNormOpParameter)
REG_OPERATOR_POPULATE(kNameLayerNormFusion, PrimitiveType_LayerNormFusion, PopulateLayerNormOpParameter)
}  // namespace lite
}  // namespace mindspore
