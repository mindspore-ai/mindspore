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
#include "nnacl/fp32_grad/layernormgrad_parameter.h"
#include "ops/grad/layer_norm_grad.h"
using mindspore::ops::kNameLayerNormGrad;
using mindspore::schema::PrimitiveType_LayerNormGrad;
namespace mindspore {
namespace lite {
OpParameter *PopulateLayerNormGradOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<LayerNormGradParameter *>(PopulateOpParameter<LayerNormGradParameter>(base_operator));
  if (param == nullptr) {
    MS_LOG(ERROR) << "new LayerNormGradParameter failed.";
    return nullptr;
  }

  auto op = dynamic_cast<ops::LayerNormGrad *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "base_operator cast to LayerNormGrad failed";
    free(param);
    return nullptr;
  }
  param->begin_norm_axis_ = static_cast<int>(op->get_begin_norm_axis());
  param->begin_params_axis_ = op->get_begin_params_axis();
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameLayerNormGrad, PrimitiveType_LayerNormGrad, PopulateLayerNormGradOpParameter)
}  // namespace lite
}  // namespace mindspore
