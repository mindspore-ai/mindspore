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

#include "nnacl/fp32_grad/layernormgrad_parameter.h"
#include "src/ops/populate/populate_register.h"
using mindspore::schema::PrimitiveType_LayerNormGrad;

namespace mindspore {
namespace lite {
OpParameter *PopulateLayerNormGradParameter(const void *prim) {
  auto *primitive = static_cast<const schema::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto value = primitive->value_as_LayerNormGrad();
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr";
    return nullptr;
  }

  auto param = reinterpret_cast<LayerNormGradParameter *>(malloc(sizeof(LayerNormGradParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc LayerNormParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(LayerNormGradParameter));

  param->op_parameter_.type_ = primitive->value_type();
  param->begin_norm_axis_ = value->begin_norm_axis();
  param->begin_params_axis_ = value->begin_params_axis();
  return reinterpret_cast<OpParameter *>(param);
}

REG_POPULATE(PrimitiveType_LayerNormGrad, PopulateLayerNormGradParameter, SCHEMA_CUR);
}  // namespace lite
}  // namespace mindspore
