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

namespace mindspore {
namespace lite {
OpParameter *PopulateLayerNormGradParameter(const void *prim) {
  auto layer_norm_grad_parameter = reinterpret_cast<LayerNormGradParameter *>(malloc(sizeof(LayerNormGradParameter)));
  if (layer_norm_grad_parameter == nullptr) {
    MS_LOG(ERROR) << "malloc LayerNormParameter failed.";
    return nullptr;
  }
  memset(layer_norm_grad_parameter, 0, sizeof(LayerNormGradParameter));
  auto *primitive = static_cast<const schema::Primitive *>(prim);
  layer_norm_grad_parameter->op_parameter_.type_ = primitive->value_type();
  auto param = primitive->value_as_LayerNormGrad();
  layer_norm_grad_parameter->begin_norm_axis_ = param->begin_norm_axis();
  layer_norm_grad_parameter->begin_params_axis_ = param->begin_params_axis();
  return reinterpret_cast<OpParameter *>(layer_norm_grad_parameter);
}

Registry g_layerNormGradParameterRegistry(schema::PrimitiveType_LayerNormGrad, PopulateLayerNormGradParameter,
                                          SCHEMA_CUR);
}  // namespace lite
}  // namespace mindspore
