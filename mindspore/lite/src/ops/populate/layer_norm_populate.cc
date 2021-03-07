/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "nnacl/layer_norm_parameter.h"
#include <cstdint>
#include "src/ops/populate/populate_register.h"
namespace mindspore {
namespace lite {
OpParameter *PopulateLayerNormParameter(const void *prim) {
  auto layer_norm_parameter = reinterpret_cast<LayerNormParameter *>(malloc(sizeof(LayerNormParameter)));
  if (layer_norm_parameter == nullptr) {
    MS_LOG(ERROR) << "malloc LayerNormParameter failed.";
    return nullptr;
  }
  memset(layer_norm_parameter, 0, sizeof(LayerNormParameter));
  auto *primitive = static_cast<const schema::Primitive *>(prim);
  layer_norm_parameter->op_parameter_.type_ = primitive->value_type();
  auto param = primitive->value_as_LayerNormFusion();
  layer_norm_parameter->epsilon_ = param->epsilon();
  layer_norm_parameter->elementwise_affine_ = param->elementwise_affine();
  layer_norm_parameter->begin_norm_axis_ = static_cast<int>(param->begin_norm_axis());
  layer_norm_parameter->begin_params_axis_ = static_cast<int>(param->begin_params_axis());
  return reinterpret_cast<OpParameter *>(layer_norm_parameter);
}

Registry g_layerNormParameterRegistry(schema::PrimitiveType_LayerNormFusion, PopulateLayerNormParameter, SCHEMA_CUR);
}  // namespace lite
}  // namespace mindspore
