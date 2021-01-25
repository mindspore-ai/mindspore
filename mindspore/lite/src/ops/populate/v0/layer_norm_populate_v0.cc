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

#include "schema/model_v0_generated.h"
#include "src/ops/populate/v0/layer_norm_populate_v0.h"
#include "nnacl/layer_norm_parameter.h"
#include "src/ops/populate/populate_register.h"

namespace mindspore {
namespace lite {
OpParameter *PopulateLayerNormParameterV0(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  auto layer_norm_prim = primitive->value_as_LayerNorm();
  auto layer_norm_parameter = reinterpret_cast<LayerNormParameter *>(malloc(sizeof(LayerNormParameter)));
  if (layer_norm_parameter == nullptr) {
    MS_LOG(ERROR) << "malloc LayerNormParameter failed.";
    return nullptr;
  }
  memset(layer_norm_parameter, 0, sizeof(LayerNormParameter));
  layer_norm_parameter->op_parameter_.type_ = schema::PrimitiveType_LayerNormFusion;
  auto normalized_shape = layer_norm_prim->normalizedShape();
  if (normalized_shape != nullptr) {
    layer_norm_parameter->normalized_dims_ = normalized_shape->size();
    if (((size_t)normalized_shape->size()) > SIZE_MAX / sizeof(int)) {
      MS_LOG(ERROR) << "normalized_shape size too big";
      free(layer_norm_parameter);
      return nullptr;
    }
    for (size_t i = 0; i < normalized_shape->size(); i++) {
      layer_norm_parameter->normalized_shape_[i] = *(normalized_shape->begin() + i);
    }
  }
  layer_norm_parameter->epsilon_ = layer_norm_prim->epsilon();
  layer_norm_parameter->elementwise_affine_ = layer_norm_prim->elementwiseAffine();

  return reinterpret_cast<OpParameter *>(layer_norm_parameter);
}

Registry g_layerNormV0ParameterRegistry(schema::v0::PrimitiveType_LayerNorm, PopulateLayerNormParameterV0, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
