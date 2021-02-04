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
#include "src/ops/populate/populate_register.h"
#include "nnacl/fp32/local_response_norm_fp32.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateLocalResponseNormParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  auto local_response_normalization_prim = primitive->value_as_LocalResponseNormalization();

  LocalResponseNormParameter *lrn_param =
    reinterpret_cast<LocalResponseNormParameter *>(malloc(sizeof(LocalResponseNormParameter)));
  if (lrn_param == nullptr) {
    MS_LOG(ERROR) << "malloc LocalResponseNormParameter failed.";
    return nullptr;
  }
  memset(lrn_param, 0, sizeof(LocalResponseNormParameter));
  lrn_param->op_parameter_.type_ = schema::PrimitiveType_LRN;
  lrn_param->depth_radius_ = local_response_normalization_prim->depth_radius();
  lrn_param->bias_ = local_response_normalization_prim->bias();
  lrn_param->alpha_ = local_response_normalization_prim->alpha();
  lrn_param->beta_ = local_response_normalization_prim->beta();
  return reinterpret_cast<OpParameter *>(lrn_param);
}
}  // namespace

Registry g_localResponseNormalizationV0ParameterRegistry(schema::v0::PrimitiveType_LocalResponseNormalization,
                                                         PopulateLocalResponseNormParameter, SCHEMA_V0);
Registry g_lrnV0ParameterRegistry(schema::v0::PrimitiveType_Lrn, PopulateLocalResponseNormParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
