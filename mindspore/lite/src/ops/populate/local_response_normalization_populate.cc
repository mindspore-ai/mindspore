/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "src/ops/local_response_normalization.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/fp32/local_response_norm_fp32.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateLocalResponseNormParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto local_response_norm_attr = reinterpret_cast<mindspore::lite::LocalResponseNormalization *>(
    const_cast<mindspore::lite::PrimitiveC *>(primitive));
  LocalResponseNormParameter *lrn_param =
    reinterpret_cast<LocalResponseNormParameter *>(malloc(sizeof(LocalResponseNormParameter)));
  if (lrn_param == nullptr) {
    MS_LOG(ERROR) << "malloc LocalResponseNormParameter failed.";
    return nullptr;
  }
  memset(lrn_param, 0, sizeof(LocalResponseNormParameter));
  lrn_param->op_parameter_.type_ = primitive->Type();
  lrn_param->depth_radius_ = local_response_norm_attr->GetDepthRadius();
  lrn_param->bias_ = local_response_norm_attr->GetBias();
  lrn_param->alpha_ = local_response_norm_attr->GetAlpha();
  lrn_param->beta_ = local_response_norm_attr->GetBeta();
  return reinterpret_cast<OpParameter *>(lrn_param);
}

Registry LocalResponseNormalizationParameterRegistry(schema::PrimitiveType_LocalResponseNormalization,
                                                     PopulateLocalResponseNormParameter);

}  // namespace lite
}  // namespace mindspore
