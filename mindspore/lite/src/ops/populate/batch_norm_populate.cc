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
#include "src/ops/populate/populate_register.h"
#include "nnacl/batchnorm_parameter.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateBatchNorm(const void *prim) {
  BatchNormParameter *batch_norm_param = reinterpret_cast<BatchNormParameter *>(malloc(sizeof(BatchNormParameter)));
  if (batch_norm_param == nullptr) {
    MS_LOG(ERROR) << "malloc BatchNormParameter failed.";
    return nullptr;
  }
  memset(batch_norm_param, 0, sizeof(BatchNormParameter));
  const schema::Primitive *primitive = static_cast<const schema::Primitive *>(prim);
  batch_norm_param->op_parameter_.type_ = primitive->value_type();
  auto prim_batchnorm = primitive->value_as_BatchNorm();
  batch_norm_param->epsilon_ = prim_batchnorm->epsilon();
  batch_norm_param->fused_ = false;
  return reinterpret_cast<OpParameter *>(batch_norm_param);
}
}  // namespace

Registry g_batchNormParameterRegistry(schema::PrimitiveType_BatchNorm, PopulateBatchNorm, SCHEMA_CUR);

}  // namespace lite
}  // namespace mindspore
