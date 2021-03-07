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
#include "nnacl/fp32_grad/binary_cross_entropy_grad.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateBinaryCrossEntropyGradParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  auto binary_cross_entropy_grad_prim = primitive->value_as_BinaryCrossEntropyGrad();
  BinaryCrossEntropyGradParameter *bce_param =
    reinterpret_cast<BinaryCrossEntropyGradParameter *>(malloc(sizeof(BinaryCrossEntropyGradParameter)));
  if (bce_param == nullptr) {
    MS_LOG(ERROR) << "malloc BinaryCrossEntropyGrad Parameter failed.";
    return nullptr;
  }
  memset(bce_param, 0, sizeof(BinaryCrossEntropyGradParameter));
  bce_param->op_parameter_.type_ = schema::PrimitiveType_BinaryCrossEntropyGrad;

  bce_param->reduction = binary_cross_entropy_grad_prim->reduction();
  return reinterpret_cast<OpParameter *>(bce_param);
}
}  // namespace

Registry g_binaryCrossEntropyGradV0ParameterRegistry(schema::v0::PrimitiveType_BinaryCrossEntropyGrad,
                                                     PopulateBinaryCrossEntropyGradParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
