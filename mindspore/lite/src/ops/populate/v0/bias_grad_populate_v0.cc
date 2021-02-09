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
#include "nnacl/arithmetic.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateBiasGradParameter(const void *prim) {
  ArithmeticParameter *arithmetic_param = reinterpret_cast<ArithmeticParameter *>(malloc(sizeof(ArithmeticParameter)));
  if (arithmetic_param == nullptr) {
    MS_LOG(ERROR) << "malloc ArithmeticParameter failed.";
    return nullptr;
  }
  memset(arithmetic_param, 0, sizeof(ArithmeticParameter));
  arithmetic_param->op_parameter_.type_ = schema::PrimitiveType_BiasAddGrad;

  return reinterpret_cast<OpParameter *>(arithmetic_param);
}
}  // namespace

Registry g_biasGradV0ParameterParameterRegistry(schema::v0::PrimitiveType_BiasGrad, PopulateBiasGradParameter,
                                                SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
