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
#include "nnacl/predict_parameter.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateCustomPredictParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  auto custom_predict_prim = primitive->value_as_CustomPredict();
  PredictParameter *param = reinterpret_cast<PredictParameter *>(malloc(sizeof(PredictParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc param failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(PredictParameter));
  param->op_parameter_.type_ = schema::PrimitiveType_CustomPredict;

  param->output_num = custom_predict_prim->outputNum();
  param->weight_threshold = custom_predict_prim->weightThreshold();
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

Registry g_customPredictV0ParameterRegistry(schema::v0::PrimitiveType_CustomPredict, PopulateCustomPredictParameter,
                                            SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
