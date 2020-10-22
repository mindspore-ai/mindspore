/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "src/ops/custom_predict.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/predict_parameter.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateCustomPredictParameter(const mindspore::lite::PrimitiveC *primitive) {
  PredictParameter *param = reinterpret_cast<PredictParameter *>(malloc(sizeof(PredictParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc param failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(PredictParameter));
  param->op_parameter_.type_ = primitive->Type();
  auto prim = reinterpret_cast<mindspore::lite::CustomPredict *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  param->output_num = prim->GetOutputNum();
  param->weight_threshold = prim->GetWeightThreshold();
  return reinterpret_cast<OpParameter *>(param);
}
Registry CustomPredictParameterRegistry(schema::PrimitiveType_CustomPredict, PopulateCustomPredictParameter);

}  // namespace lite
}  // namespace mindspore
