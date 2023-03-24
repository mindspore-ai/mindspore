/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "src/common/ops/operator_populate/operator_populate_register.h"
#include "nnacl/predict_parameter.h"
#include "ops/custom_predict.h"
using mindspore::ops::kNameCustomPredict;
using mindspore::schema::PrimitiveType_CustomPredict;
namespace mindspore {
namespace lite {
OpParameter *PopulateCustomPredictOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<PredictParameter *>(PopulateOpParameter<PredictParameter>(base_operator));
  if (param == nullptr) {
    MS_LOG(ERROR) << "new PredictParameter failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::CustomPredict *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "base_operator cast to CustomPredict failed";
    free(param);
    return nullptr;
  }
  param->output_num = op->get_output_num();
  param->weight_threshold = op->get_weight_threshold();
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameCustomPredict, PrimitiveType_CustomPredict, PopulateCustomPredictOpParameter)
}  // namespace lite
}  // namespace mindspore
