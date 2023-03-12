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
#include "nnacl/batchnorm_parameter.h"
#include "ops/batch_norm.h"
using mindspore::ops::kNameBatchNorm;
using mindspore::schema::PrimitiveType_BatchNorm;
namespace mindspore {
namespace lite {
OpParameter *PopulateBatchNormOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<BatchNormParameter *>(PopulateOpParameter<BatchNormParameter>(base_operator));
  if (param == nullptr) {
    MS_LOG(ERROR) << "new BatchNormParameter failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::BatchNorm *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "base_operator cast to BatchNorm failed";
    free(param);
    return nullptr;
  }
  param->epsilon_ = op->get_epsilon();
  param->is_training_ = op->get_is_training();
  param->fused_ = false;
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameBatchNorm, PrimitiveType_BatchNorm, PopulateBatchNormOpParameter)
}  // namespace lite
}  // namespace mindspore
