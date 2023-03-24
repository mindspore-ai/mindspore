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
#include "ops/fused_batch_norm.h"
using mindspore::ops::kNameFusedBatchNorm;
using mindspore::schema::PrimitiveType_FusedBatchNorm;
namespace mindspore {
namespace lite {
OpParameter *PopulateFusedBatchNormOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<BatchNormParameter *>(PopulateOpParameter<BatchNormParameter>(base_operator));
  if (param == nullptr) {
    MS_LOG(ERROR) << "new BatchNormParameter failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::FusedBatchNorm *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "base_operator cast to FusedBatchNorm failed";
    free(param);
    return nullptr;
  }
  param->epsilon_ = op->get_epsilon();
  param->is_training_ = static_cast<bool>(op->get_mode());
  param->momentum_ = op->get_momentum();
  if (param->momentum_ < static_cast<float>(C0NUM) || param->momentum_ > static_cast<float>(C1NUM)) {
    MS_LOG(ERROR) << "invalid momentum value: " << param->momentum_;
    free(param);
    return nullptr;
  }
  param->fused_ = true;
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameFusedBatchNorm, PrimitiveType_FusedBatchNorm, PopulateFusedBatchNormOpParameter)
}  // namespace lite
}  // namespace mindspore
