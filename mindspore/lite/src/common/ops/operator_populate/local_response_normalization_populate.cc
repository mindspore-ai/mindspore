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
#include "nnacl/fp32/local_response_norm_fp32.h"
#include "ops/lrn.h"
using mindspore::ops::kNameLRN;
using mindspore::schema::PrimitiveType_LRN;
namespace mindspore {
namespace lite {
OpParameter *PopulateLocalResponseNormOpParameter(const BaseOperatorPtr &base_operator) {
  auto param =
    reinterpret_cast<LocalResponseNormParameter *>(PopulateOpParameter<LocalResponseNormParameter>(base_operator));
  if (param == nullptr) {
    MS_LOG(ERROR) << "new LocalResponseNormParameter failed.";
    return nullptr;
  }

  auto op = dynamic_cast<ops::LRN *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "base_operator cast to LRN failed";
    free(param);
    return nullptr;
  }

  param->alpha_ = op->get_alpha();
  param->beta_ = op->get_beta();
  param->bias_ = op->get_bias();
  param->depth_radius_ = static_cast<int>(op->get_depth_radius());
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameLRN, PrimitiveType_LRN, PopulateLocalResponseNormOpParameter)
}  // namespace lite
}  // namespace mindspore
