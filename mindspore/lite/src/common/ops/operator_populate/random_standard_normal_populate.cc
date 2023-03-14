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

#include "src/common/ops/operator_populate/operator_populate_register.h"
#include "nnacl/random_parameter.h"
#include "ops/random_standard_normal.h"
using mindspore::ops::kNameRandomStandardNormal;
using mindspore::schema::PrimitiveType_RandomStandardNormal;

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateRandomStandardNormalOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<RandomNormalParam *>(PopulateOpParameter<RandomNormalParam>());
  if (param == nullptr) {
    MS_LOG(ERROR) << "new RandomNormalParam failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::RandomStandardNormal *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "operator is not RandomStandardNormal.";
    return nullptr;
  }

  if (op->get_seed2() != 0) {
    param->seed_ = op->get_seed2();
  } else {
    param->seed_ = op->get_seed();
  }
  param->mean_ = 0.0;
  param->scale_ = 1.0;

  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

REG_OPERATOR_POPULATE(kNameRandomStandardNormal, PrimitiveType_RandomStandardNormal,
                      PopulateRandomStandardNormalOpParameter);
}  // namespace lite
}  // namespace mindspore
