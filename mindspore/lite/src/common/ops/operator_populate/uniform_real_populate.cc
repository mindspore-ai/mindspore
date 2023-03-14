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
#include "src/common/ops/populate/default_populate.h"
#include "nnacl/random_parameter.h"
#include "ops/uniform_real.h"
using mindspore::ops::kNameUniformReal;
using mindspore::schema::PrimitiveType_UniformReal;

namespace mindspore {
namespace lite {
OpParameter *PopulateUniformRealOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<RandomParam *>(PopulateOpParameter<RandomParam>());
  if (param == nullptr) {
    MS_LOG(ERROR) << "new RandomParam failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::UniformReal *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "operator is not UniformReal.";
    return nullptr;
  }
  param->seed_ = static_cast<int>(op->get_seed());
  param->seed2_ = static_cast<int>(op->get_seed2());
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameUniformReal, PrimitiveType_UniformReal, PopulateUniformRealOpParameter)
}  // namespace lite
}  // namespace mindspore
