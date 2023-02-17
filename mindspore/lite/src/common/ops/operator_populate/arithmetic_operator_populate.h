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
#ifndef MINDSPORE_LITE_SRC_COMMON_OPS_OPERATOR_POPULATE_ARITHMETIC_POPULATE_H_
#define MINDSPORE_LITE_SRC_COMMON_OPS_OPERATOR_POPULATE_ARITHMETIC_POPULATE_H_

#include "nnacl/arithmetic.h"
#include "src/common/ops/operator_populate/operator_populate_register.h"
namespace mindspore {
namespace lite {
ArithmeticParameter *PopulateArithmeticCommonPara(const BaseOperatorPtr &base_operator);
OpParameter *PopulateArithmetic(const BaseOperatorPtr &base_operator);
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_COMMON_OPS_OPERATOR_POPULATE_ARITHMETIC_POPULATE_H_
