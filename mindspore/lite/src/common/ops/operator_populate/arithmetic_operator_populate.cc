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
#include "src/common/ops/operator_populate/arithmetic_operator_populate.h"
#include <memory>
#include <string>
#include "ops/real_div.h"
#include "ops/logical_and.h"
#include "ops/logical_or.h"
#include "ops/equal.h"
#include "ops/not_equal.h"
#include "ops/less.h"
#include "ops/less_equal.h"
#include "ops/greater.h"
#include "ops/greater_equal.h"
#include "ops/maximum.h"
#include "ops/minimum.h"
#include "ops/floor_div.h"
#include "ops/floor_mod.h"
#include "ops/squared_difference.h"
#include "ops/mod.h"

using mindspore::ops::kNameEqual;
using mindspore::ops::kNameFloorDiv;
using mindspore::ops::kNameFloorMod;
using mindspore::ops::kNameGreater;
using mindspore::ops::kNameGreaterEqual;
using mindspore::ops::kNameLess;
using mindspore::ops::kNameLessEqual;
using mindspore::ops::kNameLogicalAnd;
using mindspore::ops::kNameLogicalOr;
using mindspore::ops::kNameMaximum;
using mindspore::ops::kNameMinimum;
using mindspore::ops::kNameMod;
using mindspore::ops::kNameNotEqual;
using mindspore::ops::kNameRealDiv;
using mindspore::ops::kNameSquaredDifference;

namespace mindspore {
namespace lite {
ArithmeticParameter *PopulateArithmeticCommonOpPara(const BaseOperatorPtr &base_operator) {
  if (base_operator == nullptr) {
    MS_LOG(ERROR) << "base_operator is nullptr";
    return nullptr;
  }
  auto *param = reinterpret_cast<ArithmeticParameter *>(malloc(sizeof(ArithmeticParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc ArithmeticParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(ArithmeticParameter));
  auto name = base_operator->name();

  auto iter = kOpNameWithPrimitiveType.find(name);
  if (iter == kOpNameWithPrimitiveType.end()) {
    MS_LOG(ERROR) << "Can not find ParameterPtrGen : " << name;
    return nullptr;
  }

  param->op_parameter_.type_ = iter->second;
  param->broadcasting_ = false;
  param->ndim_ = 0;
  param->activation_type_ = 0;
  return param;
}

OpParameter *PopulateArithmeticOp(const BaseOperatorPtr &base_operator) {
  ArithmeticParameter *param = PopulateArithmeticCommonOpPara(base_operator);
  if (param == nullptr) {
    MS_LOG(ERROR) << "PopulateArithmeticCommonPara failed.";
    return nullptr;
  }
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameRealDiv, PopulateArithmeticOp)
REG_OPERATOR_POPULATE(kNameLogicalAnd, PopulateArithmeticOp)
REG_OPERATOR_POPULATE(kNameLogicalOr, PopulateArithmeticOp)
REG_OPERATOR_POPULATE(kNameEqual, PopulateArithmeticOp);
REG_OPERATOR_POPULATE(kNameNotEqual, PopulateArithmeticOp)
REG_OPERATOR_POPULATE(kNameLess, PopulateArithmeticOp)
REG_OPERATOR_POPULATE(kNameLessEqual, PopulateArithmeticOp)
REG_OPERATOR_POPULATE(kNameGreater, PopulateArithmeticOp)
REG_OPERATOR_POPULATE(kNameGreaterEqual, PopulateArithmeticOp)
REG_OPERATOR_POPULATE(kNameMaximum, PopulateArithmeticOp)
REG_OPERATOR_POPULATE(kNameMinimum, PopulateArithmeticOp)
REG_OPERATOR_POPULATE(kNameFloorDiv, PopulateArithmeticOp)
REG_OPERATOR_POPULATE(kNameFloorMod, PopulateArithmeticOp)
REG_OPERATOR_POPULATE(kNameMod, PopulateArithmeticOp)
REG_OPERATOR_POPULATE(kNameSquaredDifference, PopulateArithmeticOp)
}  // namespace lite
}  // namespace mindspore
