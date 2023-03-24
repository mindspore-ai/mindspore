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
#include "ops/addn.h"
#include "ops/depend.h"
#include "ops/log1p.h"
#include "ops/switch_layer.h"
#include "ops/zeros_like.h"
using mindspore::ops::kNameAddN;
using mindspore::ops::kNameDepend;
using mindspore::ops::kNameLog1p;
using mindspore::ops::kNameSwitchLayer;
using mindspore::ops::kNameZerosLike;
using mindspore::schema::PrimitiveType_AddN;
using mindspore::schema::PrimitiveType_Depend;
using mindspore::schema::PrimitiveType_Log1p;
using mindspore::schema::PrimitiveType_SwitchLayer;
using mindspore::schema::PrimitiveType_ZerosLike;

namespace mindspore {
namespace lite {
OpParameter *PopulateCommonOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc OpParameter ptr failed";
    return nullptr;
  }
  memset(param, 0, sizeof(OpParameter));

  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameAddN, PrimitiveType_AddN, PopulateCommonOpParameter)
REG_OPERATOR_POPULATE(kNameDepend, PrimitiveType_Depend, PopulateCommonOpParameter)
REG_OPERATOR_POPULATE(kNameLog1p, PrimitiveType_Log1p, PopulateCommonOpParameter)
REG_OPERATOR_POPULATE(kNameSwitchLayer, PrimitiveType_SwitchLayer, PopulateCommonOpParameter)
REG_OPERATOR_POPULATE(kNameZerosLike, PrimitiveType_ZerosLike, PopulateCommonOpParameter)
}  // namespace lite
}  // namespace mindspore
