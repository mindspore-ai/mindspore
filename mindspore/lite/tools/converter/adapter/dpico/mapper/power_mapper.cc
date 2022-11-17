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

#include "mapper/power_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include "common/fetch_content.h"
#include "common/anf_util.h"
#include "common/op_enum.h"
#include "ops/fusion/pow_fusion.h"
#include "op/power_operator.h"

namespace mindspore {
namespace dpico {
STATUS PowerMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                        const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }

  auto power_operator = std::make_unique<mapper::PowerOperator>();
  if (power_operator == nullptr) {
    MS_LOG(ERROR) << "power_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, power_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  power_operator->SetOpType(mapper::OpType::POWER);
  DataInfo data_info;
  if (cnode->inputs().size() > kInputIndex2 &&
      FetchDataFromParameterNode(cnode, kInputIndex2, &data_info) == lite::RET_OK) {
    if (data_info.data_type_ != static_cast<int>(kNumberTypeFloat32)) {
      MS_LOG(ERROR) << "data_type not correct";
      return RET_ERROR;
    }
    auto data = reinterpret_cast<float *>(data_info.data_.data());
    if (data == nullptr) {
      MS_LOG(ERROR) << "data is nullptr. " << cnode->fullname_with_scope();
      return RET_ERROR;
    }
    power_operator->SetPowerPower(*data);
  } else if (prim->GetAttr(ops::kPower) != nullptr) {
    power_operator->SetPowerPower(api::GetValue<float>(prim->GetAttr(ops::kPower)));
  } else {
    MS_LOG(ERROR) << "null param";
    return RET_ERROR;
  }
  if (prim->GetAttr(ops::kScale) != nullptr) {
    power_operator->SetPowerScale(api::GetValue<float>(prim->GetAttr(ops::kScale)));
  }
  if (prim->GetAttr(ops::kShift) != nullptr) {
    power_operator->SetPowerShift(api::GetValue<float>(prim->GetAttr(ops::kShift)));
  }
  if (PushOfflineArgs(cnode, power_operator.get(), 1) != RET_OK) {
    MS_LOG(ERROR) << "push offline args failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  base_operators->push_back(std::move(power_operator));
  return RET_OK;
}
REG_MAPPER(PowFusion, PowerMapper)
}  // namespace dpico
}  // namespace mindspore
