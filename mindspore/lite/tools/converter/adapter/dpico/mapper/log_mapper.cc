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

#include "mapper/log_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include "common/anf_util.h"
#include "ops/log.h"
#include "op/log_operator.h"

namespace mindspore {
namespace dpico {
STATUS LogMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                      const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }

  auto log_operator = std::make_unique<mapper::LogOperator>();
  if (log_operator == nullptr) {
    MS_LOG(ERROR) << "log_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, log_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  log_operator->SetOpType(mapper::OpType::LOG);
  if (prim->GetAttr(ops::kBase) != nullptr) {
    log_operator->SetLogBase(api::GetValue<float>(prim->GetAttr(ops::kBase)));
  }
  if (prim->GetAttr(ops::kScale) != nullptr) {
    log_operator->SetLogScale(api::GetValue<float>(prim->GetAttr(ops::kScale)));
  }
  if (prim->GetAttr(ops::kShift) != nullptr) {
    log_operator->SetLogShift(api::GetValue<float>(prim->GetAttr(ops::kShift)));
  }
  if (PushOfflineArgs(cnode, log_operator.get(), 1) != RET_OK) {
    MS_LOG(ERROR) << "push offline args failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  base_operators->push_back(std::move(log_operator));
  return RET_OK;
}
REG_MAPPER(Log, LogMapper)
}  // namespace dpico
}  // namespace mindspore
