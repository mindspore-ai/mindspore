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

#include "mapper/abs_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include "common/anf_util.h"
#include "ops/abs.h"
#include "op/absval_operator.h"

namespace mindspore {
namespace dpico {
STATUS AbsMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                      const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }

  auto abs_operator = std::make_unique<mapper::AbsvalOperator>();
  if (abs_operator == nullptr) {
    MS_LOG(ERROR) << "abs_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, abs_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  abs_operator->SetOpType(mapper::OpType::ABSAF);
  if (PushOfflineArgs(cnode, abs_operator.get(), 1) != RET_OK) {
    MS_LOG(ERROR) << "push offline args failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  base_operators->push_back(std::move(abs_operator));
  return RET_OK;
}
REG_MAPPER(Abs, AbsMapper)
}  // namespace dpico
}  // namespace mindspore
