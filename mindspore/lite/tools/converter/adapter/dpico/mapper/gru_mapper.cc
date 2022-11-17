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

#include "mapper/gru_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include <string>
#include "common/op_attr.h"
#include "common/anf_util.h"
#include "op/gru_operator.h"

namespace mindspore {
namespace dpico {
STATUS GruMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                      const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  MS_CHECK_TRUE_MSG(base_operators != nullptr, RET_ERROR, "base_operators is nullptr.");
  auto gru_operator = std::make_unique<mapper::GruOperator>();
  MS_CHECK_TRUE_MSG(gru_operator != nullptr, RET_ERROR, "gru_operator is nullptr.");

  if (SetCommonAttr(cnode, gru_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  if (prim->GetAttr(kNumOutput) != nullptr) {
    gru_operator->SetRecurrentNumOutput(static_cast<uint32_t>(api::GetValue<int64_t>(prim->GetAttr(kNumOutput))));
  }
  if (prim->GetAttr(kExposeHidden) != nullptr) {
    gru_operator->SetRecurrentExposeHidden(api::GetValue<bool>(prim->GetAttr(kExposeHidden)));
  }
  if (SetRecurrentDataInfo(cnode, gru_operator.get()) != RET_OK) {
    MS_LOG(ERROR) << "set gru data info failed.";
    return RET_ERROR;
  }

  base_operators->push_back(std::move(gru_operator));
  return RET_OK;
}
REG_MAPPER(Gru, GruMapper)
}  // namespace dpico
}  // namespace mindspore
