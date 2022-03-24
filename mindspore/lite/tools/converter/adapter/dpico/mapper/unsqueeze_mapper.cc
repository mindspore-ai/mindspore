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

#include "mapper/unsqueeze_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include <algorithm>
#include "common/anf_util.h"
#include "ops/unsqueeze.h"
#include "op/unsqueeze_operator.h"

namespace mindspore {
namespace dpico {
STATUS UnsqueezeMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                            const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }
  auto unsqueeze_prim = api::utils::cast<api::SharedPtr<ops::Unsqueeze>>(prim);
  MS_ASSERT(unsqueeze_prim != nullptr);

  auto unsqueeze_operator = std::make_unique<mapper::UnsqueezeOperator>();
  if (unsqueeze_operator == nullptr) {
    MS_LOG(ERROR) << "unsqueeze_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, unsqueeze_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  unsqueeze_operator->SetOpType(mapper::OpType::UNSQUEEZE);

  if (unsqueeze_prim->GetAttr(ops::kAxis) != nullptr) {
    auto axes = api::GetValue<std::vector<int64_t>>(unsqueeze_prim->GetAttr(ops::kAxis));
    std::vector<int32_t> dims;
    (void)std::transform(axes.begin(), axes.end(), std::back_inserter(dims),
                         [](const int64_t &value) { return static_cast<int32_t>(value); });
    unsqueeze_operator->SetUnsqueezeAxisVec(dims);
  } else {
    MS_LOG(ERROR) << "axis attr doesn't exist." << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  if (PushOfflineArgs(cnode, unsqueeze_operator.get(), 1) != RET_OK) {
    MS_LOG(ERROR) << "push offline args failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  base_operators->push_back(std::move(unsqueeze_operator));
  return RET_OK;
}
REG_MAPPER(Unsqueeze, UnsqueezeMapper)
}  // namespace dpico
}  // namespace mindspore
