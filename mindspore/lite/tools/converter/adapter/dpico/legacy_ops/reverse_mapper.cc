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

#include "legacy_ops/reverse_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include "common/anf_util.h"
#include "ops/reverse_v2.h"
#include "op/reverse_operator.h"

namespace mindspore {
namespace dpico {
STATUS ReverseMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                          const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }
  auto reverse_prim = api::utils::cast<api::SharedPtr<ops::ReverseV2>>(prim);
  MS_ASSERT(reverse_prim != nullptr);

  auto reverse_operator = std::make_unique<mapper::ReverseOperator>();
  if (reverse_operator == nullptr) {
    MS_LOG(ERROR) << "reverse_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, reverse_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  reverse_operator->SetOpType(mapper::OpType::REVERSE);
  if (reverse_prim->GetAttr(ops::kAxis) != nullptr) {
    auto axis = reverse_prim->get_axis();
    if (axis.size() != 1) {
      MS_LOG(ERROR) << "reverse's axis size only supports 1 by dpico. " << cnode->fullname_with_scope();
      return RET_ERROR;
    }
    reverse_operator->SetAxis(static_cast<int32_t>(axis.at(0)));
  }
  base_operators->push_back(std::move(reverse_operator));
  return RET_OK;
}
REG_MAPPER(ReverseV2, ReverseMapper)
}  // namespace dpico
}  // namespace mindspore
