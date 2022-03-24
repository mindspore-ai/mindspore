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

#include "mapper/concat_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include "ops/concat.h"
#include "op/concat_operator.h"

namespace mindspore {
namespace dpico {
STATUS ConcatMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                         const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }
  auto concat_prim = api::utils::cast<api::SharedPtr<ops::Concat>>(prim);
  MS_ASSERT(concat_prim != nullptr);

  auto concat_operator = std::make_unique<mapper::ConcatOperator>();
  if (concat_operator == nullptr) {
    MS_LOG(ERROR) << "concat_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, concat_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  concat_operator->SetOpType(mapper::OpType::CONCAT);
  if (concat_prim->GetAttr(ops::kAxis) != nullptr) {
    concat_operator->SetAxis(static_cast<int32_t>(concat_prim->get_axis()));
  }

  base_operators->push_back(std::move(concat_operator));
  return RET_OK;
}
REG_MAPPER(Concat, ConcatMapper)
}  // namespace dpico
}  // namespace mindspore
