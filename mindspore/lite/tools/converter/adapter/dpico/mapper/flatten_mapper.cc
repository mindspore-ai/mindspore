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

#include "mapper/flatten_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include "common/op_attr.h"
#include "common/anf_util.h"
#include "ops/flatten.h"
#include "op/flatten_operator.h"

namespace mindspore {
namespace dpico {
STATUS FlattenMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                          const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }
  auto flatten_prim = api::utils::cast<api::SharedPtr<ops::Flatten>>(prim);
  MS_ASSERT(flatten_prim != nullptr);

  auto flatten_operator = std::make_unique<mapper::FlattenOperator>();
  if (flatten_operator == nullptr) {
    MS_LOG(ERROR) << "flatten_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, flatten_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  flatten_operator->SetOpType(mapper::OpType::FLATTEN);
  if (flatten_prim->GetAttr(kStartAxis) != nullptr) {
    flatten_operator->SetFlattenStartAxis(
      static_cast<int32_t>(api::GetValue<int64_t>(flatten_prim->GetAttr(kStartAxis))));
  }
  if (flatten_prim->GetAttr(kEndAxis) != nullptr) {
    flatten_operator->SetFlattenEndAxis(static_cast<int32_t>(api::GetValue<int64_t>(flatten_prim->GetAttr(kEndAxis))));
  }

  base_operators->push_back(std::move(flatten_operator));
  return RET_OK;
}
REG_MAPPER(Flatten, FlattenMapper)
}  // namespace dpico
}  // namespace mindspore
