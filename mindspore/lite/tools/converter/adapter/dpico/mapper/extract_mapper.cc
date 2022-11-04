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

#include "mapper/extract_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include "common/anf_util.h"
#include "common/op_attr.h"
#include "op/extract_operator.h"

namespace mindspore {
namespace dpico {
STATUS ExtractMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                          const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }
  auto extract_operator = std::make_unique<mapper::ExtractOperator>();
  if (extract_operator == nullptr) {
    MS_LOG(ERROR) << "extract_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, extract_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  extract_operator->SetOpType(mapper::OpType::EXTRACT);
  if (prim->GetAttr(dpico::kSlicePointBegin) != nullptr) {
    extract_operator->SetSlicePointBegin(
      static_cast<uint32_t>(api::GetValue<int64_t>(prim->GetAttr(dpico::kSlicePointBegin))));
  }
  if (prim->GetAttr(dpico::kSlicePointEnd) != nullptr) {
    extract_operator->SetSlicePointEnd(
      static_cast<uint32_t>(api::GetValue<int64_t>(prim->GetAttr(dpico::kSlicePointEnd))));
  }
  if (prim->GetAttr(ops::kAxis) != nullptr) {
    extract_operator->SetAxis(static_cast<int>(api::GetValue<int64_t>(prim->GetAttr(ops::kAxis))));
  }
  base_operators->push_back(std::move(extract_operator));
  return RET_OK;
}
REG_MAPPER(Extract, ExtractMapper)
}  // namespace dpico
}  // namespace mindspore
