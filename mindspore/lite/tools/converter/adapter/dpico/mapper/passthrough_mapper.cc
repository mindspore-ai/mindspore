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

#include "mapper/passthrough_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include "common/op_attr.h"
#include "common/anf_util.h"
#include "op/passthrough_operator.h"

namespace mindspore {
namespace dpico {
STATUS PassThroughMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                              const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }

  auto pass_through_operator = std::make_unique<mapper::PassthroughOperator>();
  if (pass_through_operator == nullptr) {
    MS_LOG(ERROR) << "pass_through_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, pass_through_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  pass_through_operator->SetOpType(mapper::OpType::PASS_THROUGH);
  if (prim->GetAttr(kNumOutput) != nullptr) {
    pass_through_operator->SetPassThroughNumOutput(
      static_cast<uint32_t>(api::GetValue<int64_t>(prim->GetAttr(kNumOutput))));
  }
  if (prim->GetAttr(kBlockHeight) != nullptr) {
    pass_through_operator->SetPassThroughBlockHeight(
      static_cast<uint32_t>(api::GetValue<int64_t>(prim->GetAttr(kBlockHeight))));
  }
  if (prim->GetAttr(kBlockWidth) != nullptr) {
    pass_through_operator->SetPassThroughBlockWidth(
      static_cast<uint32_t>(api::GetValue<int64_t>(prim->GetAttr(kBlockWidth))));
  }

  base_operators->push_back(std::move(pass_through_operator));
  return RET_OK;
}
REG_MAPPER(PassThrough, PassThroughMapper)
}  // namespace dpico
}  // namespace mindspore
