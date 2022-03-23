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

#include "mapper/interp_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include "common/op_attr.h"
#include "common/anf_util.h"
#include "ops/resize.h"
#include "op/interp_operator.h"

namespace mindspore {
namespace dpico {
STATUS InterpMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                         const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }
  auto interp_prim = api::utils::cast<api::SharedPtr<ops::Resize>>(prim);
  MS_ASSERT(interp_prim != nullptr);

  auto interp_operator = std::make_unique<mapper::InterpOperator>();
  if (interp_operator == nullptr) {
    MS_LOG(ERROR) << "interp_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, interp_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  interp_operator->SetOpType(mapper::OpType::INTERP);
  if (prim->GetAttr(ops::kNewHeight) != nullptr) {
    interp_operator->SetInterpHeight(static_cast<int32_t>(interp_prim->get_new_height()));
  }
  if (prim->GetAttr(ops::kNewWidth) != nullptr) {
    interp_operator->SetInterpWidth(static_cast<int32_t>(interp_prim->get_new_width()));
  }
  if (prim->GetAttr(dpico::kZoomFactor) != nullptr) {
    interp_operator->SetInterpZoom(static_cast<int32_t>(api::GetValue<int64_t>(prim->GetAttr(dpico::kZoomFactor))));
  }
  if (prim->GetAttr(dpico::kShrinkFactor) != nullptr) {
    interp_operator->SetInterpShrink(static_cast<int32_t>(api::GetValue<int64_t>(prim->GetAttr(dpico::kShrinkFactor))));
  }
  if (prim->GetAttr(dpico::kPadBeg) != nullptr) {
    interp_operator->SetInterpPadBeg(static_cast<int32_t>(api::GetValue<int64_t>(prim->GetAttr(dpico::kPadBeg))));
  }
  if (prim->GetAttr(dpico::kPadEnd) != nullptr) {
    interp_operator->SetInterpPadEnd(static_cast<int32_t>(api::GetValue<int64_t>(prim->GetAttr(dpico::kPadEnd))));
  }

  base_operators->push_back(std::move(interp_operator));
  return RET_OK;
}
REG_MAPPER(Interp, InterpMapper)
}  // namespace dpico
}  // namespace mindspore
