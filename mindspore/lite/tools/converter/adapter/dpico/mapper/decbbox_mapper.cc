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

#include "mapper/decbbox_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include "parser/detection_output_param_holder.h"
#include "common/anf_util.h"
#include "common/op_attr.h"
#include "op/dec_bbox_operator.h"

namespace mindspore {
namespace dpico {
STATUS DecBBoxMapper::Map(const CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators, const PrimitivePtr &prim,
                          const CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }
  auto decbbox_operator = std::make_unique<mapper::DecBboxOperator>();
  if (decbbox_operator == nullptr) {
    MS_LOG(ERROR) << "decbbox_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, decbbox_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  decbbox_operator->SetOpType(mapper::OpType::DECBBOX);
  if (prim->GetAttr(kNumAnchors) != nullptr) {
    decbbox_operator->SetNumAnchors(GetValue<uint32_t>(prim->GetAttr(kNumAnchors)));
  }
  if (prim->GetAttr(kNumBboxesPerGrid) != nullptr) {
    decbbox_operator->SetNumBboxesPerGrid(GetValue<uint32_t>(prim->GetAttr(kNumBboxesPerGrid)));
  }
  if (prim->GetAttr(kNumCoords) != nullptr) {
    decbbox_operator->SetNumCoords(GetValue<uint32_t>(prim->GetAttr(kNumCoords)));
  }
  if (prim->GetAttr(kNumClasses) != nullptr) {
    decbbox_operator->SetNumClasses(GetValue<uint32_t>(prim->GetAttr(kNumClasses)));
  }
  if (prim->GetAttr(kNumGridsHeight) != nullptr) {
    decbbox_operator->SetNumGridsHeight(GetValue<uint32_t>(prim->GetAttr(kNumGridsHeight)));
  }
  if (prim->GetAttr(kNumGridsWidth) != nullptr) {
    decbbox_operator->SetNumGridsWidth(GetValue<uint32_t>(prim->GetAttr(kNumGridsWidth)));
  }
  if (prim->GetAttr(kDecBBoxParam) != nullptr) {
    auto param_ptr = GetValue<lite::DetectionOutputParamHolderPtr>(prim->GetAttr(kDecBBoxParam));
    if (param_ptr == nullptr) {
      MS_LOG(ERROR) << "decbbox param holder ptr is nullptr.";
      return RET_ERROR;
    }
    decbbox_operator->SetDecBboxParam(param_ptr->GetDetectionOutputParam());
  }
  base_operators->push_back(std::move(decbbox_operator));
  return RET_OK;
}
REG_MAPPER(DecBBox, DecBBoxMapper)
}  // namespace dpico
}  // namespace mindspore
