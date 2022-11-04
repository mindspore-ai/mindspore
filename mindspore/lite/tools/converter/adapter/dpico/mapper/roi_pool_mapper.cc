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

#include "mapper/roi_pool_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include "ops/roi_pooling.h"
#include "op/roi_pool_operator.h"

namespace mindspore {
namespace dpico {
STATUS RoiPoolMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                          const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }
  auto roi_pool_prim = api::utils::cast<api::SharedPtr<ops::ROIPooling>>(prim);
  MS_ASSERT(roi_pool_prim != nullptr);

  auto roi_pool_operator = std::make_unique<mapper::RoiPoolOperator>();
  if (roi_pool_operator == nullptr) {
    MS_LOG(ERROR) << "roi_pool_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, roi_pool_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  roi_pool_operator->SetOpType(mapper::OpType::ROIPOOLING);
  if (roi_pool_prim->GetAttr(ops::kPooledH) != nullptr) {
    roi_pool_operator->SetRoiPooledHeight(static_cast<uint32_t>(roi_pool_prim->get_pooled_h()));
  }
  if (roi_pool_prim->GetAttr(ops::kPooledW) != nullptr) {
    roi_pool_operator->SetRoiPooledWidth(static_cast<uint32_t>(roi_pool_prim->get_pooled_w()));
  }
  if (roi_pool_prim->GetAttr(ops::kScale) != nullptr) {
    roi_pool_operator->SetRoiScale(roi_pool_prim->get_scale());
  }

  base_operators->push_back(std::move(roi_pool_operator));
  return RET_OK;
}
REG_MAPPER(ROIPooling, RoiPoolMapper)
}  // namespace dpico
}  // namespace mindspore
