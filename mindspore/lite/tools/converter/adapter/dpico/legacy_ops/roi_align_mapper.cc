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

#include "legacy_ops/roi_align_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include <string>
#include "common/op_attr.h"
#include "op/roi_align_operator.h"

namespace mindspore {
namespace dpico {
STATUS RoiAlignMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                           const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }

  auto roi_align_operator = std::make_unique<mapper::RoiAlignOperator>();
  if (roi_align_operator == nullptr) {
    MS_LOG(ERROR) << "roi_align_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, roi_align_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  roi_align_operator->SetOpType(mapper::OpType::ROIALIGN);
  if (prim->GetAttr(ops::kMode) != nullptr) {
    auto pool_mode = api::GetValue<std::string>(prim->GetAttr(ops::kMode));
    if (pool_mode == "avg") {
      roi_align_operator->SetPoolMode(mapper::RoiAlignPoolMode::ROI_ALIGN_AVG);
    } else if (pool_mode == "max") {
      roi_align_operator->SetPoolMode(mapper::RoiAlignPoolMode::ROI_ALIGN_MAX);
    } else {
      MS_LOG(ERROR) << "unsupported pool mode:" << pool_mode << " by dpico. " << cnode->fullname_with_scope();
      return RET_ERROR;
    }
  }
  if (prim->GetAttr(dpico::kOutputHeight) != nullptr) {
    roi_align_operator->SetPooledHeight(
      static_cast<uint32_t>(api::GetValue<int64_t>(prim->GetAttr(dpico::kOutputHeight))));
  }
  if (prim->GetAttr(dpico::kOutputWidth) != nullptr) {
    roi_align_operator->SetPooledWidth(
      static_cast<uint32_t>(api::GetValue<int64_t>(prim->GetAttr(dpico::kOutputWidth))));
  }
  if (prim->GetAttr(dpico::kSamplingRatio) != nullptr) {
    roi_align_operator->SetSamplingRatio(
      static_cast<uint32_t>(api::GetValue<int64_t>(prim->GetAttr(dpico::kSamplingRatio))));
  }
  if (prim->GetAttr(dpico::kSpatialScale) != nullptr) {
    roi_align_operator->SetSpatialScale(api::GetValue<float>(prim->GetAttr(dpico::kSpatialScale)));
  }
  base_operators->push_back(std::move(roi_align_operator));
  return RET_OK;
}
REG_MAPPER(RoiAlign, RoiAlignMapper)
}  // namespace dpico
}  // namespace mindspore
