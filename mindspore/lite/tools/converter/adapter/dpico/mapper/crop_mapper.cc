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

#include "mapper/crop_mapper.h"
#include <memory>
#include <utility>
#include <algorithm>
#include <vector>
#include "ops/crop.h"
#include "op/crop_operator.h"

namespace mindspore {
namespace dpico {
STATUS CropMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                       const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }
  auto crop_prim = api::utils::cast<api::SharedPtr<ops::Crop>>(prim);
  MS_ASSERT(crop_prim != nullptr);

  auto crop_operator = std::make_unique<mapper::CropOperator>();
  if (crop_operator == nullptr) {
    MS_LOG(ERROR) << "crop_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, crop_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  crop_operator->SetOpType(mapper::OpType::CROP);
  if (prim->GetAttr(ops::kAxis) != nullptr) {
    crop_operator->SetAxis(static_cast<int32_t>(crop_prim->get_axis()));
  }
  if (prim->GetAttr(ops::kOffsets) != nullptr) {
    auto offsets = crop_prim->get_offsets();
    std::vector<uint32_t> offset_vec{};
    (void)std::transform(offsets.begin(), offsets.end(), std::back_inserter(offset_vec),
                         [](const int64_t offset) { return static_cast<int32_t>(offset); });
    crop_operator->SetOrigOffsetVec(offset_vec);
  }

  base_operators->push_back(std::move(crop_operator));
  return RET_OK;
}
REG_MAPPER(Crop, CropMapper)
}  // namespace dpico
}  // namespace mindspore
