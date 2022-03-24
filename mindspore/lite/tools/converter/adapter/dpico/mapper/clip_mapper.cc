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

#include "mapper/clip_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include "ops/clip.h"
#include "op/clip_operator.h"

namespace mindspore {
namespace dpico {
STATUS ClipMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                       const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }
  auto clip_prim = api::utils::cast<api::SharedPtr<ops::Clip>>(prim);
  MS_ASSERT(clip_prim != nullptr);

  auto clip_operator = std::make_unique<mapper::ClipOperator>();
  if (clip_operator == nullptr) {
    MS_LOG(ERROR) << "clip_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, clip_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  if (prim->GetAttr(ops::kMax) != nullptr) {
    clip_operator->SetClipCeil(clip_prim->get_max());
  }
  if (prim->GetAttr(ops::kMin) != nullptr) {
    clip_operator->SetClipFloor(clip_prim->get_min());
  }
  if (PushOfflineArgs(cnode, clip_operator.get(), 1) != RET_OK) {
    MS_LOG(ERROR) << "push offline args failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  base_operators->push_back(std::move(clip_operator));
  return RET_OK;
}
REG_MAPPER(Clip, ClipMapper)
}  // namespace dpico
}  // namespace mindspore
