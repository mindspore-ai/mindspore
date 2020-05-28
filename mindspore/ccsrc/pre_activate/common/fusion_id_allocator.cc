/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "pre_activate/common/fusion_id_allocator.h"
#include "session/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
FusionIdAllocator::FusionIdAllocator() { fusion_id = 0; }

FusionIdAllocator::~FusionIdAllocator() {}

void FusionIdAllocator::Init() { fusion_id = 0; }

int32_t FusionIdAllocator::AllocateFusionId() {
  fusion_id++;
  return fusion_id;
}

bool FusionIdAllocator::HasFusionIdAttr(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  return AnfAlgo::HasNodeAttr(kAttrFusionId, cnode);
}

int32_t FusionIdAllocator::GetFusionId(const AnfNodePtr &node) {
  if (HasFusionIdAttr(node)) {
    return AnfAlgo::GetNodeAttr<int32_t>(node, kAttrFusionId);
  }
  return -1;
}

void FusionIdAllocator::SetFusionId(const AnfNodePtr &node, int32_t id) {
  ValuePtr fusion_id_v = MakeValue(id);
  AnfAlgo::SetNodeAttr(kAttrFusionId, fusion_id_v, node);
}
}  // namespace opt
}  // namespace mindspore
