/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/optimizer/mindir/space_batch_nd_attr_update.h"
#include <memory>
#include <vector>
#include "include/backend/optimizer/helper.h"
#include "include/backend/kernel_info.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "mindspore/core/ops/core_ops.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kBlockShapeDimNum = 2;
constexpr auto kAttrBlockShape = "block_shape";
constexpr auto kAttrPaddings = "paddings";
constexpr auto kAttrCrops = "crops";
constexpr auto kV = "V";
constexpr auto kMSpace = "m_space";
constexpr auto kRSpace = "r_space";
constexpr auto kMBatch = "m_batch";
constexpr auto kRBatch = "r_batch";

AnfNodePtr BuildSpace(const PatternMap &m, const AnfNodePtr &) {
  auto node = m.Get(kMSpace);
  auto block_shape = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(node, kAttrBlockShape);
  if (block_shape.size() == kBlockShapeDimNum) {
    (void)block_shape.insert(block_shape.cbegin(), 1);
    common::AnfAlgo::SetNodeAttr(kAttrBlockShape, MakeValue(block_shape), node);
  }
  auto paddings = common::AnfAlgo::GetNodeAttr<std::vector<std::vector<int64_t>>>(node, kAttrPaddings);
  if (paddings.size() == kBlockShapeDimNum) {
    paddings.emplace(paddings.begin(), std::vector<int64_t>{0, 0});
    common::AnfAlgo::SetNodeAttr(kAttrPaddings, MakeValue(paddings), node);
  }
  return node;
}

AnfNodePtr BuildBatch(const PatternMap &m, const AnfNodePtr &) {
  auto node = m.Get(kMBatch);
  auto block_shape = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(node, kAttrBlockShape);
  if (block_shape.size() == kBlockShapeDimNum) {
    (void)block_shape.insert(block_shape.cbegin(), 1);
    common::AnfAlgo::SetNodeAttr(kAttrBlockShape, MakeValue(block_shape), node);
  }
  auto crops = common::AnfAlgo::GetNodeAttr<std::vector<std::vector<int64_t>>>(node, kAttrCrops);
  if (crops.size() == kBlockShapeDimNum) {
    (void)crops.emplace(crops.begin(), std::vector<int64_t>{0, 0});
    common::AnfAlgo::SetNodeAttr(kAttrCrops, MakeValue(crops), node);
  }
  return node;
}
}  // namespace

bool SpaceToBatchNDAttrUpdate::CheckMatchedDAG(const PatternMap &, const FuncGraphPtr &, const AnfNodePtr &) const {
  return true;
}

void SpaceToBatchNDAttrUpdate::DefineSrcPattern(SrcPattern *src_pattern) {
  (*src_pattern).AddVar(kV).AddCNode(kMSpace, {prim::kPrimSpaceToBatchND, kV});
}

void SpaceToBatchNDAttrUpdate::DefineDstPattern(DstPattern *dst_pattern) {
  (*dst_pattern).AddCNode(kRSpace, {prim::kPrimSpaceToBatchND, kV}, BuildSpace);
}

bool BatchToSpaceNDAttrUpdate::CheckMatchedDAG(const PatternMap &, const FuncGraphPtr &, const AnfNodePtr &) const {
  return true;
}

void BatchToSpaceNDAttrUpdate::DefineSrcPattern(SrcPattern *src_pattern) {
  (*src_pattern).AddVar(kV).AddCNode(kMBatch, {prim::kPrimBatchToSpaceND, kV});
}

void BatchToSpaceNDAttrUpdate::DefineDstPattern(DstPattern *dst_pattern) {
  (*dst_pattern).AddCNode(kRBatch, {prim::kPrimBatchToSpaceND, kV}, BuildBatch);
}
}  // namespace opt
}  // namespace mindspore
