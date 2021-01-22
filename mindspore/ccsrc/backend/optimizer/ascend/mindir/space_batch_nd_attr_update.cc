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

#include "backend/optimizer/ascend/mindir/space_batch_nd_attr_update.h"
#include <memory>
#include <vector>
#include "backend/optimizer/common/helper.h"
#include "runtime/device/kernel_info.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "base/core_ops.h"
#include "utils/utils.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kBlockShapeDimNum = 2;
constexpr auto kAttrBlockShape = "block_shape";
constexpr auto kAttrPaddings = "paddings";
constexpr auto kAttrCrops = "crops";
}  // namespace

const BaseRef SpaceToBatchNDAttrUpdate::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  VectorRef pattern({prim::kPrimSpaceToBatchND, X});
  return pattern;
}

const AnfNodePtr SpaceToBatchNDAttrUpdate::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                   const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  auto block_shape = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(node, kAttrBlockShape);
  if (block_shape.size() == kBlockShapeDimNum) {
    block_shape.insert(block_shape.begin(), 1);
    AnfAlgo::SetNodeAttr(kAttrBlockShape, MakeValue(block_shape), node);
  }
  auto paddings = AnfAlgo::GetNodeAttr<std::vector<std::vector<int64_t>>>(node, kAttrPaddings);
  if (paddings.size() == kBlockShapeDimNum) {
    paddings.emplace(paddings.begin(), std::vector<int64_t>{0, 0});
    AnfAlgo::SetNodeAttr(kAttrPaddings, MakeValue(paddings), node);
  }
  return node;
}

const BaseRef BatchToSpaceNDAttrUpdate::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  VectorRef pattern({prim::kPrimBatchToSpaceND, X});
  return pattern;
}

const AnfNodePtr BatchToSpaceNDAttrUpdate::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                   const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  auto block_shape = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(node, kAttrBlockShape);
  if (block_shape.size() == kBlockShapeDimNum) {
    block_shape.insert(block_shape.begin(), 1);
    AnfAlgo::SetNodeAttr(kAttrBlockShape, MakeValue(block_shape), node);
  }
  auto crops = AnfAlgo::GetNodeAttr<std::vector<std::vector<int64_t>>>(node, kAttrCrops);
  if (crops.size() == kBlockShapeDimNum) {
    crops.emplace(crops.begin(), std::vector<int64_t>{0, 0});
    AnfAlgo::SetNodeAttr(kAttrCrops, MakeValue(crops), node);
  }
  return node;
}
}  // namespace opt
}  // namespace mindspore
