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
#include "plugin/device/gpu/optimizer/replace_addn_fusion.h"
#include <vector>
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"
#include "backend/common/optimizer/helper.h"
#include "include/backend/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
namespace {
constexpr auto A = "A";
constexpr auto B = "B";
constexpr auto m_addn = "m_addn";
constexpr auto r_add = "r_add";
}  // namespace
bool ReplaceAddNFusion::CheckMatchedDAG(const PatternMap &, const FuncGraphPtr &graph, const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto num_input = common::AnfAlgo::GetNodeAttr<int64_t>(node, "n");
  if (LongToSize(num_input) != kAddNInputNum) {
    return false;
  }
  return true;
}

AnfNodePtr BuildAdd(const PatternMap &m, const AnfNodePtr &default_node) {
  MS_EXCEPTION_IF_NULL(default_node);
  const auto &from_node = m.Get(m_addn);
  MS_EXCEPTION_IF_NULL(from_node);
  std::vector<TypeId> outputs_type;
  std::vector<BaseShapePtr> outputs_shape;
  outputs_type.push_back(common::AnfAlgo::GetOutputInferDataType(m.Get(A), 0));
  outputs_shape.push_back(AnfAlgo::GetOutputDetailShape(m.Get(A), 0));
  common::AnfAlgo::SetOutputTypeAndDetailShape(outputs_type, outputs_shape, default_node.get());
  AnfAlgo::SetSelectKernelBuildInfo(AnfAlgo::GetSelectKernelBuildInfo(from_node), default_node.get());

  if (common::AnfAlgo::HasNodeAttr(kAttrInputIsDynamicShape, from_node->cast<CNodePtr>())) {
    common::AnfAlgo::CopyNodeAttr(kAttrInputIsDynamicShape, from_node, default_node);
  }
  if (common::AnfAlgo::HasNodeAttr(kAttrOutputIsDynamicShape, from_node->cast<CNodePtr>())) {
    common::AnfAlgo::CopyNodeAttr(kAttrOutputIsDynamicShape, from_node, default_node);
  }
  return default_node;
}

void ReplaceAddNFusion::DefineSrcPattern(SrcPattern *src_pattern) {
  (*src_pattern).AddVar(A).AddVar(B).AddCNode(m_addn, {prim::kPrimAddN, A, B});
}

void ReplaceAddNFusion::DefineDstPattern(DstPattern *dst_pattern) {
  (*dst_pattern).AddCNode(r_add, {prim::kPrimAdd, A, B}, BuildAdd);
}
}  // namespace opt
}  // namespace mindspore
