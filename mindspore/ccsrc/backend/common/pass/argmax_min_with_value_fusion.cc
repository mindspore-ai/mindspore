/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "backend/common/pass/argmax_min_with_value_fusion.h"

#include <vector>
#include <memory>

#include "mindspore/core/ops/array_ops.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
namespace {
const AnfNodePtr SetAttrDimension(const AnfNodePtr &node) {
  constexpr auto kAxis = "axis";
  constexpr auto kDimension = "dimension";
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  auto axis = common::AnfAlgo::GetNodeAttr<int64_t>(cnode, kAxis);
  common::AnfAlgo::SetNodeAttr(kDimension, MakeValue(axis), cnode);

  return cnode;
}
}  // namespace

const BaseRef ArgMaxWithValueFusion::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimArgMaxWithValue, Xs});
}

const AnfNodePtr ArgMaxWithValueFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  return SetAttrDimension(node);
}

const BaseRef ArgMinWithValueFusion::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimArgMinWithValue, Xs});
}

const AnfNodePtr ArgMinWithValueFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  return SetAttrDimension(node);
}
}  // namespace opt
}  // namespace mindspore
