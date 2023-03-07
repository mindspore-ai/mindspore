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

#include "plugin/device/ascend/optimizer/ge/convert_attr_to_input.h"

#include <memory>
#include <map>
#include <utility>
#include <string>

#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
namespace {
// old version node name | new version node name | attr_name
static const std::map<std::string, std::pair<std::string, string>> kNeedTransMap = {
  {kResizeNearestNeighborOpName, {kResizeNearestNeighborV2OpName, "size"}},
  {kResizeNearestNeighborGradOpName, {kResizeNearestNeighborV2GradOpName, ""}}};

bool NeedConvert(const BaseRef &ref) {
  if (utils::isa<AnfNodePtr>(ref)) {
    AnfNodePtr node = utils::cast<AnfNodePtr>(ref);
    MS_EXCEPTION_IF_NULL(node);
    if (!IsValueNode<Primitive>(node)) {
      return false;
    }
    auto prim = GetValuePtr<Primitive>(node);
    MS_EXCEPTION_IF_NULL(prim);
    if (kNeedTransMap.find(prim->name()) != kNeedTransMap.cend()) {
      return true;
    }
  }
  return false;
}
}  // namespace

const BaseRef ConvertAttrToInput::DefinePattern() const {
  VarPtr convert = std::make_shared<CondVar>(NeedConvert);
  VarPtr inputs = std::make_shared<SeqVar>();
  return VectorRef({convert, inputs});
}

const AnfNodePtr ConvertAttrToInput::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                             const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(DEBUG) << "Begin to convert attr to input for node: " << node->DebugString();

  const auto &origin_prim = common::AnfAlgo::GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(origin_prim);
  const auto &item = kNeedTransMap.at(origin_prim->name());
  std::string new_prim_name = item.first;
  std::string attr_name = item.second;
  const auto &origin_attrs = origin_prim->attrs();
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  // Create new primitive and inherit the origin attributes.
  auto new_prim = std::make_shared<Primitive>(new_prim_name);
  MS_EXCEPTION_IF_NULL(new_prim);
  (void)new_prim->SetAttrs(origin_attrs);

  if (origin_attrs.count(attr_name) == 0) {
    MS_LOG(DEBUG) << "Origin primitive: " << origin_prim->name() << "has no attr : " << attr_name
                  << ", so only update to new primitive: " << new_prim_name;
  } else {
    // Convert the specific attr to input and erase the specific attr.
    auto attr_value = new_prim->GetAttr(attr_name);
    MS_EXCEPTION_IF_NULL(attr_value);
    auto new_value_node = std::make_shared<ValueNode>(attr_value);
    MS_EXCEPTION_IF_NULL(new_value_node);
    new_value_node->set_abstract(attr_value->ToAbstract());
    manager->AddEdge(cnode, new_value_node);
    new_prim->EraseAttr(attr_name);
  }
  // Update the primitive of cnode
  manager->SetEdge(cnode, kIndex0, std::make_shared<ValueNode>(new_prim));
  MS_LOG(DEBUG) << "End, new node: " << node->DebugString();
  return node;
}
}  // namespace opt
}  // namespace mindspore
