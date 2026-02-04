/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "plugin/ascend/graph_optimizer/pass/heterogeneous/move_to_utils.h"

#include <memory>
#include <vector>

#include "primitive/framework_ops.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"

namespace mindspore {
namespace opt {
constexpr size_t kFirstTensorIdx = 1;

CNodePtr MoveToUtils::InsertMoveTo(const KernelGraphPtr &kernel_graph, const MoveToInfo &info) {
  /*
    data previous node o   o to ValueNode                         to ValueNode   control previous node
                        \ /                                                  o   o
                         o MoveTo                                             \ /
                          \                             data previous node o   o Depend
                           o data following node                            \ /
                                                           original input o  o MoveTo
                                                                          \ / \
                                                                    Depend o   o data following node
                                                                          /
                                                                         o control following node
  */

  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(info.to_);
  MS_EXCEPTION_IF_NULL(info.data_previous_node_);
  MS_EXCEPTION_IF_NULL(info.data_following_node_);

  // Make ValueNode for MoveTo:
  const auto &value = MakeValue(info.to_);
  MS_EXCEPTION_IF_NULL(value);
  AnfNodePtr to_input = kernel_graph->NewValueNode(value);
  to_input->set_scope(info.data_following_node_->scope());
  MS_EXCEPTION_IF_NULL(to_input);

  // Create Depend node before MoveTo to control the execution order of MoveTo when the control previous node is
  // different from the data previous node.
  if (info.control_previous_node_ != nullptr && info.control_previous_node_ != info.data_previous_node_) {
    const std::vector<AnfNodePtr> move_to_depend_input = {
      NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())), to_input, info.control_previous_node_};
    auto move_to_depend_node = kernel_graph->NewCNode(move_to_depend_input);
    MS_EXCEPTION_IF_NULL(move_to_depend_node);
    move_to_depend_node->AddAttr("move_to_pre_control", MakeValue(true));
    move_to_depend_node->set_scope(to_input->scope());
    move_to_depend_node->set_abstract(to_input->abstract());
    to_input = move_to_depend_node;
  }

  // Create MoveTo node
  const auto &blocking_value = MakeValue(false);
  auto blocking_value_node = kernel_graph->NewValueNode(blocking_value);
  blocking_value_node->set_abstract(blocking_value->ToAbstract());
  const std::vector<AnfNodePtr> move_to_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimMoveTo->name())),
                                                  info.data_previous_node_, to_input, blocking_value_node};
  auto move_to_node = kernel_graph->NewCNode(move_to_inputs);
  MS_EXCEPTION_IF_NULL(move_to_node);
  move_to_node->set_scope(info.data_following_node_->scope());
  move_to_node->set_abstract(info.data_previous_node_->abstract());

  // Set MoveTo as input of data following node.
  auto manager = kernel_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->SetEdge(info.data_following_node_, info.input_index_, move_to_node);

  // Create Depend node after MoveTo to control the execution order of MoveTo when the control following node is
  // different from the data following node.
  if (info.control_following_node_ != nullptr && info.control_following_node_ != info.data_following_node_) {
    if (InsertDependNode(kernel_graph, move_to_node, info.control_following_node_) == nullptr) {
      MS_LOG(EXCEPTION) << "Insert depend for MoveTo " << move_to_node->DebugString() << " failed.";
    }
  }

  return move_to_node;
}

CNodePtr MoveToUtils::InsertMoveAssign(const KernelGraphPtr &kernel_graph, const MoveAssignInfo &info) {
  /*
           value  o   o  control previous node
                   \ /
       parameter o  o  o to ValueNode
                  \ | /
   origin input o   o MoveAssign
                \ /
                 o Depend
                 |
                 o control following node
  */
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(info.to_);
  MS_EXCEPTION_IF_NULL(info.parameter_);
  MS_EXCEPTION_IF_NULL(info.value_);
  MS_EXCEPTION_IF_NULL(info.control_following_node_);

  auto value_input = info.value_;
  // Create Depend node before MoveAssign so that MoveTo will be executed after control previous node.
  if (info.control_previous_node_ != nullptr && info.control_previous_node_ != info.value_) {
    std::vector<AnfNodePtr> input_depend_input = {NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())),
                                                  info.value_, info.control_previous_node_};
    auto input_depend_node = kernel_graph->NewCNode(input_depend_input);
    MS_EXCEPTION_IF_NULL(input_depend_node);
    input_depend_node->AddAttr("move_assign_pre_control", MakeValue(true));
    input_depend_node->set_scope(info.control_previous_node_->scope());
    input_depend_node->set_abstract(info.value_->abstract());
    value_input = input_depend_node;
  }

  // Make ValueNode for MoveTo:
  auto value = MakeValue(info.to_);
  MS_EXCEPTION_IF_NULL(value);
  auto to_value_node = kernel_graph->NewValueNode(value);
  MS_EXCEPTION_IF_NULL(to_value_node);

  // Create MoveAssign node
  std::vector<AnfNodePtr> move_assign_inputs = {
    NewValueNode(std::make_shared<Primitive>(prim::kPrimMoveAssign->name())), info.parameter_, value_input,
    to_value_node};
  auto move_assign_node = kernel_graph->NewCNode(move_assign_inputs);
  MS_EXCEPTION_IF_NULL(move_assign_node);
  move_assign_node->set_scope(info.value_->scope());

  // Create Depend node after MoveAssign so that MoveAssign will be executed before the next execution node after user
  // node.
  if (InsertDependNode(kernel_graph, move_assign_node, info.control_following_node_) == nullptr) {
    MS_LOG(EXCEPTION) << "Insert depend for MoveAssign " << move_assign_node->DebugString() << " failed.";
  }
  return move_assign_node;
}

CNodePtr MoveToUtils::InsertDependNode(const KernelGraphPtr &kernel_graph, const CNodePtr &pre_node,
                                       const CNodePtr &post_node) {
  if (post_node->inputs().size() <= kFirstTensorIdx) {
    MS_LOG(WARNING) << "Post node " << post_node->fullname_with_scope() << " has less than one input";
    return nullptr;
  }
  const auto &origin_input = post_node->input(kFirstTensorIdx);
  MS_EXCEPTION_IF_NULL(origin_input);
  const std::vector<AnfNodePtr> depend_input = {NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())),
                                                origin_input, pre_node};
  auto depend_node = kernel_graph->NewCNode(depend_input);
  MS_EXCEPTION_IF_NULL(depend_node);
  depend_node->AddAttr("move_post_control", MakeValue(true));
  depend_node->set_scope(post_node->scope());
  depend_node->set_abstract(origin_input->abstract());
  auto manager = kernel_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->SetEdge(post_node, kFirstTensorIdx, depend_node);
  return depend_node;
}
}  // namespace opt
}  // namespace mindspore
