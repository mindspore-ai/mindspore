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

#include "backend/common/optimizer/dynamic_shape/link_custom_op.h"

#include <memory>
#include <vector>
#include "utils/anf_utils.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/optimizer/helper.h"
#include "backend/common/optimizer/dynamic_shape/dynamic_shape_helper.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace opt::dynamic_shape {
constexpr size_t kTupleFirstItemIndex = 0;
constexpr size_t kFirstDataInputIndex = 1;

void LinkCustomOp::InsertDepend(const FuncGraphPtr &g, const AnfNodePtr &prev, const AnfNodePtr &next,
                                AnfNodePtrList *depend_nodes) {
  MS_EXCEPTION_IF_NULL(g);
  MS_EXCEPTION_IF_NULL(prev);
  MS_EXCEPTION_IF_NULL(next);
  MS_EXCEPTION_IF_NULL(depend_nodes);

  DependPair cur_pair = std::make_pair(prev, next);
  if (added_set_.count(cur_pair) > 0) {
    return;
  }

  // add depend from prev to next
  auto depend_node = g->NewCNode(
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())), next, prev});
  MS_EXCEPTION_IF_NULL(depend_node);
  depend_nodes->push_back(depend_node);
  (void)added_set_.insert(cur_pair);
}

bool LinkCustomOp::LinkInternalOp(const FuncGraphPtr &g, const AnfNodePtr &node, AnfNodePtrList *depend_nodes) {
  MS_EXCEPTION_IF_NULL(g);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(depend_nodes);

  bool changed = false;
  auto custom_nodes = CustomActorNodeManager::Instance().GetCustomActorNodes(node);
  if (custom_nodes.infer_node != nullptr && custom_nodes.init_node != nullptr) {
    InsertDepend(g, custom_nodes.infer_node, custom_nodes.init_node, depend_nodes);  // link infer => init
    InsertDepend(g, custom_nodes.init_node, node, depend_nodes);                     // link init => launch
    changed = true;
  }

  return changed;
}

bool LinkCustomOp::LinkInputOp(const FuncGraphPtr &g, const CNodePtr &cnode, AnfNodePtrList *depend_nodes) {
  MS_EXCEPTION_IF_NULL(g);
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(depend_nodes);
  bool changed = false;
  auto custom_nodes = CustomActorNodeManager::Instance().GetCustomActorNodes(cnode);
  if (custom_nodes.infer_node == nullptr) {
    return changed;
  }
  size_t input_num = common::AnfAlgo::GetInputNum(cnode);
  for (size_t i = 0; i < input_num; ++i) {
    auto prev = common::AnfAlgo::GetPrevNodeOutput(cnode, i);
    const auto &prev_node = prev.first;
    if (prev_node == nullptr) {
      continue;
    }
    if (!CustomActorNodeManager::Instance().IsRegistered(prev_node)) {
      // when its subgraph and its input is a dynamic_shape_parameter, link prev_parameter => curr.infer
      if (prev_node->isa<Parameter>()) {
        auto prev_parameter = prev_node->cast<ParameterPtr>();
        MS_EXCEPTION_IF_NULL(prev_parameter);
        if (prev_parameter->has_dynamic_shape()) {
          InsertDepend(g, prev_node, custom_nodes.infer_node, depend_nodes);
          changed = true;
        }
      }
      continue;
    }
    auto prev_custom_nodes = CustomActorNodeManager::Instance().GetCustomActorNodes(prev_node);
    if (prev_custom_nodes.infer_node != nullptr) {
      // link prev.infer => curr.infer
      InsertDepend(g, prev_custom_nodes.infer_node, custom_nodes.infer_node, depend_nodes);
      changed = true;
    }

    // if the shape of prev_node is set after launch, need to link prev_node's launch to cur_node's infer
    if (AnfAlgo::IsNeedUpdateShapeAndTypeAfterLaunch(prev_node)) {
      // link prev.launch => curr.infer
      InsertDepend(g, prev_node, custom_nodes.infer_node, depend_nodes);
      changed = true;
    }
  }
  return changed;
}

bool LinkCustomOp::LinkDependSync(const FuncGraphPtr &g, const CNodePtr &cnode, AnfNodePtrList *depend_nodes) {
  MS_EXCEPTION_IF_NULL(g);
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(depend_nodes);
  bool changed = false;
  auto custom_nodes = CustomActorNodeManager::Instance().GetCustomActorNodes(cnode);
  if (custom_nodes.infer_node == nullptr) {
    return changed;
  }

  auto dynamic_shape_depends = abstract::GetValueDependArgIndices(cnode);
  if (dynamic_shape_depends.empty()) {
    return changed;
  }

  for (auto depend_index : dynamic_shape_depends) {
    auto prev = common::AnfAlgo::GetPrevNodeOutput(cnode, LongToSize(depend_index));
    const auto &prev_node = prev.first;
    if (prev_node == nullptr || !CustomActorNodeManager::Instance().IsRegistered(prev_node)) {
      continue;
    }

    // If previous node is dynamic, so it was already link.
    if (AnfAlgo::IsNeedUpdateShapeAndTypeAfterLaunch(prev_node)) {
      continue;
    }

    // Link prev_node.launch => cur_node.infer.
    InsertDepend(g, prev_node, custom_nodes.infer_node, depend_nodes);
    changed = true;
  }
  return changed;
}

/**
 * @brief Attach Custom's Depend nodes with additional MakeTuple and TupleGetItem before graph return.
 *
 *          %0 = A
 *          return %0
 *          ---->
 *          %0 = A
 *          %1 = MakeTuple(%0, %depend0, %depend1...)
 *          %2 = TupleGetItem(%1, 0)
 *          return %2
 *
 * @param g Graph.
 * @param depend_nodes Custom's Depend nodes.
 */
void LinkCustomOp::AttachDependNodes(const FuncGraphPtr &g, const AnfNodePtrList &depend_nodes) const {
  if (depend_nodes.empty()) {
    return;
  }

  MS_EXCEPTION_IF_NULL(g);
  auto return_node = g->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  auto output_node = return_node->input(kFirstDataInputIndex);
  MS_EXCEPTION_IF_NULL(output_node);

  // New MakeTuple node
  auto mk_inputs = AnfNodePtrList{NewValueNode(std::make_shared<Primitive>(prim::kPrimMakeTuple->name())), output_node};
  (void)mk_inputs.insert(mk_inputs.cend(), depend_nodes.cbegin(), depend_nodes.cend());
  auto make_tuple_node = g->NewCNode(mk_inputs);

  // Get first element item form that maketuple and return.
  auto get_1st_item = g->NewCNode(AnfNodePtrList{NewValueNode(std::make_shared<Primitive>(prim::kTupleGetItem)),
                                                 make_tuple_node, NewValueNode(SizeToLong(kTupleFirstItemIndex))});
  // The getitem node always obtains the first input of the maketuple, which is the output in the original graph,
  // so set the abstract of the output to the getitem node.
  get_1st_item->set_abstract(output_node->abstract());
  // Attach back.
  return_node->set_input(kFirstDataInputIndex, get_1st_item);
}

bool LinkCustomOp::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  bool changed = false;
  AnfNodePtrList depend_nodes;
  auto node_list = TopoSort(func_graph->get_return());
  added_set_.clear();
  for (const auto &node : node_list) {
    CNodePtr cnode = node->cast<CNodePtr>();
    if (cnode == nullptr || !CustomActorNodeManager::Instance().IsRegistered(cnode)) {
      continue;
    }

    changed = LinkInternalOp(func_graph, cnode, &depend_nodes) || changed;
    changed = LinkInputOp(func_graph, cnode, &depend_nodes) || changed;
    changed = LinkDependSync(func_graph, cnode, &depend_nodes) || changed;
  }

  CustomActorNodeManager::Instance().Reset();

  if (changed) {
    AttachDependNodes(func_graph, depend_nodes);

    // Rebuild graph's edge.
    auto mng = func_graph->manager();
    if (mng == nullptr) {
      mng = Manage(func_graph, true);
      func_graph->set_manager(mng);
    }
    mng->RemoveRoots();
    mng->KeepRoots({func_graph});
  }

  return changed;
}
}  // namespace opt::dynamic_shape
}  // namespace mindspore
