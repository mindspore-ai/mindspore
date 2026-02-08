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

#include "include/backend/common/expander/fallback/expander_fallback.h"
#include <algorithm>
#include <queue>
#include <map>
#include <memory>
#include "base/base.h"
#include "primitive/op_name.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_i.h"
#include "utils/ms_utils.h"
#include "utils/anf_utils.h"
#include "utils/compile_config.h"
#include "utils/ms_context.h"
#include "include/utils/anfalgo.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "backend/common/expander/fallback/fallback_irbuilder.h"

namespace mindspore {
namespace expander {
const std::set<PrimitivePtr> AUGASSIGN_INPLACE_LIST = {prim::kPrimInplaceAddExt,
                                                       prim::kPrimInplaceAddsExt,
                                                       prim::kPrimInplaceSubExt,
                                                       prim::kPrimInplaceSubScalar,
                                                       prim::kPrimInplaceMul,
                                                       prim::kPrimInplaceMuls,
                                                       prim::kPrimInplaceDiv,
                                                       prim::kPrimInplaceDivs,
                                                       prim::kPrimInplaceFloorDivide,
                                                       prim::kPrimInplaceFloorDivides,
                                                       prim::kPrimInplaceRemainderTensorTensor,
                                                       prim::kPrimInplaceRemainderTensorScalar};

bool IsInAugassignInplaceList(const CNodePtr &cnode) {
  for (auto &prim : AUGASSIGN_INPLACE_LIST) {
    if (IsPrimitiveCNode(cnode, prim)) {
      return true;
    }
  }
  return false;
}

bool Check(const AnfNodePtr &node) {
  if (common::GetEnv("MS_DEV_EXPANDER_FALLBACK") == "off") {
    return false;
  }
  if (!node->isa<CNode>()) {
    return false;
  }
  // Operators with 'batch_rank' attribute, which only appears in the vmap scenario, are not supported currently.
  if (common::AnfAlgo::HasNodeAttr(ops::kBatchRank, node->cast<CNodePtr>())) {
    return false;
  }
  return true;
}

void DumpGraph(const CNodePtr &ori_node, const CNodePtr &new_output) {
  auto expand_fg = std::make_shared<FuncGraph>();
  std::map<AnfNodePtr, AnfNodePtr> node_map;
  CNodePtrList newcnodes;
  for (size_t i = 1; i < ori_node->size(); i++) {
    auto p = expand_fg->add_parameter();
    p->set_abstract(ori_node->input(i)->abstract());
    node_map[ori_node->input(i)] = p;
  }
  std::queue<CNodePtr> que;
  que.push(new_output);
  while (!que.empty()) {
    auto node = que.front();
    que.pop();
    if (node_map.count(node) > 0) {
      continue;
    }
    auto new_node = expand_fg->NewCNode(node->inputs());
    new_node->CloneCNodeInfo(node);
    new_node->set_fullname_with_scope(node->fullname_with_scope());
    newcnodes.push_back(new_node);
    node_map[node] = new_node;
    for (size_t i = 1; i < node->size(); ++i) {
      const auto &inp = node->input(i);
      if (inp->isa<CNode>() && node_map.count(inp) == 0) {
        que.push(inp->cast<CNodePtr>());
      }
    }
  }
  for (const auto &cnode : newcnodes) {
    for (size_t i = 1; i < cnode->size(); i++) {
      if (node_map.count(cnode->input(i)) != 0) {
        cnode->set_input(i, node_map[cnode->input(i)]);
      }
    }
  }
  expand_fg->set_output(node_map[new_output]);
  DumpIR("verbose_ir_files/expand_" + AnfUtils::GetCNodeName(ori_node) + ".ir", expand_fg, true);
}

AnfNodePtr GetInplaceNextUpdateState(const CNodePtr &cnode, const FuncGraphManagerPtr &manager) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(manager);
  AnfNodePtr inplace_next_updatestate = nullptr;
  for (const auto &node_index : manager->node_users()[cnode]) {
    const auto &used_node = node_index.first;
    MS_EXCEPTION_IF_NULL(used_node);
    if (IsPrimitiveCNode(used_node, prim::kPrimUpdateState)) {
      inplace_next_updatestate = used_node;
      break;
    }
  }
  return inplace_next_updatestate;
}

CNodePtrList InsertAssignForInplaceFallback(const CNodePtr &cnode, const SelectKernelFunc &select_kernel_func) {
  // %0: PrimFunc_Inplace(x, y, inplace_umonad)
  // %1: UpdateState(inplace_umonad, %0)
  // convert to:
  // %0: PrimFunc_Inplace(x, y, inplace_umonad)
  // %1: UpdateState(inplace_umonad, %0)
  // %2: PrimFunc_Assign(x, %0, %1)
  // %3: UpdateState(%1, %2)
  MS_EXCEPTION_IF_NULL(cnode);
  auto func_graph = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  // fallback_users record users of inplace node(%0), will be replaced by fallback output
  // fallback_users: [%1, %2] after conversion
  CNodePtrList fallback_users;
  // inplace_next_updatestate: %1 before conversion
  auto inplace_next_updatestate_node = GetInplaceNextUpdateState(cnode, manager);
  if (inplace_next_updatestate_node == nullptr || !inplace_next_updatestate_node->isa<CNode>()) {
    MS_LOG(WARNING) << "No valid inplace_next_updatestate_node found for cnode: " << cnode->DebugString()
                    << ", Assign can not be added to keep order which may cause a precision issue.";
    return fallback_users;
  }
  auto inplace_next_updatestate = inplace_next_updatestate_node->cast<CNodePtr>();
  fallback_users.push_back(inplace_next_updatestate);
  // create assign and new updatestate node
  // inplace_next_updatestate_users: [%2, %3] after conversion
  CNodePtrList inplace_next_updatestate_users;
  CNodePtr new_umonad = nullptr;
  auto prim = GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(prim);
  const auto &inplace_indexes = prim->rw_write_input_indexes();
  for (size_t index = 0; index < inplace_indexes.size(); ++index) {
    auto inplace_node = cnode->input(inplace_indexes[index] + 1);
    // cnode will be replaced to fallback output later
    auto assign_cnode =
      func_graph->NewCNode({NewValueNode(prim::kPrimAssign), inplace_node, cnode, inplace_next_updatestate});
    MS_EXCEPTION_IF_NULL(assign_cnode);
    assign_cnode->set_abstract(inplace_node->abstract());
    // create buildInfo
    select_kernel_func(assign_cnode);
    fallback_users.push_back(assign_cnode);

    new_umonad = func_graph->NewCNode({NewValueNode(prim::kPrimUpdateState), inplace_next_updatestate, assign_cnode});
    MS_EXCEPTION_IF_NULL(new_umonad);
    new_umonad->set_abstract(inplace_next_updatestate->abstract());

    inplace_next_updatestate_users.push_back(assign_cnode);
    inplace_next_updatestate_users.push_back(new_umonad);
  }
  MS_EXCEPTION_IF_NULL(new_umonad);

  // SetEdge for original inplace_next_updatestate users to new_umonad
  // insert Assign op
  auto updatestate_users = manager->node_users()[inplace_next_updatestate];
  for (const auto &node_index : updatestate_users) {
    auto used_node = node_index.first;
    MS_EXCEPTION_IF_NULL(used_node);
    auto used_cnode = used_node->cast<CNodePtr>();
    if (used_cnode == nullptr || std::find(inplace_next_updatestate_users.begin(), inplace_next_updatestate_users.end(),
                                           used_cnode) != inplace_next_updatestate_users.end()) {
      continue;
    }
    manager->SetEdge(used_cnode, node_index.second, new_umonad);
  }
  return fallback_users;
}

void InsertLoadForInplaceFallback(const CNodePtr &cnode) {
  // %0: PrimFunc_Inplace(x, y, inplace_umonad)
  // %1: UpdateState(inplace_umonad, %0)
  // convert to:
  // %0: Load(x, inplace_umonad)
  // %1: PrimFunc_Inplace(%0, y, inplace_umonad)
  // %2: UpdateState(inplace_umonad, %1)
  MS_EXCEPTION_IF_NULL(cnode);
  auto func_graph = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  AnfNodePtr inplace_umonad = cnode->inputs().back();
  if (!HasAbstractUMonad(inplace_umonad)) {
    MS_LOG(WARNING) << "Need to be umonad, but got: " << inplace_umonad->DebugString()
                    << ", Load can not be added to keep order which may cause a precision issue.";
    return;
  }
  // insert Load op
  bool is_inserted = false;
  for (size_t i = 1; i < cnode->inputs().size(); i++) {
    auto &input_node = cnode->input(i);
    const auto abs = input_node->abstract();
    if (abs == nullptr || !abs->isa<abstract::AbstractRefTensor>()) {
      continue;
    }
    auto load_cnode = func_graph->NewCNode({NewValueNode(prim::kPrimLoad), input_node, inplace_umonad});
    MS_EXCEPTION_IF_NULL(load_cnode);
    auto ref_abs = dyn_cast<abstract::AbstractRefTensor>(abs);
    auto load_abs = ref_abs->CloneAsTensor();
    load_cnode->set_abstract(load_abs);
    manager->SetEdge(cnode, i, load_cnode);
    is_inserted = true;
  }
  if (!is_inserted) {
    MS_LOG(WARNING) << "In-place inputs need to be RefTensor, cnode: " << cnode->DebugString()
                    << ", Load can not be added to keep order which may cause a precision issue.";
  }
}

bool IbTryExpandCNode(const IRBuilderHandle &handle, const CNodePtr &cnode, const SelectKernelFunc &func) {
  auto mng = cnode->func_graph()->manager();
  MS_EXCEPTION_IF_NULL(mng);

  auto enable_augassign_inplace_fallback = common::GetCompileConfig("JIT_ENABLE_AUGASSIGN_INPLACE_FALLBACK") == "1";
  enable_augassign_inplace_fallback = enable_augassign_inplace_fallback && IsInAugassignInplaceList(cnode);
  CNodePtrList fallback_users;
  if (enable_augassign_inplace_fallback) {
    fallback_users = InsertAssignForInplaceFallback(cnode, func);
    InsertLoadForInplaceFallback(cnode);
  }

  FallbackIRBuilder ib(AnfUtils::GetCNodeName(cnode), cnode->func_graph(), func);
  auto output = ib.Run(cnode, handle);
  if (output == nullptr) {
    MS_LOG(INFO) << "Undo expanding cnode " << cnode->fullname_with_scope();
    return false;
  }
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kAdvanced)) {
    if (output->isa<CNode>()) {
      DumpGraph(cnode, output->cast<CNodePtr>());
    } else {
      MS_LOG(INFO) << "The output is not a CNode, cannot dump graph. original node: " << cnode->fullname_with_scope()
                   << ", output->DebugString: " << output->DebugString();
    }
  }
#endif
  if (!(*cnode->abstract()->Broaden() == *output->abstract())) {
    MS_LOG(WARNING) << "After expanding cnode " << cnode->fullname_with_scope() << ", the new abstract of "
                    << output->fullname_with_scope() << " does not match original cnode's abstract. "
                    << "new: " << output->abstract()->ToString() << ", old: " << cnode->abstract()->ToString();
    if (cnode->abstract()->isa<abstract::AbstractRefTensor>() && output->abstract()->isa<abstract::AbstractTensor>()) {
      const auto &ref_abs = cnode->abstract()->cast<std::shared_ptr<abstract::AbstractRefTensor>>();
      auto new_abs = std::make_shared<abstract::AbstractRefTensor>(
        output->abstract()->cast<abstract::AbstractTensorPtr>(), ref_abs->ref_key_value());
      output->set_abstract(new_abs);
      MS_LOG(WARNING) << "Restore new abstract to AbstractRefTensor new:" << new_abs->ToString();
    }
  }

  if (enable_augassign_inplace_fallback && fallback_users.size() > 1) {
    // %0: Load(x, inplace_umonad)
    // %1: PrimFunc_Inplace(%0, y, inplace_umonad)
    // %2: UpdateState(inplace_umonad, %1)
    // %3: PrimFunc_Assign(x, %1, %2)
    // %4: UpdateState(%2, %3)
    // %5: PrimFunc_Add(%1, y)
    // convert to:
    // %0: Load(x, inplace_umonad)
    // %1: PrimFunc_Add(%0, y)
    // %2: UpdateState(inplace_umonad, %1)
    // %3: PrimFunc_Assign(x, %1, %2)
    // %4: UpdateState(%2, %3)
    // %5: PrimFunc_Add(%3, y)
    auto cnode_users = mng->node_users()[cnode];
    for (const auto &node_index : cnode_users) {
      auto used_node = node_index.first;
      MS_EXCEPTION_IF_NULL(used_node);
      auto used_cnode = used_node->cast<CNodePtr>();
      if (used_cnode == nullptr) {
        continue;
      }
      // fallback_users: [%2, %3, ...]
      if (std::find(fallback_users.begin(), fallback_users.end(), used_cnode) != fallback_users.end()) {
        // replace %1 in %2, %3
        mng->SetEdge(used_cnode, node_index.second, output);
      } else {
        // replace %1 in %5
        mng->SetEdge(used_cnode, node_index.second, fallback_users[1]);
      }
    }
#ifdef ENABLE_DUMP_IR
    MS_EXCEPTION_IF_NULL(context);
    if (context->CanDump(kAdvanced)) {
      DumpIR("expand_" + AnfUtils::GetCNodeName(cnode) + ".ir", cnode->func_graph(), true);
    }
#endif
  } else {
    (void)mng->Replace(cnode, output);
  }
  return true;
}

bool TryExpandCNode(const AnfNodePtr &node, const std::function<bool(const CNodePtr &)> &func) {
  if (!Check(node)) {
    return false;
  }
  MS_LOG(DEBUG) << "Try to expand node " << node->fullname_with_scope() << ". DebugString: " << node->DebugString();
  auto graph = node->func_graph();
  auto mng = graph->manager();
  if (mng == nullptr) {
    mng = Manage(graph, true);
    MS_EXCEPTION_IF_NULL(mng);
    graph->set_manager(mng);
  }
  const auto *handle = IRBuilderFactory::Instance().GetBuilder(AnfUtils::GetCNodeName(node));
  if (handle == nullptr) {
    return false;
  }
  return IbTryExpandCNode(*handle, node->cast<CNodePtr>(), func);
}
}  // namespace expander
}  // namespace mindspore
