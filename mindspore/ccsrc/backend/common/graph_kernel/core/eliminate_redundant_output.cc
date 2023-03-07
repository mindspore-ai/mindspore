/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "backend/common/graph_kernel/core/eliminate_redundant_output.h"

#include <memory>
#include <algorithm>
#include <vector>
#include <string>
#include <utility>

#include "mindspore/core/ops/core_ops.h"
#include "ir/anf.h"
#include "ir/graph_utils.h"
#include "utils/anf_utils.h"
#include "backend/common/graph_kernel/core/graph_kernel_callback.h"
#include "backend/common/graph_kernel/core/graph_builder.h"

namespace mindspore::graphkernel {
namespace {
inline size_t GetIndex(const AnfNodePtr &getitem_node) {
  MS_EXCEPTION_IF_NULL(getitem_node);
  if (!IsPrimitiveCNode(getitem_node, prim::kPrimTupleGetItem)) {
    MS_LOG(EXCEPTION) << "User of MakeTuple should be GetItem but got " << getitem_node->fullname_with_scope();
  }
  return LongToSize(GetValue<int64_t>(
    getitem_node->cast<CNodePtr>()->input(kInputNodeOutputIndexInTupleGetItem)->cast<ValueNodePtr>()->value()));
}

void SetIndex(const AnfNodePtr &getitem_node, size_t index) {
  auto getitem = getitem_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(getitem);
  auto idx_node = NewValueNode(MakeValue<int64_t>(SizeToLong(index)));
  auto abstract = std::make_shared<abstract::AbstractScalar>(std::make_shared<Int64Imm>(index));
  idx_node->set_abstract(abstract);
  Callback::Instance()->SetEmptyKernelInfo(idx_node);
  getitem->set_input(kInputNodeOutputIndexInTupleGetItem, idx_node);
}
}  // namespace

bool GetGraphKernelGetitemList(const FuncGraphManagerPtr &mng, const AnfNodePtr &node, AnfNodePtrList *getitem_list,
                               bool merge_repeated_getitem) {
  MS_EXCEPTION_IF_NULL(mng);
  MS_EXCEPTION_IF_NULL(getitem_list);
  auto func_graph = GetCNodeFuncGraph(node);
  MS_EXCEPTION_IF_NULL(func_graph);
  auto output = func_graph->output();
  if (!IsPrimitiveCNode(output, prim::kPrimMakeTuple)) {
    MS_LOG(EXCEPTION) << "The output should be a MakeTuple, but got " << output->fullname_with_scope();
  }
  auto output_num = output->cast<CNodePtr>()->size() - 1;
  getitem_list->clear();
  getitem_list->resize(output_num, nullptr);
  auto users = mng->node_users()[node];
  bool changed = false;
  for (const auto &user : users) {
    if (!IsPrimitiveCNode(user.first, prim::kPrimTupleGetItem)) {
      MS_LOG(EXCEPTION) << "User of MakeTuple should be GetItem, but got: " << user.first->DebugString();
    }
    auto &getitem = user.first;
    auto idx = GetIndex(getitem);
    if (idx >= output_num) {
      MS_LOG(EXCEPTION) << "Index of GetItem is " << idx << ", which is out of range of MakeTuple [0, " << output_num
                        << "). GetItem node: " << getitem->DebugString();
    }
    if (merge_repeated_getitem && (*getitem_list)[idx] != nullptr) {
      (void)mng->Replace(getitem, (*getitem_list)[idx]);
      changed = true;
    } else {
      (*getitem_list)[idx] = getitem;
    }
  }
  return changed;
}

AnfNodePtrList FindGraphKernelsWithMultiOutput(const FuncGraphPtr &func_graph) {
  auto todos = TopoSort(func_graph->get_return());
  AnfNodePtrList result;
  (void)std::copy_if(todos.begin(), todos.end(), std::back_inserter(result), [](const AnfNodePtr &node) {
    return AnfUtils::IsGraphKernel(node) && IsPrimitiveCNode(GetCNodeFuncGraph(node)->output(), prim::kPrimMakeTuple);
  });
  return result;
}

bool IsSideEffectNode(const AnfNodePtr &node) {
  std::vector<PrimitivePtr> side_effect_nodes = {prim::kPrimAssign, prim::kPrimInplaceAssign};
  return std::any_of(side_effect_nodes.begin(), side_effect_nodes.end(),
                     [&node](const PrimitivePtr &p) { return IsPrimitiveCNode(node, p); });
}

/* Unify the repeated output in a func_graph.
 *   %1 = call @graph_kernel(p1, p2)
 *   %2 = tuple_getitem(%1, 0)
 *   %3 = tuple_getitem(%1, 1)
 *   graph_kernel:
 *      %1 = TensorAdd(p1, p2)
 *      %2 = Reshape(%1)
 *      return make_tuple(%2, %2)
 * -->
 *   %1 = call @graph_kernel(p1, p2)
 *   %2 = tuple_getitem(%1, 0)
 *   %3 = tuple_getitem(%1, 0)   // changed the index to 0.
 *   graph_kernel:
 *      %1 = TensorAdd(p1, p2)
 *      %2 = Reshape(%1)
 *      return make_tuple(%2, %2)
 */
class UnifyRepeatedOutput : public opt::Pass {
 public:
  bool Run(const FuncGraphPtr &func_graph) override {
    auto mng = func_graph->manager();
    MS_EXCEPTION_IF_NULL(mng);
    auto todos = FindGraphKernelsWithMultiOutput(func_graph);
    bool changed = false;
    for (auto node : todos) {
      if (CheckRepeatedOutput(GetCNodeFuncGraph(node))) {
        changed = true;
        AnfNodePtrList getitem_list;
        (void)GetGraphKernelGetitemList(mng, node, &getitem_list, false);
        if (getitem_list.size() != index_map_.size()) {
          MS_LOG(EXCEPTION) << "getitem_list.size (" << getitem_list.size() << ") should be equal to index_map.size ("
                            << index_map_.size() << ").";
        }
        for (size_t i = 0; i < index_map_.size(); ++i) {
          if (index_map_[i] != i && getitem_list[i] != nullptr) {
            SetIndex(getitem_list[i], index_map_[i]);
          }
        }
      }
    }
    return changed;
  }

 private:
  bool CheckRepeatedOutput(const FuncGraphPtr &sub_func_graph) {
    // the output should be a MakeTuple.
    auto maketuple = sub_func_graph->output()->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(maketuple);
    AnfNodePtrList outputs(maketuple->inputs().begin() + 1, maketuple->inputs().end());
    index_map_.resize(outputs.size());
    bool found = false;
    for (size_t i = 0; i < outputs.size(); ++i) {
      index_map_[i] =
        static_cast<size_t>(std::find(outputs.begin(), outputs.begin() + SizeToLong(i), outputs[i]) - outputs.begin());
      if (index_map_[i] != i) {
        found = true;
      }
    }
    return found;
  }
  std::vector<size_t> index_map_;
};

/* Unify the get_item nodes that have same index.
 *   %1 = call @graph_kernel(p1, p2)
 *   %2 = tuple_getitem(%1, 0)
 *   %3 = tuple_getitem(%1, 0)
 *   %4 = tuple_getitem(%1, 1)
 *   %5 = user_x(%2)
 *   %6 = user_y(%3)
 *   %7 = user_z(%4)
 *   --->
 *   %1 = call @graph_kernel(p1, p2)
 *   %2 = tuple_getitem(%1, 0) // unify the original %2 and %3
 *   %3 = tuple_getitem(%1, 1)
 *   %4 = user_x(%2)
 *   %5 = user_y(%2)
 *   %6 = user_z(%3)
 */
class UnifyRepeatedGetitem : public opt::Pass {
 public:
  bool Run(const FuncGraphPtr &func_graph) override {
    auto mng = func_graph->manager();
    MS_EXCEPTION_IF_NULL(mng);
    auto todos = FindGraphKernelsWithMultiOutput(func_graph);
    bool changed = false;
    for (auto node : todos) {
      AnfNodePtrList getitem_list;
      changed = GetGraphKernelGetitemList(mng, node, &getitem_list, true) || changed;
    }
    return changed;
  }
};

/* if a graphkernel node of multi-output is directly used by other kernel,
 * change it to use getitem-maketuple.
 *   %1 = call @graph_kernel(p1, p2)  // assume it has 3 outputs.
 *   %2 = AddN(%1)
 *   --->
 *   %1 = call @graph_kernel(p1, p2)
 *   %2 = tuple_getitem(%1, 0)
 *   %3 = tuple_getitem(%1, 1)
 *   %4 = tuple_getitem(%1, 2)
 *   %5 = make_tuple(%2, %3, %4)
 *   %6 = AddN(%5)
 */
class TupleNodeFormatter : public opt::Pass {
 public:
  bool Run(const FuncGraphPtr &func_graph) override {
    auto mng = func_graph->manager();
    MS_EXCEPTION_IF_NULL(mng);
    auto todos = FindGraphKernelsWithMultiOutput(func_graph);
    bool changed = false;
    for (auto &node : todos) {
      auto &users = mng->node_users()[node];
      for (auto &user : users) {
        if (!IsPrimitiveCNode(user.first, prim::kPrimTupleGetItem)) {
          auto mt = TransToMaketuple(node);
          (void)mng->Replace(node, mt);
          changed = true;
          break;
        }
      }
    }
    if (changed) {
      (void)EliminateMaketupleGetitem(func_graph);
    }
    return changed;
  }

  AnfNodePtr TransToMaketuple(const AnfNodePtr &node) {
    auto fg = node->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    auto node_abs = node->abstract()->cast<abstract::AbstractTuplePtr>();
    MS_EXCEPTION_IF_NULL(node_abs);
    auto output_num = SizeToLong(node_abs->size());
    AnfNodePtrList mt_inputs{NewValueNode(prim::kPrimMakeTuple)};
    mt_inputs.reserve(output_num + 1);
    for (int64_t i = 0; i < output_num; i++) {
      AnfNodePtrList gt_inputs{NewValueNode(prim::kPrimTupleGetItem), node, NewValueNode(MakeValue(i))};
      gt_inputs.back()->set_abstract(std::make_shared<abstract::AbstractScalar>(std::make_shared<Int64Imm>(i)));
      auto &gt = mt_inputs.emplace_back(fg->NewCNode(gt_inputs));
      gt->set_abstract(node_abs->elements()[i]);
      Callback::Instance()->SetEmptyKernelInfo(gt);
    }
    auto mt = fg->NewCNode(mt_inputs);
    mt->set_abstract(node_abs);
    Callback::Instance()->SetEmptyKernelInfo(mt);
    return mt;
  }
};

bool EliminateRedundantOutput::Run(const FuncGraphPtr &func_graph) {
  auto mng = func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(func_graph, true);
    func_graph->set_manager(mng);
  }
  bool changed = false;
  changed = std::make_shared<TupleNodeFormatter>()->Run(func_graph) || changed;
  changed = std::make_shared<UnifyRepeatedGetitem>()->Run(func_graph) || changed;
  changed = std::make_shared<UnifyRepeatedOutput>()->Run(func_graph) || changed;
  changed = std::make_shared<UnifyRepeatedGetitem>()->Run(func_graph) || changed;
  changed = std::make_shared<EliminateHangingOutput>()->Run(func_graph) || changed;
  return changed;
}

void EliminateHangingOutput::UpdateGetitemIndex(const AnfNodePtr &getitem, size_t offset) const {
  if (offset == 0) {
    return;
  }
  MS_EXCEPTION_IF_NULL(getitem);
  auto index = GetIndex(getitem);
  if (offset > index) {
    MS_LOG(EXCEPTION) << "The offset is greater than the original index of GetItem: " << getitem->DebugString() << ". "
                      << offset << " vs " << index;
  }
  index -= offset;
  SetIndex(getitem, index);
}

AnfNodePtr EliminateHangingOutput::ReplaceMakeTuple(const AnfNodePtr &node, const AnfNodePtrList &getitems) const {
  auto func_graph = GetCNodeFuncGraph(node);
  MS_EXCEPTION_IF_NULL(func_graph);
  auto old_maketuple = func_graph->output()->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(old_maketuple);
  AnfNodePtrList new_maketuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  AbstractBasePtrList abstract_list;
  size_t offset = 0;
  for (size_t i = 0; i < getitems.size(); ++i) {
    // If a node has no user, it should be eliminated, but except for side-effect node.
    if (getitems[i] == nullptr && !IsSideEffectNode(old_maketuple->input(i + 1))) {
      offset++;
    } else {
      new_maketuple_inputs.push_back(old_maketuple->input(i + 1));
      abstract_list.push_back(old_maketuple->input(i + 1)->abstract());
      if (getitems[i] != nullptr) {
        UpdateGetitemIndex(getitems[i], offset);
      }
    }
  }
  if (offset == 0) {
    return nullptr;
  }
  if (new_maketuple_inputs.size() == 1) {
    MS_LOG(EXCEPTION) << "Input of MakeTuple could not be empty";
  }
  const size_t maketuple_one_input_size = 2;
  if (new_maketuple_inputs.size() == maketuple_one_input_size) {
    func_graph->set_output(new_maketuple_inputs.back());
  } else {
    auto make_tuple = func_graph->NewCNode(new_maketuple_inputs);
    make_tuple->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
    Callback::Instance()->SetEmptyKernelInfo(make_tuple);
    func_graph->set_output(make_tuple);
  }

  auto old_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(old_cnode);
  AnfNodePtrList inputs(old_cnode->inputs().begin() + 1, old_cnode->inputs().end());
  auto graph_kernel_node = CreateNewFuseCNode(node->func_graph(), func_graph, inputs);
  return graph_kernel_node;
}

bool EliminateHangingOutput::Run(const FuncGraphPtr &func_graph) {
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  auto todos = FindGraphKernelsWithMultiOutput(func_graph);
  bool changed = false;
  for (auto node : todos) {
    AnfNodePtrList getitems;
    (void)GetGraphKernelGetitemList(mng, node, &getitems, false);
    auto new_node = ReplaceMakeTuple(node, getitems);
    if (new_node != nullptr) {
      if (!IsPrimitiveCNode(GetCNodeFuncGraph(new_node)->output(), prim::kPrimMakeTuple)) {
        // only one output, remove the getitem.
        auto i = std::find_if(getitems.begin(), getitems.end(), [](const AnfNodePtr &node) { return node != nullptr; });
        if (i != getitems.end()) {
          (void)mng->Replace(*i, new_node);
        }
      } else {
        (void)mng->Replace(node, new_node);
      }
      changed = true;
    }
  }
  return changed;
}
}  // namespace mindspore::graphkernel
