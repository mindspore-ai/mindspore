/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_INCORPORATE_GETITEM_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_INCORPORATE_GETITEM_H_

#include <algorithm>
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>
#include <utility>

#include "ir/func_graph.h"
#include "ir/func_graph_cloner.h"
#include "frontend/optimizer/optimizer_caller.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
namespace irpass {
namespace internal {
class GetitemTransform {
 public:
  GetitemTransform() : cache_() {}
  ~GetitemTransform() = default;

  FuncGraphPtr operator()(const AnfNodePtr &node, const FuncGraphPtr &fg, int64_t idx) {
    if (cache_.find(fg) == cache_.end()) {
      cache_[fg] = {};
    }

    auto &cache = cache_[fg];
    if (cache.find(idx) == cache.end()) {
      std::ostringstream ss("tp", std::ostringstream::app);
      ss << idx;

      auto new_fg = TransformableClone(fg, std::make_shared<TraceTransform>(ss.str()));
      auto output = new_fg->output();
      if (IsPrimitiveCNode(output, prim::kPrimMakeTuple)) {
        auto cnode = output->cast<CNodePtr>();
        auto ids = LongToSize(idx + 1);
        // Inputs should be [make_tuple, item1, item2, ...], so have to offset idx in tuple_getitem by 1.
        if (ids >= cnode->size()) {
          MS_LOG(EXCEPTION) << "index " << ids << " is out of inputs length " << cnode->size();
        }
        new_fg->set_output(cnode->input(ids));
      } else {
        auto idx_node = NewValueNode(idx);
        idx_node->set_abstract(std::make_shared<abstract::AbstractScalar>(idx));
        auto output_node = new_fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), output, idx_node});
        output_node->set_abstract(node->abstract());
        new_fg->set_output(output_node);
      }

      cache[idx] = new_fg;
    }
    return cache[idx];
  }

 private:
  std::unordered_map<FuncGraphPtr, std::unordered_map<int64_t, FuncGraphPtr>> cache_;
};

class GetItemTransformACrossGraph {
 public:
  GetItemTransformACrossGraph() : cache_() {}
  ~GetItemTransformACrossGraph() = default;

  FuncGraphPtr operator()(const AnfNodePtr &node, const FuncGraphPtr &fg, int64_t idx) {
    if (cache_.find(fg) == cache_.end()) {
      cache_[fg] = {};
    }

    auto &cache = cache_[fg];
    if (cache.find(idx) == cache.end()) {
      std::ostringstream ss("tp", std::ostringstream::app);
      ss << idx;

      auto new_fg_outer = TransformableClone(fg, std::make_shared<TraceTransform>(ss.str()));
      MS_EXCEPTION_IF_NULL(fg->manager());
      fg->manager()->AddFuncGraph(new_fg_outer);
      auto output_outer = new_fg_outer->output();
      if (!IsValueNode<FuncGraph>(output_outer)) {
        MS_LOG(WARNING) << "Output of outer graph should be a func_graph";
        return nullptr;
      }
      auto fg_inner = GetValueNode<FuncGraphPtr>(output_outer);
      auto new_fg = TransformableClone(fg_inner, std::make_shared<TraceTransform>(ss.str()));
      new_fg_outer->set_output(NewValueNode(new_fg));
      auto output = new_fg->output();
      if (IsPrimitiveCNode(output, prim::kPrimMakeTuple)) {
        auto cnode = output->cast<CNodePtr>();
        auto ids = LongToSize(idx + 1);
        // Inputs should be [make_tuple, item1, item2, ...], so have to offset idx in tuple_getitem by 1.
        if (ids >= cnode->size()) {
          MS_LOG(EXCEPTION) << "index " << ids << " is out of inputs length " << cnode->size();
        }
        new_fg->set_output(cnode->input(ids));
      } else {
        auto idx_node = NewValueNode(idx);
        idx_node->set_abstract(std::make_shared<abstract::AbstractScalar>(idx));
        auto output_node = new_fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), output, idx_node});
        output_node->set_abstract(node->abstract());
        new_fg->set_output(output_node);
      }

      cache[idx] = new_fg_outer;
    }
    return cache[idx];
  }

 private:
  std::unordered_map<FuncGraphPtr, std::unordered_map<int64_t, FuncGraphPtr>> cache_;
};

bool HasMoreJ(const OptimizerPtr &optimizer) {
  bool more_j = false;
  auto res = optimizer->resource();
  auto resource_ptr = std::dynamic_pointer_cast<pipeline::Resource>(res);
  if (resource_ptr != nullptr) {
    const auto &manager = optimizer->manager();
    MS_EXCEPTION_IF_NULL(manager);
    more_j = manager->func_graph_j_total(resource_ptr->func_graph());
  }
  return more_j;
}

bool IsOutputShrinkable(const AnfNodePtr &output) {
  if (IsPrimitiveCNode(output, prim::kPrimMakeTuple)) {
    return true;
  }
  if (GetValueNode<ValueTuplePtr>(output)) {
    return true;
  }
  return false;
}

size_t GetOutputSize(const AnfNodePtr &output) {
  if (IsPrimitiveCNode(output, prim::kPrimMakeTuple)) {
    const auto &output_cnode = output->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(output_cnode);
    return output_cnode->size() - 1;
  }
  const auto &value_tuple = GetValueNode<ValueTuplePtr>(output);
  if (value_tuple == nullptr) {
    MS_LOG(EXCEPTION) << "fg output is not MakeTuple or ValueTuple, but: " << output->DebugString();
  }
  return value_tuple->size();
}

struct TpCNodeAndIndex {
  // CNode {TupleGetItem, call, index}
  CNodePtr tp_cnode;
  int64_t index;
};

int64_t UpdateUserNodeIndex(const CNodePtr &fg_call_cnode, const int64_t current_index,
                            const std::vector<TpCNodeAndIndex> &tp_cnodes_and_index) {
  const auto &manager = fg_call_cnode->func_graph()->manager();
  MS_EXCEPTION_IF_NULL(manager);
  int64_t new_index = current_index;
  auto txn = manager->Transact();
  for (int64_t i = 0; i < SizeToLong(tp_cnodes_and_index.size()); ++i) {
    const auto &cnode_and_index = tp_cnodes_and_index[i];
    if (cnode_and_index.index != i) {
      constexpr auto kInputIndex = 2;
      txn.SetEdge(cnode_and_index.tp_cnode, kInputIndex, NewValueNode(i));
    }
    if (cnode_and_index.index == current_index) {
      new_index = i;
    }
  }
  txn.Commit();
  return new_index;
}

AbstractBasePtr ShrinkAbstract(const AbstractBasePtr &original_abstract,
                               const std::vector<TpCNodeAndIndex> &tp_cnodes_and_index) {
  if (original_abstract != nullptr && original_abstract->isa<abstract::AbstractTuple>()) {
    const auto &abs_tuple = original_abstract->cast<abstract::AbstractTuplePtr>();
    MS_EXCEPTION_IF_NULL(abs_tuple);
    const auto &abs_tuple_elements = abs_tuple->elements();
    const int64_t before_shrink_tuple_size = SizeToLong(abs_tuple_elements.size());
    AbstractBasePtrList shrunk_abstract_elements;
    std::transform(tp_cnodes_and_index.cbegin(), tp_cnodes_and_index.cend(),
                   std::back_inserter(shrunk_abstract_elements),
                   [abs_tuple_elements, before_shrink_tuple_size](const auto &node_and_index) {
                     if (node_and_index.index >= before_shrink_tuple_size) {
                       MS_LOG(EXCEPTION) << "index should less than inputs size, index: " << node_and_index.index
                                         << ", abstract tuple size: " << before_shrink_tuple_size;
                     }
                     return abs_tuple_elements[node_and_index.index];
                   });
    return std::make_shared<abstract::AbstractTuple>(shrunk_abstract_elements);
  }
  return nullptr;
}

FuncGraphPtr ShrinkUnsedOutput(const FuncGraphPtr &fg, const std::vector<TpCNodeAndIndex> &tp_cnodes_and_index) {
  const auto &manager = fg->manager();
  MS_EXCEPTION_IF_NULL(manager);

  auto new_fg = TransformableClone(fg, std::make_shared<TraceTransform>("tp_use"));
  auto new_fg_output = new_fg->output();
  AnfNodePtr shrunk_output = nullptr;
  int64_t before_shrink_inputs_size = 0;
  if (IsPrimitiveCNode(new_fg_output, prim::kPrimMakeTuple)) {
    // Shrink output;
    auto new_fg_output_cnode = new_fg_output->cast<CNodePtr>();
    const auto &new_fg_output_inputs = new_fg_output_cnode->inputs();
    constexpr auto kMinimalSize = 2;
    if (new_fg_output_inputs.size() <= kMinimalSize) {
      MS_LOG(EXCEPTION) << "New fg output should at least 2 elements, but: " << new_fg_output->DebugString();
    }
    before_shrink_inputs_size = SizeToLong(new_fg_output_inputs.size() - 1);
    AnfNodePtrList shrunk_inputs{NewValueNode({prim::kPrimMakeTuple})};
    // Bypass maketuple primitive in new_fg_output_inputs;
    std::transform(tp_cnodes_and_index.cbegin(), tp_cnodes_and_index.cend(), std::back_inserter(shrunk_inputs),
                   [new_fg_output, new_fg_output_inputs, before_shrink_inputs_size](const auto &node_and_index) {
                     if (node_and_index.index >= before_shrink_inputs_size) {
                       MS_LOG(EXCEPTION) << "index should less than inputs size, index: " << node_and_index.index
                                         << ", output: " << new_fg_output->DebugString();
                     }
                     return new_fg_output_inputs[node_and_index.index + 1];
                   });
    shrunk_output = new_fg->NewCNode(shrunk_inputs);
  } else {
    auto value_tuple = GetValueNode<ValueTuplePtr>(new_fg_output);
    if (value_tuple == nullptr) {
      MS_LOG(EXCEPTION) << "New fg output is not MakeTuple or ValueTuple, but " << new_fg_output->DebugString();
    }
    ValuePtrList shrunk_inputs;
    before_shrink_inputs_size = value_tuple->size();
    std::transform(tp_cnodes_and_index.cbegin(), tp_cnodes_and_index.cend(), std::back_inserter(shrunk_inputs),
                   [new_fg_output, value_tuple, before_shrink_inputs_size](const auto &node_and_index) {
                     if (node_and_index.index >= before_shrink_inputs_size) {
                       MS_LOG(EXCEPTION) << "index should less than inputs size, index: " << node_and_index.index
                                         << ", output: " << new_fg_output->DebugString();
                     }
                     return (*value_tuple)[node_and_index.index];
                   });
    shrunk_output = NewValueNode(std::make_shared<ValueTuple>(shrunk_inputs));
  }
  auto shrunk_abstract = ShrinkAbstract(new_fg_output->abstract(), tp_cnodes_and_index);
  MS_EXCEPTION_IF_NULL(shrunk_abstract);
  shrunk_output->set_abstract(shrunk_abstract);
  new_fg->set_output(shrunk_output);
  MS_LOG(DEBUG) << "Partly item used; original size: " << before_shrink_inputs_size
                << ", new size: " << tp_cnodes_and_index.size() << ", fg: " << fg->ToString() << ", new graph"
                << new_fg->ToString();
  return new_fg;
}

struct FuncGraphIntVectorPairHasher {
  std::size_t Int64VectorHash(const std::vector<int64_t> &int_vector) const {
    std::size_t hash_value = 0;
    constexpr auto kMaxElementsNum = 4;
    for (size_t i = 0; (i < int_vector.size()) && (i < kMaxElementsNum); ++i) {
      hash_value = hash_combine(hash_value, std::hash<int64_t>{}(int_vector[i]));
    }
    return hash_value;
  }

  std::size_t operator()(const std::pair<FuncGraphPtr, std::vector<int64_t>> &p) const {
    auto h1 = std::hash<FuncGraphPtr>{}(p.first);
    auto h2 = Int64VectorHash(p.second);
    return hash_combine(h1, h2);
  }
};

bool ShouldTransform(const AnfNodePtr &node, const std::vector<TpCNodeAndIndex> &tp_cnodes_and_index) {
  if (node->abstract() && node->abstract()->isa<abstract::AbstractTuple>()) {
    const auto &abs_tuple = *(node->abstract()->cast<abstract::AbstractTuplePtr>());
    if (tp_cnodes_and_index[0].index == 0 && abs_tuple.size() > 0) {
      if (abs_tuple[0]->isa<abstract::AbstractScalar>() && abs_tuple[0]->GetTypeTrack()->isa<EnvType>()) {
        return true;
      }
    }
    // fprop_fg will return MakeTuple(xx, bprop_fg).
    if (tp_cnodes_and_index.size() > 1 && tp_cnodes_and_index[1].index == 1 && abs_tuple.size() > 1 &&
        abs_tuple[1]->isa<abstract::AbstractFunction>()) {
      return true;
    }
  }
  return false;
}

// Incorporate getitem if the indexed node is a ZerosLike node, so another opt pass AddN(MakeTuple(Xs, ZerosLike))
// can work.
bool AlwaysTransformThisIndex(const AnfNodePtr &output, const int64_t index) {
  if (IsPrimitiveCNode(output, prim::kPrimMakeTuple)) {
    const auto &output_cnode = output->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(output_cnode);
    if (index >= SizeToLong(output_cnode->size() - 1)) {
      MS_LOG(EXCEPTION) << "Index of GetItem: " << index
                        << " exceeds size of MakeTuple: " << output_cnode->DebugString();
    }
    if (IsPrimitiveCNode(output_cnode->input(index + 1), prim::kPrimZerosLike)) {
      return true;
    }
  }
  return false;
}
}  // namespace internal

// {prim::kPrimTupleGetItem, {G, Xs}, C}
class IncorporateGetitem : public AnfVisitor {
 public:
  IncorporateGetitem() : getitem_transform_() {}
  ~IncorporateGetitem() override = default;

  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimTupleGetItem, {IsCNode, IsValueNode<Int64Imm>})(node);
    if (node->func_graph() == nullptr || idx_ == -1 || fg_ == nullptr || fg_->has_flag(FUNC_GRAPH_FLAG_DEFER_INLINE) ||
        fg_->has_flag(FUNC_GRAPH_OUTPUT_NO_RECOMPUTE)) {
      return nullptr;
    }

    const auto &manager = fg_->manager();
    MS_EXCEPTION_IF_NULL(manager);
    if (internal::AlwaysTransformThisIndex(fg_->output(), idx_)) {
      return TransformFuncGraph(manager, node);
    }
    // This node had been substituted.
    if (processed_nodes_.find(fg_call_cnode_) != processed_nodes_.end()) {
      MS_LOG(DEBUG) << "fg call with same cnode is already replaced, node: " << node->DebugString()
                    << ", fg_call: " << fg_call_cnode_->DebugString();
      return nullptr;
    }
    bool output_is_shrinkable = internal::IsOutputShrinkable(fg_->output());
    std::vector<internal::TpCNodeAndIndex> tp_cnodes_and_index;
    auto fg_call_cnode_users_counter = MultipleUse(fg_call_cnode_, fg_, &tp_cnodes_and_index);
    bool multiple_use = (tp_cnodes_and_index.size() > 1);
    if (output_is_shrinkable && multiple_use && (tp_cnodes_and_index.size() == fg_call_cnode_users_counter)) {
      if (!internal::ShouldTransform(fg_call_cnode_, tp_cnodes_and_index) && !internal::HasMoreJ(optimizer)) {
        MS_LOG(DEBUG) << "No more j and multiple use, will shrink, node: " << node->DebugString()
                      << ", fg_call: " << fg_call_cnode_->DebugString();
        const auto output_size = internal::GetOutputSize(fg_->output());
        if (fg_call_cnode_users_counter == output_size) {
          processed_nodes_.emplace(fg_call_cnode_);
          MS_LOG(DEBUG) << "All elements in output is used, no need to transform, node: " << node->DebugString()
                        << ", fg_call: " << fg_call_cnode_->DebugString();
          return nullptr;
        }
        auto new_node = ShrinkFuncGraphOutput(node, tp_cnodes_and_index);
        if (new_node != nullptr) {
          return new_node;
        }
      }
    }
    MS_LOG(DEBUG) << "Cannot shrink, transform_getitem, node: " << node->DebugString()
                  << ", fg_call: " << fg_call_cnode_->DebugString();
    return TransformFuncGraph(manager, node);
  }

  size_t MultipleUse(const CNodePtr &fg_call, const FuncGraphPtr &fg,
                     std::vector<internal::TpCNodeAndIndex> *cnodes_and_index) const {
    const auto &manager = fg->manager();
    MS_EXCEPTION_IF_NULL(manager);
    auto &cnode_and_index_vector = *cnodes_and_index;
    std::set<int64_t> index_set;
    std::size_t total_usage = 0;
    const auto &node_users_map = manager->node_users();
    const auto &it = node_users_map.find(fg_call);
    if (it == node_users_map.end()) {
      return 0;
    }
    const auto &node_users = it->second;
    for (const auto &user : node_users) {
      if (IsPrimitiveCNode(user.first, prim::kPrimTupleGetItem)) {
        const auto &cnode = user.first->cast<CNodePtr>();
        if (cnode->input(2)->isa<ValueNode>()) {
          auto idx = GetValue<int64_t>(cnode->input(2)->cast<ValueNodePtr>()->value());
          cnode_and_index_vector.push_back({cnode, idx});
          index_set.insert(idx);
          total_usage++;
        } else {
          MS_LOG(EXCEPTION) << "tuple_getitem index is not valuenode, but: " << user.first->DebugString();
        }
      } else {
        MS_LOG(DEBUG) << "fg_call usre is not tuple_getitem, user: " << user.first->DebugString();
      }
    }
    if (index_set.size() != total_usage) {
      MS_LOG(DEBUG) << "some index usage is duplicated, total_usage: " << total_usage;
      MS_LOG(DEBUG) << "index_set:";
      for (auto idx : index_set) {
        MS_LOG(DEBUG) << " " << idx;
      }
    }
    // sort by index;
    std::sort(cnode_and_index_vector.begin(), cnode_and_index_vector.end(),
              [](const auto &tp1, const auto &tp2) { return tp1.index < tp2.index; });
    return node_users.size();
  }

  AnfNodePtr ShrinkFuncGraphOutput(const AnfNodePtr &node,
                                   const std::vector<internal::TpCNodeAndIndex> &tp_cnodes_and_index) {
    const auto &manager = fg_->manager();
    MS_EXCEPTION_IF_NULL(manager);
    std::vector<int64_t> index_vector;
    (void)std::transform(tp_cnodes_and_index.begin(), tp_cnodes_and_index.end(), std::back_inserter(index_vector),
                         [](const auto &cnode_and_index) { return cnode_and_index.index; });
    auto iter = processed_fgs_.find(std::make_pair(fg_, index_vector));
    if (iter != processed_fgs_.end()) {
      MS_LOG(DEBUG) << "fg is already processed, just update caller index, node: " << node->DebugString()
                    << ", fg_call: " << fg_call_cnode_->DebugString();
      MS_LOG(DEBUG) << "original fg: " << fg_->ToString() << ", processed_fg: " << iter->second->ToString();
      processed_nodes_.emplace(fg_call_cnode_);
      manager->SetEdge(fg_call_cnode_, 0, NewValueNode(iter->second));
      auto shrunk_abstract = internal::ShrinkAbstract(fg_call_cnode_->abstract(), tp_cnodes_and_index);
      if (shrunk_abstract != nullptr) {
        fg_call_cnode_->set_abstract(shrunk_abstract);
      }
      auto new_idx = internal::UpdateUserNodeIndex(fg_call_cnode_, idx_, tp_cnodes_and_index);
      auto new_node =
        node->func_graph()->NewCNode({NewValueNode(prim::kPrimTupleGetItem), fg_call_cnode_, NewValueNode(new_idx)});
      new_node->set_abstract(node->abstract());
      return new_node;
    }
    const auto new_fg = internal::ShrinkUnsedOutput(fg_, tp_cnodes_and_index);
    if (new_fg != nullptr) {
      MS_LOG(DEBUG) << "fg output is shrunk, original fg: " << fg_->ToString() << ", new fg: " << new_fg->ToString();
      processed_nodes_.emplace(fg_call_cnode_);
      processed_fgs_.emplace(std::make_pair(fg_, index_vector), new_fg);
      manager->SetEdge(fg_call_cnode_, 0, NewValueNode(new_fg));
      auto shrunk_abstract = internal::ShrinkAbstract(fg_call_cnode_->abstract(), tp_cnodes_and_index);
      if (shrunk_abstract != nullptr) {
        fg_call_cnode_->set_abstract(shrunk_abstract);
      }
      auto new_idx = internal::UpdateUserNodeIndex(fg_call_cnode_, idx_, tp_cnodes_and_index);
      auto new_node =
        node->func_graph()->NewCNode({NewValueNode(prim::kPrimTupleGetItem), fg_call_cnode_, NewValueNode(new_idx)});
      new_node->set_abstract(node->abstract());
      return new_node;
    }
    MS_LOG(DEBUG) << "Shrink failed. node: " << node->DebugString()
                  << ", switch_call: " << fg_call_cnode_->DebugString();
    return nullptr;
  }

  AnfNodePtr TransformFuncGraph(const FuncGraphManagerPtr &manager, const AnfNodePtr &origin_node) {
    auto new_fg = getitem_transform_(origin_node, fg_, idx_);
    MS_LOG(DEBUG) << "Original fg: " << fg_->ToString() << ", new fg: " << new_fg->ToString();
    (void)args_.insert(args_.begin(), NewValueNode(new_fg));
    auto new_node = origin_node->func_graph()->NewCNode(args_);
    // Check if the another only usage of {G, Xs} is UpdateState{s, {G, Xs}}, if yes, replace
    // UpdateState{s, {G, Xs}} with UpdateState{s, new_node};
    auto &node_users_map = manager->node_users();
    auto it = node_users_map.find(fg_call_cnode_);
    if (it != node_users_map.end()) {
      AnfNodePtr update_state_node = nullptr;
      auto &node_users = it->second;
      if (node_users.size() == 2) {
        for (auto &node_user : node_users) {
          if (IsPrimitiveCNode(node_user.first, prim::kPrimUpdateState)) {
            update_state_node = node_user.first;
          }
        }
      }
      if (update_state_node != nullptr) {
        auto update_state_cnode = update_state_node->cast<CNodePtr>();
        // double check;
        if (update_state_cnode->input(2) == fg_call_cnode_) {
          MS_LOG(DEBUG) << "Replace UpdateState node: " << update_state_cnode->DebugString(2)
                        << ", input 2 with: " << new_node->DebugString();
          manager->SetEdge(update_state_cnode, 2, new_node);
        }
      }
    }
    new_node->set_abstract(origin_node->abstract());
    return new_node;
  }

  void Visit(const CNodePtr &cnode) override {
    if (cnode->size() == 0 || !IsValueNode<FuncGraph>(cnode->input(0))) {
      return;
    }

    fg_call_cnode_ = cnode;
    auto &inputs = cnode->inputs();
    fg_ = GetValueNode<FuncGraphPtr>(inputs[0]);
    (void)std::copy(inputs.begin() + 1, inputs.end(), std::back_inserter(args_));
  }

  void Visit(const ValueNodePtr &vnode) override { idx_ = GetValue<int64_t>(vnode->value()); }

  void Reset() {
    idx_ = -1;
    fg_ = nullptr;
    fg_call_cnode_ = nullptr;
    args_.clear();
  }

 private:
  int64_t idx_{-1};
  FuncGraphPtr fg_{nullptr};
  CNodePtr fg_call_cnode_{nullptr};
  std::vector<AnfNodePtr> args_{};
  std::set<AnfNodePtr> processed_nodes_;
  std::unordered_map<std::pair<FuncGraphPtr, std::vector<int64_t>>, FuncGraphPtr,
                     internal::FuncGraphIntVectorPairHasher>
    processed_fgs_;
  internal::GetitemTransform getitem_transform_;
};

// A special case, cannot wait for TupleListGetitemDependReorder pass.
// {prim::kPrimTupleGetItem, {prim::kPrimDepend, {G, Xs}, {prim::kPrimUpdateState, Y, {G, Xs}}}, C} ->
// {prim::kPrimDepend, {tp_idx_G, Xs}, {prim::kPrimUpdateState, Y, {tp_idx_G, Xs}}} ->
class IncorporateGetitemDepend : public AnfVisitor {
 public:
  IncorporateGetitemDepend() : getitem_transform_() {}
  ~IncorporateGetitemDepend() override = default;

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimTupleGetItem, {IsCNode, IsValueNode<Int64Imm>})(node);
    if (node->func_graph() == nullptr || idx_ == -1 || fg_ == nullptr || fg_->has_flag(FUNC_GRAPH_FLAG_DEFER_INLINE) ||
        fg_->has_flag(FUNC_GRAPH_OUTPUT_NO_RECOMPUTE)) {
      return nullptr;
    }

    auto new_fg = getitem_transform_(node, fg_, idx_);
    (void)args_.insert(args_.begin(), NewValueNode(new_fg));
    auto new_fg_cnode = node->func_graph()->NewCNode(args_);
    AnfNodePtr new_depend_cnode;
    if (used_in_update_) {
      auto update_cnode = depend_2nd_input_->cast<CNodePtr>();
      AnfNodePtrList new_update_inputs;
      (void)std::copy(update_cnode->inputs().begin(), update_cnode->inputs().end() - 1,
                      std::back_inserter(new_update_inputs));
      new_update_inputs.push_back(new_fg_cnode);
      auto new_update_cnode = node->func_graph()->NewCNode(new_update_inputs);
      new_depend_cnode =
        node->func_graph()->NewCNode({NewValueNode(prim::kPrimDepend), new_fg_cnode, new_update_cnode});
    } else {
      new_depend_cnode =
        node->func_graph()->NewCNode({NewValueNode(prim::kPrimDepend), new_fg_cnode, depend_2nd_input_});
    }
    new_depend_cnode->set_abstract(node->abstract());
    return new_depend_cnode;
  }

  void Visit(const CNodePtr &cnode) override {
    // cnode : {kPrimDepend, {G, Xs}, {kPrimUpdatestate, Y, {G, Xs}}}
    if (!IsPrimitiveCNode(cnode, prim::kPrimDepend)) {
      return;
    }
    if (cnode->size() != 3 || !IsCNode(cnode->input(1))) {
      return;
    }
    depend_2nd_input_ = cnode->input(2);

    auto fg_cnode = cnode->input(1)->cast<CNodePtr>();
    // fg_cnode : {G, Xs}
    if (!IsValueNode<FuncGraph>(fg_cnode->input(0))) {
      return;
    }

    auto &inputs = fg_cnode->inputs();
    fg_ = GetValueNode<FuncGraphPtr>(inputs[0]);
    (void)std::copy(inputs.begin() + 1, inputs.end(), std::back_inserter(args_));

    if (!IsPrimitiveCNode(depend_2nd_input_, prim::kPrimUpdateState)) {
      return;
    }
    auto update_cnode = depend_2nd_input_->cast<CNodePtr>();
    if (update_cnode->size() != 3) {
      return;
    }
    // match {kPrimUpdateState, Y, {G, Xs}}
    if (update_cnode->input(2) == fg_cnode) {
      used_in_update_ = true;
    }
  }

  void Visit(const ValueNodePtr &vnode) override { idx_ = GetValue<int64_t>(vnode->value()); }

  void Reset() {
    idx_ = -1;
    fg_ = nullptr;
    args_.clear();
  }

 private:
  bool used_in_update_{false};
  int64_t idx_{-1};
  FuncGraphPtr fg_{nullptr};
  AnfNodePtr depend_2nd_input_{nullptr};
  std::vector<AnfNodePtr> args_{};
  internal::GetitemTransform getitem_transform_;
};

// {prim::kPrimTupleGetItem, {{prim::kPrimSwitch, X, G1, G2}, Xs}, C}
class IncorporateGetitemSwitch : public AnfVisitor {
 public:
  IncorporateGetitemSwitch() : getitem_transform_() {}
  ~IncorporateGetitemSwitch() override = default;

  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) override {
    Reset();
    is_in_get_ = true;
    AnfVisitor::Match(prim::kPrimTupleGetItem, {IsCNode, IsValueNode<Int64Imm>})(node);
    is_in_get_ = false;

    auto fg = node->func_graph();
    if (idx_ == -1 || switch_ == nullptr || fg == nullptr) {
      return nullptr;
    }

    is_in_switch_ = true;
    AnfVisitor::Match(prim::kPrimSwitch, {IsNode, IsValueNode<FuncGraph>, IsValueNode<FuncGraph>})(switch_);
    is_in_switch_ = false;

    if (g2_ == nullptr) {
      return nullptr;
    }
    if (processed_nodes_.find(switch_) != processed_nodes_.end()) {
      MS_LOG(DEBUG) << "fg in switch node has been replaced. node: " << node->DebugString()
                    << ", switch: " << switch_->DebugString();
      return nullptr;
    }

    bool g1_output_is_shrinkable = internal::IsOutputShrinkable(g1_->output());
    bool g2_output_is_shrinkable = internal::IsOutputShrinkable(g2_->output());

    auto tuple_getitem = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(tuple_getitem);
    const auto &switch_call = tuple_getitem->input(1);
    MS_EXCEPTION_IF_NULL(switch_call);
    const auto &switch_call_cnode = switch_call->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(switch_call_cnode);
    // If exist env_getitem/env_setitem in this funcgraph or
    // if g1_/g2_ is fprop func_graph and the corresponding bprop funcgraph has any env_getitem or env_setitem;
    std::vector<internal::TpCNodeAndIndex> tp_cnodes_and_index;
    auto switch_call_users_counter = MultipleUseOfSwitch(switch_call, fg, &tp_cnodes_and_index);
    bool multiple_use = (tp_cnodes_and_index.size() > 1);
    if (g1_output_is_shrinkable && g2_output_is_shrinkable && multiple_use &&
        (tp_cnodes_and_index.size() == switch_call_users_counter)) {
      if (!internal::HasMoreJ(optimizer) && !ExistEnvNode(fg) && !ExistEnvNodeInTupleItem(g1_) &&
          !ExistEnvNodeInTupleItem(g2_) && !internal::ShouldTransform(switch_call, tp_cnodes_and_index)) {
        MS_LOG(DEBUG) << "No more j, will shrink. Node: " << node->DebugString()
                      << ", switch: " << switch_->DebugString();
        const auto g1_output_size = internal::GetOutputSize(g1_->output());
        const auto g2_output_size = internal::GetOutputSize(g2_->output());
        if (g1_output_size != g2_output_size) {
          MS_LOG(EXCEPTION) << "output of g1 and g2 should have same tuple size, but g1 output: "
                            << g1_->output()->DebugString() << ", g2 output: " << g2_->output()->DebugString();
        }
        if (switch_call_users_counter == g1_output_size) {
          processed_nodes_.emplace(switch_call);
          MS_LOG(DEBUG) << "All elements in output is used, no need to transform, node: " << node->DebugString()
                        << ", switch: " << switch_->DebugString();
          return nullptr;
        }

        auto new_node = ShrinkFuncGraphOutput(node, switch_call_cnode, tp_cnodes_and_index);
        if (new_node != nullptr) {
          return new_node;
        }
      }
    }
    MS_LOG(DEBUG) << "Cannot shrink output, transform_getitem_switch, node: " << node->DebugString()
                  << ", switch: " << switch_->DebugString();
    auto new_g1 = getitem_transform_(node, g1_, idx_);
    auto new_g2 = getitem_transform_(node, g2_, idx_);
    MS_LOG(DEBUG) << "Original fg1: " << g1_->ToString() << ", new_fg1: " << new_g1->ToString();
    MS_LOG(DEBUG) << "Original fg2: " << g2_->ToString() << ", new_fg2: " << new_g2->ToString();
    auto sw_node = fg->NewCNode({NewValueNode(prim::kPrimSwitch), x_, NewValueNode(new_g1), NewValueNode(new_g2)});
    (void)args_.insert(args_.begin(), sw_node);

    auto new_node = fg->NewCNode(args_);
    new_node->set_abstract(node->abstract());
    return new_node;
  }
  AnfNodePtr ShrinkFuncGraphOutput(const AnfNodePtr &node, const CNodePtr &switch_call_cnode,
                                   const std::vector<internal::TpCNodeAndIndex> &tp_cnodes_and_index) {
    const auto &manager = node->func_graph()->manager();
    MS_EXCEPTION_IF_NULL(manager);
    auto switch_cnode = switch_->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(switch_cnode);
    std::vector<int64_t> index_vector;
    (void)std::transform(tp_cnodes_and_index.begin(), tp_cnodes_and_index.end(), std::back_inserter(index_vector),
                         [](const auto &cnode_and_index) { return cnode_and_index.index; });
    const auto &iter1 = processed_fgs_.find(std::make_pair(g1_, index_vector));
    const auto &iter2 = processed_fgs_.find(std::make_pair(g2_, index_vector));
    if (iter1 != processed_fgs_.end() && iter2 != processed_fgs_.end()) {
      MS_LOG(DEBUG) << "fg output had been processed, no need to transform, node: " << node->DebugString()
                    << ", switch: " << switch_->DebugString();
      MS_LOG(DEBUG) << "Original fg1: " << g1_->ToString() << ", new_fg1: " << iter1->second->ToString();
      MS_LOG(DEBUG) << "Original fg2: " << g2_->ToString() << ", new_fg2: " << iter2->second->ToString();
      processed_nodes_.emplace(switch_);
      manager->SetEdge(switch_cnode, 2, NewValueNode(iter1->second));
      manager->SetEdge(switch_cnode, 3, NewValueNode(iter2->second));
      auto shrunk_abstract = internal::ShrinkAbstract(switch_call_cnode->abstract(), tp_cnodes_and_index);
      if (shrunk_abstract != nullptr) {
        switch_call_cnode->set_abstract(shrunk_abstract);
      }
      auto new_idx = internal::UpdateUserNodeIndex(switch_call_cnode, idx_, tp_cnodes_and_index);
      auto new_node =
        node->func_graph()->NewCNode({NewValueNode(prim::kPrimTupleGetItem), switch_call_cnode, NewValueNode(new_idx)});
      new_node->set_abstract(node->abstract());
      return new_node;
    }
    const auto &new_g1 = internal::ShrinkUnsedOutput(g1_, tp_cnodes_and_index);
    const auto &new_g2 = internal::ShrinkUnsedOutput(g2_, tp_cnodes_and_index);
    if (new_g1 != nullptr && new_g2 != nullptr) {
      MS_LOG(DEBUG) << "Shrink output. node: " << node->DebugString() << ", switch: " << switch_->DebugString();
      MS_LOG(DEBUG) << "Original fg1: " << g1_->ToString() << ", new_fg1: " << new_g1->ToString();
      MS_LOG(DEBUG) << "Original fg2: " << g2_->ToString() << ", new_fg2: " << new_g2->ToString();
      processed_nodes_.emplace(switch_);
      processed_fgs_.emplace(std::make_pair(g1_, index_vector), new_g1);
      processed_fgs_.emplace(std::make_pair(g2_, index_vector), new_g2);
      manager->SetEdge(switch_cnode, 2, NewValueNode(new_g1));
      manager->SetEdge(switch_cnode, 3, NewValueNode(new_g2));
      auto shrunk_abstract = internal::ShrinkAbstract(switch_call_cnode->abstract(), tp_cnodes_and_index);
      if (shrunk_abstract != nullptr) {
        switch_call_cnode->set_abstract(shrunk_abstract);
      }
      auto new_idx = internal::UpdateUserNodeIndex(switch_call_cnode, idx_, tp_cnodes_and_index);
      auto new_node =
        node->func_graph()->NewCNode({NewValueNode(prim::kPrimTupleGetItem), switch_call_cnode, NewValueNode(new_idx)});
      new_node->set_abstract(node->abstract());
      return new_node;
    }
    MS_LOG(DEBUG) << "Shrink failed. node: " << node->DebugString()
                  << ", switch_call: " << switch_call_cnode->DebugString();
    return nullptr;
  }
  void Visit(const AnfNodePtr &node) override {
    if (is_in_switch_ && x_ == nullptr) {
      x_ = node;
      return;
    }
    AnfVisitor::Visit(node);
  }

  void Visit(const CNodePtr &cnode) override {
    if (is_in_get_ && cnode->size() != 0) {
      auto &inputs = cnode->inputs();
      switch_ = inputs[0];
      (void)std::copy(inputs.begin() + 1, inputs.end(), std::back_inserter(args_));
    }
  }

  void Visit(const ValueNodePtr &vnode) override {
    if (is_in_get_) {
      idx_ = GetValue<int64_t>(vnode->value());
    }

    if (is_in_switch_) {
      auto g = GetValueNode<FuncGraphPtr>(vnode);
      if (g1_ == nullptr) {
        g1_ = g;
      } else {
        g2_ = g;
      }
    }
  }

  void Reset() {
    x_ = nullptr;
    g1_ = nullptr;
    g2_ = nullptr;
    switch_ = nullptr;
    args_.clear();
    is_in_get_ = false;
    is_in_switch_ = false;
  }

 private:
  size_t MultipleUseOfSwitch(const AnfNodePtr &switch_call, const FuncGraphPtr &fg,
                             std::vector<internal::TpCNodeAndIndex> *cnodes_and_index) const {
    auto switch_call_cnode = switch_call->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(switch_call_cnode);
    auto manager = fg->manager();
    MS_EXCEPTION_IF_NULL(manager);
    auto &cnode_and_index_vector = *cnodes_and_index;
    std::set<int64_t> index_set;
    std::size_t total_usage = 0;
    auto &node_users_map = manager->node_users();
    auto it = node_users_map.find(switch_call);
    if (it == node_users_map.end()) {
      return 0;
    }
    auto &node_users = it->second;
    // If switch was used by more than 1 tuple_getitem nodes, this pass shouldn't be execute.
    for (auto user : node_users) {
      if (IsPrimitiveCNode(user.first, prim::kPrimTupleGetItem)) {
        auto cnode = user.first->cast<CNodePtr>();
        constexpr auto kInputIndex = 2;
        if (cnode->input(kInputIndex)->isa<ValueNode>()) {
          const auto &idx_node = cnode->input(kInputIndex)->cast<ValueNodePtr>();
          MS_EXCEPTION_IF_NULL(idx_node);
          auto idx = GetValue<int64_t>(idx_node->value());
          cnode_and_index_vector.push_back({cnode, idx});
          index_set.insert(idx);
          total_usage++;
        } else {
          MS_LOG(EXCEPTION) << "Tuple_getitem index is not valuenode, but: " << user.first->DebugString(2);
        }
      } else {
        MS_LOG(DEBUG) << "switch_call user is not tuple_getitem, user: " << user.first->DebugString(2);
      }
    }
    if (index_set.size() != total_usage) {
      MS_LOG(DEBUG) << "some index is duplicated, total_usage: " << total_usage;
      MS_LOG(DEBUG) << "index_set: ";
      for (auto idx : index_set) {
        MS_LOG(DEBUG) << " " << idx;
      }
    }
    // sort by index;
    std::sort(cnode_and_index_vector.begin(), cnode_and_index_vector.end(),
              [](const auto &tp1, const auto &tp2) { return tp1.index < tp2.index; });
    return node_users.size();
  }

  static bool inline ExistEnvNode(const FuncGraphPtr &fg) {
    MS_EXCEPTION_IF_NULL(fg);
    auto &nodes = fg->value_nodes();
    return std::any_of(nodes.begin(), nodes.end(), [](const auto &node) {
      return IsPrimitive(node.first, prim::kPrimEnvSetItem) || IsPrimitive(node.first, prim::kPrimEnvGetItem);
    });
  }

  static bool inline ExistEnvNodeInTupleItem(const FuncGraphPtr &fg) {
    MS_EXCEPTION_IF_NULL(fg);
    const auto &output = fg->output();
    if (!IsPrimitiveCNode(output, prim::kPrimMakeTuple)) {
      return false;
    }
    const auto &cnode = output->cast<CNodePtr>();
    const auto &inputs = cnode->inputs();
    return std::any_of(inputs.cbegin() + 1, inputs.cend(), [](const auto &input) {
      auto sub_fg = GetValueNode<FuncGraphPtr>(input);
      if (sub_fg != nullptr && ExistEnvNode(sub_fg)) {
        return true;
      }
      return false;
    });
  }

  int64_t idx_{-1};
  AnfNodePtr switch_{nullptr}, x_{nullptr};
  FuncGraphPtr g1_{nullptr}, g2_{nullptr};
  bool is_in_get_{false}, is_in_switch_{false};
  std::vector<AnfNodePtr> args_{};
  std::set<AnfNodePtr> processed_nodes_;
  std::unordered_map<std::pair<FuncGraphPtr, std::vector<int64_t>>, FuncGraphPtr,
                     internal::FuncGraphIntVectorPairHasher>
    processed_fgs_;
  internal::GetitemTransform getitem_transform_;
};

// {prim::kPrimTupleGetItem, {{prim::kPrimSwitchLayer, X, {prim::kPrimMakeTuple, G1, G2...}}, Xs}, C}
class IncorporateGetitemSwitchLayerA : public AnfVisitor {
 public:
  IncorporateGetitemSwitchLayerA() : getitem_transform_() {}
  ~IncorporateGetitemSwitchLayerA() override = default;

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    is_in_get_ = true;
    AnfVisitor::Match(prim::kPrimTupleGetItem, {IsCNode, IsValueNode<Int64Imm>})(node);
    is_in_get_ = false;

    auto fg = node->func_graph();
    if (idx_ == -1 || switch_layer_ == nullptr || fg == nullptr) {
      return nullptr;
    }

    is_in_switch_ = true;
    AnfVisitor::Match(prim::kPrimSwitchLayer, {IsNode, IsCNode})(switch_layer_);
    is_in_switch_ = false;

    if (graphs_.empty()) {
      return nullptr;
    }

    std::vector<AnfNodePtr> layers;
    for (auto &graph : graphs_) {
      auto fg_transform = getitem_transform_(node, graph, idx_);
      if (fg_transform == nullptr) {
        return nullptr;
      }
      layers.push_back(NewValueNode(fg_transform));
    }
    auto layers_node = fg->NewCNode(prim::kPrimMakeTuple, layers);
    std::vector<AnfNodePtr> sw_args{NewValueNode(prim::kPrimSwitchLayer), x_, layers_node};
    auto sw_node = fg->NewCNode(sw_args);
    (void)args_.insert(args_.begin(), sw_node);

    return fg->NewCNode(args_);
  }

  void Visit(const AnfNodePtr &node) override {
    if (is_in_switch_ && x_ == nullptr) {
      x_ = node;
      return;
    }
    AnfVisitor::Visit(node);
  }

  void Visit(const CNodePtr &cnode) override {
    if (is_in_get_ && cnode->size() != 0) {
      auto &inputs = cnode->inputs();
      switch_layer_ = inputs[0];
      (void)std::copy(inputs.begin() + 1, inputs.end(), std::back_inserter(args_));
    }
    if (is_in_switch_ && cnode->size() >= 2) {
      auto &inputs = cnode->inputs();
      if (IsPrimitiveCNode(cnode, prim::kPrimMakeTuple) && IsValueNode<FuncGraph>(inputs[1])) {
        (void)std::transform(inputs.begin() + 1, inputs.end(), std::back_inserter(graphs_),
                             [](const AnfNodePtr &vnode) { return GetValueNode<FuncGraphPtr>(vnode); });
      }
    }
  }

  void Visit(const ValueNodePtr &vnode) override {
    if (is_in_get_) {
      idx_ = GetValue<int64_t>(vnode->value());
    }
  }

  void Reset() {
    x_ = nullptr;
    graphs_.clear();
    switch_layer_ = nullptr;
    args_.clear();
    is_in_get_ = false;
    is_in_switch_ = false;
  }

 private:
  int64_t idx_{-1};
  AnfNodePtr switch_layer_{nullptr}, x_{nullptr};
  std::vector<FuncGraphPtr> graphs_{};
  bool is_in_get_{false}, is_in_switch_{false};
  std::vector<AnfNodePtr> args_{};
  internal::GetitemTransform getitem_transform_;
};

// {prim::kPrimTupleGetItem, {{{prim::kPrimSwitchLayer, X, {prim::kPrimMakeTuple, G1, G2...}}, Xs}, Ys}, C}
class IncorporateGetitemSwitchLayerB : public AnfVisitor {
 public:
  IncorporateGetitemSwitchLayerB() : getitem_transform_() {}
  ~IncorporateGetitemSwitchLayerB() override = default;

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    is_in_get_ = true;
    AnfVisitor::Match(prim::kPrimTupleGetItem, {IsCNode, IsValueNode<Int64Imm>})(node);
    is_in_get_ = false;

    auto fg = node->func_graph();
    if (idx_ == -1 || switch_layer_call_ == nullptr || !switch_layer_call_->isa<CNode>() || fg == nullptr) {
      return nullptr;
    }

    auto &switch_layer_call_inputs = switch_layer_call_->cast<CNodePtr>()->inputs();
    (void)std::copy(switch_layer_call_inputs.begin() + 1, switch_layer_call_inputs.end(), std::back_inserter(args_));

    is_in_switch_ = true;
    AnfVisitor::Match(prim::kPrimSwitchLayer, {IsNode, IsCNode})(switch_layer_call_inputs[0]);
    is_in_switch_ = false;

    if (graphs_.empty()) {
      return nullptr;
    }

    std::vector<AnfNodePtr> layers;
    for (auto &graph : graphs_) {
      auto fg_transform = getitem_transform_(node, graph, idx_);
      if (fg_transform == nullptr) {
        return nullptr;
      }
      layers.push_back(NewValueNode(fg_transform));
    }
    auto layers_node = fg->NewCNode(prim::kPrimMakeTuple, layers);
    std::vector<AnfNodePtr> sw_args{NewValueNode(prim::kPrimSwitchLayer), x_, layers_node};
    auto sw_node = fg->NewCNode(sw_args);
    (void)args_.insert(args_.begin(), sw_node);
    auto call_switch_layer = fg->NewCNode(args_);
    (void)outer_call_args_.insert(outer_call_args_.begin(), call_switch_layer);
    return fg->NewCNode(outer_call_args_);
  }

  void Visit(const AnfNodePtr &node) override {
    if (is_in_switch_ && x_ == nullptr) {
      x_ = node;
      return;
    }
    AnfVisitor::Visit(node);
  }

  void Visit(const CNodePtr &cnode) override {
    if (is_in_get_ && cnode->size() != 0) {
      auto &inputs = cnode->inputs();
      switch_layer_call_ = inputs[0];
      (void)std::copy(inputs.begin() + 1, inputs.end(), std::back_inserter(outer_call_args_));
    }
    if (is_in_switch_ && cnode->size() >= 2) {
      auto &inputs = cnode->inputs();
      if (IsPrimitiveCNode(cnode, prim::kPrimMakeTuple) && IsValueNode<FuncGraph>(inputs[1])) {
        (void)std::transform(inputs.begin() + 1, inputs.end(), std::back_inserter(graphs_),
                             [](const AnfNodePtr &vnode) { return GetValueNode<FuncGraphPtr>(vnode); });
      }
    }
  }

  void Visit(const ValueNodePtr &vnode) override {
    if (is_in_get_) {
      idx_ = GetValue<int64_t>(vnode->value());
    }
  }

  void Reset() {
    x_ = nullptr;
    graphs_.clear();
    switch_layer_call_ = nullptr;
    args_.clear();
    outer_call_args_.clear();
    is_in_get_ = false;
    is_in_switch_ = false;
  }

 private:
  int64_t idx_{-1};
  AnfNodePtr switch_layer_call_{nullptr}, x_{nullptr};
  std::vector<FuncGraphPtr> graphs_{};
  bool is_in_get_{false}, is_in_switch_{false};
  std::vector<AnfNodePtr> args_{};
  std::vector<AnfNodePtr> outer_call_args_{};
  internal::GetItemTransformACrossGraph getitem_transform_;
};

class IncorporateGetitemSet : public OptimizerCaller {
 public:
  IncorporateGetitemSet()
      : incorporate_getitem_(std::make_shared<IncorporateGetitem>()),
        incorporate_getitem_depend_(std::make_shared<IncorporateGetitemDepend>()),
        incorporate_getitem_switch_(std::make_shared<IncorporateGetitemSwitch>()),
        incorporate_getitem_switch_layer_a_(std::make_shared<IncorporateGetitemSwitchLayerA>()),
        incorporate_getitem_switch_layer_b_(std::make_shared<IncorporateGetitemSwitchLayerB>()) {
    eliminaters_.emplace_back(incorporate_getitem_);
    eliminaters_.emplace_back(incorporate_getitem_depend_);
    eliminaters_.emplace_back(incorporate_getitem_switch_);
    eliminaters_.emplace_back(incorporate_getitem_switch_layer_a_);
    eliminaters_.emplace_back(incorporate_getitem_switch_layer_b_);
  }
  ~IncorporateGetitemSet() = default;

  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) override {
    AnfNodePtr new_node;
    for (auto &eliminater : eliminaters_) {
      new_node = (*eliminater)(optimizer, node);
      if (new_node != nullptr) {
        return new_node;
      }
    }
    return nullptr;
  }

 private:
  OptimizerCallerPtr incorporate_getitem_, incorporate_getitem_depend_, incorporate_getitem_switch_,
    incorporate_getitem_switch_layer_a_, incorporate_getitem_switch_layer_b_;
  std::vector<OptimizerCallerPtr> eliminaters_{};
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_INCORPORATE_GETITEM_H_
