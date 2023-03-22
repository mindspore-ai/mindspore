/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_IR_MANAGER_H_
#define MINDSPORE_CORE_IR_MANAGER_H_

#include <set>
#include <map>
#include <list>
#include <string>
#include <vector>
#include <utility>
#include <memory>
#include <functional>

#include "utils/any.h"
#include "utils/misc.h"
#include "utils/signal.h"
#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "utils/compact_set.h"
#include "utils/ordered_set.h"
#include "utils/ordered_map.h"
#include "ir/anf.h"
#include "ir/graph_utils.h"
#include "utils/hashing.h"
#include "base/base_ref.h"

namespace mindspore {
namespace change {
struct ChangeCounter;
struct Change {
  virtual ~Change() = default;
  virtual void Apply(ChangeCounter *counter) = 0;
};
using ChangePtr = std::unique_ptr<Change>;
}  // namespace change

class FuncGraphTransaction;
class FuncGraphManager;
class FuncGraphPassIndex;
using FuncGraphManagerPtr = std::shared_ptr<FuncGraphManager>;
using FuncGraphIndexPtr = std::shared_ptr<FuncGraphPassIndex>;

using AnfNodeIndexSet = CompactSet<std::pair<AnfNodePtr, int>>;
using NodeUsersMap = mindspore::HashMap<AnfNodePtr, AnfNodeIndexSet, PointerHash<AnfNodePtr>>;
using FuncGraphIndexMap = mindspore::HashMap<FuncGraphPtr, FuncGraphIndexPtr>;

using FuncGraphSetPair = std::pair<FuncGraphPtr, FuncGraphSet>;
using FuncGraphSetPtr = std::shared_ptr<FuncGraphSet>;

// manage the func graphs.
// if no manager exist, just create one and associate it to all func graphs; else reuse simply.
// func_graph, be managed graph
// manage: if true, created manager will be set in func_graph
// FuncGraphManagerPtr: return created manager
MS_CORE_API FuncGraphManagerPtr Manage(FuncGraphPtr func_graph, bool manage = true);

MS_CORE_API FuncGraphManagerPtr Manage(const std::vector<FuncGraphPtr> &func_graphs, bool manage = true);

MS_CORE_API FuncGraphManagerPtr MakeManager(const std::vector<FuncGraphPtr> &func_graphs = {}, bool manage = true);

struct Signals {
  Signal<void()> InvalidateComputer;
};

using CNodeIndexPair = std::pair<AnfNodePtr, int>;
using CNodeIndexPairPtr = std::shared_ptr<CNodeIndexPair>;
using FuncGraphToFuncGraphSetMap = OrderedMap<FuncGraphPtr, FuncGraphSet>;

// For Fast Pass
class FuncGraphPassIndex {
 public:
  FuncGraphPassIndex() : has_gen_index_(false) {}
  void set_has_gen_index(bool is_gen_index) { has_gen_index_ = is_gen_index; }
  bool has_gen_index() const { return has_gen_index_; }
  mindspore::HashMap<AnfNodePtr, FuncGraphWeakPtr> node_to_fg_;
  mindspore::HashMap<std::string, std::set<AnfNodePtr>> name_to_cnode_;
  mindspore::HashMap<AnfNodePtr, std::set<AnfNodePtr>> subgraph_out_caller_map_;
  mindspore::HashMap<AnfNodePtr, size_t> node_degree_;

 private:
  bool has_gen_index_;
};

// analysis base class, graphs analysis which need dynamic compute by DepCollector in each read
class DepComputer {
 public:
  explicit DepComputer(const FuncGraphManager *manager);
  virtual ~DepComputer() { manager_ = nullptr; }

  virtual size_t size() const { return 0; }

  void Reset() {
    ExtraReset();
    validate_ = false;
    func_graphs_validate_.clear();
  }

  void OnInvalidateComputer() { Reset(); }

  void Recompute();

  void Recompute(const FuncGraphPtr &fg);

  bool IsValidate() const { return validate_; }

  bool IsValidate(const FuncGraphPtr &fg) { return func_graphs_validate_[fg]; }

 protected:
  // subclass can reset their own member;
  virtual void ExtraReset() {}
  // subclass do the real compute
  virtual void RealRecompute() {}
  virtual void RealRecompute(FuncGraphPtr) {}

  const FuncGraphManager *manager_;
  bool validate_;
  OrderedMap<FuncGraphPtr, bool> func_graphs_validate_;

 private:
  friend FuncGraphManager;
};

// graph g's all direct or proxy parents
class FuncGraphParentsTotalComputer final : public DepComputer {
 public:
  explicit FuncGraphParentsTotalComputer(const FuncGraphManager *m) : DepComputer(m) {}
  ~FuncGraphParentsTotalComputer() override = default;

  FuncGraphToFuncGraphSetMap &func_graph_parents_total_analysis() { return func_graph_parents_total_analysis_; }

  size_t size() const override { return func_graph_parents_total_analysis_.size(); }

  FuncGraphToFuncGraphSetMap func_graph_parents_total_analysis_;

 protected:
  void ExtraReset() override { func_graph_parents_total_analysis_.clear(); }

  void RealRecompute(FuncGraphPtr fg) override;

 private:
  FuncGraphSetPtr SeekParents(const FuncGraphPtr &fg, mindspore::HashMap<FuncGraphPtr, FuncGraphSetPtr> *seen_fgs);
};

using FuncGraphToFuncGraphMap = OrderedMap<FuncGraphPtr, FuncGraphPtr>;

// graph's nearest parent in parents total
class ParentComputer final : public DepComputer {
 public:
  explicit ParentComputer(const FuncGraphManager *m) : DepComputer(m) {}
  ~ParentComputer() override = default;

  FuncGraphToFuncGraphMap &parent_analysis() { return parent_analysis_; }

  size_t size() const override { return parent_analysis_.size(); }

  FuncGraphToFuncGraphMap parent_analysis_;

 protected:
  void ExtraReset() override { parent_analysis_.clear(); }

  void RealRecompute(FuncGraphPtr fg) override;
};

// graph's children graph except self
class ChildrenComputer final : public DepComputer {
 public:
  explicit ChildrenComputer(const FuncGraphManager *m) : DepComputer(m) {}
  ~ChildrenComputer() override = default;

  FuncGraphToFuncGraphSetMap &children_analysis() { return children_analysis_; }

  size_t size() const override { return children_analysis_.size(); }

  FuncGraphToFuncGraphSetMap children_analysis_;

 protected:
  void ExtraReset() override { children_analysis_.clear(); }

  void RealRecompute(FuncGraphPtr fg) override;
};

// graph's children graph include self
class ScopeComputer final : public DepComputer {
 public:
  explicit ScopeComputer(const FuncGraphManager *m) : DepComputer(m) {}
  ~ScopeComputer() override = default;

  FuncGraphToFuncGraphSetMap &scope_analysis() { return scope_analysis_; }

  size_t size() const override { return scope_analysis_.size(); }

  FuncGraphToFuncGraphSetMap scope_analysis_;

 protected:
  void ExtraReset() override { scope_analysis_.clear(); }

  void RealRecompute(FuncGraphPtr fg) override;
};

using FVTotalMap = OrderedMap<FuncGraphPtr, OrderedMap<BaseRef, int, BaseRefHash>>;

class FVTotalComputer final : public DepComputer {
 public:
  explicit FVTotalComputer(const FuncGraphManager *m) : DepComputer(m) {}
  ~FVTotalComputer() override = default;

  FVTotalMap &fv_total_analysis() { return fv_total_analysis_; }

  size_t size() const override { return fv_total_analysis_.size(); }

  FVTotalMap fv_total_analysis_;

 protected:
  void ExtraReset() override { fv_total_analysis_.clear(); }

  void RealRecompute() override;
};

class FuncGraphsUsedTotalComputer final : public DepComputer {
 public:
  explicit FuncGraphsUsedTotalComputer(const FuncGraphManager *m) : DepComputer(m) {}
  ~FuncGraphsUsedTotalComputer() override = default;

  FuncGraphToFuncGraphSetMap &func_graph_used_total_analysis() { return func_graph_used_total_analysis_; }

  size_t size() const override { return func_graph_used_total_analysis_.size(); }

  FuncGraphToFuncGraphSetMap func_graph_used_total_analysis_;

 protected:
  void ExtraReset() override { func_graph_used_total_analysis_.clear(); }

  void RealRecompute(FuncGraphPtr fg) override;
};

using FuncGraphToBoolMap = OrderedMap<FuncGraphPtr, bool>;
using RecursiveMap = OrderedMap<FuncGraphPtr, std::shared_ptr<std::list<FuncGraphPtr>>>;

class RecursiveComputer final : public DepComputer {
 public:
  explicit RecursiveComputer(const FuncGraphManager *m) : DepComputer(m) {}
  ~RecursiveComputer() override = default;

  RecursiveMap &recursive_map() { return recursive_map_; }
  FuncGraphToBoolMap &recursive_analysis() { return recursive_analysis_; }

  void CheckRecursiveGraphs(const FuncGraphPtr &fg, std::list<FuncGraphPtr> *trace);

  size_t size() const override { return recursive_analysis_.size(); }

  RecursiveMap recursive_map_;
  FuncGraphToBoolMap recursive_analysis_;

 protected:
  void ExtraReset() override {
    recursive_analysis_.clear();
    recursive_map_.clear();
  }

  void RealRecompute(FuncGraphPtr fg) override;
};

class FuncGraphMetaFgPrimTotalComputer final : public DepComputer {
 public:
  explicit FuncGraphMetaFgPrimTotalComputer(const FuncGraphManager *m) : DepComputer(m) {}
  ~FuncGraphMetaFgPrimTotalComputer() override = default;

  FuncGraphToBoolMap &meta_fg_prim_total_analysis() { return meta_fg_prim_total_analysis_; }

  size_t size() const override { return meta_fg_prim_total_analysis_.size(); }

  FuncGraphToBoolMap meta_fg_prim_total_analysis_;

 protected:
  void ExtraReset() override { meta_fg_prim_total_analysis_.clear(); }

  void RealRecompute(FuncGraphPtr fg) override;

  bool SeekMetaFgPrim(const FuncGraphPtr &fg, SeenNum seen_num);
};

class MS_CORE_API FuncGraphManager : public std::enable_shared_from_this<FuncGraphManager> {
 public:
  explicit FuncGraphManager(const std::vector<FuncGraphPtr> &roots, bool manage = true);
  virtual ~FuncGraphManager() {
    if (is_manage_) {
      RemoveRoots();
    }
    Clear();
  }

  void Reset();
  void Init();
  void Clear() noexcept;
  void AddFuncGraph(const FuncGraphPtr &func_graph, bool is_root = false);
  void KeepRoots(const std::vector<FuncGraphPtr> &roots = {});
  void RemoveRoots();
  void SetParameters(const FuncGraphPtr &fg, const std::vector<AnfNodePtr> &parameters);
  void AddParameter(const FuncGraphPtr &fg, const AnfNodePtr &parameter);
  void InsertFrontParameter(const FuncGraphPtr &fg, const AnfNodePtr &parameter);
  void MaybeDropFuncGraphs(const FuncGraphSet &func_graphs, bool ignore_users = false);
  bool Replace(const AnfNodePtr &old_node, const AnfNodePtr &new_node);
  bool Replace(const AnfNodePtr &old_node, const AnfNodePtr &new_node, const AnfNodePtr &mask_node);
  void SetEdge(const AnfNodePtr &node, int index, const AnfNodePtr &value);
  void AddEdge(const AnfNodePtr &node, const AnfNodePtr &value);
  void MoveAllCNodeDropGraph(const FuncGraphPtr &source, const FuncGraphPtr &target, const AnfNodePtr &call_node,
                             const ScopePtr &scope);

  FuncGraphTransaction Transact();
  void CommitChanges(std::vector<change::ChangePtr> &&changes);

  bool IsManaged() const { return is_manage_; }

  const FuncGraphSet &roots() const { return roots_; }

  const FuncGraphSet &func_graphs() const { return func_graphs_; }

  AnfNodeSet &all_nodes() { return all_nodes_; }

  NodeUsersMap &node_users() { return node_users_; }

  const NodeUsersMap &node_users() const { return node_users_; }

  FVTotalMap &free_variables_total() const;

  FuncGraphSet &func_graph_parents_total(const FuncGraphPtr &fg) const;

  FuncGraphSet &scopes(const FuncGraphPtr &fg) const;

  FuncGraphPtr parent(const FuncGraphPtr &fg) const;

  FuncGraphSet &children(const FuncGraphPtr &fg) const;

  FuncGraphSet &func_graphs_used_total(const FuncGraphPtr &fg) const;

  const FuncGraphIndexPtr &func_graph_index(const FuncGraphPtr &fg) const;

  bool recursive(const FuncGraphPtr &fg) const;
  std::shared_ptr<std::list<FuncGraphPtr>> recursive_graphs(const FuncGraphPtr &fg) const;

  bool func_graph_meta_fg_prim_total(const FuncGraphPtr &fg) const;

  std::shared_ptr<Signals> signals() const { return signals_; }

  // Static Analysis
  NodeUsersMap node_users_;
  AnfNodeSet all_nodes_;  // managed nodes

  // Dynamic Analysis
  std::shared_ptr<ParentComputer> func_graph_parent_;

 private:
  // Erase OneGraph From Manager
  void EraseOneGraph(const FuncGraphPtr &fg);
  void AddIntoManaged(const FuncGraphPtr &fg);
  void ProcessEdgeAdd(const AnfNodePtr &node, int index, const AnfNodePtr &input);
  void ProcessEdgeRemove(const AnfNodePtr &node, int index, const AnfNodePtr &input);
  void ProcessInputsEdgeAdd(const CNodePtr &cnode);
  void ProcessInputsEdgeRemove(const CNodePtr &cnode);
  void AcquireNodes(std::vector<AnfNodePtr> &&nodes);
  FuncGraphSet MaybeDropNodes(std::vector<AnfNodePtr> &&nodes);
  void OnEdgeAdded(const AnfNodePtr &node, int index, const AnfNodePtr &input);
  void OnEdgeRemoved(const AnfNodePtr &node, int index, const AnfNodePtr &input);
  void MoveAllNodes(const FuncGraphPtr &source, const FuncGraphPtr &target);

  FuncGraphSet roots_;                   // Managed roots.
  FuncGraphSet func_graphs_;             // Managed func graphs.
  FuncGraphIndexMap func_graphs_index_;  // For Fast Pass

  std::shared_ptr<Signals> signals_;

  // Dynamic Analysis
  std::shared_ptr<FuncGraphParentsTotalComputer> func_graph_parents_total_;
  std::shared_ptr<ChildrenComputer> children_;
  std::shared_ptr<ScopeComputer> scopes_;
  std::shared_ptr<FVTotalComputer> free_variables_total_;
  std::shared_ptr<FuncGraphsUsedTotalComputer> func_graphs_used_total_;
  std::shared_ptr<RecursiveComputer> recursive_;
  std::shared_ptr<FuncGraphMetaFgPrimTotalComputer> meta_fg_prim_total_;

  bool is_manage_;
};

class MS_CORE_API FuncGraphTransaction {
 public:
  explicit FuncGraphTransaction(FuncGraphManager *manager) : manager_(manager) {}
  FuncGraphTransaction() : manager_(nullptr) {}
  ~FuncGraphTransaction() = default;

  FuncGraphTransaction(const FuncGraphTransaction &other) = delete;
  FuncGraphTransaction &operator=(const FuncGraphTransaction &other) = delete;

  FuncGraphTransaction(FuncGraphTransaction &&other) = default;
  FuncGraphTransaction &operator=(FuncGraphTransaction &&other) = default;

  // set parameters of a func graph
  void SetParameters(FuncGraphPtr fg, const std::vector<AnfNodePtr> &params);
  void AddParameter(FuncGraphPtr fg, const AnfNodePtr &param);
  void InsertFrontParameter(FuncGraphPtr fg, const AnfNodePtr &param);

  // replace old_node with new_node
  bool Replace(const AnfNodePtr &old_node, const AnfNodePtr &new_node);
  bool Replace(const AnfNodePtr &old_node, const AnfNodePtr &new_node, const AnfNodePtr &mask_node);

  // set edge, i.e., declare setting node.inputs[key] to value.
  void SetEdge(const AnfNodePtr &src_node, int k, const AnfNodePtr &v);
  // Add edge, i.e., append value to node.inputs.
  void AddEdge(const AnfNodePtr &src_node, const AnfNodePtr &v);

  // commit all changes
  void Commit();

 private:
  FuncGraphManager *manager_;
  std::vector<change::ChangePtr> changes_;
};

inline FuncGraphTransaction FuncGraphManager::Transact() { return FuncGraphTransaction(this); }

}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_MANAGER_H_
