/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_IR_MANAGER_H_
#define MINDSPORE_CCSRC_IR_MANAGER_H_

#include <unordered_set>
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
#include "utils/ordered_set.h"
#include "utils/ordered_map.h"
#include "utils/graph_utils.h"
#include "utils/counter.h"
#include "utils/hashing.h"
#include "ir/anf.h"

namespace mindspore {
struct Change;
class FuncGraphTransaction;
class FuncGraphManager;
using FuncGraphManagerPtr = std::shared_ptr<FuncGraphManager>;

struct AnfNodeIndexPairHasher {
  std::size_t operator()(const std::pair<AnfNodePtr, int> &p1) const {
    return std::hash<const AnfNode *>{}(p1.first.get());
  }
};

struct AnfNodeIndexPairEqual {
  bool operator()(const std::pair<AnfNodePtr, int> &lhs, const std::pair<AnfNodePtr, int> &rhs) const {
    return lhs == rhs;
  }
};
using AnfNodeIndexSet = OrderedSet<std::pair<AnfNodePtr, int>, AnfNodeIndexPairHasher, AnfNodeIndexPairEqual>;
// NodeUsersMap, for node B input i use node A, it will be one item in map with key: A, and value: (B, i)
using NodeUsersMap = OrderedMap<AnfNodePtr, AnfNodeIndexSet>;
using FuncGraphSetPair = std::pair<FuncGraphPtr, FuncGraphSet>;
using FuncGraphSetPtr = std::shared_ptr<FuncGraphSet>;
using EdgeTuple = std::pair<AnfNodePtr, std::pair<int, AnfNodePtr>>;
struct EdgeTupleHasher {
  std::size_t operator()(const EdgeTuple &p1) const {
    return hash_combine({std::hash<AnfNode *>{}(p1.first.get()), std::hash<int>{}(p1.second.first),
                         std::hash<AnfNode *>{}(p1.second.second.get())});
  }
};

struct EdgeTupleEqual {
  bool operator()(const EdgeTuple &lhs, const EdgeTuple &rhs) const {
    return lhs.first == rhs.first && lhs.second.first == rhs.second.first && lhs.second.second == rhs.second.second;
  }
};
using EdgeTupleCounter = Counter<EdgeTuple, EdgeTupleHasher, EdgeTupleEqual>;
// manage the func graphs.
// if no manager exist, just create one and associate it to all func graphs; else reuse simply.
// func_graph, be managed graph
// manage: if true, created manager will be set in func_graph
// FuncGraphManagerPtr: return created manager
FuncGraphManagerPtr Manage(FuncGraphPtr func_graph, bool manage = true);

FuncGraphManagerPtr Manage(const std::vector<FuncGraphPtr> &func_graphs, bool manage = true);

FuncGraphManagerPtr MakeManager(const std::vector<FuncGraphPtr> &func_graphs = {}, bool manage = true);

struct Signals {
  Signal<void(FuncGraphPtr)> AddFuncGraph;
  Signal<void(FuncGraphPtr)> DropFuncGraph;
  Signal<void(AnfNodePtr)> AddNode;
  Signal<void(AnfNodePtr)> DropNode;
  Signal<void(AnfNodePtr, int, AnfNodePtr)> AddEdge;
  Signal<void(AnfNodePtr, int, AnfNodePtr)> DropEdge;
  Signal<void(FuncGraphPtr, FuncGraphPtr)> MoveAllCNode;
  Signal<void()> InvalidateCollector;
  Signal<void()> InvalidateComputer;
};

enum EdgeProcessDirection { kDecEdge = -1, kIncEdge = 1 };

using CNodeIndexPair = std::pair<AnfNodePtr, int>;
using CNodeIndexPairPtr = std::shared_ptr<CNodeIndexPair>;

using FuncGraphToFuncGraphCounterMap = OrderedMap<FuncGraphPtr, OrderedMap<FuncGraphPtr, int>>;
template <typename ValueT, class CollectorHash = std::hash<ValueT>, class CollectorEqual = std::equal_to<ValueT>>
using FuncGraphToAnfNodeCounterMap = OrderedMap<FuncGraphPtr, OrderedMap<ValueT, int, CollectorHash, CollectorEqual>>;

// analysis base class
class FuncGraphAnalysis {
 public:
  explicit FuncGraphAnalysis(const FuncGraphManager *const manager);

  virtual ~FuncGraphAnalysis() { manager_ = nullptr; }

  virtual size_t size() const { return 0; }

  virtual void OnAddFuncGraph(FuncGraphPtr) {}

  virtual void OnDropFuncGraph(FuncGraphPtr) {}

  virtual void OnMoveAllCNode(FuncGraphPtr, FuncGraphPtr) {}

 protected:
  // subclass can reset their own member;
  virtual void ExtraReset() {}

  virtual void OnAddNode(AnfNodePtr n) {}

  virtual void OnDropNode(AnfNodePtr n) {}

  virtual void OnAddEdge(AnfNodePtr, int, AnfNodePtr) {}

  virtual void OnDropEdge(AnfNodePtr, int, AnfNodePtr) {}

  const FuncGraphManager *manager_;
  bool include_func_graph_none_;
};

using FuncGraphToAnfNodeMap = OrderedMap<FuncGraphPtr, AnfNodeSet>;

struct CNodeIndexHasher {
  std::size_t operator()(const CNodeIndexPairPtr pair) const {
    MS_EXCEPTION_IF_NULL(pair);
    MS_EXCEPTION_IF_NULL(pair->first);
    return hash_combine(pair->first->hash(), std::hash<int>()(pair->second));
  }
};

struct CNodeIndexEqual {
  bool operator()(const CNodeIndexPairPtr lhs, const CNodeIndexPairPtr rhs) const {
    if (lhs == nullptr || rhs == nullptr) {
      return false;
    }
    if (lhs == rhs) {
      return true;
    }
    if (lhs->first != rhs->first) {
      return false;
    }
    if (lhs->second != rhs->second) {
      return false;
    }
    return true;
  }
};

// graphs analysis which compute in write, read needn't recompute
class DepCollector : public FuncGraphAnalysis {
 public:
  explicit DepCollector(const FuncGraphManager *manager);
  ~DepCollector() override = default;

  void Reset() { ExtraReset(); }
  void OnInvalidateCollector() { Reset(); }

 protected:
  // inherit from FuncGraphAnalysis
  void OnAddEdge(AnfNodePtr node, int index, AnfNodePtr inp) override;
  void OnDropEdge(AnfNodePtr node, int index, AnfNodePtr inp) override;
  // subclass can override;
  virtual void OnModEdge(AnfNodePtr, int, AnfNodePtr, EdgeProcessDirection) {}
};

class CounterFuncGraphCollector : public DepCollector {
 public:
  explicit CounterFuncGraphCollector(const FuncGraphManager *m) : DepCollector(m) {}
  ~CounterFuncGraphCollector() override = default;
  FuncGraphToFuncGraphCounterMap &count_func_graphs_map() { return count_func_graphs_map_; }
  // inherit from FuncGraphAnalysis
  size_t size() const override { return count_func_graphs_map_.size(); }
  void OnAddFuncGraph(FuncGraphPtr fg) final { count_func_graphs_map_[fg] = OrderedMap<FuncGraphPtr, int>(); }
  void OnDropFuncGraph(FuncGraphPtr fg) final { (void)count_func_graphs_map_.erase(fg); }
  bool Inc(const FuncGraphPtr &func_graph, const FuncGraphPtr &key, int count);
  bool Dec(const FuncGraphPtr &func_graph, const FuncGraphPtr &key, int count);
  bool Mod(const FuncGraphPtr &func_graph, const FuncGraphPtr &key, int count);

  FuncGraphToFuncGraphCounterMap count_func_graphs_map_;

 protected:
  void ExtraReset() override { count_func_graphs_map_.clear(); }
};

template <typename ValueT, class CollectorHash = std::hash<ValueT>, class CollectorEqual = std::equal_to<ValueT>>
class CounterAnfNodeCollector : public DepCollector {
 public:
  explicit CounterAnfNodeCollector(const FuncGraphManager *m) : DepCollector(m) {}
  ~CounterAnfNodeCollector() override = default;
  FuncGraphToAnfNodeCounterMap<ValueT, CollectorHash, CollectorEqual> &count_nodes_map() { return count_nodes_map_; }

  size_t size() const override { return count_nodes_map_.size(); }
  void OnAddFuncGraph(FuncGraphPtr fg) final {
    count_nodes_map_[fg] = OrderedMap<ValueT, int, CollectorHash, CollectorEqual>();
  }
  void OnDropFuncGraph(FuncGraphPtr fg) final { (void)count_nodes_map_.erase(fg); }

  bool Inc(const FuncGraphPtr &func_graph, const ValueT &key, int count);
  bool Dec(const FuncGraphPtr &func_graph, const ValueT &key, int count);
  bool Mod(const FuncGraphPtr &func_graph, const ValueT &key, int count);

  FuncGraphToAnfNodeCounterMap<ValueT, CollectorHash, CollectorEqual> count_nodes_map_;

 protected:
  void ExtraReset() override { count_nodes_map_.clear(); }
};

using FuncGraphToFuncGraphSetMap = OrderedMap<FuncGraphPtr, FuncGraphSet>;

// graphs analysis which need dynamic compute by DepCollector in each read
class DepComputer : public FuncGraphAnalysis {
 public:
  explicit DepComputer(const FuncGraphManager *manager);
  ~DepComputer() override = default;

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

  void OnAddFuncGraph(FuncGraphPtr) final { Reset(); }

  void OnDropFuncGraph(FuncGraphPtr) final { Reset(); }

 protected:
  // subclass do the real compute
  virtual void RealRecompute() {}
  virtual void RealRecompute(FuncGraphPtr) {}

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
  FuncGraphSetPtr SeekParents(const FuncGraphPtr &fg, size_t seen_num);
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

class FVTotalComputer final : public DepComputer,
                              public CounterAnfNodeCollector<AnfNodePtr>,
                              public CounterFuncGraphCollector {
 public:
  explicit FVTotalComputer(const FuncGraphManager *m)
      : DepComputer(m), CounterAnfNodeCollector(m), CounterFuncGraphCollector(m) {}
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

class FuncGraphJTotalComputer final : public DepComputer {
 public:
  explicit FuncGraphJTotalComputer(const FuncGraphManager *m) : DepComputer(m) {}
  ~FuncGraphJTotalComputer() override = default;

  FuncGraphToBoolMap &j_total_analysis() { return j_total_analysis_; }

  size_t size() const override { return j_total_analysis_.size(); }

  FuncGraphToBoolMap j_total_analysis_;

 protected:
  void ExtraReset() override { j_total_analysis_.clear(); }

  void RealRecompute(FuncGraphPtr fg) override;
  bool SeekJ(const FuncGraphPtr &fg, size_t seen_num);
};

class FuncGraphManager : public std::enable_shared_from_this<FuncGraphManager> {
 public:
  explicit FuncGraphManager(const std::vector<FuncGraphPtr> &roots, bool manage = true);
  ~FuncGraphManager() {
    if (is_manage_) {
      RemoveRoots();
    }
  }

  void Reset();
  void Init();
  void Clear();
  void AddFuncGraph(FuncGraphPtr func_graph, bool is_root = false);
  void KeepRoots(const std::vector<FuncGraphPtr> &roots = {});
  void RemoveRoots();
  void SetParameters(const FuncGraphPtr &fg, const std::vector<AnfNodePtr> &parameters);
  void MaybeDropFuncGraphs(const FuncGraphSet &func_graphs, bool ignore_users = false);
  bool Replace(const AnfNodePtr &old_node, const AnfNodePtr &new_node);
  void SetEdge(const AnfNodePtr &node, int index, const AnfNodePtr &value);
  void MoveAllCNodeDropGraph(FuncGraphPtr source, FuncGraphPtr target, const ScopePtr &scope);

  FuncGraphTransaction Transact();
  void CommitChanges(const std::vector<Change> &changes);

  bool IsManaged() const { return is_manage_; }

  const FuncGraphSet &roots() const { return roots_; }

  const FuncGraphSet &func_graphs() const { return func_graphs_; }

  AnfNodeSet &all_nodes() { return all_nodes_; }

  NodeUsersMap &node_users() { return node_users_; }

  FVTotalMap &free_variables_total() const;

  FuncGraphSet &func_graph_parents_total(const FuncGraphPtr &fg) const;

  FuncGraphSet &scopes(const FuncGraphPtr &fg) const;

  FuncGraphPtr parent(const FuncGraphPtr &fg) const;

  FuncGraphSet &children(const FuncGraphPtr &fg) const;

  FuncGraphSet &func_graphs_used_total(const FuncGraphPtr &fg) const;

  bool recursive(const FuncGraphPtr &fg) const;
  std::shared_ptr<std::list<FuncGraphPtr>> recursive_graphs(const FuncGraphPtr &fg) const;

  bool func_graph_j_total(const FuncGraphPtr &fg) const;

  std::shared_ptr<Signals> signals() const { return signals_; }

  IncludeType Limit(const AnfNodePtr &node);

  // Static Analysis
  NodeUsersMap node_users_;
  AnfNodeSet all_nodes_;  // managed nodes

  // Dynamic Analysis
  std::shared_ptr<ParentComputer> func_graph_parent_;

 private:
  void AddIntoManaged(const FuncGraphPtr &fg);
  void ProcessEdge(AnfNodePtr node, int index, AnfNodePtr inp, EdgeProcessDirection direction);
  void ProcessInputs(const AnfNodePtr &node, EdgeProcessDirection direction);
  void AcquireNodes(const std::vector<AnfNodePtr> &nodes);
  FuncGraphSetPtr MaybeDropNodes(const std::vector<AnfNodePtr> &nodes);
  void ParseChanges(const std::vector<Change> &changes, EdgeTupleCounter *add_edges, EdgeTupleCounter *rm_edges,
                    Counter<AnfNodePtr> *adds, Counter<AnfNodePtr> *rms);
  void AddEdge(AnfNodePtr node, int index, AnfNodePtr input);
  void DropEdge(AnfNodePtr node, int index, AnfNodePtr input);
  void MoveAllNodes(FuncGraphPtr source, FuncGraphPtr target);

  FuncGraphSet roots_;        // managed roots
  FuncGraphSet func_graphs_;  // managed func graphs

  std::shared_ptr<Signals> signals_;

  // Dynamic Analysis
  std::shared_ptr<FuncGraphParentsTotalComputer> func_graph_parents_total_;
  std::shared_ptr<ChildrenComputer> children_;
  std::shared_ptr<ScopeComputer> scopes_;
  std::shared_ptr<FVTotalComputer> free_variables_total_;
  std::shared_ptr<FuncGraphsUsedTotalComputer> func_graphs_used_total_;
  std::shared_ptr<RecursiveComputer> recursive_;
  std::shared_ptr<FuncGraphJTotalComputer> j_total_;

  bool is_manage_;
};

class FuncGraphTransaction {
 public:
  explicit FuncGraphTransaction(FuncGraphManager *manager) : manager_(manager), changes_() {
    MS_EXCEPTION_IF_NULL(manager_);
    if (!manager_->IsManaged()) {
      MS_LOG(DEBUG) << "The manager is not managed yet";
    }
  }

  FuncGraphTransaction() : manager_(nullptr) {}
  ~FuncGraphTransaction() { manager_ = nullptr; }

  // set parameters of a func graph
  void SetParameters(FuncGraphPtr fg, const std::vector<AnfNodePtr> &params);

  // replace old_node with new_node
  bool Replace(const AnfNodePtr &old_node, const AnfNodePtr &new_node);

  // set esge, i.e., declare setting node.inputs[key] to value.
  void SetEdge(const AnfNodePtr &src_node, int k, const AnfNodePtr &v);

  // commit all changes
  void Commit();

 private:
  FuncGraphManager *manager_;
  std::vector<Change> changes_;
};

// args for set param
struct ArgsOfSetParams {
  FuncGraphPtr func_graph;
  std::vector<AnfNodePtr> params;
  bool operator==(const ArgsOfSetParams &other) const { return &other == this; }

  friend std::ostream &operator<<(std::ostream &os, const ArgsOfSetParams &) {
    os << "[ArgsOfSetParams]";
    return os;
  }
};

// args for set edge
struct ArgsOfSetEdge {
  CNodePtr root_node;
  AnfNodePtr new_node;
  size_t index;
  bool operator==(const ArgsOfSetEdge &other) const { return &other == this; }

  friend std::ostream &operator<<(std::ostream &os, const ArgsOfSetEdge &other) {
    os << "[ArgsOfSetEdge]";
    return os;
  }
};

struct Change {
  enum OpName { kTxSetParams, kTxSetEdge };
  OpName op;
  Any args;
  Change(OpName name, const Any &para) : op(name), args(para) {}
};

}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_IR_MANAGER_H_
