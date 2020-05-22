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

#ifndef MINDSPORE_CCSRC_IR_FUNC_GRAPH_H_
#define MINDSPORE_CCSRC_IR_FUNC_GRAPH_H_

#include <map>
#include <string>
#include <vector>
#include <list>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <functional>

#include "ir/anf.h"
#include "ir/manager.h"
#include "utils/any.h"
#include "utils/ordered_set.h"
#include "pipeline/static_analysis/abstract_value.h"

namespace mindspore {
using BaseRefCounterMap = OrderedMap<BaseRef, int, BaseRefHash>;
using FuncGraphCounterMap = OrderedMap<FuncGraphPtr, int>;

template <typename ValueT, class CounterHash = std::hash<ValueT>, class CounterEqual = std::equal_to<ValueT>>
using CounterOrderedMap = OrderedMap<ValueT, int, CounterHash, CounterEqual>;
using AnfNodeCounterMap = CounterOrderedMap<AnfNodePtr>;
using CNodeIndexCounterMap = CounterOrderedMap<CNodeIndexPairPtr, CNodeIndexHasher, CNodeIndexEqual>;

using FuncGraphMap = OrderedMap<FuncGraphPtr, int>;

const char FUNC_GRAPH_FLAG_IGNORE_VALUES[] = "ignore_values";
const char FUNC_GRAPH_FLAG_DEFER_INLINE[] = "defer_inline";
const char FUNC_GRAPH_FLAG_CORE[] = "core";
const char FUNC_GRAPH_FLAG_SPECIALIZE_PARAMETER[] = "spec_param";

// ANF transform class
// either a primitive or a func_graph
class FuncGraphTransform {
 public:
  enum Type { kGtPrimitive, kGtFuncGraph };

  explicit FuncGraphTransform(const PrimitivePtr prim, const FuncGraphPtr func_graph = nullptr)
      : prim_(prim), func_graph_(FuncGraphWeakPtr(func_graph)) {}

  explicit FuncGraphTransform(const FuncGraphPtr &func_graph, const PrimitivePtr &prim = func_graph_prim_)
      : prim_(prim), func_graph_(FuncGraphWeakPtr(func_graph)) {}

  FuncGraphTransform(const FuncGraphTransform &t) : prim_(t.prim_), func_graph_(t.func_graph_) {}

  ~FuncGraphTransform() = default;

  Type type() const {
    if (IsFuncGraph()) {
      return kGtFuncGraph;
    } else {
      return kGtPrimitive;
    }
  }

  bool IsPrimitive() const { return (func_graph_.lock() == nullptr); }
  bool IsFuncGraph() const { return (func_graph_.lock() != nullptr); }
  FuncGraphPtr func_graph() const { return func_graph_.lock(); }
  PrimitivePtr primitive() const { return prim_; }

  FuncGraphTransform &operator=(const FuncGraphTransform &t) {
    if (this != &t) {
      prim_ = t.prim_;
      func_graph_ = t.func_graph_;
    }
    return *this;
  }

 private:
  PrimitivePtr prim_;
  // FuncGraph will be hold by FuncGraphManager, so weak_ptr is enough here.
  // And use weak_ptr can break the reference cycle between "primal" and "grad" graph in
  // FPropRemapper::FinalizeGraph().
  FuncGraphWeakPtr func_graph_;
  static const PrimitivePtr func_graph_prim_;
};

class FuncGraphBase : public Value {
 public:
  FuncGraphBase() = default;

  ~FuncGraphBase() override = default;
  MS_DECLARE_PARENT(FuncGraphBase, Value);
};

extern const char kFuncGraphFlagUndetermined[];

class FuncGraph : public FuncGraphBase {
 public:
  FuncGraph();

  ~FuncGraph() override = default;
  MS_DECLARE_PARENT(FuncGraph, FuncGraphBase);

  // get the graph's abstract
  abstract::AbstractFunctionPtr abstract();
  abstract::AbstractBasePtr MakeAbstractClosure(const abstract::AnalysisContextPtr &context);

  // return the graph's output, or nullptr if not yet deduced
  AnfNodePtr output() const;
  void set_output(const AnfNodePtr &value, bool force_new_ret = false);

  const std::vector<AnfNodePtr> &parameters() const { return parameters_; }
  virtual ParameterPtr add_parameter();
  void add_parameter(const ParameterPtr &p);
  void set_parameters(const std::vector<AnfNodePtr> &params) { parameters_ = params; }
  // add a weight parameter with specific name
  ParameterPtr AddWeightParameter(const std::string &name);

  // create a cnode with given inputs, bound to this graph
  virtual CNodePtr NewCNode(const std::vector<AnfNodePtr> &inputs = std::vector<AnfNodePtr>());

  // create a cnode with given inputs, bound to this graph, and set to specific scope
  CNodePtr NewCNodeWithScope(const std::vector<AnfNodePtr> &inputs, const ScopePtr &scope);

  // Functions for handling variable argument, keyword-only arguments and variable keyword argument
  AnfNodePtr GetDefaultValueByName(const std::string &name);
  void set_param_default_value(const std::string &name, const AnfNodePtr &node) {
    parameter_default_value_[name] = node;
  }
  void SetDefaultValues(const std::vector<std::string> &name_list, const std::vector<AnfNodePtr> &value_list);
  void ClearDefaultValues();
  size_t GetDefaultValueCount();
  std::map<std::string, AnfNodePtr> &parameter_default_value() { return parameter_default_value_; }
  void set_has_vararg(bool has_) { has_vararg_ = has_; }
  bool has_vararg() const { return has_vararg_; }
  AnfNodePtr GetVariableArgParameter();
  std::string GetVariableArgName();
  void set_has_kwarg(bool has_) { has_kwarg_ = has_; }
  bool has_kwarg() const { return has_kwarg_; }
  void set_kwonlyargs_count(int count) { kwonlyargs_count_ = count; }
  int kwonlyargs_count() const { return kwonlyargs_count_; }
  AnfNodePtr GetVariableKwargParameter();
  std::string GetVariableKwargName();
  void set_hyper_param_count(size_t count) { hyper_param_count_ = count; }
  size_t hyper_param_count() const { return hyper_param_count_; }
  int GetPositionalArgsCount() const;
  AnfNodePtr GetParameterByName(const std::string &name);
  bool NeedGenerate(const std::vector<abstract::AbstractKeywordArgPtr> &kwarg_list);
  FuncGraphPtr GenerateGraph(const AbstractBasePtrList &args_spec_list);
  void set_is_generate(bool generated) { is_generated_ = generated; }
  bool is_generated() const { return is_generated_; }

  bool has_flag(const std::string &flag);
  std::unordered_map<std::string, bool> &flags() { return flags_; }
  void set_flags(const std::unordered_map<std::string, bool> &flags) { flags_ = flags; }
  void set_flags(const std::string &key, const bool value) { flags_[key] = value; }

  std::unordered_map<std::string, FuncGraphTransform> &transforms() { return transforms_; }
  void set_transforms(const std::unordered_map<std::string, FuncGraphTransform> &transforms) {
    transforms_ = transforms;
  }

  CNodePtr get_return() const { return return_; }
  void set_return(const CNodePtr &cnode) { return_ = cnode; }

  FuncGraphManagerPtr manager() const { return manager_.lock(); }
  void set_manager(const FuncGraphManagerPtr &m) { manager_ = std::weak_ptr<FuncGraphManager>(m); }

  std::string ToString() const override;
  GraphDebugInfoPtr debug_info();
  void set_debug_info(const GraphDebugInfoPtr &info) {
    if (info == nullptr) {
      MS_LOG(EXCEPTION) << "Graph set null debug info";
    }
    this->debug_info_ = info;
  }

  // get all nodes belonging to this func graph
  const AnfNodeSet &nodes();
  void CopyNodes(const AnfNodeSet &other_nodes);
  void ClearNodes();
  void AddNode(AnfNodePtr node);
  void DropNode(AnfNodePtr node);

  // get all value_nodes belonging to this func graph
  const AnfNodeCounterMap &value_nodes();
  void CopyValueNodes(const AnfNodeCounterMap &other_value_nodes);
  void ClearValueNodes();
  void AddValueNode(AnfNodePtr node, int count = 1);
  void DropValueNode(AnfNodePtr node);

  // get all free vars directly used in this func graph
  const AnfNodeCounterMap &free_variables();
  void CopyFreeVariables(const AnfNodeCounterMap &others);
  void ClearFreeVariables();
  bool AddFreeVariable(AnfNodePtr node, int count = 1);
  bool DropFreeVariable(AnfNodePtr node);

  // get all vars required by this func graph
  const BaseRefCounterMap &free_variables_total();

  // Return the set of graphs free_variables_total belong to.
  std::vector<AnfNodePtr> free_variables_nodes();

  // get all vars that are func graphs
  std::vector<FuncGraphPtr> free_variables_func_graphs();

  // get all value nodes of func graph directly used by this func graph
  const AnfNodeCounterMap &func_graph_value_nodes();
  void CopyFuncGraphValueNodes(const AnfNodeCounterMap &others);
  void ClearFuncGraphValueNodes();
  bool AddFuncGraphValueNode(AnfNodePtr node, int count = 1);
  bool DropFuncGraphValueNode(AnfNodePtr node);

  // get all value nodes of J func graph directly used by this func graph
  const AnfNodeCounterMap &j_func_graph_value_nodes();
  void CopyJFuncGraphValueNodes(const AnfNodeCounterMap &others);
  void ClearJFuncGraphValueNodes();
  void AddJFuncGraphValueNode(AnfNodePtr node, int count = 1);
  void DropJFuncGraphValueNode(AnfNodePtr node);

  // get all func graphs nested used by this func graph
  const FuncGraphSet &func_graphs_used_total();

  // get all user value nodes of this func graph, by CNode and its input's index
  const CNodeIndexCounterMap &func_graph_cnodes_index();
  void CopyFuncGraphCNodesIndex(const CNodeIndexCounterMap &other_value_nodes);
  void ClearFuncGraphCNodesIndex();
  void AddFuncGraphCNodeIndex(CNodeIndexPairPtr node, int count = 1);
  void DropFuncGraphCNodeIndex(CNodeIndexPairPtr node);

  // Return the parent of this graph.
  FuncGraphPtr parent();

  // Return the children of this graph.
  const FuncGraphSet &children();

  // Return the scope of this graph, scope have graph self but children not have.
  const FuncGraphSet &scope();

  // Return whether this graph is recursive
  bool recursive();

  // Return graphs which forms a recursive loop
  std::shared_ptr<std::list<FuncGraphPtr>> recursive_graphs();

  std::size_t hash() const override { return std::hash<const FuncGraph *>{}(this); }

  void DumpFuncGraph(const std::string &path = "./func_graph.dot");

  bool operator==(const Value &other) const override {
    if (other.isa<FuncGraph>()) {
      return &other == this;
    } else {
      return false;
    }
  }
  void GenerateVarParams(const FuncGraphPtr &specialized_graph, std::vector<AnfNodePtr> *specialized_parameter_list,
                         std::unordered_map<AnfNodePtr, AnfNodePtr> *repl_nodes, int variable_args_count,
                         int pos_args_input_count);

  void GenerateKwParams(const FuncGraphPtr &specialized_graph, std::vector<AnfNodePtr> *specialized_parameter_list,
                        const std::vector<abstract::AbstractKeywordArgPtr> &kwarg_list,
                        std::unordered_map<AnfNodePtr, AnfNodePtr> *repl_nodes);

  void GenerateDefaultValue(const FuncGraphPtr &specialized_graph,
                            const std::vector<AnfNodePtr> &specialized_parameter_list,
                            std::unordered_map<AnfNodePtr, AnfNodePtr> *repl_nodes);

  const std::vector<AnfNodePtr> &paramter_obj_nodes() const { return paramter_obj_nodes_; }
  void add_parameter_obj_node(const AnfNodePtr &p);

  std::unordered_map<AnfNodePtr, AnfNodePtr> &make_ref_params() { return make_ref_params_; }

  std::unordered_map<std::string, bool> flags_;
  std::unordered_map<std::string, FuncGraphTransform> transforms_;
  // parameter default value
  std::map<std::string, AnfNodePtr> parameter_default_value_;
  std::unordered_map<AnfNodePtr, AnfNodePtr> make_ref_params_;

  std::list<CNodePtr> GetOrderedCnodes();
  void EraseUnusedNodeInOrder(const AnfNodePtr &n);
  void EraseUnusedNodeInOrder();
  void CheckOrder();
  void DumpCNodeList();
  void ReleaseFullOrderToEffectOrder();
  void SetEffectDepends(const std::vector<AnfNodePtr> &depend_inputs);
  bool HasEffect(const CNodePtr &cnode);

 private:
  // graph is manipulated by manager and others
  friend FuncGraphManager;

  // all nodes of the function
  AnfNodeSet nodes_;

  // all value nodes of the function
  AnfNodeCounterMap value_nodes_;

  // all func graph value nodes of the function
  AnfNodeCounterMap func_graph_value_nodes_;

  // all free variables of the function
  AnfNodeCounterMap free_variables_;

  // all value nodes calling J in the function
  AnfNodeCounterMap j_func_graph_value_nodes_;

  // all user value nodes of this func graph, recording by CNode and its input's index
  CNodeIndexCounterMap func_graph_cnodes_index_;

  // parameters of this function
  std::vector<AnfNodePtr> parameters_;
  std::vector<AnfNodePtr> paramter_obj_nodes_;

  // whether there is a *args and **kwargs, and count kwonlyargs'number
  bool has_vararg_;
  bool has_kwarg_;
  int kwonlyargs_count_;
  // the hyper param is placed on the top graph,
  // and positioned in the end of the param list, so we record the number to trace the position
  size_t hyper_param_count_;
  // the argument input list for the graph used to generate this graph
  bool is_generated_;

  // the cnode that calls 'return' primitive
  // we use shared pointer to manage it.
  CNodePtr return_;

  // back-ref to its manager
  // hold a weak ref to FuncGraphManager as FuncGraphManager also hold many ref to FuncGraph.
  // Otherwise, FuncGraph and FuncGraphManager will make a reference cycles.
  // Notes: Normally, there will be a global FuncGraphManager, it will hold all FuncGraphs.
  // In some ut test cases, they may use local FuncGraphManager in function which
  // generating the func graph, when go outside of that function, func graph will have no
  // FuncGraphManager. In that special case, Manage() should be called to make the func graph
  // managed.
  std::weak_ptr<FuncGraphManager> manager_;

  GraphDebugInfoPtr debug_info_;
  void GenerateKwargReplNode(const FuncGraphPtr &specialized_graph,
                             std::unordered_map<AnfNodePtr, AnfNodePtr> *repl_nodes,
                             const std::vector<AnfNodePtr> &kwarg_keys_tuple_nodes,
                             const std::vector<AnfNodePtr> &kwarg_values_tuple_nodes);

  // CNode order which relates to origin code order
  std::list<CNodePtr> order_;
};

inline CNodePtr NewCNode(const std::vector<AnfNodePtr> &inputs, const FuncGraphPtr &fg) {
  MS_EXCEPTION_IF_NULL(fg);
  return fg->NewCNode(inputs);
}

// Find the root cnodes of a segment of cnodes.
std::shared_ptr<OrderedSet<CNodePtr>> FindRoots(const std::vector<CNodePtr> &segment);
// Find the leaf cnodes of a segment of cnodes.
std::shared_ptr<OrderedSet<CNodePtr>> FindLeaves(const std::vector<CNodePtr> &segment);
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_IR_FUNC_GRAPH_H_
