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

#ifndef MINDSPORE_CORE_IR_FUNC_GRAPH_H_
#define MINDSPORE_CORE_IR_FUNC_GRAPH_H_

#include <set>
#include <map>
#include <string>
#include <vector>
#include <list>
#include <unordered_map>
#include <memory>
#include <functional>
#include <utility>

#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "utils/ordered_set.h"
#include "utils/ordered_map.h"
#include "utils/macros.h"
#include "base/base_ref.h"
#include "base/effect_info.h"
#include "ir/anf.h"
#include "ir/manager.h"
#include "ir/func_graph_transform.h"
#include "ir/func_graph_base.h"
#include "abstract/abstract_value.h"

namespace mindspore {
using BaseRefCounterMap = OrderedMap<BaseRef, int, BaseRefHash>;
using FuncGraphCounterMap = OrderedMap<FuncGraphPtr, int>;

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

template <typename ValueT, class CounterHash = std::hash<ValueT>, class CounterEqual = std::equal_to<ValueT>>
using CounterOrderedMap = OrderedMap<ValueT, int, CounterHash, CounterEqual>;
using AnfNodeCounterMap = CounterOrderedMap<AnfNodePtr>;
using CNodeIndexCounterMap = CounterOrderedMap<CNodeIndexPairPtr, CNodeIndexHasher, CNodeIndexEqual>;

using FuncGraphMap = OrderedMap<FuncGraphPtr, int>;

const char FUNC_GRAPH_FLAG_IGNORE_VALUE[] = "ignore_value";
const char FUNC_GRAPH_FLAG_DEFER_INLINE[] = "defer_inline";
const char FUNC_GRAPH_FLAG_PRIMAL_OF_BPROP[] = "primal_of_bprop";
const char FUNC_GRAPH_FLAG_SPARSE_BPROP[] = "sparse_bprop";
const char FUNC_GRAPH_FLAG_NO_INLINE[] = "no_inline";
const char FUNC_GRAPH_FLAG_AFTER_BLOCK[] = "after_block";
const char FUNC_GRAPH_FLAG_CORE[] = "core";
const char FUNC_GRAPH_FLAG_K_GRAPH[] = "k_graph";
const char FUNC_GRAPH_ATTR_GRAPH_KERNEL[] = "graph_kernel";
const char FUNC_GRAPH_FLAG_SPECIALIZE_PARAMETER[] = "spec_param";
const char FUNC_GRAPH_OUTPUT_NO_RECOMPUTE[] = "output_no_recompute";
const char FUNC_GRAPH_FLAG_FORCE_INLINE[] = "force_inline";
const char FUNC_GRAPH_FLAG_DUMP[] = "dump";
const char FUNC_GRAPH_FLAG_DYNAMIC_SHAPE[] = "dynamic_shape";

const char kFuncGraphFlagUndetermined[] = "undeterminate";
const char kFuncGraphFlagBackPropEntry[] = "back_prop_entry";
const char kFuncGraphFlagReAutoMonad[] = "re_auto_monad";
const char kFuncGraphFlagRecursive[] = "recursive";

class MS_CORE_API FuncGraph : public FuncGraphBase, public EffectInfoHolder {
 public:
  using Drawer = std::function<void(const std::string &, const FuncGraphPtr &)>;

  FuncGraph();
  explicit FuncGraph(GraphDebugInfoPtr &&debug_info);
  ~FuncGraph() override = default;
  MS_DECLARE_PARENT(FuncGraph, FuncGraphBase);

  void DoBreakLoop() override;

  // Get the graph's abstract.
  abstract::AbstractFunctionPtr abstract();
  abstract::AbstractBasePtr ToAbstract() override;

  // get function graph inputs, but parameters
  const std::vector<AnfNodePtr> get_inputs() const;
  // Return the graph's output, or nullptr if not yet deduced.
  AnfNodePtr output() const;
  void set_output(const AnfNodePtr &value, bool force_new_ret = false);

  const std::vector<AnfNodePtr> &parameters() const { return parameters_; }
  // Append
  virtual ParameterPtr add_parameter();
  ParameterPtr add_parameter(NodeDebugInfoPtr &&debug_info);
  void add_parameter(const ParameterPtr &param);
  void append_parameter(const ParameterPtr &p) { parameters_.push_back(p); }
  // Prepend
  virtual ParameterPtr InsertFrontParameter();
  void InsertFrontParameter(const ParameterPtr &param);
  void PrependParameter(const ParameterPtr &p) { parameters_.insert(parameters_.begin(), p); }
  void set_parameters(const std::vector<AnfNodePtr> &params) { parameters_ = params; }
  void set_parameters(std::vector<AnfNodePtr> &&params) { parameters_ = std::move(params); }
  // Add a FV weight parameter with specific name.
  ParameterPtr AddFvParameter(const std::string &name, const ValuePtr &default_value);

  // Create a cnode with given inputs, bound to this graph.
  virtual CNodePtr NewCNode(std::vector<AnfNodePtr> &&inputs);
  virtual CNodePtr NewCNode(const std::vector<AnfNodePtr> &inputs);
  CNodePtr NewCNode(const PrimitivePtr &primitive, const std::vector<AnfNodePtr> &inputs);

  // Create a cnode with given inputs, bound to this graph and push back to order list.
  CNodePtr NewCNodeInOrder(std::vector<AnfNodePtr> &&inputs);
  CNodePtr NewCNodeInOrder(const std::vector<AnfNodePtr> &inputs = std::vector<AnfNodePtr>());
  CNodePtr NewCNodeInOrder(const PrimitivePtr &primitive, const std::vector<AnfNodePtr> &inputs);

  // Create a cnode with given inputs, bound to this graph and push back to front of order list.
  CNodePtr NewCNodeInFront(const std::vector<AnfNodePtr> &inputs = std::vector<AnfNodePtr>());

  // Create a cnode with given inputs, put it to order list before the position node.
  CNodePtr NewCNodeBefore(const AnfNodePtr &position, const std::vector<AnfNodePtr> &inputs);

  // Create a cnode with given inputs, put it to order list after the position node.
  CNodePtr NewCNodeAfter(const AnfNodePtr &position, const std::vector<AnfNodePtr> &inputs);

  // Functions for handling variable argument, keyword-only arguments and variable keyword argument.
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
  // Parameters are ordered as: Positional Parameters, Kwonlyargs, *Varargs, **Kwargs, HyperParam;
  AnfNodePtr GetVariableArgParameter();
  std::string GetVariableArgName();
  void set_has_kwarg(bool has_) { has_kwarg_ = has_; }
  bool has_kwarg() const { return has_kwarg_; }
  void set_kwonlyargs_count(int count) { kw_only_args_count_ = count; }
  int kwonlyargs_count() const { return kw_only_args_count_; }
  AnfNodePtr GetVariableKwargParameter();
  std::string GetVariableKwargName();
  AnfNodePtrList GetKwOnlyArgsParameters();
  void set_fv_param_count(size_t count) { fv_param_count_ = count; }
  size_t fv_param_count() const { return fv_param_count_; }
  int GetPositionalArgsCount() const;
  AnfNodePtr GetParameterByName(const std::string &name);
  bool NeedGenerate(const std::vector<abstract::AbstractKeywordArgPtr> &kwarg_list);
  FuncGraphPtr GenerateGraph(const AbstractBasePtrList &args_spec_list);
  void set_is_generate(bool generated) { is_generated_ = generated; }
  bool is_generated() const { return is_generated_; }

  mindspore::HashMap<std::string, ValuePtr> &attrs() { return attrs_; }
  void set_attrs(const mindspore::HashMap<std::string, ValuePtr> &attrs) {
    for (auto &attr : attrs) {
      attrs_[attr.first] = attr.second;
    }
  }
  bool has_flag(const std::string &key) const;
  void set_flag(const std::string &key, bool flag) { attrs_[key] = MakeValue(flag); }
  void erase_flag(const std::string &key) { (void)attrs_.erase(key); }

  bool has_attr(const std::string &key) const;
  ValuePtr get_attr(const std::string &key) const;
  void set_attr(const std::string &key, const ValuePtr &value) { attrs_[key] = value; }

  mindspore::HashMap<std::string, FuncGraphTransform> &transforms() { return transforms_; }
  void set_transforms(const mindspore::HashMap<std::string, FuncGraphTransform> &transforms) {
    transforms_ = transforms;
  }

  CNodePtr get_return() const { return return_; }
  void set_return(const CNodePtr &cnode) { return_ = cnode; }
  const CNodePtr &return_node() const { return return_; }

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
  // Get all nodes belonging to this func graph.
  const AnfNodeSet &nodes() const;
  void CopyNodes(const FuncGraphPtr &source);
  void ClearNodes();
  void AddNode(const AnfNodePtr &node);
  void DropNode(const AnfNodePtr &node);

  // Get all value_nodes belonging to this func graph.
  const AnfNodeCounterMap &value_nodes() const;
  void CopyValueNodes(const FuncGraphPtr &source);
  void ClearValueNodes();
  void AddValueNode(const AnfNodePtr &node, int count = 1);
  void DropValueNode(const AnfNodePtr &node);

  // Get all free vars directly used in this func graph.
  const AnfNodeCounterMap &free_variables() const;
  void CopyFreeVariables(const FuncGraphPtr &source);
  void ClearFreeVariables();
  bool AddFreeVariable(const AnfNodePtr &node, int count = 1);
  bool DropFreeVariable(const AnfNodePtr &node);

  // Get all vars required by this func graph.
  const BaseRefCounterMap &free_variables_total();

  // Return the set of graphs free_variables_total belong to.
  std::vector<AnfNodePtr> free_variables_nodes();

  // Get all vars that are func graphs
  std::vector<FuncGraphPtr> free_variables_func_graphs();

  // Get all value nodes of func graph directly used by this func graph.
  const FuncGraphCounterMap &func_graphs_used() const;
  void CopyFuncGraphsUsed(const FuncGraphPtr &source);
  void ClearFuncGraphsUsed();
  bool AddFuncGraphUsed(const FuncGraphPtr &fg, int count = 1);
  bool DropFuncGraphUsed(const FuncGraphPtr &fg);

  // Get all value nodes in the inputs of MetaFgPrim directly used by this func graph.
  const mindspore::HashMap<AnfNodePtr, int> &meta_fg_prim_value_nodes() const;
  void CopyMetaFgPrimValueNodes(const FuncGraphPtr &source);
  void ClearMetaFgPrimValueNodes();
  void AddMetaFgPrimValueNode(const AnfNodePtr &value_node, int count = 1);
  void DropMetaFgPrimValueNode(const AnfNodePtr &value_node);

  // Get all func graphs nested used by this func graph.
  const FuncGraphSet &func_graphs_used_total();

  // Get all user value nodes of this func graph, by CNode and its input's index.
  const CNodeIndexCounterMap &func_graph_cnodes_index() const;
  void CopyFuncGraphCNodesIndex(const FuncGraphPtr &source);
  void ClearFuncGraphCNodesIndex();
  void AddFuncGraphCNodeIndex(const CNodeIndexPairPtr &pair, int count = 1);
  void DropFuncGraphCNodeIndex(const CNodeIndexPairPtr &pair);

  // Return the parent of this graph.
  FuncGraphPtr parent();

  // Return the children of this graph.
  const FuncGraphSet &children();

  // Return the scope of this graph, scope have graph self but children not have.
  const FuncGraphSet &scope();

  // Return whether this graph is recursive.
  bool recursive();

  // Return graphs which forms a recursive loop.
  std::shared_ptr<std::list<FuncGraphPtr>> recursive_graphs();

  std::size_t hash() const override { return PointerHash<FuncGraph>{}(this); }

  bool operator==(const Value &other) const override {
    if (other.isa<FuncGraph>()) {
      return &other == this;
    } else {
      return false;
    }
  }
  void GenerateVarParams(const FuncGraphPtr &specialized_graph, int variable_args_count, int pos_args_input_count,
                         std::vector<AnfNodePtr> *specialized_parameter_list,
                         mindspore::HashMap<AnfNodePtr, AnfNodePtr> *repl_nodes) const;

  void GenerateKwParams(const FuncGraphPtr &specialized_graph,
                        const std::vector<abstract::AbstractKeywordArgPtr> &kwarg_list,
                        std::vector<AnfNodePtr> *specialized_parameter_list,
                        mindspore::HashMap<AnfNodePtr, AnfNodePtr> *repl_nodes) const;

  void GenerateDefaultValue(const FuncGraphPtr &specialized_graph,
                            const std::vector<AnfNodePtr> &specialized_parameter_list,
                            mindspore::HashMap<AnfNodePtr, AnfNodePtr> *repl_nodes) const;

  const std::vector<AnfNodePtr> &paramter_obj_nodes() const { return paramter_obj_nodes_; }
  void add_parameter_obj_node(const AnfNodePtr &p) { paramter_obj_nodes_.push_back(p); }

  mindspore::HashMap<std::string, ValuePtr> attrs_;
  mindspore::HashMap<std::string, FuncGraphTransform> transforms_;
  // Parameter default value.
  std::map<std::string, AnfNodePtr> parameter_default_value_;
  SeenNum seen_;

  std::list<CNodePtr> GetOrderedCnodes();
  void EraseUnusedNodeInOrder(const AnfNodePtr &node);
  void EraseUnusedNodeInOrder();
  void DumpCNodeList();
  const OrderedSet<CNodePtr> &order_list() const { return order_; }

  void set_order_list(OrderedSet<CNodePtr> &&order_list) { order_ = std::move(order_list); }

  // Add a cnode at the end of order list.
  void AppendOrderList(const CNodePtr &cnode) { order_.push_back(cnode); }

  // Prepend cnode at the front of order list.
  void PrependOrderList(const CNodePtr &cnode) { order_.push_front(cnode); }

  // Maintain cnode order list when a cnode is replaced by a new one.
  void ReplaceInOrder(const AnfNodePtr &old_node, const AnfNodePtr &new_node);

  // Clear cnode order list.
  void ClearOrderList() { order_.clear(); }

  bool stub() const { return stub_; }
  void set_stub(bool stub) { stub_ = stub; }
  std::shared_ptr<bool> switch_input() const { return switch_input_; }
  void set_switch_input(const std::shared_ptr<bool> &switch_input) { switch_input_ = switch_input; }
  std::shared_ptr<bool> switch_layer_input() const { return switch_layer_input_; }
  void set_switch_layer_input(const std::shared_ptr<bool> &switch_layer_input) {
    switch_layer_input_ = switch_layer_input;
  }
  void SetMultiTarget() const;
  bool exist_multi_target() const { return exist_multi_target_; }
  void set_exist_multi_target(bool exist_multi_target) { exist_multi_target_ = exist_multi_target; }
  int64_t stage() const { return stage_; }
  void set_stage(int64_t stage) { stage_ = stage; }

  bool dropped() const { return dropped_; }
  void set_dropped(bool dropped) { dropped_ = dropped; }

  std::string bprop_hash() const { return bprop_hash_; }
  void set_bprop_hash(const std::string &bprop_hash) { bprop_hash_ = bprop_hash; }

  bool modify_output() const { return modify_output_; }
  void set_modify_output(bool modify_output) { modify_output_ = modify_output; }
  const mindspore::HashSet<AnfNodePtr> &used_forward_nodes() const { return used_forward_nodes_; }
  void set_used_forward_nodes(const std::vector<AnfNodePtr> &used_forward_nodes);
  void ClearUsedForwardNodes() { used_forward_nodes_.clear(); }

  bool is_tensor_condition_branch() const { return is_tensor_condition_branch_; }
  void set_is_tensor_condition_branch(bool is_tensor_condition_branch) {
    is_tensor_condition_branch_ = is_tensor_condition_branch;
  }

  /// \brief Topological sort a graph from the given end node.
  ///
  /// \param[in] node The end node of the graph to be sorted.
  ///
  /// \return The sorted nodes.
  static std::vector<AnfNodePtr> TopoSort(const AnfNodePtr &node);

  void set_python_obj(const ValuePtr &python_obj) { python_obj_ = python_obj; }
  ValuePtr python_obj() const { return python_obj_; }

 private:
  // Only used for func_graph manager to control resource free.
  int attached_mng_cnt() const { return attached_mng_cnt_; }
  void IncAttachedMngCnt() { attached_mng_cnt_++; }
  void DecAttachedMngCnt() { attached_mng_cnt_--; }
  // Clear all info from manager.
  void ClearAllManagerInfo();

  // Graph is manipulated by manager and others.
  friend FuncGraphManager;

  // All nodes of the function.
  AnfNodeSet nodes_;

  // All value nodes of the function.
  AnfNodeCounterMap value_nodes_;

  // All func graph value nodes of the function.
  FuncGraphCounterMap func_graphs_used_;

  // All free variables of the function.
  AnfNodeCounterMap free_variables_;

  // All value nodes calling MetaFgPrim in the function.
  mindspore::HashMap<AnfNodePtr, int> meta_fg_prim_value_nodes_;

  // All user value nodes of this func graph, recording by CNode and its input's index.
  CNodeIndexCounterMap func_graph_cnodes_index_;

  // Parameters of this function.
  std::vector<AnfNodePtr> parameters_;
  std::vector<AnfNodePtr> paramter_obj_nodes_;

  // Whether there is a *args and **kwargs, and count kw_only_args'number.
  bool has_vararg_;
  bool has_kwarg_;
  bool exist_multi_target_;
  int kw_only_args_count_;
  // Hyper param is used as free variable and placed on the top graph.
  // and positioned in the end of the param list, so we record the number to trace the position.
  size_t fv_param_count_;
  // Argument input list for the graph used to generate this graph.
  bool is_generated_;
  // CNode that calls 'return' primitive.
  // We use shared pointer to manage it.
  CNodePtr return_;

  // Back-ref to its manager.
  // Hold a weak ref to FuncGraphManager as FuncGraphManager also hold many ref to FuncGraph.
  // Otherwise, FuncGraph and FuncGraphManager will make a reference cycles.
  // Notes: Normally, there will be a global FuncGraphManager, it will hold all FuncGraphs.
  // In some ut test cases, they may use local FuncGraphManager in function which
  // generating the func graph, when go outside of that function, func graph will have no
  // FuncGraphManager. In that special case, Manage() should be called to make the func graph
  // managed.
  std::weak_ptr<FuncGraphManager> manager_;
  int attached_mng_cnt_ = 0;

  GraphDebugInfoPtr debug_info_;
  void GenerateKwargReplNode(const FuncGraphPtr &specialized_graph,
                             const std::vector<AnfNodePtr> &kwarg_keys_tuple_nodes,
                             const std::vector<AnfNodePtr> &kwarg_values_tuple_nodes,
                             mindspore::HashMap<AnfNodePtr, AnfNodePtr> *repl_nodes) const;

  // CNode order which relates to origin code order.
  OrderedSet<CNodePtr> order_;
  bool stub_;
  // Design switch_input and switch_layer_input as a ptr to
  // share between derived backpropagator and cloned graphs.
  std::shared_ptr<bool> switch_input_;
  std::shared_ptr<bool> switch_layer_input_;
  int64_t stage_;
  std::unordered_map<AbstractBasePtrList, FuncGraphPtr, abstract::AbstractBasePtrListHasher,
                     abstract::AbstractBasePtrListEqual>
    func_graph_cache_;

  // If the graph was changed, it should be dropped in cache data_converter::object_map_
  // which used by ConvertToFuncGraph.
  bool dropped_ = false;
  // If the graph is a bprop graph, it should has a hash of the bprop directory.
  std::string bprop_hash_;

  // If the graph is decorated by @ms_function and runs grad process in pynative mode,
  // forward nodes used in grad graph will be added to output for holding output values.
  bool modify_output_ = false;
  mindspore::HashSet<AnfNodePtr> used_forward_nodes_;
  // If the func_graph is input of switch node, and the condition of switch is AbstractTensor, need set true.
  bool is_tensor_condition_branch_ = false;
  // Corresponding python obj.
  ValuePtr python_obj_ = nullptr;
};

inline CNodePtr NewCNode(const std::vector<AnfNodePtr> &inputs, const FuncGraphPtr &fg) {
  MS_EXCEPTION_IF_NULL(fg);
  return fg->NewCNode(inputs);
}

inline CNodePtr NewCNode(std::vector<AnfNodePtr> &&inputs, const FuncGraphPtr &fg) {
  MS_EXCEPTION_IF_NULL(fg);
  return fg->NewCNode(std::move(inputs));
}

SeenNum NewFgSeenGeneration();

// Find the root cnodes of a segment of cnodes.
std::shared_ptr<OrderedSet<CNodePtr>> FindRoots(const std::vector<CNodePtr> &segment);
// Find the leaf cnodes of a segment of cnodes.
std::shared_ptr<OrderedSet<CNodePtr>> FindLeaves(const std::vector<CNodePtr> &segment);
}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_FUNC_GRAPH_H_
