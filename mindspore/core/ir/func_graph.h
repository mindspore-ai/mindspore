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
#include "mindapi/base/macros.h"
#include "base/base_ref.h"
#include "base/effect_info.h"
#include "ir/anf.h"
#include "ir/manager.h"
#include "ir/func_graph_transform.h"
#include "ir/func_graph_base.h"
#include "abstract/abstract_value.h"
#include "mindspore/core/symbolic_shape/symbol_engine.h"

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

template <typename KeyT, class CounterHash = std::hash<KeyT>, class CounterEqual = std::equal_to<KeyT>>
using CounterOrderedMap = OrderedMap<KeyT, int, CounterHash, CounterEqual>;
using AnfNodeCounterMap = CounterOrderedMap<AnfNodePtr>;
using CNodeIndexCounterMap = CounterOrderedMap<CNodeIndexPairPtr, CNodeIndexHasher, CNodeIndexEqual>;

using FuncGraphMap = OrderedMap<FuncGraphPtr, int>;

const char FUNC_GRAPH_FLAG_IGNORE_VALUE[] = "ignore_value";
const char FUNC_GRAPH_FLAG_VMAP_TRANSFORMED[] = "vmap_transformed";
const char FUNC_GRAPH_FLAG_DEFER_INLINE[] = "defer_inline";
const char FUNC_GRAPH_FLAG_PRIMAL_OF_BPROP[] = "primal_of_bprop";
const char FUNC_GRAPH_FLAG_SPARSE_BPROP[] = "sparse_bprop";
const char FUNC_GRAPH_FLAG_NO_INLINE[] = "no_inline";
const char FUNC_GRAPH_FLAG_CELL_REUSE[] = "cell_reuse";
const char FUNC_GRAPH_FLAG_CELL_LAZY_INLINE_ORDER[] = "lazy_inline_order";
const char FUNC_GRAPH_FLAG_AFTER_BLOCK[] = "after_block";
const char FUNC_GRAPH_FLAG_CORE[] = "core";
const char FUNC_GRAPH_FLAG_K_GRAPH[] = "k_graph";
const char FUNC_GRAPH_ATTR_GRAPH_KERNEL[] = "graph_kernel";
const char FUNC_GRAPH_FLAG_SPECIALIZE_PARAMETER[] = "spec_param";
const char FUNC_GRAPH_OUTPUT_NO_RECOMPUTE[] = "output_no_recompute";
const char FUNC_GRAPH_RECOMPUTE_K_GRAPH[] = "recompute_k_graph";
const char FUNC_GRAPH_RECOMPUTE_GRAD_GRAPH[] = "recompute_grad_graph";
const char FUNC_GRAPH_NOT_RECOMPUTE_K_GRAPH[] = "not_recompute_k_graph";
const char FUNC_GRAPH_FLAG_FORCE_INLINE[] = "force_inline";
const char FUNC_GRAPH_FLAG_DUMP[] = "dump";
const char FUNC_GRAPH_FLAG_DYNAMIC_SHAPE[] = "dynamic_shape";
const char FUNC_GRAPH_FLAG_NO_RECURSIVE[] = "no_recursive";
const char FUNC_GRAPH_FLAG_ARGS_NO_EXPAND[] = "args_no_expand";
const char FUNC_GRAPH_FLAG_PROXY_GRAPH[] = "proxy_graph";
const char FUNC_GRAPH_FLAG_NO_CHILD_GRAPH[] = "no_child_graph";

const char kFuncGraphFlagUndetermined[] = "undeterminate";
const char kFuncGraphFlagBackPropEntry[] = "back_prop_entry";
const char kFuncGraphFlagReAutoMonad[] = "re_auto_monad";
const char kFuncGraphFlagRecursive[] = "recursive";
const char kFuncGraphFlagMetaFuncGraphBprop[] = "meta_fg_bprop";
const char kFuncGraphFlagAddedForwardU[] = "added_forward_u";

class MS_CORE_API FuncGraph : public FuncGraphBase, public EffectInfoHolder {
 public:
  using Drawer = std::function<void(const std::string &, const FuncGraphPtr &)>;

  FuncGraph();
  explicit FuncGraph(GraphDebugInfoPtr &&debug_info);
  ~FuncGraph();
  MS_DECLARE_PARENT(FuncGraph, FuncGraphBase);

  void DoBreakLoop() override;

  // Get the graph's abstract.
  abstract::AbstractFunctionPtr abstract();
  abstract::AbstractBasePtr ToAbstract() override;

  // get function graph inputs, but parameters
  const AnfNodePtrList get_inputs() const;
  const AnfNodePtrList &parameters() const { return parameters_; }
  // Append
  virtual ParameterPtr add_parameter();
  ParameterPtr add_parameter(NodeDebugInfoPtr &&debug_info);
  void add_parameter(const ParameterPtr &param);
  void append_parameter(const ParameterPtr &p) { parameters_.push_back(p); }
  // Prepend
  virtual ParameterPtr InsertFrontParameter();
  void InsertFrontParameter(const ParameterPtr &param);
  void PrependParameter(const ParameterPtr &p) { parameters_.insert(parameters_.begin(), p); }
  void set_parameters(const AnfNodePtrList &params) { parameters_ = params; }
  void set_parameters(AnfNodePtrList &&params) { parameters_ = std::move(params); }
  // Add a FV weight parameter with specific name.
  ParameterPtr AddFvParameter(const std::string &name, const ValuePtr &default_value);

  // Create a CNode with given inputs, bound to this graph.
  virtual CNodePtr NewCNodeWeak(AnfNodeWeakPtrList &&weak_inputs);
  virtual CNodePtr NewCNodeWeak(const AnfNodeWeakPtrList &weak_inputs);

  // @deprecated
  // To use 'CNodePtr NewCNodeWeak(AnfNodeWeakPtrList &&weak_inputs)' instead.
  virtual CNodePtr NewCNode(AnfNodePtrList &&inputs);
  // @deprecated
  // To use 'CNodePtr NewCNodeWeak(const AnfNodeWeakPtrList &weak_inputs)' instead.
  virtual CNodePtr NewCNode(const AnfNodePtrList &inputs);

  CNodePtr NewCNode(const PrimitivePtr &primitive, const AnfNodePtrList &inputs);

  // Create a CNode with given weak inputs, bound to this graph and push back to order list.
  CNodePtr NewCNodeInOrderWeak(AnfNodeWeakPtrList &&weak_inputs);
  CNodePtr NewCNodeInOrderWeak(const AnfNodeWeakPtrList &weak_inputs);

  // Create a CNode with given inputs, bound to this graph and push back to order list.
  CNodePtr NewCNodeInOrder(AnfNodePtrList &&inputs);
  CNodePtr NewCNodeInOrder(const AnfNodePtrList &inputs = AnfNodePtrList());
  CNodePtr NewCNodeInOrder(const PrimitivePtr &primitive, const AnfNodePtrList &inputs);

  // Create a CNode with given inputs, bound to this graph and push back to front of order list.
  CNodePtr NewCNodeInFront(const AnfNodePtrList &inputs = AnfNodePtrList());

  // Create a CNode with given inputs, put it to order list before the position node.
  CNodePtr NewCNodeBefore(const AnfNodePtr &position, const AnfNodePtrList &inputs);

  // Create a CNode with given inputs, put it to order list after the position node.
  CNodePtr NewCNodeAfter(const AnfNodePtr &position, const AnfNodePtrList &inputs);

  // Functions for handling variable argument, keyword-only arguments and variable keyword argument.
  AnfNodePtr GetDefaultValueByName(const std::string &name);
  void set_param_default_value(const std::string &name, const AnfNodePtr &node) {
    parameter_default_value_[name] = node;
  }
  void SetDefaultValues(const std::vector<std::string> &name_list, const AnfNodePtrList &value_list);
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
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_abs_list);
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

  // Return the graph's output, or nullptr if not yet deduced.
  AnfNodePtr output() const;
  void set_output(const AnfNodePtr &value, bool force_new_ret = false);

  CNodePtr get_return() const { return return_.lock(); }
  const CNodePtr return_node() const { return return_.lock(); }
  void set_return(const CNodePtr &cnode) {
    return_owner_ = cnode;
    return_ = CNodeWeakPtr(cnode);
  }
  void ResetReturnOwner() { return_owner_.reset(); }

  const std::list<AnfNodePtr> &own_nodes() const;
  void AddOwnNode(const AnfNodePtr &node);
  void AddOwnNode(const AnfNodePtrList &nodes);
  void AddOwnNode(const AnfNodeWeakPtrList &weak_nodes);
  void RemoveOwnNode(const AnfNodePtr &node);
  void ResetOwnNodes();

  FuncGraphManagerPtr manager() const { return manager_.lock(); }
  void set_manager(const FuncGraphManagerPtr &m) { manager_ = std::weak_ptr<FuncGraphManager>(m); }

  std::string ToString() const override;
  GraphDebugInfoPtr debug_info();
  void set_debug_info(const GraphDebugInfoPtr &info) {
    if (info == nullptr) {
      MS_LOG(INTERNAL_EXCEPTION) << "Graph set null debug info";
    }
    this->debug_info_ = info;
  }
  // Get all nodes belonging to this func graph.
  const AnfNodeSet &nodes() const;
  const AnfNodeSet &switch_nodes() const;
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
  AnfNodePtrList free_variables_nodes();

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
                         AnfNodePtrList *specialized_parameter_list,
                         mindspore::HashMap<AnfNodePtr, AnfNodePtr> *repl_nodes) const;

  void GenerateKwParams(const FuncGraphPtr &specialized_graph,
                        const std::vector<abstract::AbstractKeywordArgPtr> &kwarg_list, int pos_args_input_count,
                        AnfNodePtrList *specialized_parameter_list,
                        mindspore::HashMap<AnfNodePtr, AnfNodePtr> *repl_nodes) const;

  void GenerateDefaultValue(const FuncGraphPtr &specialized_graph, const AnfNodePtrList &specialized_parameter_list,
                            mindspore::HashMap<AnfNodePtr, AnfNodePtr> *repl_nodes) const;

  const AnfNodePtrList &parameter_obj_nodes() const { return parameter_obj_nodes_; }
  void add_parameter_obj_node(const AnfNodePtr &p) { parameter_obj_nodes_.push_back(p); }

  mindspore::HashMap<std::string, ValuePtr> attrs_;
  mindspore::HashMap<std::string, FuncGraphTransform> transforms_;
  // Parameter default value.
  std::map<std::string, AnfNodePtr> parameter_default_value_;

  SeenNum seen_{0};
  SeenNum extra_seen_{0};

  std::list<CNodePtr> GetOrderedCnodes();
  void EraseUnusedNodeInOrder(const AnfNodePtr &node);
  void EraseUnusedNodeInOrder();
  void DumpCNodeList();
  const std::list<CNodeWeakPtr> &order_list() const { return order_; }

  void set_order_list(std::list<CNodeWeakPtr> &&order_list) { order_ = std::move(order_list); }

  // Add a CNode at the end of order list.
  void AppendOrderList(const CNodePtr &cnode) { (void)order_.emplace_back(CNodeWeakPtr(cnode)); }

  // Prepend CNode at the front of order list.
  void PrependOrderList(const CNodePtr &cnode) { (void)order_.emplace_front(CNodeWeakPtr(cnode)); }

  // Maintain CNode order list when a CNode is replaced by a new one.
  void ReplaceInOrder(const AnfNodePtr &old_node, const AnfNodePtr &new_node);

  // Clear CNode order list.
  void ClearOrderList() { order_.clear(); }

  bool stub() const { return stub_; }
  void set_stub(bool stub) { stub_ = stub; }

  std::shared_ptr<bool> indirect() {
    // Lazy initialization.
    if (!indirect_) {
      indirect_ = std::make_shared<bool>(false);
    }
    return indirect_;
  }
  void set_indirect(std::shared_ptr<bool> indirect) { indirect_ = indirect; }

  void SetMultiTarget() const;
  bool exist_multi_target() const { return exist_multi_target_; }
  void set_exist_multi_target(bool exist_multi_target) { exist_multi_target_ = exist_multi_target; }
  int64_t stage() const { return stage_; }
  void set_stage(int64_t stage) { stage_ = stage; }
  int64_t segment() const { return segment_; }
  void set_segment(int64_t segment) { segment_ = segment; }
  bool dynamic_shape() { return dynamic_shape_; }
  void set_dynamic_shape(bool dynamic_shape) { dynamic_shape_ = dynamic_shape; }

  bool dropped() const { return dropped_; }
  void set_dropped(bool dropped) { dropped_ = dropped; }

  std::string bprop_hash() const { return bprop_hash_; }
  void set_bprop_hash(const std::string &bprop_hash) { bprop_hash_ = bprop_hash; }

  std::string bprop_filepath() const { return bprop_filepath_; }
  void set_bprop_filepath(const std::string &bprop_filepath) { bprop_filepath_ = bprop_filepath; }

  bool modify_output() const { return modify_output_; }
  void set_modify_output(bool modify_output) { modify_output_ = modify_output; }
  const mindspore::OrderedSet<AnfNodePtr> &used_forward_nodes() const { return used_forward_nodes_; }
  void set_used_forward_nodes(const AnfNodePtrList &used_forward_nodes);
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
  static AnfNodePtrList TopoSort(const AnfNodePtr &node);

  void set_python_obj(const ValuePtr &python_obj) { python_obj_ = python_obj; }
  ValuePtr python_obj() const { return python_obj_; }

  const std::string &phase() const { return phase_; }

  void set_symbol_engine(const SymbolEnginePtr &se) { symbol_engine_ = se; }
  const SymbolEnginePtr &symbol_engine() const { return symbol_engine_; }

  // Only used for func_graph manager to control resource free.
  int attached_mng_cnt() const { return attached_mng_cnt_; }

  // Reserve the func graph, not to release in manager.
  void set_reserved(bool reserved) { reserved_ = reserved; }
  bool reserved() const { return reserved_; }

 private:
  // Only used for func_graph manager to control resource free.
  void IncAttachedMngCnt() { attached_mng_cnt_++; }
  void DecAttachedMngCnt() { attached_mng_cnt_--; }
  // Clear all info from manager.
  void ClearAllResource();

  // Graph is manipulated by manager and others.
  friend FuncGraphManager;

  // All nodes of the function.
  AnfNodeSet nodes_;

  // All switch nodes of the function.
  AnfNodeSet switch_nodes_;

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
  AnfNodePtrList parameters_;
  AnfNodePtrList parameter_obj_nodes_;

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
  CNodeWeakPtr return_;
  // Before release all func graphs in Manager, reset the owner firstly.
  CNodePtr return_owner_;

  // Back-ref to its manager.
  // Hold a weak ref to FuncGraphManager as FuncGraphManager also hold many ref to FuncGraph.
  // Otherwise, FuncGraph and FuncGraphManager will make a reference cycles.
  // Notes: Normally, there will be a global FuncGraphManager, it will hold all FuncGraphs.
  // In some ut test cases, they may use local FuncGraphManager in function which
  // generating the func graph, when go outside of that function, func graph will have no
  // FuncGraphManager. In that special case, Manage() should be called to make the func graph
  // managed.
  std::weak_ptr<FuncGraphManager> manager_;
  int attached_mng_cnt_{0};

  GraphDebugInfoPtr debug_info_;
  void GenerateKwargReplNode(const FuncGraphPtr &specialized_graph, const AnfNodePtrList &kwarg_keys_tuple_nodes,
                             const AnfNodePtrList &kwarg_values_tuple_nodes,
                             mindspore::HashMap<AnfNodePtr, AnfNodePtr> *repl_nodes) const;

  // CNode order which relates to origin code order.
  std::list<CNodeWeakPtr> order_;
  bool stub_;

  // The graph is used as some input of Switch, SwitchLayer, or Partial.
  std::shared_ptr<bool> indirect_;

  int64_t stage_;
  int64_t segment_;
  bool dynamic_shape_ = false;
  std::unordered_map<AbstractBasePtrList, FuncGraphPtr, abstract::AbstractBasePtrListHasher,
                     abstract::AbstractBasePtrListEqual>
    func_graph_cache_;

  // If the graph was changed, it should be dropped in cache data_converter::object_map_
  // which used by ConvertToFuncGraph.
  bool dropped_{false};
  // If the graph is a bprop graph, it should has a hash of the bprop function.
  std::string bprop_hash_;
  // If the graph is a bprop graph, it should has a filepath of the bprop function.
  std::string bprop_filepath_;

  // If the graph is decorated with @jit and runs grad process in pynative mode,
  // forward nodes used in grad graph will be added to output for holding output values.
  bool modify_output_{false};
  mindspore::OrderedSet<AnfNodePtr> used_forward_nodes_;
  // If the func_graph is input of switch node, and the condition of switch is AbstractTensor, need set true.
  bool is_tensor_condition_branch_{false};
  // Corresponding python obj.
  ValuePtr python_obj_{nullptr};
  std::string phase_;
  // Own all nodes in the func graph.
  std::list<AnfNodePtr> own_nodes_;
  // the manager of symbolic shape's symbols and operations.
  SymbolEnginePtr symbol_engine_;
  // Reserve the func graph, not to release in manager.
  bool reserved_{false};
};

inline CNodePtr NewCNode(const AnfNodePtrList &inputs, const FuncGraphPtr &fg) {
  MS_EXCEPTION_IF_NULL(fg);
  return fg->NewCNode(inputs);
}

inline CNodePtr NewCNode(AnfNodePtrList &&inputs, const FuncGraphPtr &fg) {
  MS_EXCEPTION_IF_NULL(fg);
  return fg->NewCNode(std::move(inputs));
}

MS_CORE_API SeenNum NewFgSeenGeneration();

// Find the root cnodes of a segment of cnodes.
std::shared_ptr<OrderedSet<CNodePtr>> FindRoots(const std::vector<CNodePtr> &segment);
// Find the leaf cnodes of a segment of cnodes.
std::shared_ptr<OrderedSet<CNodePtr>> FindLeaves(const std::vector<CNodePtr> &segment);
}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_FUNC_GRAPH_H_
