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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_AD_AUTO_GRAD_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_AD_AUTO_GRAD_H_

#include <memory>
#include <utility>
#include <map>
#include <vector>
#include <string>
#include <tuple>
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "frontend/expander/bprop/bprop.h"
#include "pipeline/pynative/base.h"
#include "mindspore/ccsrc/include/backend/kernel_graph.h"
#include "runtime/pynative/async/async_hqueue.h"

namespace mindspore {
namespace pynative {
namespace autograd {
using KernelGraph = session::KernelGraph;

struct GradAttr {
  GradAttr(bool get_all, bool get_by_list, bool sens_param, bool get_by_position, bool weight_param_is_tuple)
      : grad_all_inputs(get_all),
        grad_weights(get_by_list),
        has_sens(sens_param),
        get_by_position(get_by_position),
        weight_param_is_tuple(weight_param_is_tuple) {}

  bool grad_all_inputs;
  bool grad_weights;
  bool has_sens;
  bool get_by_position;
  bool weight_param_is_tuple;
};

class VariableAdjoint;
class FunctionNode {
 public:
  FunctionNode(KernelGraphPtr tape, const AnfNodePtr &dout)
      : tape_(std::move(tape)), accumulate_dout_(dout), fake_dout_(dout) {}
  void AddNextEdge(const std::shared_ptr<VariableAdjoint> &next_variable, const AnfNodePtr &din);
  void UpdateAccumulativeDout(const AnfNodePtr &new_dout);
  const std::vector<std::pair<std::shared_ptr<VariableAdjoint>, AnfNodePtr>> &next_edges() const { return next_edges_; }
  const KernelGraphPtr &tape() { return tape_; }
  const AnfNodePtr &accumulate_dout() const { return accumulate_dout_; }
  void set_accumulate_dout(const AnfNodePtr &accumulate_dout) { accumulate_dout_ = accumulate_dout; }
  void ReplaceEdges();
  const AnfNodePtr &fake_dout() const { return fake_dout_; }

 private:
  AnfNodePtr HyperAdd(const AnfNodePtr &left_node, const AnfNodePtr &right_node);
  // Bprop func graph
  const KernelGraphPtr tape_;
  // Input of dout for this bprop function
  AnfNodePtr accumulate_dout_;
  // First we generate a fake dout
  const AnfNodePtr fake_dout_;
  // The pair.first is a variable, pair.second is dout of variable
  std::vector<std::pair<std::shared_ptr<VariableAdjoint>, AnfNodePtr>> next_edges_;
  // Replace next_edges where din == dout in brprop function
  std::vector<int> need_replace_edges_;
};
using FunctionNodePtr = std::shared_ptr<FunctionNode>;

// Variable represent a parameter or output of a middle cnode
class VariableAdjoint {
 public:
  VariableAdjoint() = default;
  VariableAdjoint(FunctionNodePtr fn, ValuePtr out_value, bool is_leaf = false)
      : fn_(std::move(fn)), out_value_(std::move(out_value)), is_leaf_(is_leaf) {}

  ValuePtr out_value() const { return out_value_; }
  FunctionNodePtr fn() const { return fn_; }
  const string &fake_prim_name() const { return fake_prim_name_; }
  void set_fake_prim_name(const string &fake_prim_name) { fake_prim_name_ = fake_prim_name; }
  bool is_fake_bprop() const { return is_fake_bprop_; }
  void set_is_fake_bprop(bool is_fake_bprop) { is_fake_bprop_ = is_fake_bprop; }
  bool is_need_propagate() const { return is_need_propagate_; }
  void set_is_need_propagate(bool is_need_grad) { is_need_propagate_ = is_need_grad; }
  bool is_need_grad() const { return is_need_grad_; }
  void set_is_need_grad(bool is_need_grad) { is_need_grad_ = is_need_grad; }
  bool is_leaf() const { return is_leaf_; }
  void set_is_leaf(bool is_leaf) { is_leaf_ = is_leaf; }
  AnfNodePtr k_node() const { return k_node_; }
  void set_k_node(const AnfNodePtr &k_node) { k_node_ = k_node; }
  AnfNodePtr RealDout();
  std::string ToString() const;

 private:
  // Abstract bprop function
  FunctionNodePtr fn_{nullptr};
  ValuePtr out_value_{nullptr};
  // If node has not bprop, we record its prim name
  std::string fake_prim_name_;
  // Record this node is a fake bprop
  bool is_fake_bprop_{false};
  // Flag to judge need to propagrate
  bool is_need_propagate_{false};
  // Flag to judge variable whether need grad
  bool is_need_grad_{true};
  // Flag the variable is a leaf in bprop.
  bool is_leaf_{false};
  // K mapped cnode for primal CNode; primal CNode is owned by primal funcgraph, this is owned by tape_;
  AnfNodePtr k_node_{nullptr};
};
using VariableAdjointPtr = std::shared_ptr<VariableAdjoint>;
using expander::bprop::UserType;

struct AdParam {
  AdParam() : tape_(std::make_shared<KernelGraph>()), fg_(std::make_shared<FuncGraph>()) {}
  // Bprop funcgraph
  KernelGraphPtr tape_;
  FuncGraphPtr fg_;
  VariableAdjointPtr last_variable_{nullptr};
  // Just for ad graph
  AnfNodePtr last_node_{nullptr};
  ValuePtr sens_value_;
  // Bprop dins of each variable or middle out
  OrderedMap<AnfNodePtr, VariableAdjointPtr> anfnode_to_variable_adjoint_;
  OrderedSet<VariableAdjointPtr> variable_adjoint_set_;
  // Record cnode's input map for tape_
  expander::bprop::UserMap users_;
  std::vector<std::tuple<AnfNodePtr, CNodePtr, size_t>> lazy_user_data_;
};
using AdParamPtr = std::shared_ptr<AdParam>;

class AutoGradCellImpl {
 public:
  AutoGradCellImpl(const std::vector<ValuePtr> &input_param_values, const AbstractBasePtrList &abs_list,
                   size_t op_num_in_bprop_graph, const AsyncHqueuePtr &assist_queue, bool enable_async,
                   bool grad_by_value);
  ~AutoGradCellImpl() = default;
  // Reverse connect bprop of op
  bool KPynativeOp(const GradParamPtr &grad_param);
  // Reverse connect jit or higher order sub bprop funcgraph
  bool KPynativeWithFProp(const GradParamPtr &grad_param);
  // Update top cell output, record last_node
  void UpdateOutputNodeOfTopCell(const ValuePtr &sens_out);
  FuncGraphPtr Finish(const tensor::TensorPtrList &weights, const std::vector<size_t> &grad_position,
                      const GradAttr &grad_attr);
  // Grad for function graph
  inline AdParamPtr ad_param() const {
    MS_EXCEPTION_IF_NULL(ad_param_);
    return ad_param_;
  }
  // Input node is user cnode one of input, index is user input index
  // User->input(index) is input node
  void AddUser(const AnfNodePtr &input, const CNodePtr &user, size_t index);
  inline bool grad_by_value() { return grad_by_value_; }

 private:
  FuncGraphPtr GradFuncGraph(const GradParamPtr &grad_param);
  AnfNodePtr GetKnode(const PrimitivePtr &prim, const CNodePtr &cnode, const AnfNodePtrList &cnode_inputs,
                      bool jit_by_value);
  CNodePtr GetBpropGraphCNode(const GradParamPtr &grad_param, const AnfNodePtrList &args, AnfNodePtr *const tape_dout);
  CNodePtr GetBPropFromExpander(const GradParamPtr &grad_param, const AnfNodePtrList &args,
                                AnfNodePtr *const tape_dout);
  CNodePtr GetBPropFromFProp(const GradParamPtr &grad_param, const AnfNodePtrList &args, AnfNodePtr *const tape_dout);
  CNodePtr GetBPropCNode(const GradParamPtr &grad_param, const AnfNodePtrList &args, const FuncGraphPtr &bprop_graph,
                         bool cache_hit, AnfNodePtr *const tape_dout);
  void GradGraphByExpander(const GradParamPtr &grad_param);
  ValuePtrList GetInputArgs(const CNodePtr &cnode, AnfNodePtrList *cnode_inputs) const;
  void CreateParameterAdjoint(const GradParamPtr &grad_param) const;
  void ProcessMetaFuncGraphOp(const GradParamPtr &grad_param, const PrimitivePtr &prim, const CNodePtr &cnode,
                              const ValuePtrList &op_args, const ValuePtr &out);
  // Construct input as cnode for expander
  CNodePtr ConstructBpropGraphInput(const GradParamPtr &grad_param, const AnfNodePtr &dout,
                                    const VariableAdjointPtr &variable_adjoint, bool is_custom_prim);
  // Back propagate for one node;
  void UpdateNextEdgesAsync(const VariableAdjointPtr &variable, const std::vector<CNodePtr> &dins,
                            const GradParamPtr &grad_param);
  void UpdateNextEdges(const VariableAdjointPtr &variable, const std::vector<CNodePtr> &dins,
                       const ValuePtrList &input_value, const abstract::AbstractBasePtrList &abs, bool grad_by_value);
  void UpdateNextEdge(const FunctionNodePtr &fn, const AnfNodePtr &din, const ValuePtr &input_arg,
                      const AbstractBasePtr &abs);

  AbstractBasePtr BuildForwardLastNode();
  // Add parameter(weights) to anfnode_to_variable_adjoint_
  ParameterPtr CreateTapeParameter(const tensor::TensorPtr &tensor, const abstract::AbstractBasePtr &abs);
  ParameterPtr AddParameterNode(const tensor::TensorPtr &tensor, const abstract::AbstractBasePtr &abs);
  AnfNodePtr MapParameter(const ValuePtr &value, const abstract::AbstractBasePtr &abs);
  ParameterPtr ExtractParameter(const tensor::TensorPtr &tensor) const;
  AnfNodePtrList ExtractParamters(const tensor::TensorPtrList &weights) const;
  void UpdateSensParameter(const ValuePtr &value);
  AnfNodePtr TraceShape(const FunctionNodePtr &fn, const ValuePtr &out_value, const abstract::AbstractBasePtr &out_abs,
                        const tensor::TensorPtr &input_tensor, const AnfNodePtr &din);
  void BuildBPropCutCNode(const CNodePtr &cnode, const PrimitivePtr &prim, std::vector<CNodePtr> *outputs);
  void BuildCustomBpropCNode(const CNodePtr &cnode, const PrimitivePtr &prim, std::vector<CNodePtr> *outputs);
  void BuildFakeBpropCNode(const CNodePtr &cnode, std::vector<CNodePtr> *outputs) const;
  // Replace input or weights parameter from primal funcgraph to parameters of tape_;
  void ReplacePrimalParameter(bool has_sens_arg);
  void UpdateTapeParameter(const tensor::TensorPtr &tensor);
  void DoParameterReplaceByManager(bool has_sens_arg);
  void DoParameterReplaceByUser(bool has_sens_arg);
  // Set sens and weights parameter nodes by user input info
  void SetSensAndWeights(const tensor::TensorPtrList &weights, bool has_sens_arg);
  // get last reverse iterator
  OrderedSet<VariableAdjointPtr>::reverse_iterator GetLastNodeReverseIter();

  void BackPropagate();
  // Set return node according to grad flag
  void SetOutput(const tensor::TensorPtrList &weights, const std::vector<size_t> &grad_position,
                 const GradAttr &grad_attr);
  AnfNodePtr GetGradNodeByIndex(const tensor::TensorPtr &tensor);
  AnfNodePtr GetInputGrad(bool grad_all_inputs, bool get_by_position, const std::vector<size_t> &grad_position);
  AnfNodePtr GetWeightGrad(bool grad_weights, const tensor::TensorPtrList &weights, bool weight_param_is_tuple);
  void LazyAddUser(const AnfNodePtr &node, const CNodePtr &user, size_t index);
  void UpdateLazyUser();
  void AddTupleGetItemUser(const AnfNodePtr &input, const CNodePtr &user, size_t index);
  void Replace(const AnfNodePtr &old_node, const AnfNodePtr &new_node, UserType *user, bool need_update = false);
  // To elimate tuplegetitem cnode
  void ElimateTupleGetItem();

  // Fbprop
  void SetKNodeInfo(const ValuePtr &value, const AnfNodePtr &k_node, const AbstractBasePtr &out_abs);
  AnfNodePtr BuildKNode(const AnfNodePtr &prim, const GradParamPtr &grad_param, bool from_single_op);
  void BuildKNodeListFromPrimalCNode(const ValuePtrList &op_args, const abstract::AbstractBasePtrList &input_abs,
                                     AnfNodePtrList *const node_list);
  void BuildKNodeListForHighOrderGraph(const ValuePtrList &op_args, const abstract::AbstractBasePtrList &input_abs,
                                       AnfNodePtrList *const node_list);
  AnfNodePtr BuildKNodeForCNodeInput(const AnfNodePtr &input_node);
  AnfNodePtr BuildKNodeForCNodeInput(const ValuePtr &input, const abstract::AbstractBasePtr &abs);
  AnfNodePtr BuildKNodeForMakeTuple(const AnfNodePtr &input_node);
  AnfNodePtr BuildKNodeForTupleGetItem(const AnfNodePtr &input_node);

  // Last cnode of this Cell, may be a primitive op or cell with user defined bprop.
  AdParamPtr ad_param_{nullptr};
  // Top cell inputs
  std::vector<std::pair<AnfNodePtr, VariableAdjointPtr>> cell_inputs_;
  // These weights need to calculate gradient.
  mindspore::HashSet<std::string> need_grad_weights_;
  AnfNodePtrList weights_used_in_graph_;
  AnfNodePtrList k_nodes_used_in_graph_;
  // Flag for ms_funtcion and high order
  bool grad_by_value_{false};
  bool need_do_manager_replace_{false};
  AsyncHqueuePtr assist_queue_{nullptr};
  bool enable_async_{false};
  std::string device_target_;
};
using AutoGradCellImplPtr = std::shared_ptr<AutoGradCellImpl>;

void ClearPyNativeAutoGradStaticRes();
}  // namespace autograd
}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_AD_AUTO_GRAD_H_
