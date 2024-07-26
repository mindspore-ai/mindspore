/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_IR_IR_BPROP_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_IR_IR_BPROP_H_

#include <utility>
#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include "ir/anf.h"
#include "pipeline/pynative/base.h"
#include "pipeline/pynative/grad/variable.h"
#include "pipeline/pynative/grad/ir/ir_pass.h"
#include "frontend/expander/bprop/bprop.h"

namespace mindspore::pynative::autograd {
void ClearAutoGradCache();
using KernelGraph = session::KernelGraph;
struct AdParam {
  AdParam() : tape_(std::make_shared<KernelGraph>()), fg_(std::make_shared<FuncGraph>()) {}
  // Bprop funcgraph
  KernelGraphPtr tape_;
  FuncGraphPtr fg_;
  IrVariablePtr last_variable_{nullptr};
  // Just for ad graph
  AnfNodePtr last_node_{nullptr};
  ValuePtr sens_value_;
  // Bprop dins of each variable or middle out
  OrderedMap<AnfNodePtr, IrVariablePtr> anfnode_to_variable_adjoint_;
  OrderedSet<IrVariablePtr> variable_adjoint_set_;
  // Record cnode's input map for tape_
  expander::bprop::UserMap users_;
  expander::bprop::UserType reverse_users_;
  AnfNodePtrList weights_used_in_graph_;
  std::vector<std::tuple<AnfNodePtr, CNodePtr, size_t>> lazy_user_data_;
};
using AdParamPtr = std::shared_ptr<AdParam>;

class IrBprop {
 public:
  IrBprop(AdParamPtr ad_param, std::string device_target, bool grad_by_value, bool is_run_recompute = false)
      : ad_param_(std::move(ad_param)), grad_by_value_(grad_by_value), is_run_recompute_(is_run_recompute) {
    pass_forward_ = std::make_shared<bprop_pass::IrPassForward>(this, std::move(device_target), grad_by_value_);
  }

  // Get graph bporp graph by ad::grad or by expander
  std::pair<bool, FuncGraphPtr> GetBpropGraph(const GradParamPtr &grad_param);

  // Build custom
  void BuildCustomBpropCNode(const CNodePtr &cnode, const PrimitivePtr &prim, std::vector<CNodePtr> *outputs);

  // Create bprop_cut cnode in bprop graph
  void BuildBPropCutCNode(const CNodePtr &cnode, const PrimitivePtr &prim, std::vector<CNodePtr> *outputs,
                          size_t weight_size = 0, bool is_need_recompute = false);
  // Get parameter from a value
  AnfNodePtr MapParameter(const ValuePtr &value, const abstract::AbstractBasePtr &abs,
                          std::vector<std::pair<tensor::BaseTensorPtr, AutoGradMetaDataPtr>> *param_meta_grad_info);

  // Create variable for parameter
  ParameterPtr AddParameterNode(const tensor::BaseTensorPtr &tensor, const abstract::AbstractBasePtr &abs);

  // Create a new parameter
  ParameterPtr CreateTapeParameter(const tensor::BaseTensorPtr &tensor, const abstract::AbstractBasePtr &abs);

  // Update cnode dout
  void UpdateNextEdges(const VariablePtr &variable, const std::vector<CNodePtr> &dins, const ValuePtrList &inputs_value,
                       const abstract::AbstractBasePtrList &abs, const string &op_name = "");

  // Used for ture dout repalce
  void AddUser(const AnfNodePtr &node, const CNodePtr &user, size_t index);

  // Used for high grad
  void AddReverseUser(const AnfNodePtr &node, const CNodePtr &user, size_t index);

  // Create link for op grad graph and generate a bprop graph
  void BackPropagate();

  // Get lase node variable
  AbstractBasePtr BuildForwardLastNode();

  // Replace for true dout
  void Replace(const AnfNodePtr &old_node, const AnfNodePtr &new_node, expander::bprop::UserType *user,
               bool need_update = false);

  AdParamPtr ad_param() const { return ad_param_; }
  inline bool bprop_graph_run_by_single_op() { return bprop_graph_run_by_single_op_; }
  inline void set_bprop_graph_run_by_single_op(bool bprop_graph_run_by_single_op) {
    bprop_graph_run_by_single_op_ = bprop_graph_run_by_single_op_ || bprop_graph_run_by_single_op;
  }

 private:
  // Get bprop graph by ad::grad
  FuncGraphPtr GetBpropGraphFromFprop(const GradParamPtr &grad_param);

  // Get Bprop by expander
  FuncGraphPtr GetBpropGraphFromExpander(const GradParamPtr &grad_param);

  // Use topo grad for every cnode
  void GradGraphByExpander(const GradParamPtr &grad_param);

  // Create variable for param
  void CreateParameterAdjoint(const GradParamPtr &grad_param) const;

  // Use pass for cnode inputs
  void PrepareGradCNodeInputs(const PrimitivePtr &prim, const CNodePtr &cnode, ValuePtrList *inputs_value,
                              AnfNodePtrList *cnode_inputs);

  // Get knode and value for cnode inputs
  ValuePtrList GetInputArgs(const CNodePtr &cnode, AnfNodePtrList *cnode_inputs) const;

  // Do grad for a cnode
  void GradCNode(const PrimitivePtr &prim, const CNodePtr &cnode, const GradParamPtr &grad_param,
                 const ValuePtrList &inputs_value, AnfNodePtrList *cnode_inputs);

  // Build knode for MakeTuple
  AnfNodePtr BuildKNodeForMakeTuple(const AnfNodePtr &input_node);

  // Build knode for TupleGetItem
  AnfNodePtr BuildKNodeForTupleGetItem(const AnfNodePtr &input_node);

  // Get knode for cnode inputs
  AnfNodePtr BuildKNodeForCNodeInput(const AnfNodePtr &input);

  // Get a compute cnode
  AnfNodePtr GetKnode(const PrimitivePtr &prim, const CNodePtr &cnode, const AnfNodePtrList &cnode_inputs,
                      bool jit_by_value);

  // Set dout for every input arg
  void UpdateNextEdge(const IrFunctionNodePtr &fn, const AnfNodePtr &din, const ValuePtr &input_arg,
                      const AbstractBasePtr &abs);

  // Used for dict inputs
  void UpdateNextEdgeForDict(const IrFunctionNodePtr &fn, const AnfNodePtr &din, const ValuePtr &input_arg,
                             const AbstractBasePtr &abs);

  // Set din for corresponding input
  AnfNodePtr TraceInput(const IrFunctionNodePtr &fn, const ValuePtr &out_value,
                        const abstract::AbstractBasePtr &out_abs, const tensor::BaseTensorPtr &input_tensor,
                        const AnfNodePtr &din);

  // Used for dict input
  AnfNodePtr TraceInputForDict(const IrFunctionNodePtr &fn, const ValuePtr &out_value,
                               const abstract::AbstractBasePtr &out_abs, const tensor::BaseTensorPtr &input_tensor,
                               const AnfNodePtr &din);

  // Get last node variable
  OrderedSet<IrVariablePtr>::reverse_iterator GetLastNodeReverseIter();

  // Used for tuplegetiem elimate
  void AddTupleGetItemUser(const AnfNodePtr &node, const CNodePtr &user, size_t index);

  // For lazy user
  void UpdateLazyUser();

  // Input node is user cnode one of input, index is user input index
  // User->input(index) is input node
  void LazyAddUser(const AnfNodePtr &node, const CNodePtr &user, size_t index);

  AdParamPtr ad_param_{nullptr};
  bool grad_by_value_{false};
  bool is_run_recompute_{false};
  // Flag for ms_funtcion and high order
  bool bprop_graph_run_by_single_op_{false};
  bprop_pass::PyNativePassForwardPtr pass_forward_;
};
using IrBpropPtr = std::unique_ptr<IrBprop>;
}  // namespace mindspore::pynative::autograd
#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_IR_IR_BPROP_H_
