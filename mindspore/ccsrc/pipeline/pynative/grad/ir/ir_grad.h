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
#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_IR_IR_GRAD_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_IR_IR_GRAD_H_

#include <utility>
#include <memory>
#include <map>
#include <vector>
#include <string>
#include <tuple>
#include "ir/anf.h"
#include "pipeline/pynative/base.h"
#include "pipeline/pynative/grad/ir/ir_bprop.h"
#include "ir/func_graph.h"
#include "pipeline/pynative/grad/variable.h"
#include "pipeline/pynative/grad/ir/ir_pass.h"
#include "include/backend/kernel_graph.h"
#include "runtime/pipeline/async_hqueue.h"
#include "pipeline/pynative/grad/auto_grad.h"

namespace mindspore {
namespace pynative {
namespace autograd {
class IrGrad : public AutoGrad {
 public:
  IrGrad(const std::vector<ValuePtr> &input_param_values, const AbstractBasePtrList &abs_list,
         size_t op_num_in_bprop_graph, bool grad_by_value, bool is_run_recompute);
  ~IrGrad() = default;

  // Reverse connect bprop of op
  bool KPynativeOp(const GradParamPtr &grad_param) override;
  // Reverse connect jit or higher order sub bprop funcgraph
  bool KPynativeWithFProp(const GradParamPtr &grad_param) override;
  // Update top cell output, record last_node
  void UpdateOutputNodeOfTopCell(const ValuePtr &sens_out) override;
  FuncGraphPtr Finish(const tensor::BaseTensorPtrList &weights, const std::vector<size_t> &grad_position,
                      const GradAttr &grad_attr);

  // Grad for function graph
  inline AdParamPtr ad_param() const {
    MS_EXCEPTION_IF_NULL(ad_param_);
    return ad_param_;
  }
  inline bool bprop_graph_run_by_single_op() { return ir_bprop_->bprop_graph_run_by_single_op(); }
  void set_bprop_graph_run_by_single_op(bool bprop_graph_run_by_single_op) {
    bool flag = ir_bprop()->bprop_graph_run_by_single_op() || bprop_graph_run_by_single_op;
    ir_bprop_->set_bprop_graph_run_by_single_op(flag);
  }

 private:
  CNodePtr GetBpropGraphCNode(const GradParamPtr &grad_param, const AnfNodePtrList &args, AnfNodePtr *const tape_dout);
  CNodePtr GetBPropCNode(const GradParamPtr &grad_param, const AnfNodePtrList &args, const FuncGraphPtr &bprop_graph,
                         bool cache_hit, AnfNodePtr *const tape_dout);
  // Construct input as cnode for expander
  CNodePtr ConstructBpropGraphInput(const GradParamPtr &grad_param, const AnfNodePtr &dout,
                                    const VariablePtr &variable_adjoint, const AnfNodePtr &k_node, bool is_custom_prim);
  ParameterPtr ExtractParameter(const tensor::BaseTensorPtr &tensor) const;
  void UpdateSensParameter(const ValuePtr &value);
  // Replace input or weights parameter from primal funcgraph to parameters of tape_;
  void ReplacePrimalParameter(bool has_sens_arg);
  void UpdateTapeParameter(const tensor::BaseTensorPtr &tensor);
  void DoParameterReplaceByManager(bool has_sens_arg);
  void DoParameterReplaceByUser(bool has_sens_arg, expander::bprop::UserType *user);
  // Set sens and weights parameter nodes by user input info
  void SetSensAndWeights(const tensor::BaseTensorPtrList &weights, bool has_sens_arg);

  // Set return node according to grad flag
  void SetOutput(const tensor::BaseTensorPtrList &weights, const std::vector<size_t> &grad_position,
                 const GradAttr &grad_attr);
  AnfNodePtr GetGradNodeByIndex(const tensor::BaseTensorPtr &tensor);
  AnfNodePtr GetInputGrad(bool grad_all_inputs, bool get_by_position, const std::vector<size_t> &grad_position);
  AnfNodePtr GetWeightGrad(bool grad_weights, const tensor::BaseTensorPtrList &weights, bool weight_param_is_tuple);
  // To elimate tuplegetitem cnode
  void ElimateTupleGetItem();

  // Fbprop
  void SetKNodeInfo(const ValuePtr &value, const AnfNodePtr &k_node, const AbstractBasePtr &out_abs);
  AnfNodePtr BuildKNode(const AnfNodePtr &prim, const GradParamPtr &grad_param, bool from_single_op);
  void BuildKNodeListFromPrimalCNode(const ValuePtrList &op_args, const abstract::AbstractBasePtrList &input_abs,
                                     AnfNodePtrList *const node_list);
  AnfNodePtr BuildKNodeForCNodeInput(const ValuePtr &input, const abstract::AbstractBasePtr &abs);
  void BuildKNodeListForHighOrderGraph(const ValuePtrList &op_args, const abstract::AbstractBasePtrList &input_abs,
                                       AnfNodePtrList *const node_list);

  // Last cnode of this Cell, may be a primitive op or cell with user defined bprop.
  AdParamPtr ad_param_{nullptr};
  // Top cell inputs
  std::vector<std::pair<AnfNodePtr, VariablePtr>> cell_inputs_;
  // These weights need to calculate gradient.
  mindspore::HashSet<std::string> need_grad_weights_;
  // Keep reference for cnode
  AnfNodePtrList k_nodes_used_in_graph_;
  bool need_do_manager_replace_{false};
};
}  // namespace autograd
}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_IR_IR_GRAD_H_
