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
#include "pipeline/jit/pi/auto_grad/bprop_func_graph_manager.h"
#include <algorithm>
#include <iterator>
#include <memory>
#include "base/base.h"
#include "frontend/expander/bprop/bprop_meta_func_graph.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/ad/grad.h"
#include "frontend/optimizer/irpass.h"
#include "ir/func_graph_cloner.h"
#include "ir/value.h"
#include "ops/framework_ops.h"
#include "ops/sequence_ops.h"
#include "pipeline/jit/ps/pass.h"
#include "pipeline/jit/ps/resource.h"
#include "pipeline/pynative/pynative_utils.h"
#include "ops/array_ops.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace pijit {
namespace grad {
FuncGraphPtr BpropFuncGraphManager::PrimBpropGraphPass(const FuncGraphPtr &prim_grad_graph) {
  opt::irpass::OptimizeIRPassLib irpass;
  opt::OptPassGroupMap map({
    {"inline", opt::OptPassConfig({irpass.inline_})},
    {"renormalize", opt::OptPassConfig::Renormalize()},
  });
  auto resource = std::make_shared<pipeline::Resource>();
  resource->set_func_graph(prim_grad_graph);
  auto graph_opt = opt::Optimizer::MakeOptimizer("prim_bprop_graph_opt", resource, map);
  return graph_opt->step(prim_grad_graph, false);
}

FuncGraphPtr BpropFuncGraphManager::GetAccumulateGraph(const ValuePtr &dout, const ValuePtr &factor) {
  auto func_graph = std::make_shared<FuncGraph>();
  func_graph->debug_info()->set_name("Accumulate_" + dout->ToString());
  auto param_dout = func_graph->add_parameter();
  param_dout->set_abstract(dout->ToAbstract()->Broaden());
  auto param_factor = func_graph->add_parameter();
  param_factor->set_abstract(factor->ToAbstract()->Broaden());
  auto prim = prim::GetPythonOps("hyper_add");
  auto new_dout = func_graph->NewCNode({NewValueNode(prim), param_dout, param_factor});
  func_graph->set_output(new_dout);
  return PrimBpropGraphPass(func_graph);
}

FuncGraphPtr BpropFuncGraphManager::GetPrimBpropGraph(const PrimitivePtr &prim, const ValuePtrList &inputs,
                                                      const ValuePtr &out, const ValuePtr &dout) {
  abstract::AbstractBasePtrList args_abs;
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(args_abs),
                 [](const ValuePtr &node) { return node->ToAbstract()->Broaden(); });
  args_abs.push_back(out->ToAbstract()->Broaden());
  args_abs.push_back(dout->ToAbstract()->Broaden());
  return GetPrimBpropGraph(prim, args_abs);
}

FuncGraphPtr BpropFuncGraphManager::GetPrimBpropGraph(const PrimitivePtr &prim,
                                                      const abstract::AbstractBasePtrList &args_abs) {
  auto prim_name = prim->name();
  if (prim_to_bprop_.find(prim_name) != prim_to_bprop_.end()) {
    auto func_graph = BasicClone(prim_to_bprop_.at(prim_name));
    MS_EXCEPTION_IF_CHECK_FAIL(func_graph->parameters().size() == args_abs.size(),
                               "Arguments is not match parameters.");
    for (size_t index = 0; index < args_abs.size(); index++) {
      func_graph->parameters()[index]->set_abstract(args_abs[index]);
    }
    return PrimBpropGraphPass(func_graph);
  }
  const expander::bprop::BpropHandle *handle = expander::bprop::BpropIRBuilderFactory::Instance().GetBuilder(prim_name);
  auto meta_graph = std::make_shared<expander::bprop::BpropMetaFuncGraph>(prim, handle);
  auto grad_graph = meta_graph->GenerateFuncGraph(args_abs);
  prim_to_bprop_[prim_name] = BasicClone(grad_graph);
  return PrimBpropGraphPass(grad_graph);
}

// Modify the output node of func_graph to add forward nodes used in bprop graph.
void ModifyOutputNode(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  const auto &used_forward_nodes = func_graph->used_forward_nodes();
  if (used_forward_nodes.empty()) {
    return;
  }

  // Create a new make tuple node to hold all forward used nodes.
  abstract::AbstractBasePtrList added_abs_list;
  AnfNodePtrList added_node_list{NewValueNode(prim::kPrimMakeTuple)};
  for (const auto &node : used_forward_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    (void)added_node_list.emplace_back(node);
    (void)added_abs_list.emplace_back(node->abstract());
  }
  AnfNodePtr added_output_node = func_graph->NewCNode(std::move(added_node_list));
  AbstractBasePtr added_output_abs = std::make_shared<abstract::AbstractTuple>(added_abs_list);
  added_output_node->set_abstract(added_output_abs);

  // Get original output node and abstract, and merge original output node and used forward nodes to return node.
  auto original_output_node = func_graph->output();
  MS_EXCEPTION_IF_NULL(original_output_node);
  auto original_output_abs = original_output_node->abstract();
  MS_EXCEPTION_IF_NULL(original_output_abs);
  AnfNodePtrList new_output_nodes{NewValueNode(prim::kPrimMakeTuple), original_output_node, added_output_node};
  auto merge_node = func_graph->NewCNode(std::move(new_output_nodes));
  abstract::AbstractBasePtrList new_output_abs{original_output_abs, added_output_abs};
  merge_node->set_abstract(std::make_shared<abstract::AbstractTuple>(new_output_abs));
  func_graph->set_output(merge_node);

  // Clear
  func_graph->set_modify_output(true);
  func_graph->ClearUsedForwardNodes();
}

FuncGraphPtr BpropFuncGraphManager::GetFuncGraphBpropGraph(const FuncGraphPtr &forward_graph,
                                                           const ValuePtrList &inputs, const ValuePtr &out,
                                                           const ValuePtr &dout) {
  if (func_graph_to_bprop_.find(forward_graph) != func_graph_to_bprop_.end()) {
    return func_graph_to_bprop_.at(forward_graph);
  }
  abstract::AbstractBasePtrList args_abs;
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(args_abs),
                 [](const ValuePtr input) { return input->ToAbstract()->Broaden(); });
  auto resource = std::make_shared<pipeline::Resource>();
  resource->set_args_abs(args_abs);
  resource->set_func_graph(forward_graph);
  ModifyOutputNode(forward_graph);
  auto ps_jit_instance = pynative::PyNativeExecutor::GetInstance()->grad_executor()->jit();
  bool jit_eliminate_forward = ps_jit_instance->eliminate_forward();
  ps_jit_instance->set_eliminate_forward(false);
  // Control flow not eliminate forward
  auto is_control_flow = pynative::PyNativeAlgo::Common::IsControlFlowGraph(forward_graph);
  auto grad_graph =
    ad::Grad(is_control_flow ? BasicClone(forward_graph) : forward_graph, opt::Optimizer::MakeEmptyOptimizer(resource));
  MS_EXCEPTION_IF_NULL(grad_graph);
  ps_jit_instance->set_eliminate_forward(jit_eliminate_forward);
  auto output = grad_graph->output();
  auto grad_out = grad_graph->NewCNodeInOrder(prim::kPrimTupleGetItem, {output, NewValueNode(MakeValue<int64_t>(1))});
  auto param_dout = grad_graph->add_parameter();
  param_dout->set_abstract(forward_graph->output()->abstract());
  auto new_dout = grad_graph->NewCNodeInOrder({grad_out, param_dout});
  auto size =
    GetValueNode<FuncGraphPtr>(output->cast<CNodePtr>()->input(2))->output()->cast<CNodePtr>()->inputs().size();
  AnfNodePtrList ins;
  for (size_t index = 1; index < size - 1; index++) {
    auto d_out =
      grad_graph->NewCNodeInOrder(prim::kPrimTupleGetItem, {new_dout, NewValueNode(MakeValue<int64_t>(index))});
    ins.push_back(d_out);
  }
  new_dout = grad_graph->NewCNodeInOrder(prim::kPrimMakeTuple, ins);
  grad_graph->set_output(new_dout);
  resource->set_func_graph(grad_graph);
  for (size_t index = 0; index < args_abs.size(); index++) {
    grad_graph->parameters()[index]->set_abstract(args_abs[index]);
  }
  grad_graph = pipeline::JitBpropGraphPass(resource, true);
  func_graph_to_bprop_[forward_graph] = grad_graph;
  return grad_graph;
}
}  // namespace grad
}  // namespace pijit
}  // namespace mindspore
