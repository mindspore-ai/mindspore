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

#include "pipeline/pynative/grad/ms_function_grad.h"
#include <utility>
#include "pipeline/pynative/pynative_utils.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/parallel_context.h"
#include "ir/func_graph_cloner.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "runtime/pynative/async/async_queue.h"
#include "pipeline/pynative/grad/bprop_task.h"
#include "pipeline/pynative/grad/bprop_expander/bprop.h"
#include "pipeline/jit/pass.h"

namespace mindspore {
namespace pynative {
namespace {
const mindspore::HashSet<std::string> kNotRealOP{prim::kPrimMakeTuple->name(),
                                                 prim::kPrimTupleGetItem->name(),
                                                 prim::kPrimStopGradient->name(),
                                                 prim::kPrimUpdateState->name(),
                                                 prim::kPrimLoad->name(),
                                                 prim::kPrimDepend->name(),
                                                 prim::kPrimReturn->name(),
                                                 prim::kPrimNPUAllocFloatStatus->name(),
                                                 prim::kPrimNPUGetFloatStatus->name(),
                                                 prim::kPrimNPUClearFloatStatus->name()};

FrontendOpRunInfoPtr GetOpRunInfo(const py::object &out, const py::args &args, const std::string &graph_phase,
                                  ValuePtr *added_out_v) {
  // Get actual output value and added output value.
  if (!py::isinstance<py::tuple>(out)) {
    MS_LOG(EXCEPTION) << "The output value of ms_function func graph should be a tuple.";
  }
  auto tuple_out = py::cast<py::tuple>(out);
  constexpr size_t tuple_out_size = 2;
  if (tuple_out.size() != tuple_out_size) {
    MS_LOG(EXCEPTION) << "The tuple size of output value of ms_function func graph should be 2.";
  }
  MS_EXCEPTION_IF_NULL(added_out_v);
  // Forward output of op in ms_function graph
  *added_out_v = PyNativeAlgo::DataConvert::PyObjToValue(tuple_out[1]);
  MS_LOG(DEBUG) << "Added output value is: " << (*added_out_v)->ToString();
  auto op_run_info = std::make_shared<FrontendOpRunInfo>();
  PyNativeAlgo::PyParser::ParseOpInputByPythonObj(op_run_info, args);
  op_run_info->base_op_run_info.op_name = graph_phase;
  // Output of ms_function
  op_run_info->out_value = PyNativeAlgo::DataConvert::PyObjToValue(tuple_out[0]);
  op_run_info->base_op_run_info.abstract =
    PyNativeAlgo::Common::SetAbstractValueToAnyValue(op_run_info->out_value->ToAbstract());
  op_run_info->grad_flag = true;
  return op_run_info;
}

size_t GetOutputTensorNumForTuple(const CNodePtr &make_tuple) {
  size_t output_num = 0;
  MS_EXCEPTION_IF_NULL(make_tuple);
  if (IsPrimitiveCNode(make_tuple, prim::kPrimMakeTuple)) {
    for (size_t i = 1; i < make_tuple->size(); ++i) {
      const auto &input_i = make_tuple->input(i);
      MS_EXCEPTION_IF_NULL(input_i);
      if (input_i->isa<CNode>()) {
        auto cnode = input_i->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(cnode);
        output_num += GetOutputTensorNumForTuple(cnode);
      } else if (input_i->isa<Parameter>()) {
        output_num += 1;
      } else if (input_i->isa<ValueNode>()) {
        auto v = input_i->cast<ValueNodePtr>();
        MS_EXCEPTION_IF_NULL(v->value());
        if (v->value()->isa<tensor::Tensor>()) {
          output_num += 1;
        }
      }
    }
  } else {
    output_num += AnfAlgo::GetOutputElementNum(make_tuple);
  }
  return output_num;
}

// Modify the output node of func_graph to add forward nodes used in bprop graph.
void ModifyOutputNode(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  const auto &used_forward_nodes = func_graph->used_forward_nodes();

  // Get original output node and abstract
  auto original_output_node = func_graph->output();
  MS_EXCEPTION_IF_NULL(original_output_node);
  auto original_output_abs = original_output_node->abstract();
  MS_EXCEPTION_IF_NULL(original_output_abs);

  // Create a new make tuple node to hold all forward used nodes.
  abstract::AbstractBasePtrList added_abs_list;
  std::vector<AnfNodePtr> added_node_list{NewValueNode(prim::kPrimMakeTuple)};
  std::for_each(used_forward_nodes.begin(), used_forward_nodes.end(),
                [&added_abs_list, &added_node_list](const AnfNodePtr &node) {
                  MS_EXCEPTION_IF_NULL(node);
                  added_node_list.push_back(node);
                  added_abs_list.push_back(node->abstract());
                });
  AnfNodePtr added_output_node;
  AbstractBasePtr added_output_abs;
  if (added_abs_list.empty()) {
    added_output_node = NewValueNode(MakeValue<int32_t>(1));
    added_output_abs = std::make_shared<abstract::AbstractScalar>(std::make_shared<Int32Imm>(1));
  } else {
    added_output_node = func_graph->NewCNode(std::move(added_node_list));
    added_output_abs = std::make_shared<abstract::AbstractTuple>(added_abs_list);
  }
  added_output_node->set_abstract(added_output_abs);

  // Merge original output node and used forward nodes to return node.
  std::vector<AnfNodePtr> new_output_nodes{NewValueNode(prim::kPrimMakeTuple), original_output_node, added_output_node};
  auto merge_node = func_graph->NewCNode(std::move(new_output_nodes));
  abstract::AbstractBasePtrList new_output_abs{original_output_abs, added_output_abs};
  merge_node->set_abstract(std::make_shared<abstract::AbstractTuple>(new_output_abs));
  func_graph->set_output(merge_node);

  // Clear
  func_graph->set_modify_output(true);
  func_graph->ClearUsedForwardNodes();
}
}  // namespace

void MsFunction::RunReplace(const CNodePtr &added_make_tuple,
                            const std::vector<tensor::TensorPtr> &total_output_tensors, const FuncGraphPtr &grad_graph,
                            bool is_dynamic_shape) const {
  MS_EXCEPTION_IF_NULL(added_make_tuple);
  size_t index = 0;
  for (size_t i = 1; i < added_make_tuple->size(); ++i) {
    const auto &input_i = added_make_tuple->input(i);
    MS_EXCEPTION_IF_NULL(input_i);
    auto cnode = input_i->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    MS_LOG(DEBUG) << "Replace output tensors for cnode: " << cnode->DebugString();
    auto output_vnode = cnode->forward().first;
    MS_EXCEPTION_IF_NULL(output_vnode);
    // To clean up all value nodes in PyNative after run grad graph
    if (is_not_support_by_expander_) {
      MS_EXCEPTION_IF_NULL(grad_graph);
      grad_graph->AddValueNode(output_vnode);
    }
    MS_LOG(DEBUG) << "Original output value node: " << output_vnode->ToString();
    size_t output_num = GetOutputTensorNumForTuple(cnode);
    if (index + output_num > total_output_tensors.size()) {
      MS_LOG(EXCEPTION) << "The size of total_output_tensors: " << total_output_tensors.size()
                        << ", but the current index: " << index << ", output num: " << output_num;
    }
    // Get new tensors.
    std::vector<ValuePtr> new_values;
    for (size_t j = index; j < index + output_num; ++j) {
      (void)new_values.emplace_back(total_output_tensors[j]);
    }
    index = index + output_num;
    // Replace new tensors.
    if (output_num == 1) {
      output_vnode->set_value(new_values[0]);
    } else if (output_num > 1) {
      output_vnode->set_value(std::make_shared<ValueTuple>(new_values));
    } else {
      MS_LOG(EXCEPTION) << "The output value of forward cnode is empty, forward cnode info: " << cnode->ToString();
    }
    if (is_dynamic_shape) {
      if (output_num == 1) {
        output_vnode->set_abstract(PyNativeAlgo::Common::SetAbstractValueToAnyValue(new_values[0]->ToAbstract()));
      } else {
        AbstractBasePtrList abs_list;
        for (size_t j = 0; j < output_num; ++j) {
          (void)abs_list.emplace_back(PyNativeAlgo::Common::SetAbstractValueToAnyValue(new_values[j]->ToAbstract()));
        }
        output_vnode->set_abstract(std::make_shared<abstract::AbstractTuple>(abs_list));
      }
    }
    MS_LOG(DEBUG) << "New output value node: " << output_vnode->ToString();
  }
  // Save op info with new tensors for current running ms_function func graph.
  if (index != total_output_tensors.size()) {
    MS_LOG(EXCEPTION) << "The index: " << index
                      << " should be equal to the size of total_output_tensors: " << total_output_tensors.size();
  }
}

void MsFunction::ReplaceWithRealTensorsInGradGraph(const GradExecutor *grad_executor, const ValuePtr &added_out,
                                                   const FuncGraphPtr &ms_func_graph, const FuncGraphPtr &grad_graph,
                                                   const FrontendOpRunInfoPtr &op_run_info) const {
  MS_EXCEPTION_IF_NULL(ms_func_graph);
  // Get added forward nodes.
  auto merge_node = ms_func_graph->output();
  MS_EXCEPTION_IF_NULL(merge_node);
  auto merge_make_tuple = merge_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(merge_make_tuple);
  constexpr size_t merge_output_size = 3;
  // First is make_tuple, second is actual output, third is added output
  if (merge_make_tuple->size() != merge_output_size) {
    MS_LOG(EXCEPTION) << "The input size of merge make tuple node should be 3, but it is: " << merge_make_tuple->size();
  }
  constexpr size_t added_output_index = 2;
  const auto &added_forward_node = merge_make_tuple->input(added_output_index);
  bool is_dynamic_shape = common::AnfAlgo::IsDynamicShape(merge_node);
  if (is_dynamic_shape) {
    const_cast<GradExecutor *>(grad_executor)->set_use_dynamic_shape_process(true);
    MS_LOG(DEBUG) << "Ms function is dynamic shape";
  }
  // Just one added output
  MS_EXCEPTION_IF_NULL(grad_executor);
  MS_EXCEPTION_IF_NULL(added_forward_node);
  if (added_forward_node->isa<ValueNode>()) {
    MS_LOG(DEBUG) << "The added forward output node is value node: " << added_forward_node->DebugString();
    std::vector<tensor::TensorPtr> total_output_tensors;
    TensorValueToTensor(added_out, &total_output_tensors);
    grad_executor->top_cell()->set_op_info_with_ms_func_forward_tensors(op_run_info->op_info, total_output_tensors);
    return;
  }
  // Replace new output tensors for forward nodes, it will also work in grad graph with same value node.
  auto added_make_tuple = added_forward_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(added_make_tuple);
  MS_LOG(DEBUG) << "The added forward make tuple node info: " << added_make_tuple->DebugString();
  std::vector<tensor::TensorPtr> total_output_tensors;
  TensorValueToTensor(added_out, &total_output_tensors);
  // The forward node in ms_function graph is created during compilation and is a
  // placeholder(mindspore/ccsrc/frontend/optimizer/ad/pynative_dfunctor.cc).After running ms_function, need to update
  // to real value.
  RunReplace(added_make_tuple, total_output_tensors, grad_graph, is_dynamic_shape);
  grad_executor->top_cell()->set_op_info_with_ms_func_forward_tensors(op_run_info->op_info, total_output_tensors);
}

void MsFunction::UpdateMsFunctionForwardTensors(const GradExecutor *grad_executor, const TopCellInfoPtr &top_cell,
                                                const string &op_info, const ValuePtr &new_forward_value) const {
  MS_EXCEPTION_IF_NULL(new_forward_value);
  MS_LOG(DEBUG) << "Ms func graph has already ran before. The graph phase is: " << graph_phase_;
  MS_LOG(DEBUG) << "The output values of added forward nodes are: " << new_forward_value->ToString();
  std::vector<tensor::TensorPtr> new_tensors;
  TensorValueToTensor(new_forward_value, &new_tensors);
  if (new_tensors.empty()) {
    MS_LOG(DEBUG) << "The size of added forward tensors is zero, no need to update.";
    return;
  }
  MS_EXCEPTION_IF_NULL(top_cell);
  const auto &old_tensors = top_cell->op_info_with_ms_func_forward_tensors().at(op_info);
  if (old_tensors.size() != new_tensors.size()) {
    MS_LOG(EXCEPTION) << "The size of old tensors is: " << old_tensors.size()
                      << ", but the size of new tensors is: " << new_tensors.size()
                      << ", the current op info is: " << op_info;
  }
  MS_EXCEPTION_IF_NULL(grad_executor);
  for (size_t i = 0; i < new_tensors.size(); ++i) {
    grad_executor->UpdatePreTensorInfo(new_tensors[i], {old_tensors[i]});
    MS_EXCEPTION_IF_NULL(old_tensors[i]);
    old_tensors[i]->set_sync_status(kNeedSyncDeviceToHost);
  }
}

void MsFunction::GetInputArgsNode(const FrontendOpRunInfoPtr &op_run_info, AnfNodePtrList *input_nodes,
                                  const GradExecutor *grad_executor) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(input_nodes);
  MS_EXCEPTION_IF_NULL(grad_executor);
  for (size_t i = 0; i < op_run_info->input_size; ++i) {
    const auto &input_i_value = op_run_info->input_value[i];
    const auto &id = PyNativeAlgo::Common::GetIdByValue(input_i_value);
    const auto &input_i_node = grad_executor->GetInput(input_i_value, id);
    MS_EXCEPTION_IF_NULL(input_i_node);
    MS_LOG(DEBUG) << "The input " << i << " id " << id << " value is: " << input_i_value->ToString()
                  << ", node is: " << input_i_node->DebugString();
    (void)input_nodes->emplace_back(input_i_node);
  }
}

void MsFunction::GetWeightsNode(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                                const FuncGraphPtr &ms_func_graph, AnfNodePtrList *input_nodes) const {
  MS_EXCEPTION_IF_NULL(grad_executor);
  MS_EXCEPTION_IF_NULL(input_nodes);
  const auto &top_cell = grad_executor->top_cell();
  const auto &graph_info = top_cell->graph_info_map().at(top_cell->fg());
  MS_EXCEPTION_IF_NULL(graph_info);
  // Get weights info of ms_function
  auto manage = Manage(ms_func_graph, false);
  MS_EXCEPTION_IF_NULL(ms_func_graph);
  const auto &original_params = ms_func_graph->parameters();
  size_t params_size = original_params.size();
  std::vector<AnfNodePtr> new_params;
  MS_EXCEPTION_IF_NULL(op_run_info);
  for (size_t i = 0; i < params_size; ++i) {
    if (i < op_run_info->input_size) {  // non-weights node.
      (void)new_params.emplace_back(original_params[i]);
      continue;
    }
    // Must weight param
    auto param = original_params[i]->cast<ParameterPtr>();
    const auto tensor_value = PyNativeAlgo::Common::GetTensorFromParam(original_params[i]);
    MS_EXCEPTION_IF_NULL(tensor_value);
    const auto it = graph_info->weight_params.find(tensor_value->id());
    if (it != graph_info->weight_params.end()) {
      // Share same weight parameter in different ms_function call.
      (void)manage->Replace(original_params[i], it->second);
      param = it->second;
    } else {
      top_cell->fg()->add_parameter(param);
      param->debug_info()->set_name(param->name());
      top_cell->SetParamNodeMapInGraphInfoMap(tensor_value->id(), param, true);
    }
    (void)new_params.emplace_back(param);
    (void)input_nodes->emplace_back(param);
    (void)op_run_info->input_value.emplace_back(tensor_value);
    MS_LOG(DEBUG) << "Top graph set free parameter " << param->DebugString() << ". Its default value is "
                  << tensor_value->ToString() << ". Its name is: " << param->name();
  }
  ms_func_graph->set_parameters(new_params);
  manage->Clear();
}

void MsFunction::MakeCNodeForMsFunction(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                                        const FuncGraphPtr &ms_func_graph, CNodePtr *ms_function_cnode) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  // Get input node info of ms_function
  std::vector<AnfNodePtr> input_nodes{NewValueNode(ms_func_graph)};
  MS_EXCEPTION_IF_NULL(grad_executor);
  GetInputArgsNode(op_run_info, &input_nodes, grad_executor);
  // Get weights node info of ms_function.
  GetWeightsNode(op_run_info, grad_executor, ms_func_graph, &input_nodes);
  // Make a CNode which includes ms_function fprop graph and inputs node
  MS_EXCEPTION_IF_NULL(ms_function_cnode);
  *ms_function_cnode = grad_executor->top_cell()->fg()->NewCNode(input_nodes);
  (*ms_function_cnode)->set_abstract(ms_func_graph->output()->abstract());
  MS_LOG(DEBUG) << "Make ms function forward CNode: " << (*ms_function_cnode)->DebugString();
}

CNodePtr MsFunction::MakeAdjointForMsFunction(const FrontendOpRunInfoPtr &op_run_info,
                                              const GradExecutor *grad_executor, const FuncGraphPtr &ms_func_graph,
                                              const FuncGraphPtr &grad_graph) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(grad_executor);
  CNodePtr ms_function_cnode = nullptr;
  MakeCNodeForMsFunction(op_run_info, grad_executor, ms_func_graph, &ms_function_cnode);
  MS_EXCEPTION_IF_NULL(ms_function_cnode);
  const auto &top_cell = grad_executor->top_cell();
  const auto &out_id = PyNativeAlgo::Common::GetIdByValue(op_run_info->out_value);
  top_cell->SetNodeMapInGraphInfoMap(out_id, ms_function_cnode);

  // Connect grad graph of ms_function to context.
  auto auto_grad_cell_ptr = top_cell->auto_grad_cell_ptr();
  auto grad_param = std::make_shared<autograd::GradParam>(
    ms_function_cnode, op_run_info->input_value, op_run_info->out_value,
    is_not_support_by_expander_ ? grad_graph : ms_func_graph, !top_cell->is_high_order_top_cell());
  grad_param->graph_cache_key = op_run_info->op_info;
  grad_param->use_dynamic_shape_process = grad_executor->use_dynamic_shape_process();
  grad_param->is_ms_function_graph = true;
  grad_param->is_not_support_by_expander = is_not_support_by_expander_;
  {
    py::gil_scoped_release gil_release;
    grad_executor->async_executor()->Wait();
  }
  if (!auto_grad_cell_ptr->KPynativeWithFProp(grad_param)) {
    MS_LOG(EXCEPTION) << "Failed to make adjoint for ms_function cnode, ms_function cnode info: "
                      << ms_function_cnode->DebugString();
  }
  top_cell->set_need_do_final_opt(true);
  return ms_function_cnode;
}

void MsFunction::AsyncKPynativeWithFProp(const GradExecutor *grad_executor,
                                         const autograd::AutoGradCellImplPtr &auto_grad_cell_ptr,
                                         const autograd::GradParamPtr &grad_param) const {
  MS_EXCEPTION_IF_NULL(grad_executor);
  const auto fn = [grad_param, auto_grad_cell_ptr]() {
    MS_EXCEPTION_IF_NULL(auto_grad_cell_ptr);
    if (!auto_grad_cell_ptr->KPynativeWithFProp(grad_param)) {
      MS_LOG(EXCEPTION) << "Failed to make adjoint for ms_function cnode";
    }
  };
  auto task = std::make_shared<BpropTask>(fn);
  grad_executor->async_executor()->Push(task);
}

void MsFunction::AsyncGradMsFunctionInner(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                                          const ValuePtr &added_out_v, const FuncGraphPtr &ms_func_graph,
                                          const FuncGraphPtr &grad_graph) const {
  const auto fn = [this, op_run_info, grad_executor, added_out_v, ms_func_graph, grad_graph]() {
    this->GradMsFunctionInner(op_run_info, grad_executor, added_out_v, ms_func_graph, grad_graph);
  };
  auto task = std::make_shared<BpropTask>(fn);
  grad_executor->async_executor()->Push(task);
}

void MsFunction::GradMsFunctionInner(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                                     const ValuePtr &added_out_v, const FuncGraphPtr &ms_func_graph,
                                     const FuncGraphPtr &grad_graph) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(grad_executor);

  // Step 1: Update actual output tensors used in grad graph.
  MS_EXCEPTION_IF_NULL(op_run_info->out_value);
  MS_LOG(DEBUG) << "ms_function actual output value: " << op_run_info->out_value->ToString();
  // The output of ms_function may be used in subsequent PyNative process
  grad_executor->UpdateForwardTensorInfoInBpropGraph(op_run_info);

  // Step 2: Update output tensors of added forward nodes, which are added to return node of ms_function func graph.
  if (grad_executor->use_dynamic_shape_process()) {
    MS_LOG(DEBUG) << "Get dynamic shape process";
  } else {
    const auto &pre_top_cell = grad_executor->GetAlreadyRunTopCell(grad_executor->top_cell()->already_run_cell_id());
    if (pre_top_cell != nullptr && pre_top_cell->op_info_with_ms_func_forward_tensors().find(op_run_info->op_info) !=
                                     pre_top_cell->op_info_with_ms_func_forward_tensors().end()) {
      UpdateMsFunctionForwardTensors(grad_executor, pre_top_cell, op_run_info->op_info, added_out_v);
    }
  }

  ReplaceWithRealTensorsInGradGraph(grad_executor, added_out_v, ms_func_graph, grad_graph, op_run_info);

  // Change ms function graph to real output
  auto clone_ms_func_graph = BasicClone(ms_func_graph);
  auto clone_grad_graph = grad_graph;
  if (is_not_support_by_expander_) {
    // Clone value node for find it in grad.cc:SaveForwardTensorInfoInBpropGraph, which used by clean device address
    clone_grad_graph = BasicClone(grad_graph, true);
  }
  auto new_make_tuple = clone_ms_func_graph->output()->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(new_make_tuple);
  clone_ms_func_graph->set_output(new_make_tuple->input(1));
  // Make Adjoint for grad graph
  const auto &ms_function_cnode =
    MakeAdjointForMsFunction(op_run_info, grad_executor, clone_ms_func_graph, clone_grad_graph);
  grad_executor->CheckGraphDynamic(ms_function_cnode, true, op_run_info->base_op_run_info.op_name);
}

void MsFunction::SetMsFuncGraphParameters(const FuncGraphPtr &ms_func_graph) {
  MS_EXCEPTION_IF_NULL(ms_func_graph);
  auto parallel_mode = parallel::ParallelContext::GetInstance()->parallel_mode();
  if (parallel_mode == parallel::kSemiAutoParallel || parallel_mode == parallel::kAutoParallel) {
    for (auto &parameter : ms_func_graph->parameters()) {
      auto param = parameter->cast<ParameterPtr>();
      if (param->has_default()) {
        (void)ms_function_params_.emplace_back(param->name());
      }
    }
  }
}

void MsFunction::ModifyMsFunctionForwardOutput(const FuncGraphPtr &ms_func_graph) {
  MS_EXCEPTION_IF_NULL(ms_func_graph);
  PyNativeAlgo::Common::DumpGraphIR("ms_func_modify_before_forward_graph.ir", ms_func_graph);
  AnfNodePtrList node_list{};
  const auto &order = TopoSort(ms_func_graph->output());
  for (const auto &node : order) {
    if (node == nullptr || !node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto prim = GetCNodePrimitive(cnode);
    if (prim == nullptr) {
      MS_LOG(EXCEPTION) << "Should be primitive, but: " << cnode->DebugString();
    }
    if (kNotRealOP.find(prim->name()) != kNotRealOP.end()) {
      continue;
    }

    MS_LOG(DEBUG) << "Get cnode " << cnode->DebugString();
    const auto &unused_inputs = BpropExpander().GetUnusedInputs(cnode);
    // Check output used in bprop graph
    if (std::find(unused_inputs.begin(), unused_inputs.end(), cnode->size()) == unused_inputs.end()) {
      auto out = pynative::PyNativeAlgo::Common::CreatOutputTensorValueByAbstract(cnode->abstract());
      auto v_node = NewValueNode(out);
      v_node->set_abstract(cnode->abstract());
      cnode->set_forward(v_node, "");
      node_list.emplace_back(cnode);
    }
  }
  ms_func_graph->set_used_forward_nodes(node_list);
  ModifyOutputNode(ms_func_graph);
  PyNativeAlgo::Common::DumpGraphIR("ms_func_modify_after_forward_graph.ir", ms_func_graph);
}

py::object MsFunction::GradMsFunction(const py::object &out, const py::args &args) {
  if (graph_phase_.empty()) {
    MS_LOG(EXCEPTION) << "The graph phase is empty, can not obtain ms_function func graph.";
  }
  // Get forward graph
  MS_LOG(DEBUG) << "ms_function func graph phase: " << graph_phase_;
  auto executor = pipeline::GraphExecutorPy::GetInstance();
  MS_EXCEPTION_IF_NULL(executor);
  FuncGraphPtr ms_func_graph = executor->GetFuncGraph(graph_phase_);
  MS_EXCEPTION_IF_NULL(ms_func_graph);
  // Get actual forward output object.
  py::object ret = out;
  if (ms_func_graph->modify_output()) {
    auto tuple_out = py::cast<py::tuple>(out);
    ret = tuple_out[0];
  }
  // Save dynamic shape info if output tensors of forward graph have dynamic shapes
  const auto &grad_executor = PyNativeAlgo::Common::GetPyNativeExecutor()->grad_executor();
  // Make Adjoint for grad graph of ms_function.
  if (!grad_executor->grad_flag()) {
    MS_LOG(DEBUG) << "Only run forward infer computation, no need to construct grad graph.";
    graph_phase_.clear();
    return ret;
  }
  ValuePtr added_out_v;
  const auto &op_run_info = GetOpRunInfo(out, args, graph_phase_, &added_out_v);
  is_not_support_by_expander_ =
    PyNativeAlgo::Common::IsControlFlowGraph(ms_func_graph) || parallel::IsAutoParallelCareGraph(ms_func_graph);
  MS_LOG(DEBUG) << "Graph is control flow " << PyNativeAlgo::Common::IsControlFlowGraph(ms_func_graph)
                << ", auto parallel " << parallel::IsAutoParallelCareGraph(ms_func_graph);
  FuncGraphPtr grad_graph = nullptr;
  if (is_not_support_by_expander_) {
    grad_graph = executor->GetGradGraph(graph_phase_);
  }
  PyNativeAlgo::Common::DumpGraphIR("ms_func_forward_graph.ir", ms_func_graph);
  GradMsFunctionInner(op_run_info, grad_executor.get(), added_out_v, ms_func_graph, grad_graph);
  SetMsFuncGraphParameters(ms_func_graph);
  graph_phase_.clear();
  return ret;
}
}  // namespace pynative
}  // namespace mindspore
