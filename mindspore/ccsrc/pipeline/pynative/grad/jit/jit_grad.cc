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

#include "pipeline/pynative/grad/jit/jit_grad.h"

#include <utility>
#include "frontend/optimizer/ad/grad.h"
#include "ops/structure_op_name.h"
#include "ops/framework_op_name.h"
#include "ops/sequence_ops.h"
#include "pipeline/pynative/pynative_utils.h"
#include "pipeline/pynative/grad/jit/jit_dfunctor.h"
#include "ir/func_graph_cloner.h"
#include "pipeline/pynative/grad/bprop_task.h"
#include "pipeline/jit/ps/pass.h"
#include "frontend/expander/bprop/bprop.h"

namespace mindspore {
namespace pynative {
namespace {
const char kAddedValue[] = "added_value";

const mindspore::HashSet<std::string> kExpanderWhiteList{
  kVmapStackAssignOpName,
  kVmapUnstackAssignOpName,
  kPyExecuteOpName,
  kPrintOpName,
};

FrontendOpRunInfoPtr GetOpRunInfo(const py::object &out, const py::args &args, const std::string &graph_phase,
                                  bool modify_output, const FuncGraphPtr &jit_forward_graph, ValuePtr *added_out_v) {
  auto op_run_info = std::make_shared<FrontendOpRunInfo>();
  op_run_info->requires_grad = true;
  op_run_info->is_jit_input = true;
  op_run_info->base_op_run_info.op_name = graph_phase;
  PyNativeAlgo::PyParser::ParseOpInputByPythonObj(op_run_info, args);
  // Set input abs
  op_run_info->op_grad_info->input_abs.resize(op_run_info->input_size);
  const auto &original_params = jit_forward_graph->parameters();
  for (size_t i = 0; i < op_run_info->input_size; ++i) {
    if (original_params.size() <= i) {
      MS_LOG(INTERNAL_EXCEPTION) << "Index out of range for index: " << i
                                 << " and origin params size: " << original_params.size();
    }
    op_run_info->op_grad_info->input_abs[i] = original_params[i]->abstract();
  }
  if (modify_output) {
    if (!py::isinstance<py::tuple>(out)) {
      MS_LOG(EXCEPTION) << "The output value of jit func graph should be a tuple.";
    }
    auto tuple_out = py::cast<py::tuple>(out);
    constexpr size_t tuple_out_size = 2;
    if (tuple_out.size() != tuple_out_size) {
      MS_LOG(EXCEPTION) << "The tuple size of output value of jit func graph should be 2.";
    }
    MS_EXCEPTION_IF_NULL(added_out_v);
    // Forward output of op in jit graph
    *added_out_v = PyNativeAlgo::DataConvert::PyObjToValue(tuple_out[1]);
    op_run_info->real_out = PyNativeAlgo::DataConvert::PyObjToValue(tuple_out[0]);
  } else {
    op_run_info->real_out = PyNativeAlgo::DataConvert::PyObjToValue(out);
  }
  return op_run_info;
}

size_t GetTensorNumFromAbstract(const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  if (abs->isa<abstract::AbstractTensor>()) {
    // Is a tensor
    constexpr size_t kTensorOutputNum = 1;
    return kTensorOutputNum;
  } else if (abs->isa<abstract::AbstractSequence>()) {
    const auto &abs_seq = abs->cast<abstract::AbstractSequencePtr>()->elements();
    return std::accumulate(abs_seq.begin(), abs_seq.end(), 0, [](size_t out_num, const abstract::AbstractBasePtr &abs) {
      return out_num + GetTensorNumFromAbstract(abs);
    });
  } else if (abs->isa<abstract::AbstractCSRTensor>()) {
    // Currently, CSRTensor only supports 2-D matrix (shape has 2 values). 5 outputs = 3 Tensors + 2 shape values.
    constexpr size_t kCSRTensorOutputNum = 5;
    return kCSRTensorOutputNum;
  } else if (abs->isa<abstract::AbstractCOOTensor>()) {
    // Currently, COOTensor only supports 2-D matrix (shape has 2 values). 4 outputs = 2 Tensors + 2 shape values.
    constexpr size_t kCOOTensorOutputNum = 4;
    return kCOOTensorOutputNum;
  }
  return 0;
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

CNodePtr GetAddedNode(const FuncGraphPtr &jit_forward_graph) {
  MS_EXCEPTION_IF_NULL(jit_forward_graph);
  if (!jit_forward_graph->modify_output()) {
    return nullptr;
  }
  // Get added forward nodes.
  auto merge_node = jit_forward_graph->output();
  MS_EXCEPTION_IF_NULL(merge_node);
  auto merge_make_tuple = merge_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(merge_make_tuple);
  constexpr size_t merge_output_size = 3;
  // First is make_tuple, second is actual output, third is added output
  if (merge_make_tuple->size() != merge_output_size) {
    MS_LOG(EXCEPTION) << "The input size of merge make tuple node should be 3, but it is: " << merge_make_tuple->size();
  }
  constexpr size_t added_output_index = 2;
  return merge_make_tuple->input(added_output_index)->cast<CNodePtr>();
}

bool IsGraphDynamic(const FuncGraphPtr &func_graph) {
  for (const auto &param : func_graph->parameters()) {
    if (param->isa<Parameter>() && !param->cast<ParameterPtr>()->has_default()) {
      const auto &abs = param->abstract();
      if (abs != nullptr && abs->BuildShape()->IsDynamic()) {
        return true;
      }
    }
  }
  MS_EXCEPTION_IF_NULL(func_graph->output());
  if (auto abs = func_graph->output()->abstract(); abs != nullptr && abs->BuildShape()->IsDynamic()) {
    return true;
  }
  return false;
}

bool JitOutputHasDict(const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  if (abs->isa<abstract::AbstractDictionary>()) {
    return true;
  } else if (abs->isa<abstract::AbstractSequence>()) {
    const auto &abs_sequence = abs->cast<abstract::AbstractSequencePtr>();
    return std::any_of(abs_sequence->elements().begin(), abs_sequence->elements().end(),
                       [](const abstract::AbstractBasePtr &item) { return JitOutputHasDict(item); });
  }
  return false;
}
}  // namespace

void Jit::RunReplace(const CNodePtr &added_node, const ValuePtrList &total_output_tensors) const {
  MS_EXCEPTION_IF_NULL(added_node);
  size_t index = 0;
  for (size_t i = 1; i < added_node->size(); ++i) {
    const auto &input_i = added_node->input(i);
    MS_EXCEPTION_IF_NULL(input_i);
    auto cnode = input_i->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    MS_LOG(DEBUG) << "Replace output tensors for cnode: " << cnode->DebugString();
    const auto &output_vnode = cnode->forward().first;
    MS_EXCEPTION_IF_NULL(output_vnode);
    MS_LOG(DEBUG) << "Old output value node: " << output_vnode->ToString();
    MS_EXCEPTION_IF_NULL(output_vnode->abstract());
    bool is_tuple_out = output_vnode->abstract()->isa<abstract::AbstractSequence>();
    size_t output_num = GetTensorNumFromAbstract(cnode->abstract());
    if (output_num == 0) {
      MS_LOG(DEBUG) << "The output value out is not include tensor";
      continue;
    }
    if (index + output_num > total_output_tensors.size()) {
      MS_LOG(EXCEPTION) << "The size of total_output_tensors: " << total_output_tensors.size()
                        << ", but the current index: " << index << ", output num: " << output_num;
    }
    // Get new tensors.
    std::vector<ValuePtr> new_values;
    for (size_t j = index; j < index + output_num; ++j) {
      // If jit graph reused in dynamic shape, added output tensor should be update tensor address in run actor
      auto tensor = total_output_tensors[j]->cast<tensor::TensorPtr>();
      if (tensor != nullptr) {
        tensor->set_is_forward_output(true);
      }
      (void)new_values.emplace_back(total_output_tensors[j]);
    }
    index = index + output_num;
    // Replace new tensors.
    // Can not use output_num > 1, because output can be (a), tuple just have only one element
    if (is_tuple_out) {
      output_vnode->set_value(std::make_shared<ValueTuple>(new_values));
    } else {
      output_vnode->set_value(new_values[0]);
    }
    MS_LOG(DEBUG) << "New output value node: " << output_vnode->ToString();
  }
  // Save op info with new tensors for current running jit func graph.
  if (index != total_output_tensors.size()) {
    MS_LOG(EXCEPTION) << "The index: " << index
                      << " should be equal to the size of total_output_tensors: " << total_output_tensors.size();
  }
}

void Jit::ReplaceAddedCnodeActualOutput(const CNodePtr &added_node, const ValuePtrList &total_output_tensors) const {
  MS_EXCEPTION_IF_NULL(added_node);
  // Replace new output tensors for forward nodes, it will also work in grad graph with same value node.
  MS_LOG(DEBUG) << "The added forward make tuple node info: " << added_node->DebugString();
  // The forward node in jit graph is created during compilation and is a placeholder.
  // After running jit, need to update to real value.
  RunReplace(added_node, total_output_tensors);
}

void Jit::GetInputArgsNode(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                           AnfNodePtrList *input_nodes) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(input_nodes);
  MS_EXCEPTION_IF_NULL(grad_executor);
  for (size_t i = 0; i < op_run_info->input_size; ++i) {
    const auto &input_i_value = op_run_info->op_grad_info->input_value[i];
    const auto &id = PyNativeAlgo::Common::GetIdByValue(input_i_value);
    const auto &input_i_node = grad_executor->GetInput(input_i_value, id);
    MS_EXCEPTION_IF_NULL(input_i_node);
    MS_LOG(DEBUG) << "The input " << i << " id " << id << " , node is: " << input_i_node->DebugString();
    (void)input_nodes->emplace_back(input_i_node);
  }
}

void Jit::GetWeightsNode(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                         const FuncGraphPtr &jit_forward_graph, AnfNodePtrList *input_nodes) const {
  MS_EXCEPTION_IF_NULL(grad_executor);
  MS_EXCEPTION_IF_NULL(input_nodes);
  const auto &top_cell = grad_executor->top_cell();
  const auto &graph_info = top_cell->graph_info_map().at(top_cell->fg());
  MS_EXCEPTION_IF_NULL(graph_info);
  // Get weights info of jit
  MS_EXCEPTION_IF_NULL(jit_forward_graph);
  const auto &original_params = jit_forward_graph->parameters();
  size_t params_size = original_params.size();
  MS_EXCEPTION_IF_NULL(op_run_info);
  for (size_t i = 0; i < params_size; ++i) {
    if (i < op_run_info->input_size) {  // non-weights node.
      continue;
    }
    // Must weight param
    auto param = original_params[i]->cast<ParameterPtr>();
    const auto tensor_value = PyNativeAlgo::Common::GetTensorFromParam(original_params[i]);
    MS_EXCEPTION_IF_NULL(tensor_value);
    const auto it = graph_info->weight_params.find(tensor_value->id());
    if (it != graph_info->weight_params.end()) {
      param = it->second;
    } else {
      top_cell->fg()->add_parameter(param);
      param->debug_info()->set_name(param->name());
      top_cell->SetParamNodeMapInGraphInfoMap(tensor_value->id(), param, true);
    }
    (void)input_nodes->emplace_back(param);
    MS_LOG(DEBUG) << "Top graph set free parameter " << param->DebugString() << ". Its default value is "
                  << tensor_value->ToString() << ". Its name is: " << param->name();
  }
}

void Jit::MakeCNodeForJit(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                          const FuncGraphPtr &jit_forward_graph, CNodePtr *jit_cnode) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(jit_forward_graph);
  // Get input node info of jit
  AnfNodePtrList input_nodes{NewValueNode(jit_forward_graph)};
  MS_EXCEPTION_IF_NULL(grad_executor);
  GetInputArgsNode(op_run_info, grad_executor, &input_nodes);
  // Get weights node info of jit.
  GetWeightsNode(op_run_info, grad_executor, jit_forward_graph, &input_nodes);
  // Make a CNode which includes jit fprop graph and inputs node
  MS_EXCEPTION_IF_NULL(jit_cnode);
  *jit_cnode = grad_executor->top_cell()->fg()->NewCNode(input_nodes);
  (*jit_cnode)->set_abstract(jit_forward_graph->output()->abstract());
  MS_LOG(DEBUG) << "Make jit forward CNode: " << (*jit_cnode)->DebugString();
}

void Jit::MakeAdjointForJit(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                            const FuncGraphPtr &jit_forward_graph, const FuncGraphPtr &jit_grad_graph,
                            bool has_added_v) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(grad_executor);

  const auto &top_cell = grad_executor->top_cell();
  PyNativeAlgo::Common::SetGraphInputAndWeightsInfo(op_run_info, jit_forward_graph, top_cell);
  RecordForwardGraphForJit(op_run_info, grad_executor, jit_forward_graph);
  // Connect grad graph of jit to context.
  (void)PyNativeAlgo::Common::SetValueGradInfo(op_run_info->real_out, top_cell, InputType::kOpOutput);
  MS_EXCEPTION_IF_NULL(jit_forward_graph);
  MS_EXCEPTION_IF_NULL(jit_forward_graph->output()->abstract());
  if (grad_executor->dynamic_shape()->enable_unknown_shape() &&
      jit_forward_graph->output()->abstract()->BuildShape()->IsDynamic()) {
    MS_LOG(DEBUG) << "Set jit unknown shape out to abs cache";
    grad_executor->dynamic_shape()->SaveUnknownShapeAbsFromJit(op_run_info->real_out,
                                                               jit_forward_graph->output()->abstract(), 0);
  }
  auto op_grad_info = std::make_shared<OpGradInfo>();
  op_grad_info->input_value = op_run_info->op_grad_info->input_value;
  op_grad_info->input_abs = op_run_info->op_grad_info->input_abs;
  op_grad_info->out_value = op_run_info->real_out;
  op_grad_info->output_size = PyNativeAlgo::Common::GetValueSize(op_grad_info->out_value);
  op_grad_info->input_value_grad_type = op_run_info->op_grad_info->input_value_grad_type;
  if (jit_forward_graph->output()->abstract()->isa<abstract::AbstractAny>()) {
    op_grad_info->out_abs = PyNativeAlgo::Common::SetAbstractValueToAnyValue(op_grad_info->out_value->ToAbstract());
  } else {
    op_grad_info->out_abs = jit_forward_graph->output()->abstract();
  }
  auto grad_param = std::make_shared<GradParam>(op_grad_info, grad_executor->use_dynamic_shape_process());
  grad_param->is_control_flow = compile_info_.is_control_flow_;

  grad_param->has_added_v = has_added_v;
  grad_param->is_jit_graph = true;
  // As long as the jit is in the process of dynamic shape,
  // let it run actor execution to avoid backend pass
  grad_param->is_jit_self_dynamic_shape = compile_info_.is_dynamic_shape_;

  grad_param->fg = jit_grad_graph;
  grad_param->source_fg = jit_forward_graph;
  grad_param->graph_cache_key = graph_phase_;
  grad_param->jit_out_has_dict = JitOutputHasDict(op_grad_info->out_abs);
  auto auto_grad_cell_ptr = top_cell->auto_grad_cell_ptr();
  KPynativeWithFProp(grad_executor, auto_grad_cell_ptr, grad_param);
  top_cell->set_need_do_final_opt(true);
  top_cell->set_has_call_graph(grad_executor->use_dynamic_shape_process());
  top_cell->set_has_control_flow(compile_info_.is_control_flow_);
  top_cell->set_jit_out_has_dict(grad_param->jit_out_has_dict);
}

void Jit::KPynativeWithFProp(const GradExecutor *grad_executor, const autograd::AutoGradPtr &auto_grad_cell_ptr,
                             const GradParamPtr &grad_param) const {
  grad_executor->WaitBpropTask();
  MS_EXCEPTION_IF_NULL(auto_grad_cell_ptr);
  if (!auto_grad_cell_ptr->KPynativeWithFProp(grad_param)) {
    MS_LOG(EXCEPTION) << "Failed to make adjoint for jit cnode";
  }
}

void Jit::RecordForwardGraphForJit(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                                   const FuncGraphPtr &jit_forward_graph) const {
  int save_graphs = MsContext::GetInstance()->get_param<int>(MS_CTX_SAVE_GRAPHS_FLAG);
  if (save_graphs) {
    CNodePtr jit_cnode = nullptr;
    MakeCNodeForJit(op_run_info, grad_executor, jit_forward_graph, &jit_cnode);
    MS_EXCEPTION_IF_NULL(jit_cnode);
    const auto &out_id = PyNativeAlgo::Common::GetIdByValue(op_run_info->real_out);
    const auto &top_cell = grad_executor->top_cell();
    top_cell->SetNodeMapInGraphInfoMap(out_id, jit_cnode);
  }
}

void Jit::GradJitInner(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                       const FuncGraphPtr &primal_func_graph, const FuncGraphPtr &jit_grad_graph,
                       const CNodePtr &added_node, const ValuePtr &added_out_v) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(grad_executor);
  // Step 1: Replace added cnode forward with actual output
  ValuePtr flatten_v = added_out_v;
  bool added_v_is_empty = true;
  if (added_out_v != nullptr) {
    ValuePtrList total_output_tensors;
    PyNativeAlgo::DataConvert::FlattenValueSeqArg(added_out_v, false, &total_output_tensors);
    flatten_v = std::make_shared<ValueTuple>(total_output_tensors);
    added_v_is_empty = total_output_tensors.empty();
    ReplaceAddedCnodeActualOutput(added_node, total_output_tensors);
  }

  // Step 2: Check or set set_use_dynamic_shape_process flag
  auto node_info = std::make_shared<DynamicDetectNodeInfo>(nullptr, op_run_info->op_grad_info->input_abs,
                                                           op_run_info->base_op_run_info.abstract);
  node_info->is_graph_node = true;
  node_info->graph_phase = graph_phase_;
  grad_executor->dynamic_shape()->CheckNodeDynamic(grad_executor->top_cell(), op_run_info->op_grad_info->input_value,
                                                   node_info);

  // Step 3: Update actual output tensors used in grad graph.
  MS_LOG(DEBUG) << "jit actual output value: " << op_run_info->real_out->ToString();
  grad_executor->top_cell()->GetOpInfo(op_run_info, true);
  grad_executor->UpdateTopCellForwardTensorInfoInBpropGraph(op_run_info->op_info, op_run_info->real_out,
                                                            op_run_info->base_op_run_info.stream_id);

  // Step 4: Update output tensors of added forward nodes, which are added to return node of jit func graph.
  if (!added_v_is_empty) {
    if (grad_executor->use_dynamic_shape_process()) {
      // If jit is not control flow, the jit is executed by actor under dynamic shape, and valuenode
      // will be updated
      if (!compile_info_.is_control_flow_) {
        UpdateJitForwardTensorInfoInBpropGraph(op_run_info->op_info + kAddedValue, flatten_v,
                                               op_run_info->base_op_run_info.stream_id);
      }
    } else {
      // Static shape will run by replace
      grad_executor->UpdateTopCellForwardTensorInfoInBpropGraph(op_run_info->op_info + kAddedValue, flatten_v,
                                                                op_run_info->base_op_run_info.stream_id);
    }
  }

  // Make Adjoint for grad graph
  MakeAdjointForJit(op_run_info, grad_executor, primal_func_graph, jit_grad_graph, !added_v_is_empty);
}

void Jit::UpdateJitForwardTensorInfoInBpropGraph(const std::string &op_info, const ValuePtr &v,
                                                 const size_t &stream_id) {
  const auto it = graph_phase_with_replace_info_.find(graph_phase_);
  if (it == graph_phase_with_replace_info_.end()) {
    MS_LOG(DEBUG) << "Jit " << graph_phase_ << " run firstly";
    auto &replace_info = graph_phase_with_replace_info_[graph_phase_];
    SetIdWithOpInfo(v, op_info, kIndex0, &(replace_info.id_with_op_info));
    return;
  }
  // Not first run
  MS_LOG(DEBUG) << "Update jit forward output tensor info " << op_info;
  UpdateForwardOutputTensorInfo(op_info, v, it->second, stream_id);
}

void Jit::SaveForwardOutputTensorInfoInBpropGraph(const FuncGraphPtr &func_graph) {
  const auto it = graph_phase_with_replace_info_.find(graph_phase_);
  if (it == graph_phase_with_replace_info_.end()) {
    MS_LOG(EXCEPTION) << "Can not find graph phase " << graph_phase_ << " in graph_phase_with_replace_info";
  }
  MS_LOG(DEBUG) << "Save jit forward output tensor info";
  auto manager = MakeManager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(func_graph);
  SaveForwardOutputTensorInfo(func_graph, true, &(it->second));
}

void Jit::ProcessCnodeFromAdGrad(const CNodePtr &k_app, const CNodePtr &cnode_morph) {
  // Run grad process for func_graph and replace forward nodes with its output tensors.
  if (eliminate_forward_) {
    ReplaceEquivOut(k_app, cnode_morph);
  }
}

bool Jit::GetJitGradGraph(const pipeline::ResourcePtr &resource) {
  auto graph_executor = pipeline::GraphExecutorPy::GetInstance();
  MS_EXCEPTION_IF_NULL(graph_executor);
  graph_phase_ = graph_executor->phase();
  MS_LOG(DEBUG) << "The phase of current pipeline graph is: " << graph_phase_;
  // Exporting graph in PyNative mode or only running forward process no need to do this action.
  const auto &pynative_grad_executor = PyNativeAlgo::Common::GetPyNativeExecutor()->grad_executor();
  if (graph_phase_.find("export") == 0 || !pynative_grad_executor->RequiresGrad()) {
    MS_LOG(DEBUG) << "When exporting graph or only running forward process";
    return true;
  }

  MS_EXCEPTION_IF_NULL(resource);
  auto jit_forward_graph = resource->func_graph();
  MS_EXCEPTION_IF_NULL(jit_forward_graph);
  graph_executor->SetJitPrimalFuncGraph(BasicClone(jit_forward_graph), graph_phase_);
  auto clone_graph = GetJitForwardGraphCNodeInfo(jit_forward_graph);
  if (clone_graph != nullptr) {
    graph_executor->SetJitGradGraph(clone_graph, graph_phase_);
    return true;
  }

  // Control flow not eliminate forward
  auto is_control_flow = PyNativeAlgo::Common::IsControlFlowGraph(jit_forward_graph);
  auto jit_output_has_dict = JitOutputHasDict(jit_forward_graph->output()->abstract());
  set_eliminate_forward(!is_control_flow && !jit_output_has_dict);
  MS_LOG(DEBUG) << "Run ad grad eliminate_forward " << eliminate_forward_;
  auto grad_graph = ad::Grad(is_control_flow ? BasicClone(jit_forward_graph) : jit_forward_graph,
                             opt::Optimizer::MakeEmptyOptimizer(resource));
  MS_EXCEPTION_IF_NULL(grad_graph);
  graph_executor->SetJitGradGraph(grad_graph, graph_phase_);
  ModifyOutputNode(jit_forward_graph);

  // Keep roots for only keeping forward func graph in resource.
  auto manager = resource->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->KeepRoots({jit_forward_graph});
  eliminate_forward_ = true;
  return true;
}

void Jit::Reset() { graph_phase_.clear(); }

FuncGraphPtr Jit::GetJitForwardGraphCNodeInfo(const FuncGraphPtr &jit_forward_graph) {
  MS_EXCEPTION_IF_NULL(jit_forward_graph);
  PyNativeAlgo::Common::DumpGraphIR("jit_modify_before_forward_graph.ir", jit_forward_graph);
  if (PyNativeAlgo::Common::IsControlFlowGraph(jit_forward_graph)) {
    MS_LOG(DEBUG) << "Get control flow";
    jit_compile_info_[graph_phase_].is_control_flow_ = true;
    return nullptr;
  }
  if (IsGraphDynamic(jit_forward_graph)) {
    MS_LOG(DEBUG) << "Get dynamic shape";
    jit_compile_info_[graph_phase_].is_dynamic_shape_ = true;
    return nullptr;
  }
  jit_compile_info_[graph_phase_] = JitCompileInfo();
  AnfNodePtrList node_list{};
  const auto &order = TopoSort(jit_forward_graph->output());
  for (const auto &node : order) {
    if (node == nullptr || !node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    const auto &prim = GetCNodePrimitive(cnode);
    if (prim == nullptr) {
      MS_LOG(EXCEPTION) << "Should be primitive, but: " << node->DebugString();
    }
    if (!PyNativeAlgo::GradCommon::IsRealOp(cnode)) {
      continue;
    }
    MS_LOG(DEBUG) << "Get cnode " << cnode->DebugString();
    const auto &unused_inputs = BpropExpander::GetUnusedInputs(prim->name());
    if (!unused_inputs.empty() && unused_inputs.find(INT_MAX) != unused_inputs.end() &&
        kExpanderWhiteList.find(prim->name()) == kExpanderWhiteList.end()) {
      MS_LOG(DEBUG) << "Prim " << prim->name() << " is not support by expander";
      jit_compile_info_[graph_phase_].is_control_flow_ = true;
      return nullptr;
    }
    pynative::PyNativeAlgo::GradCommon::GetUsedCNodeInBpropGraph(cnode, unused_inputs, &node_list);
  }
  if (node_list.empty()) {
    MS_LOG(DEBUG) << "No need do replace";
    // Make sure forward graph does not change
    return BasicClone(jit_forward_graph);
  }
  pynative::PyNativeAlgo::GradCommon::SetForward(node_list);
  // jit_forward_graph will be changed output
  auto clone_graph = BasicClone(jit_forward_graph);
  jit_forward_graph->set_used_forward_nodes(node_list);
  ModifyOutputNode(jit_forward_graph);
  PyNativeAlgo::Common::DumpGraphIR("jit_modify_after_forward_graph.ir", jit_forward_graph);
  return clone_graph;
}

py::object Jit::GradJit(const py::object &out, const py::args &args) {
  if (graph_phase_.empty()) {
    MS_LOG(EXCEPTION) << "The graph phase is empty, can not obtain jit func graph.";
  }
  PyNativeAlgo::Common::GetPyNativeExecutor()->forward_executor()->WaitForwardTask();
  // Get forward graph
  MS_LOG(DEBUG) << "jit func graph phase: " << graph_phase_;
  auto executor = pipeline::GraphExecutorPy::GetInstance();
  MS_EXCEPTION_IF_NULL(executor);
  FuncGraphPtr jit_forward_graph = executor->GetFuncGraph(graph_phase_);
  MS_EXCEPTION_IF_NULL(jit_forward_graph);
  // Get actual forward output object.
  py::object ret = out;
  if (jit_forward_graph->modify_output()) {
    auto tuple_out = py::cast<py::tuple>(out);
    ret = tuple_out[0];
  }
  // Save dynamic shape info if output tensors of forward graph have dynamic shapes
  const auto &grad_executor = PyNativeAlgo::Common::GetPyNativeExecutor()->grad_executor();
  // Make Adjoint for grad graph of jit.
  if (!grad_executor->RequiresGrad()) {
    MS_LOG(DEBUG) << "Only run forward infer computation, no need to construct grad graph.";
    graph_phase_.clear();
    return ret;
  }
  compile_info_ = jit_compile_info_.at(graph_phase_);
  ValuePtr added_out_v = nullptr;
  const auto &op_run_info =
    GetOpRunInfo(out, args, graph_phase_, jit_forward_graph->modify_output(), jit_forward_graph, &added_out_v);
  PyNativeAlgo::Common::DumpGraphIR("jit_forward_graph.ir", jit_forward_graph);
  auto jit_grad_graph = executor->GetJitGradGraph(graph_phase_);
  if (compile_info_.is_dynamic_shape_) {
    grad_executor->set_use_dynamic_shape_process(true);
  }
  GradJitInner(op_run_info, grad_executor.get(), executor->GetJitPrimalFuncGraph(graph_phase_), jit_grad_graph,
               GetAddedNode(jit_forward_graph), added_out_v);
  Reset();
  return ret;
}
}  // namespace pynative
}  // namespace mindspore
