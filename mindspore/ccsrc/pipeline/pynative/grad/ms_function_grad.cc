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
#include "mindspore/core/ops/sequence_ops.h"
#include "pipeline/pynative/pynative_utils.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/parallel_context.h"
#include "ir/func_graph_cloner.h"
#include "runtime/pynative/async/async_queue.h"
#include "pipeline/pynative/grad/bprop_task.h"
#include "pipeline/jit/pass.h"
#include "frontend/expander/bprop/bprop.h"

namespace mindspore {
namespace pynative {
namespace {
const mindspore::HashSet<std::string> kNotRealOP{
  kMakeTupleOpName,
  kMakeListNewOpName,
  kTupleGetItemOpName,
  kStopGradientOpName,
  kUpdateStateOpName,
  kLoadOPName,
  kDependOpName,
  kReturnOpName,
  kNPUAllocFloatStatusOpName,
  kNPUGetFloatStatusOpName,
  kNPUClearFloatStatusOpName,
  kMirrorOperatorOpName,
  kSequenceSliceOpName,
  kSequenceMulOpName,
};

const mindspore::HashSet<std::string> kExpanderWhiteList{
  kVmapStackAssignOpName,
  kVmapUnstackAssignOpName,
  kPyExecuteOpName,
  kPrintOpName,
};

FrontendOpRunInfoPtr GetOpRunInfo(const py::object &out, const py::args &args, const std::string &graph_phase,
                                  bool modify_output, const FuncGraphPtr &ms_func_graph, ValuePtr *added_out_v) {
  auto op_run_info = std::make_shared<FrontendOpRunInfo>();
  op_run_info->requires_grad = true;
  op_run_info->is_ms_function_input = true;
  op_run_info->base_op_run_info.op_name = graph_phase;
  PyNativeAlgo::PyParser::ParseOpInputByPythonObj(op_run_info, args);
  // Set input abs
  op_run_info->op_grad_info->input_abs.resize(op_run_info->input_size);
  const auto &original_params = ms_func_graph->parameters();
  for (size_t i = 0; i < op_run_info->input_size; ++i) {
    op_run_info->op_grad_info->input_abs[i] = original_params[i]->abstract();
  }
  if (modify_output) {
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

  // Create a new make tuple node to hold all forward used nodes.
  abstract::AbstractBasePtrList added_abs_list;
  AnfNodePtrList added_node_list{NewValueNode(prim::kPrimMakeTuple)};
  const auto &used_forward_nodes = func_graph->used_forward_nodes();
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

bool IsRealOp(const PrimitivePtr &prim) {
  MS_EXCEPTION_IF_NULL(prim);
  return kNotRealOP.find(prim->name()) == kNotRealOP.end();
}

bool IsRealOp(const AnfNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &prim = GetCNodePrimitive(cnode);
  if (prim == nullptr) {
    return false;
  }
  return IsRealOp(prim);
}

void GetUsedCNodeInBpropGraph(const CNodePtr &cnode, const mindspore::HashSet<size_t> &unused_inputs,
                              AnfNodePtrList *node_list) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(node_list);
  // Check input used in single op bprop graph. For example,
  // A = a * b;
  // B = A * c;
  // So, A can also replace by its output
  size_t input_num = cnode->size() - 1;
  for (size_t i = 0; i < input_num; ++i) {
    if (unused_inputs.find(i) != unused_inputs.end() && cnode->input(i + 1)->isa<CNode>()) {
      // Input used by bprop graph, and it is a cnode have produce real output
      const auto &input_c = cnode->input(i + 1)->cast<CNodePtr>();
      if (IsPrimitive(input_c, prim::kPrimMakeTuple)) {
        size_t tuple_input_num = input_c->size() - 1;
        for (size_t j = 0; j < tuple_input_num; ++j) {
          if (auto f_node = common::AnfAlgo::VisitKernel(input_c, j).first; f_node->isa<CNode>() && IsRealOp(f_node)) {
            (void)node_list->emplace_back(f_node);
          }
        }
      } else {
        if (auto f_node = common::AnfAlgo::VisitKernel(input_c, 0).first; f_node->isa<CNode>() && IsRealOp(f_node)) {
          (void)node_list->emplace_back(f_node);
        }
      }
    }
  }
  // Check output used in single op bprop graph
  if (unused_inputs.find(cnode->size()) == unused_inputs.end()) {
    (void)node_list->emplace_back(cnode);
  }
}

CNodePtr GetAddedNode(const FuncGraphPtr &ms_func_graph) {
  MS_EXCEPTION_IF_NULL(ms_func_graph);
  if (!ms_func_graph->modify_output()) {
    return nullptr;
  }
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
  return merge_make_tuple->input(added_output_index)->cast<CNodePtr>();
}
}  // namespace

void MsFunction::RunReplace(const CNodePtr &added_node, const ValuePtrList &total_output_tensors) const {
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
  // Save op info with new tensors for current running ms_function func graph.
  if (index != total_output_tensors.size()) {
    MS_LOG(EXCEPTION) << "The index: " << index
                      << " should be equal to the size of total_output_tensors: " << total_output_tensors.size();
  }
}

void MsFunction::ReplaceAddedCnodeActualOutput(const CNodePtr &added_node,
                                               const ValuePtrList &total_output_tensors) const {
  MS_EXCEPTION_IF_NULL(added_node);
  // Replace new output tensors for forward nodes, it will also work in grad graph with same value node.
  MS_LOG(DEBUG) << "The added forward make tuple node info: " << added_node->DebugString();
  // The forward node in ms_function graph is created during compilation and is a
  // placeholder(mindspore/ccsrc/frontend/optimizer/ad/pynative_dfunctor.cc).After running ms_function, need to update
  // to real value.
  RunReplace(added_node, total_output_tensors);
}

void MsFunction::GetInputArgsNode(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                                  const FuncGraphPtr &ms_func_graph, AnfNodePtrList *input_nodes) const {
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

void MsFunction::GetWeightsNode(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                                const FuncGraphPtr &ms_func_graph, AnfNodePtrList *input_nodes) const {
  MS_EXCEPTION_IF_NULL(grad_executor);
  MS_EXCEPTION_IF_NULL(input_nodes);
  const auto &top_cell = grad_executor->top_cell();
  const auto &graph_info = top_cell->graph_info_map().at(top_cell->fg());
  MS_EXCEPTION_IF_NULL(graph_info);
  // Get weights info of ms_function
  MS_EXCEPTION_IF_NULL(ms_func_graph);
  const auto &original_params = ms_func_graph->parameters();
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

void MsFunction::MakeCNodeForMsFunction(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                                        const FuncGraphPtr &ms_func_graph, CNodePtr *ms_function_cnode) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(ms_func_graph);
  // Get input node info of ms_function
  AnfNodePtrList input_nodes{NewValueNode(ms_func_graph)};
  MS_EXCEPTION_IF_NULL(grad_executor);
  GetInputArgsNode(op_run_info, grad_executor, ms_func_graph, &input_nodes);
  // Get weights node info of ms_function.
  GetWeightsNode(op_run_info, grad_executor, ms_func_graph, &input_nodes);
  // Make a CNode which includes ms_function fprop graph and inputs node
  MS_EXCEPTION_IF_NULL(ms_function_cnode);
  *ms_function_cnode = grad_executor->top_cell()->fg()->NewCNode(input_nodes);
  (*ms_function_cnode)->set_abstract(ms_func_graph->output()->abstract());
  MS_LOG(DEBUG) << "Make ms function forward CNode: " << (*ms_function_cnode)->DebugString();
}

void MsFunction::MakeAdjointForMsFunction(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                                          const FuncGraphPtr &ms_func_graph, const FuncGraphPtr &grad_graph,
                                          bool has_added_v) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(grad_executor);

  const auto &top_cell = grad_executor->top_cell();
  PyNativeAlgo::Common::SetGraphInputAndWeightsInfo(op_run_info, ms_func_graph);
  RecordForwardGraphForMsFunction(op_run_info, grad_executor, ms_func_graph);
  // Connect grad graph of ms_function to context.
  (void)PyNativeAlgo::Common::SetValueGradInfo(op_run_info->real_out, top_cell, TensorGradType::kOpOutput);
  auto op_grad_info = std::make_shared<OpGradInfo>();
  op_grad_info->input_value = op_run_info->op_grad_info->input_value;
  op_grad_info->input_abs = op_run_info->op_grad_info->input_abs;
  op_grad_info->out_value = op_run_info->real_out;
  op_grad_info->out_abs = ms_func_graph->output()->abstract();
  op_grad_info->input_value_grad_type = op_run_info->op_grad_info->input_value_grad_type;
  auto grad_param = std::make_shared<autograd::GradParam>(op_grad_info, !top_cell->is_high_order_top_cell(),
                                                          grad_executor->use_dynamic_shape_process());
  grad_param->is_control_flow = compile_info_.is_control_flow_;

  grad_param->has_added_v = has_added_v;
  grad_param->is_ms_function_graph = true;
  // As long as the ms function is in the process of dynamic shape,
  // let it run actor execution to avoid backend pass
  grad_param->is_ms_function_self_dynamic_shape = compile_info_.is_dynamic_shape_;

  grad_param->fg = grad_graph;
  grad_param->source_fg = ms_func_graph;
  grad_param->graph_cache_key = graph_phase_;
  auto auto_grad_cell_ptr = top_cell->auto_grad_cell_ptr();
  KPynativeWithFProp(grad_executor, auto_grad_cell_ptr, grad_param);
  top_cell->set_need_do_final_opt(true);
  top_cell->set_has_call_graph(grad_executor->use_dynamic_shape_process());
  top_cell->set_has_control_flow(compile_info_.is_control_flow_);
}

void MsFunction::KPynativeWithFProp(const GradExecutor *grad_executor,
                                    const autograd::AutoGradCellImplPtr &auto_grad_cell_ptr,
                                    const autograd::GradParamPtr &grad_param) const {
  {
    py::gil_scoped_release gil_release;
    grad_executor->async_executor()->Wait();
  }
  MS_EXCEPTION_IF_NULL(auto_grad_cell_ptr);
  if (!auto_grad_cell_ptr->KPynativeWithFProp(grad_param)) {
    MS_LOG(EXCEPTION) << "Failed to make adjoint for ms_function cnode";
  }
}

void MsFunction::RecordForwardGraphForMsFunction(const FrontendOpRunInfoPtr &op_run_info,
                                                 const GradExecutor *grad_executor,
                                                 const FuncGraphPtr &ms_func_graph) const {
  int save_graphs = MsContext::GetInstance()->get_param<int>(MS_CTX_SAVE_GRAPHS_FLAG);
  if (save_graphs) {
    CNodePtr ms_function_cnode = nullptr;
    MakeCNodeForMsFunction(op_run_info, grad_executor, ms_func_graph, &ms_function_cnode);
    MS_EXCEPTION_IF_NULL(ms_function_cnode);
    const auto &out_id = PyNativeAlgo::Common::GetIdByValue(op_run_info->real_out);
    const auto &top_cell = grad_executor->top_cell();
    top_cell->SetNodeMapInGraphInfoMap(out_id, ms_function_cnode);
  }
}

void MsFunction::GradMsFunctionInner(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                                     const FuncGraphPtr &primal_func_graph, const FuncGraphPtr &grad_graph,
                                     const CNodePtr &added_node, const ValuePtr &added_out_v) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(grad_executor);
  // Step 1: Replace added cnode forward with actual output
  ValuePtr flatten_v = added_out_v;
  bool added_v_is_empty = true;
  if (added_out_v != nullptr) {
    ValuePtrList total_output_tensors;
    PyNativeAlgo::DataConvert::FlattenValueSeqArg(added_out_v, &total_output_tensors);
    flatten_v = std::make_shared<ValueTuple>(total_output_tensors);
    added_v_is_empty = total_output_tensors.empty();
    ReplaceAddedCnodeActualOutput(added_node, total_output_tensors);
  }

  // Step 2: Update actual output tensors used in grad graph.
  MS_LOG(DEBUG) << "ms_function actual output value: " << op_run_info->real_out->ToString();
  grad_executor->top_cell()->GetOpInfo(op_run_info);
  grad_executor->UpdateTopCellForwardTensorInfoInBpropGraph(op_run_info->op_info, op_run_info->real_out);

  // Step 3: Update output tensors of added forward nodes, which are added to return node of ms_function func graph.
  if (!added_v_is_empty) {
    if (grad_executor->use_dynamic_shape_process()) {
      // If ms function is not control flow, the ms function is executed by actor under dynamic shape, and valuenode
      // will be updated
      if (!compile_info_.is_control_flow_) {
        UpdateMsFunctionlForwardTensorInfoInBpropGraph(op_run_info->op_info + kAddedValue, flatten_v);
      }
    } else {
      // Static shape will run by replace
      grad_executor->UpdateTopCellForwardTensorInfoInBpropGraph(op_run_info->op_info + kAddedValue, flatten_v);
    }
  }

  // Make Adjoint for grad graph
  MakeAdjointForMsFunction(op_run_info, grad_executor, primal_func_graph, grad_graph, !added_v_is_empty);

  auto node_info = std::make_shared<DynamicDetectNodeInfo>(nullptr, op_run_info->op_grad_info->input_abs,
                                                           op_run_info->base_op_run_info.abstract);
  node_info->is_graph_node = true;
  node_info->graph_phase = op_run_info->base_op_run_info.op_name;
  grad_executor->CheckGraphDynamic(op_run_info->op_grad_info->input_value, node_info);
}

void MsFunction::UpdateMsFunctionlForwardTensorInfoInBpropGraph(const std::string &op_info, const ValuePtr &v) {
  const auto it = graph_phase_with_replace_info_.find(graph_phase_);
  if (it == graph_phase_with_replace_info_.end()) {
    MS_LOG(DEBUG) << "Ms function " << graph_phase_ << " run firstly";
    auto &replace_info = graph_phase_with_replace_info_[graph_phase_];
    SetIdWithOpInfo(v, op_info, kIndex0, &(replace_info.id_with_op_info));
    return;
  }
  // Not first run
  MS_LOG(DEBUG) << "Update ms function forward output tensor info";
  UpdateForwardOutputTensorInfo(op_info, v, it->second);
}

void MsFunction::SaveForwardOutputTensorInfoInBpropGraph(const FuncGraphPtr &func_graph) {
  const auto it = graph_phase_with_replace_info_.find(graph_phase_);
  if (it == graph_phase_with_replace_info_.end()) {
    MS_LOG(EXCEPTION) << "Can not find graph phase " << graph_phase_ << " in graph_phase_with_replace_info";
  }
  MS_LOG(DEBUG) << "Save ms function forward output tensor info";
  auto manager = MakeManager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(func_graph);
  SaveForwardOutputTensorInfo(func_graph, true, &(it->second));
}

void MsFunction::Reset() { graph_phase_.clear(); }

FuncGraphPtr MsFunction::ProcessMsFunctionFuncGraph(const FuncGraphPtr &ms_func_graph) {
  MS_EXCEPTION_IF_NULL(ms_func_graph);
  if (PyNativeAlgo::Common::IsControlFlowGraph(ms_func_graph)) {
    MS_LOG(DEBUG) << "Get control flow";
    ms_function_compile_info_[graph_phase_].is_control_flow_ = true;
    return nullptr;
  }
  if (auto abs = ms_func_graph->output()->abstract(); abs != nullptr && abs->BuildShape()->IsDynamic()) {
    MS_LOG(DEBUG) << "Get dynamic shape";
    ms_function_compile_info_[graph_phase_].is_dynamic_shape_ = true;
    return nullptr;
  }
  ms_function_compile_info_[graph_phase_] = MsFunctionCompileInfo();
  PyNativeAlgo::Common::DumpGraphIR("ms_func_modify_before_forward_graph.ir", ms_func_graph);
  AnfNodePtrList node_list{};
  const auto &order = TopoSort(ms_func_graph->output());
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
    if (!IsRealOp(prim)) {
      continue;
    }
    MS_LOG(DEBUG) << "Get cnode " << cnode->DebugString();
    const auto &unused_inputs = BpropExpander().GetUnusedInputs(prim->name());
    if (!unused_inputs.empty() && unused_inputs.find(INT_MAX) != unused_inputs.end() &&
        kExpanderWhiteList.find(prim->name()) == kExpanderWhiteList.end()) {
      MS_LOG(DEBUG) << "Prim " << prim->name() << " is not support by expander";
      ms_function_compile_info_[graph_phase_].is_control_flow_ = true;
      return nullptr;
    }
    GetUsedCNodeInBpropGraph(cnode, unused_inputs, &node_list);
  }
  if (node_list.empty()) {
    MS_LOG(DEBUG) << "No need do replace";
    return ms_func_graph;
  }
  for (const auto &cn : node_list) {
    auto out = pynative::PyNativeAlgo::Common::CreatOutputTensorValueByAbstract(cn->abstract());
    const auto &c_node = cn->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(c_node);
    c_node->set_forward(PyNativeAlgo::Common::CreateValueNodeByValue(out, cn->abstract()), "");
  }
  // Ms_func_graph will be changed output
  auto clone_graph = BasicClone(ms_func_graph);
  ms_func_graph->set_used_forward_nodes(node_list);
  ModifyOutputNode(ms_func_graph);
  PyNativeAlgo::Common::DumpGraphIR("ms_func_modify_after_forward_graph.ir", ms_func_graph);
  return clone_graph;
}

py::object MsFunction::GradMsFunction(const py::object &out, const py::args &args) {
  if (graph_phase_.empty()) {
    MS_LOG(EXCEPTION) << "The graph phase is empty, can not obtain ms_function func graph.";
  }
  PyNativeAlgo::Common::GetPyNativeExecutor()->forward_executor()->WaitForwardTask();
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
  if (!grad_executor->RequiresGrad()) {
    MS_LOG(DEBUG) << "Only run forward infer computation, no need to construct grad graph.";
    graph_phase_.clear();
    return ret;
  }
  compile_info_ = ms_function_compile_info_.at(graph_phase_);
  ValuePtr added_out_v = nullptr;
  const auto &op_run_info =
    GetOpRunInfo(out, args, graph_phase_, ms_func_graph->modify_output(), ms_func_graph, &added_out_v);
  PyNativeAlgo::Common::DumpGraphIR("ms_func_forward_graph.ir", ms_func_graph);
  auto grad_graph = executor->GetGradGraph(graph_phase_);
  if (compile_info_.is_dynamic_shape_) {
    grad_executor->set_use_dynamic_shape_process(true);
  }
  GradMsFunctionInner(op_run_info, grad_executor.get(), executor->GetPrimalFuncGraph(graph_phase_), grad_graph,
                      GetAddedNode(ms_func_graph), added_out_v);
  Reset();
  return ret;
}
}  // namespace pynative
}  // namespace mindspore
