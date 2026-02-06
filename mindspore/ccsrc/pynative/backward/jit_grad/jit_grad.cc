/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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

#include "pynative/backward/jit_grad/jit_grad.h"

#include <utility>
#include "primitive/structure_op_name.h"
#include "primitive/framework_op_name.h"
#include "primitive/sequence_ops.h"
#include "pynative/backward/grad_utils.h"
#include "include/pynative/utils/pynative_utils.h"
#include "ir/func_graph_cloner.h"
#include "include/utils/pynative/common_utils.h"
#include "include/utils/pynative/adapter.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "pynative/backward/op_grad/func_grad.h"
#include "include/frontend/optimizer/ad/grad_interface.h"
#include "include/frontend/optimizer/ad/jit_grad_interface.h"
#include "include/frontend/jit/ps/executor/jit_executor_py.h"

namespace mindspore {
namespace pynative {
namespace {
constexpr char kAddedValue[] = "added_value";

const mindspore::HashSet<std::string> kExpanderWhiteList{
  kVmapStackAssignOpName,
  kVmapUnstackAssignOpName,
  kPyExecuteOpName,
  kPrintOpName,
};

py::tuple ConvertDictArgs(const py::object &args) {
  if (!py::isinstance<py::tuple>(args)) {
    MS_LOG(EXCEPTION) << "Args should be tuple but got: " << py::str(args);
  }
  auto tuple_args = py::cast<py::tuple>(args);
  py::list new_args;
  for (const auto &e : tuple_args) {
    auto element = py::cast<py::object>(e);
    if (py::isinstance<py::dict>(element)) {
      MS_LOG(INFO) << "Convert dict input " << py::str(element) << " to tuple with values.";
      new_args.append(py::reinterpret_steal<py::tuple>(PyDict_Values(element.ptr())));
    } else {
      new_args.append(element);
    }
  }
  return py::cast<py::tuple>(new_args);
}

FrontendOpRunInfoPtr GetOpRunInfo(const py::args &args, const std::string &graph_phase,
                                  const FuncGraphPtr &jit_forward_graph) {
  auto op_run_info = std::make_shared<FrontendOpRunInfo>();
  op_run_info->requires_grad = true;
  op_run_info->is_jit_input = true;
  op_run_info->base_op_run_info.op_name = graph_phase;
  // Dict input for graph should be converted to tuple after compile.
  const auto &new_args = ConvertDictArgs(args);
  PyNativeAlgo::PyParser::ParseOpInputByPythonObj(op_run_info, new_args);
  // Set input abs
  const auto &original_params = jit_forward_graph->parameters();
  if (op_run_info->input_size > original_params.size()) {
    MS_LOG(EXCEPTION) << "The number of inputs: " << op_run_info->input_size
                      << " should not greater than the number of parameters,which is : " << original_params.size();
  }
  for (size_t i = 0; i < op_run_info->input_size; ++i) {
    op_run_info->op_grad_info->input_abs[i] = original_params[i]->abstract();
  }
  return op_run_info;
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
  }
  if (abs->isa<abstract::AbstractSequence>()) {
    const auto &abs_sequence = abs->cast<abstract::AbstractSequencePtr>();
    return std::any_of(abs_sequence->elements().begin(), abs_sequence->elements().end(),
                       [](const abstract::AbstractBasePtr &item) { return JitOutputHasDict(item); });
  }
  return false;
}
}  // namespace

void Jit::ClearAutoGradCache() {
  ad::ClearGradCache();
  AutoGradUtil::ClearAutoGradStaticCache();
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
    auto tensor_id = std::to_string(tensor_value->id());
    const auto it = graph_info->weight_params.find(tensor_id);
    if (it != graph_info->weight_params.end()) {
      param = it->second;
    } else {
      MS_EXCEPTION_IF_NULL(param);
      top_cell->fg()->add_parameter(param);
      if (param->debug_info() != nullptr) {
        param->debug_info()->set_name(param->name());
      }
      top_cell->SetParamNodeMapInGraphInfoMap(tensor_id, param, true);
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
  MS_EXCEPTION_IF_NULL(grad_executor->top_cell()->fg());
  *jit_cnode = grad_executor->top_cell()->fg()->NewCNode(input_nodes);
  MS_EXCEPTION_IF_NULL(jit_forward_graph->output());
  (*jit_cnode)->set_abstract(jit_forward_graph->output()->abstract());
  MS_LOG(DEBUG) << "Make jit forward CNode: " << (*jit_cnode)->DebugString();
}

GradParamPtr Jit::CreateJitGradParam(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                                     const FuncGraphPtr &jit_forward_graph, const FuncGraphPtr &jit_grad_graph) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(grad_executor);
  PyNativeAlgo::Common::SetGraphInputAndWeightsInfo(op_run_info, jit_forward_graph);
  MS_EXCEPTION_IF_NULL(jit_forward_graph);
  op_run_info->op_grad_info->out_abs = jit_forward_graph->output()->abstract();

  auto grad_param = std::make_shared<GradParam>(op_run_info->op_grad_info);
  grad_param->is_jit_graph = true;
  // For adapter backward interface, we can not know high order info in forward step. So
  // when top cell is nullptr, we set high order false, this cause high order can not use with backward api use jit.
  grad_param->is_high_order =
    grad_executor->TopCellNoCheck() == nullptr ? false : grad_executor->top_cell()->is_high_order_top_cell();
  grad_param->is_control_flow = compile_info_.is_control_flow_;
  // As long as the jit is in the process of dynamic shape,
  // let it run actor execution to avoid backend pass
  grad_param->is_jit_self_dynamic_shape = compile_info_.is_dynamic_shape_;
  grad_param->fg = jit_grad_graph;
  grad_param->source_fg = jit_forward_graph;
  grad_param->graph_cache_key = graph_phase_;
  grad_param->jit_out_has_dict = JitOutputHasDict(op_run_info->op_grad_info->out_abs);
  return grad_param;
}

void Jit::RecordForwardGraphForJit(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                                   const FuncGraphPtr &jit_forward_graph) const {
  int save_graphs = MsContext::GetInstance()->CanDump(kIntroductory);
  const auto &top_cell = grad_executor->TopCellNoCheck();
  if (save_graphs && top_cell != nullptr) {
    CNodePtr jit_cnode = nullptr;
    MakeCNodeForJit(op_run_info, grad_executor, jit_forward_graph, &jit_cnode);
    MS_EXCEPTION_IF_NULL(jit_cnode);
    const auto &out_id = PyNativeAlgo::Common::GetIdByValue(op_run_info->real_out);
    top_cell->SetNodeMapInGraphInfoMap(out_id, jit_cnode);
  }
}

void Jit::GradJitInner(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                       const FuncGraphPtr &jit_forward_graph, const FuncGraphPtr &jit_grad_graph) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(grad_executor);
  MS_EXCEPTION_IF_NULL(jit_forward_graph);
  MS_EXCEPTION_IF_NULL(jit_forward_graph->output());
  const auto &forward_output_abs = jit_forward_graph->output()->abstract();
  MS_EXCEPTION_IF_NULL(forward_output_abs);
  auto &&grad_param = CreateJitGradParam(op_run_info, grad_executor, jit_forward_graph, jit_grad_graph);
  if (!autograd::KPynativeWithFProp(grad_param)) {
    MS_LOG(EXCEPTION) << "Failed to make adjoint for jit cnode";
  }
  op_run_info->real_out = grad_param->op_grad_info->out_value;
  // Step 2: Flatten output value
  ValuePtrList flatten_actual_output_list;
  CommonUtils::FlattenValueSeqArg(op_run_info->real_out, false, true, &flatten_actual_output_list);
  auto flatten_replace_value = std::make_shared<ValueTuple>(flatten_actual_output_list);
  // Update output related info
  op_run_info->op_grad_info->out_value = flatten_replace_value;
  MS_LOG(DEBUG) << "jit actual output value: " << op_run_info->real_out->ToString() << ", output id "
                << PyNativeAlgo::Common::GetIdByValue(op_run_info->real_out);
  RecordForwardGraphForJit(op_run_info, grad_executor, jit_forward_graph);
  const auto &top_cell = grad_executor->TopCellNoCheck();
  // When use flow auto grad, top cell may be nullptr;
  if (top_cell != nullptr) {
    top_cell->set_jit_out_has_dict(grad_param->jit_out_has_dict);
  }
}

bool Jit::GetJitGradGraph(const pipeline::ResourcePtr &resource, const std::string &phase) {
  graph_phase_ = phase;
  pipeline::ExecutorPyPtr graph_executor = pipeline::GetExecutor(graph_phase_);
  MS_EXCEPTION_IF_NULL(graph_executor);
  MS_LOG(DEBUG) << "The phase of current pipeline graph is: " << graph_phase_;
  if (graph_phase_.find("export") == 0 || !GradState::Get().RequiresGrad()) {
    MS_LOG(DEBUG) << "When exporting graph or only running forward process";
    return true;
  }

  // Set Primal graph
  MS_EXCEPTION_IF_NULL(resource);
  auto jit_forward_graph = resource->func_graph();
  MS_EXCEPTION_IF_NULL(jit_forward_graph);
  bool is_control_flow = PyNativeAlgo::Common::IsControlFlowGraph(jit_forward_graph);
  // Set jit compile info
  jit_compile_info_[graph_phase_] = JitCompileInfo();
  jit_compile_info_[graph_phase_].is_control_flow_ = is_control_flow;
  jit_compile_info_[graph_phase_].is_dynamic_shape_ = IsGraphDynamic(jit_forward_graph);
  if (resource->is_load()) {
    MS_LOG(DEBUG) << "Compile cache is loaded, skip grad graph generation.";
    auto graphs = ad::CacheFuncGraphBeforeOpt(nullptr, nullptr);
    graph_executor->SetJitPrimalFuncGraph(graphs.second, graph_phase_);
    graph_executor->SetJitGradGraph(graphs.first, graph_phase_);
    return true;
  }
  auto primal_fg = BasicClone(jit_forward_graph, true);

  // Get top cell recompute flag
  py::object input = resource->source_input();
  if (input != py::none() && py::hasattr(input, ad::kOutputNoRecompute)) {
    MS_LOG(INFO) << "Top cell with recompute flag, do not do recompute again in gradjit, cell: " << py::str(input);
    primal_fg->set_flag(ad::kTopCellWithRecompute, true);
  }

  // Using adgrad to generate fprop func graph for jit function in pynative mode
  // Using cloned jit_forward_graph --> primal_fg as input
  // Ensure that the primal graph found by fprop in GradJit is not affected by subsequent compilation passes.
  bool is_view_inplace = resource->is_pynative_grad_view_inplace();
  auto grad_graph = ad::Grad(primal_fg, opt::Optimizer::MakeEmptyOptimizer(resource), true,
                             ad::BpropAutoMonadLevel::kLevelNone, is_view_inplace);
  MS_EXCEPTION_IF_NULL(grad_graph);
  auto primal_fg_iter = grad_graph->transforms().find("primal");
  if (primal_fg_iter == grad_graph->transforms().end()) {
    MS_LOG(EXCEPTION) << "Cannot find primal func graph: " << grad_graph->ToString();
  }
  auto actual_primal_graph = primal_fg_iter->second.func_graph();
  (void)ad::CacheFuncGraphBeforeOpt(grad_graph, actual_primal_graph);
  graph_executor->SetJitPrimalFuncGraph(actual_primal_graph, graph_phase_);
  graph_executor->SetJitGradGraph(grad_graph, graph_phase_);
  CommonUtils::DumpGraphIR("jit_modify_before_forward_graph.ir", jit_forward_graph);
  MS_LOG(DEBUG) << "Func graph is control flow: " << jit_compile_info_[graph_phase_].is_control_flow_
                << " , is dynamic shape: " << jit_compile_info_[graph_phase_].is_dynamic_shape_;

  // Keep roots for only keeping forward func graph in resource.
  auto manager = resource->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->KeepRoots({jit_forward_graph});
  return true;
}

void Jit::Reset() { graph_phase_.clear(); }

py::object Jit::GradJit(const py::args &args) {
  if (graph_phase_.empty()) {
    MS_LOG(EXCEPTION) << "The graph phase is empty, can not obtain jit func graph.";
  }
  MS_LOG(DEBUG) << "jit func graph phase: " << graph_phase_;
  pipeline::ExecutorPyPtr executor = pipeline::GetExecutor(graph_phase_);
  MS_EXCEPTION_IF_NULL(executor);
  const auto &grad_executor = mindspore::pynative::PyNativeExecutor::grad_executor();
  // For no grad function, execute forward graph directly
  if (!GradState::Get().RequiresGrad()) {
    MS_LOG(DEBUG) << "Only run forward infer computation, no need to construct grad graph.";
    py::object ret = executor->Run(args, py::str(graph_phase_));
    graph_phase_.clear();
    return ret;
  }
  if (auto iter = jit_compile_info_.find(graph_phase_); iter != jit_compile_info_.end()) {
    compile_info_ = iter->second;
  } else {
    MS_LOG(EXCEPTION) << "Can not find graph_phase_: " << graph_phase_ << " in jit_compile_info_. "
                      << "Please check if the corresponding forward compilation process is missing.";
  }
  // Get compiled forward graph and generate op_run_info
  FuncGraphPtr jit_forward_graph = executor->GetFuncGraph(graph_phase_);
  MS_EXCEPTION_IF_NULL(jit_forward_graph);
  CommonUtils::DumpGraphIR("jit_forward_graph.ir", jit_forward_graph);
  // Get grad graph after adgrad
  auto jit_grad_graph = executor->GetJitGradGraph(graph_phase_);
  // Get primal graph
  auto jit_primal_graph = executor->GetJitPrimalFuncGraph(graph_phase_);
  MS_EXCEPTION_IF_NULL(jit_grad_graph);
  MS_EXCEPTION_IF_NULL(jit_primal_graph);

  py::object ret;
  const auto &op_run_info = GetOpRunInfo(args, graph_phase_, jit_forward_graph);
  MS_LOG(DEBUG) << "Start gradjit using forward primal graph and adgrad graph";
  GradJitInner(op_run_info, grad_executor.get(), jit_primal_graph, jit_grad_graph);
  MS_EXCEPTION_IF_NULL(op_run_info->real_out);
  ret = py::reinterpret_steal<py::object>(tensor::Wrap(op_run_info->real_out));
  Reset();
  return ret;
}

struct JitGradRegister {
  JitGradRegister() {
    PyNativeAdapter::SetGetJitBpropGraphHandler(
      [](const pipeline::ResourcePtr &resource, const std::string &phase) -> bool {
        return pynative::PyNativeExecutor::GetInstance()->grad_executor()->jit()->GetJitGradGraph(resource, phase);
      });
  }
} grad_jit_register;
}  // namespace pynative
}  // namespace mindspore
