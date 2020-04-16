/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "session/ascend_session.h"
#include <algorithm>
#include "operator/ops.h"
#include "ir/meta_tensor.h"
#include "ir/anf.h"
#include "common/trans.h"
#include "device/kernel_runtime.h"
#include "device/ascend/kernel_select_ascend.h"
#include "device/ascend/kernel_build_ascend.h"
#include "device/ascend/ascend_kernel_runtime.h"
#include "device/ascend/ascend_device_address.h"
#include "pre_activate/ascend/ascend_backend_optimization.h"
#include "device/kernel_adjust.h"
#include "device/ascend/ascend_stream_assign.h"
#include "predict/predict.h"
#include "session/anf_runtime_algorithm.h"
#include "ir/scalar.h"
#include "debug/anf_ir_dump.h"
#include "debug/anf_ir_utils.h"
#include "common/utils.h"
#include "pre_activate/common/helper.h"
#include "device/kernel_runtime_manager.h"
#include "kernel/tbe/tbe_python_funcs.h"
#include "utils/config_manager.h"

namespace mindspore {
namespace session {
const size_t kInvalidIndex = SIZE_MAX;
namespace {
void DumpGraphExeOrder(const std::vector<CNodePtr> &execution_order) {
  MS_LOG(INFO) << "Dump execution_order size " << execution_order.size();
  MS_LOG(INFO) << "[index][stream_label][graph_id][node string]";
  int i = 0;
  for (auto &cnode : execution_order) {
    MS_EXCEPTION_IF_NULL(cnode);
    MS_LOG(INFO) << "[ " << i << "]"
                 << "[" << AnfAlgo::GetStreamDistinctionLabel(cnode.get()) << "]"
                 << "[" << AnfAlgo::GetGraphId(cnode.get()) << "]"
                 << "[" << cnode->DebugString() << "]";
    i++;
  }
}

void DumpGraphInputArgs(const VectorRef &args) {
  MS_LOG(INFO) << "Args size[%lu]" << args.size();
  for (size_t i = 0; i < args.size(); i++) {
    if (utils::isa<AnfNodePtr>(args[i])) {
      auto anf = utils::cast<AnfNodePtr>(args[i]);
      MS_EXCEPTION_IF_NULL(anf);
      MS_LOG(INFO) << "Parameter arg" << i << " = [%s]" << anf->DebugString();
    } else if (utils::isa<ValuePtr>(args[i])) {
      auto value = utils::cast<ValuePtr>(args[i]);
      MS_EXCEPTION_IF_NULL(value);
      MS_LOG(INFO) << "Tensor arg" << i << " = " << value->ToString();
    } else {
      MS_LOG(INFO) << "Unknown arg" << i << " = " << args[i].ToString();
    }
  }
}

void SetStreamDistinctionLabel(const KernelGraphPtr &graph, uint32_t label, bool is_override) {
  MS_EXCEPTION_IF_NULL(graph);
  for (auto &node : graph->execution_order()) {
    if (is_override || AnfAlgo::GetStreamDistinctionLabel(node.get()) == kInvalidDistincLabel) {
      MS_EXCEPTION_IF_NULL(node);
      AnfAlgo::SetStreamDistinctionLabel(label, node.get());
    }
  }
}

GraphId GetDistinctionLabel(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  // if graph is empty,use graph id as distinction label
  if (graph->execution_order().empty()) {
    return graph->graph_id();
  }
  // else use first node of execution order as label
  return AnfAlgo::GetStreamDistinctionLabel(graph->execution_order()[0].get());
}

std::vector<BaseRef> GetRealArgs(const KernelGraphPtr graph, const VectorRef &args) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> graph_inputs = graph->inputs();
  auto valid_inputs = graph->ValidInputs();
  size_t real_args_size = 0;
  std::vector<BaseRef> real_args = {};
  for (size_t i = 0; i < args.size(); i++) {
    if (utils::isa<AnfNodePtr>(args[i])) {
      auto tmp_args = AnfAlgo::GetAllOutput(utils::cast<AnfNodePtr>(args[i]), {prim::kPrimTupleGetItem});
      for (auto &real_arg : tmp_args) {
        auto anf_node = utils::cast<AnfNodePtr>(real_arg);
        MS_EXCEPTION_IF_NULL(anf_node);
        auto abstract = anf_node->abstract();
        MS_EXCEPTION_IF_NULL(abstract);
        // create multiple parameters if is a tuple output real kernel
        if (abstract->isa<abstract::AbstractTuple>() &&
            !AnfAlgo::CheckPrimitiveType(anf_node, prim::kPrimTupleGetItem)) {
          auto tuple_abstract = abstract->cast<abstract::AbstractTuplePtr>();
          real_args_size += tuple_abstract->size();
          continue;
        }
        real_args_size += 1;
        real_args.push_back(real_arg);
      }
    } else {
      real_args_size += 1;
      real_args.push_back(args[i]);
    }
  }
  if (graph_inputs.size() != valid_inputs.size()) {
    MS_LOG(EXCEPTION) << "graph_inputs.size(): " << graph_inputs.size()
                      << ", valid_inputs.size(): " << valid_inputs.size() << " not equal";
  }
  if (real_args_size != graph_inputs.size()) {
    for (size_t j = 0; j < valid_inputs.size(); j++) {
      if (valid_inputs[j]) {
        MS_LOG(INFO) << "index: " << j << ", nodes: " << graph_inputs[j]->DebugString();
      }
    }
    MS_LOG(WARNING) << "real_args_size: " << real_args_size << ", graph_inputs.size(): " << graph_inputs.size()
                    << " not equal";
  }
  return real_args;
}
}  // namespace

GraphId AscendSession::CompileGraph(const AnfNodePtrList &lst, const AnfNodePtrList &outputs) {
  MS_LOG(INFO) << "start";
  auto graph_id = graph_sum_;
  // construct graph, if successfully, graph_sum_ + 1
  auto graph = ConstructKernelGraph(lst, outputs);
  MS_EXCEPTION_IF_NULL(graph);
  opt::AscendBackendIRFusionOptimization(graph);
  // select kernel build info
  SelectKernel(*graph);
  // convert kernel Graph to model
  predictmodel::StepConvertGraph(graph);
  // optimize graph
  HardwareOptimize(graph);
  // init runtime resource
  InitRuntimeResource();
  // assign static memory of parameters
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->AssignStaticMemoryInput(graph.get());
  MS_LOG(INFO) << "Compile graph " << graph_id << " success";
  return graph_id;
}

void AscendSession::BuildGraph(GraphId graph_id) {
  MS_LOG(INFO) << "start";
  auto graph = GetGraph(graph_id);
  MS_EXCEPTION_IF_NULL(graph);
  // multiple graph handle
  if (graph_id == final_graph_id_) {
    if (!graph->executable()) {
      return;
    }
    // merge child graph
    MergeGraphExecOrder();
  } else {
    // set the distinction label of single graph
    SetStreamDistinctionLabel(GetGraph(graph_id), graph_id, false);
  }
  // adjust execution order because  merge child graph and other special operations
  AdjustKernel(graph);
  // Assign streams for control sink and hccl and so on
  AssignStream(graph);

  device::KernelAdjust::GetInstance().Profiling(NOT_NULL(graph.get()));
  // build kernel if node is cnode
  BuildKernel(graph);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->precompile_only()) {
    MS_LOG(INFO) << "Precompile only, stop in build kernel step";
  } else {
    // alloc memory, including static memory and dynamic memory
    MemoryAlloc(graph.get());
    // generate task info for task sink mode
    GenerateTaskInfo(graph);
    // load task info to device if it is sink mode
    LoadTask(graph);
  }
  MS_LOG(INFO) << "end";
}

void AscendSession::RunGraph(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs,
                             VectorRef *const outputs) {
  MS_LOG(INFO) << "start";
  auto kernel_graph = GetGraph(graph_id);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  // if none of child graph and no anf output exists
  if (!kernel_graph->executable()) {
    MS_LOG(INFO) << "No child graph has anf output";
    UpdateOutputs(kernel_graph, outputs, inputs);
    return;
  }
  // load input data from user input
  LoadInputData(kernel_graph, inputs);
  // convert inputs to model
  predictmodel::StepConvertWeight(inputs);
  {
    py::gil_scoped_release release;
    // run task on device
    ExecTask(kernel_graph);
  }
  // get result from device
  UpdateOutputs(kernel_graph, outputs, inputs);
  // summary
  Summary(kernel_graph.get());
  // dump used for debug
  Dump(kernel_graph);
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::RunOpHardwareOptimize(const std::shared_ptr<session::KernelGraph> &kernel_graph) const {
  MS_LOG(INFO) << "Start";
  // data layout optimization
  opt::RunOpAscendDataLayout(kernel_graph);
  // mixed precision optimization
  opt::AscendMixPrecision(kernel_graph);
  MS_LOG(INFO) << "Finish";
}

void AscendSession::RunOpExecTask(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  MS_LOG(INFO) << "Start!";
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  bool ret_ok = runtime_instance->LaunchKernel(kernel_graph.get());
  if (!ret_ok) {
    MS_LOG(EXCEPTION) << "run task error!";
  }
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::BuildOp(const OpRunInfo &op_run_info, const GraphInfo &graph_info,
                            std::vector<tensor::TensorPtr> *input_tensors) {
  MS_EXCEPTION_IF_NULL(input_tensors);
  MS_LOG(INFO) << "Build op " << op_run_info.op_name << " start !";
  // construct graph include one op
  auto graph = ConstructSingleOpGraph(op_run_info, input_tensors);
  MS_EXCEPTION_IF_NULL(graph);
  opt::RunOpAscendBackendIRFusionOptimization(graph);
  // kernel select
  SelectKernel(*graph);
  // optimize
  RunOpHardwareOptimize(graph);
  // init runtime resource
  InitRuntimeResource();
  // build kernel
  RunOpAdjustKernel(graph);
  BuildKernel(graph);
  run_op_graphs_[graph_info] = graph;
}

py::tuple AscendSession::RunOp(const OpRunInfo &op_run_info, const GraphInfo &graph_info,
                               const std::vector<tensor::TensorPtr> &input_tensors) {
  auto graph = run_op_graphs_[graph_info];
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Run op " << op_run_info.op_name << " start!";
  // malloc mem
  RunOpMemoryAlloc(input_tensors, graph.get());
  // load input data to device
  LoadInputData(graph, input_tensors);
  // run op
  RunOpExecTask(graph);
  // get output
  VectorRef outputs;
  UpdateOutputs(graph, &outputs, input_tensors);
  // trans output to tuple
  auto output_tensors = TransformBaseRefListToTuple(outputs);
  if (!utils::isa<PyObjectRef>(output_tensors) ||
      !py::isinstance<py::tuple>(utils::cast<PyObjectRef>(output_tensors).object_)) {
    MS_LOG(EXCEPTION) << "The output tensors should be a tuple !";
  }
  py::object tuple_obj = utils::cast<PyObjectRef>(output_tensors).object_;
  py::tuple tuple_tensors = py::cast<py::tuple>(tuple_obj);
  run_op_graphs_.clear();
  MS_LOG(INFO) << "Run op " << op_run_info.op_name << " finish!";
  return tuple_tensors;
}

// compile graph steps
void AscendSession::SelectKernel(const KernelGraph &kernel_graph) const {
  MS_LOG(INFO) << "Start!";
  for (const auto &cnode : kernel_graph.execution_order()) {
    device::ascend::SelectKernelInfo(cnode);
    MS_LOG(INFO) << "Select ApplyKernel: " << cnode->DebugString();
  }
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::InitRuntimeResource() {
  MS_LOG(INFO) << "Start!";
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  if (!runtime_instance->Init()) {
    MS_LOG(EXCEPTION) << "Kernel runtime init error.";
  }
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::HardwareOptimize(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  MS_LOG(INFO) << "HardwareOptimize start!";
  opt::AscendBackendOptimization(kernel_graph);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
  MS_LOG(INFO) << "HardwareOptimize Finish!";
}

void AscendSession::AdjustKernel(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  MS_LOG(INFO) << "Start!";
  device::KernelAdjust::GetInstance().Reorder(kernel_graph);
  opt::HideNopNode(kernel_graph.get());
  // Insert CLearZero op
  // prepare for next step from json get atomic info
  BuildKernel(kernel_graph);
  device::ascend::KernelBuildPreprocess(kernel_graph.get());
  device::KernelAdjust::GetInstance().InsertSwitchLoop(kernel_graph);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->save_graphs_flag();
  auto save_graphs_path = context_ptr->save_graphs_path();
  if (save_graphs_path.empty()) {
    save_graphs_path = ".";
  }
  if (save_graphs) {
    std::string file_path = save_graphs_path + "/" + "after_adjust_kernel.ir";
    DumpIR(file_path, kernel_graph);
  }
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::RunOpAdjustKernel(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  MS_LOG(INFO) << "Start!";
  opt::HideNopNode(kernel_graph.get());
  // Insert CLearZero op
  // prepare for next step from json get atomic info
  BuildKernel(kernel_graph);
  device::ascend::KernelBuildPreprocess(kernel_graph.get());
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::AssignStream(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  MS_LOG(INFO) << "Start!";
  device::ascend::AscendStreamAssign::GetInstance().AssignStreamNew(kernel_graph);
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::BuildKernel(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  MS_LOG(INFO) << "Start!";
  struct timeval start_time, end_time;
  (void)gettimeofday(&start_time, nullptr);
  auto ret = device::ascend::KernelBuild(kernel_graph.get());
  if (!ret) {
    MS_LOG(EXCEPTION) << "Kernel build error.";
  }
  (void)gettimeofday(&end_time, nullptr);
  const uint64_t kUSecondInSecond = 1000000;
  uint64_t cost = kUSecondInSecond * static_cast<uint64_t>(end_time.tv_sec - start_time.tv_sec);
  cost += static_cast<uint64_t>(end_time.tv_usec - start_time.tv_usec);
  MS_LOG(INFO) << "KernelBuild run in  " << PRIu64 << " us " << cost;
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::MemoryAlloc(KernelGraph *kernel_graph) const {
  MS_LOG(INFO) << "Start!";
  MS_EXCEPTION_IF_NULL(kernel_graph);
  opt::RemoveNopNode(kernel_graph);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->AssignMemory(kernel_graph);
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::RunOpMemoryAlloc(const std::vector<tensor::TensorPtr> &input_tensors,
                                     KernelGraph *kernel_graph) const {
  MS_LOG(INFO) << "Start memory alloc!";
  MS_EXCEPTION_IF_NULL(kernel_graph);
  opt::RemoveNopNode(kernel_graph);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->RunOpAssignMemory(input_tensors, kernel_graph);
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::GenerateTaskInfo(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  MS_LOG(INFO) << "Start!";
  (void)device::KernelAdjust::GetInstance().StepLoadCtrlInputs(context_, kernel_graph);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  bool ret_ok = runtime_instance->GenTask(kernel_graph.get());
  if (!ret_ok) {
    MS_LOG(EXCEPTION) << "Generate task error!";
  }
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::LoadTask(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  MS_LOG(INFO) << "Start!";
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  bool ret_ok = runtime_instance->LoadTask(kernel_graph.get());
  if (!ret_ok) {
    MS_LOG(EXCEPTION) << "Load task error!";
  }
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::ExecTask(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  MS_LOG(INFO) << "Start!";
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  bool ret_ok = runtime_instance->Run(kernel_graph.get());
  if (!ret_ok) {
    MS_LOG(EXCEPTION) << "run task error!";
  }
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::Dump(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  MS_LOG(INFO) << "Start!";
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  (void)runtime_instance->DumpData(kernel_graph.get());
  MS_LOG(INFO) << "Finish!";
}

GraphId AscendSession::SetFinalGraphInput(const std::vector<AnfNodePtr> &args) {
  MS_LOG(INFO) << "Start! Args size " << args.size();
  auto final_graph = std::make_shared<KernelGraph>();
  final_graph_id_ = graph_sum_++;
  graphs_[final_graph_id_] = final_graph;
  final_graph->set_graph_id(final_graph_id_);
  MS_LOG(INFO) << "Create a new final graph" << final_graph_id_ << "success";
  // init private variables and bind them with final_graph_id
  graph_execute_orders_[final_graph_id_] = std::vector<GraphId>();
  graph_order_types_[final_graph_id_] = std::vector<GraphType>();
  for (const auto &parameter : args) {
    MS_EXCEPTION_IF_NULL(parameter);
    if (!parameter->isa<Parameter>()) {
      MS_LOG(EXCEPTION) << parameter->DebugString() << " is not a parameter type!";
    }
    AnfNodePtr parameter_backend = nullptr;
    // if function return UINT_MAX,the parameter is not exist in child graph
    auto parameter_belong_graph_id = GetGraphIdByNode(parameter);
    if (parameter_belong_graph_id == kInvalidGraphId) {
      parameter_backend = final_graph->NewParameter(parameter->cast<ParameterPtr>());
      final_graph->FrontBackendlMapAdd(parameter, parameter_backend);
      MS_LOG(INFO) << "New parameter" << parameter->DebugString() << "in final_graph";
    } else {
      // parametr is a parameter of child graph
      auto graph = GetGraph(parameter_belong_graph_id);
      MS_EXCEPTION_IF_NULL(graph);
      MS_LOG(INFO) << "Reuse parameter [" << parameter->DebugString() << "] of child graph ["
                   << parameter_belong_graph_id << "]";
      parameter_backend = graph->GetBackendAnfByFrontAnf(parameter);
    }
    MS_EXCEPTION_IF_NULL(parameter_backend);
    MS_LOG(INFO) << "parameter backend " << parameter_backend->DebugString() << " belong_graph_id "
                 << AnfAlgo::GetGraphId(parameter_backend.get());
    // add parameter in backend to final graph inputs
    auto final_graph_inputs = final_graph->MutableInputs();
    MS_EXCEPTION_IF_NULL(final_graph_inputs);
    final_graph_inputs->push_back(parameter_backend);
  }
  MS_LOG(INFO) << "End final_graph_id " << final_graph_id_;
  return final_graph_id_;
}

void AscendSession::SetFinalGraphOutput(const BaseRef &output) {
  auto final_graph = GetGraph(final_graph_id_);
  MS_EXCEPTION_IF_NULL(final_graph);
  if (!utils::isa<AnfNodePtr>(output)) {
    if (!utils::isa<ValuePtr>(output)) {
      MS_LOG(EXCEPTION) << "Unknown output type:" << output.ToString();
    }
    auto value_ptr = utils::cast<ValuePtr>(output);
    auto value_node = NewValueNode(value_ptr);
    MS_EXCEPTION_IF_NULL(value_node);
    auto kernel_info = std::make_shared<device::KernelInfo>();
    value_node->set_kernel_info(kernel_info);
    value_node->set_abstract(abstract::FromValue(value_ptr));
    final_graph->set_output(final_graph->NewCNode({NewValueNode(prim::kPrimMakeTuple), value_node}));
    final_graph->set_executable(false);
    MS_LOG(INFO) << "Not anf output[" << output.ToString() << "]";
    return;
  }
  // get the backend anf node related to the output node of front
  auto output_anf_node = utils::cast<AnfNodePtr>(output);
  auto output_from_graph_id = GetGraphIdByNode(output_anf_node);
  auto output_from_graph = GetGraph(output_from_graph_id);
  MS_EXCEPTION_IF_NULL(output_anf_node);
  MS_LOG(INFO) << "Set the output[" << output_anf_node->DebugString() << "] of graph[" << output_from_graph_id
               << "] to final graph";
  MS_EXCEPTION_IF_NULL(output_from_graph);
  // if output is from final graph,it remarks no child graph exist
  if (final_graph_id_ == output_from_graph_id) {
    MS_LOG(INFO) << "No child graph,output is " << output_anf_node->DebugString();
    final_graph->set_output(ConstructOutput({output_anf_node}, final_graph));
    final_graph->set_executable(false);
    return;
  }
  final_graph->set_output(output_from_graph->output());
}

KernelGraphPtr AscendSession::GetGraph(mindspore::GraphId graph_id) {
  auto it = graphs_.find(graph_id);
  if (it == graphs_.end()) {
    MS_LOG(WARNING) << "Can't find graph " << graph_id;
    return nullptr;
  }
  return it->second;
}

void AscendSession::InsertSwitchToGraph(GraphId condition_graph_id, GraphId true_graph_id) {
  MS_LOG(INFO) << "Start!";
  MS_LOG(INFO) << "Condition graph id[" << condition_graph_id << "],true graph id[" << true_graph_id << "]";
  auto condition_graph = GetGraph(condition_graph_id);
  MS_EXCEPTION_IF_NULL(condition_graph);
  tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(kNumberTypeInt32, std::vector<int>{1});
  int32_t *val = nullptr;
  val = static_cast<int32_t *>(tensor->data_c(true));
  MS_EXCEPTION_IF_NULL(val);
  *val = 0;
  auto value_node = std::make_shared<ValueNode>(tensor);
  value_node->set_abstract(abstract::FromValue(tensor, false));
  auto counter_const = condition_graph->NewValueNode(value_node);
  condition_graph->AddValueNodeToGraph(counter_const);
  // create a new switch op
  auto switch_primitive = std::make_shared<Primitive>("StreamSwitch");
  auto kernel_build_info_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  kernel_build_info_builder->SetOutputsFormat(std::vector<std::string>{kOpFormat_DEFAULT});
  kernel_build_info_builder->SetOutputsDeviceType(std::vector<TypeId>{kNumberTypeInt32});
  kernel_build_info_builder->SetFusionType(kernel::FusionType::OPAQUE);
  kernel_build_info_builder->SetProcessor(kernel::Processor::AICORE);
  kernel_build_info_builder->SetKernelType(KernelType::RT_KERNEL);
  auto cond_output_it = condition_output_.find(condition_graph_id);
  if (cond_output_it == condition_output_.end()) {
    MS_LOG(EXCEPTION) << "Can't find condition graph" << condition_graph_id;
  }
  auto cond_output_kernel =
    AnfAlgo::VisitKernel(condition_graph->GetBackendAnfByFrontAnf(cond_output_it->second), 0).first;
  MS_EXCEPTION_IF_NULL(cond_output_kernel);
  std::vector<AnfNodePtr> inputs = {NewValueNode(switch_primitive), cond_output_kernel, counter_const};
  CNodePtr switch_node = condition_graph->NewCNode(inputs);
  AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info_builder->Build(), switch_node.get());
  MS_EXCEPTION_IF_NULL(switch_node);
  switch_node->set_abstract(std::make_shared<abstract::AbstractNone>());
  AnfAlgo::SetGraphId(condition_graph_id, switch_node.get());
  AnfAlgo::SetStreamDistinctionLabel(GetDistinctionLabel(GetGraph(condition_graph_id)), switch_node.get());
  // set attr: cond_ RT_GREATER
  AnfAlgo::SetNodeAttr(kAttrSwitchCondition, MakeValue<int>(static_cast<int>(RT_GREATER)), switch_node);
  // set attr:data_type
  AnfAlgo::SetNodeAttr(kAttrDataType, MakeValue<int>(static_cast<int>(RT_SWITCH_INT64)), switch_node);
  // set attr:true branch graph id ,which is same to stream distinction label
  AnfAlgo::SetNodeAttr(kAttrTrueBranchStream, MakeValue<uint32_t>(true_graph_id), switch_node);
  // append switch at the end of condition graph
  std::vector<CNodePtr> exec_order = condition_graph->execution_order();
  exec_order.push_back(switch_node);
  condition_graph->set_execution_order(exec_order);
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::CopyOutputOfIf(GraphId false_graph_id) {
  auto &graph_execute_order = GetGraphOrder(final_graph_id_);
  auto &graph_order_type = GetGraphOrderType(final_graph_id_);
  auto false_index = ExecOrderOfChildGraph(final_graph_id_, false_graph_id);
  if (false_index == kInvalidIndex || false_index == 0) {
    return;
  }
  for (int i = SizeToInt(false_index) - 1; i >= 0; i--) {
    size_t graph_index = IntToSize(i);
    if (graph_index >= graph_execute_order.size()) {
      MS_LOG(EXCEPTION) << "Graph index[" << graph_index << "] out of range[" << graph_execute_order.size() << "]";
    }
    if (graph_order_type[graph_index] == COMMON_GRAPH) {
      auto true_last_id = graph_execute_order[graph_index];
      MS_LOG(INFO) << "The last graph of if true branch is " << true_last_id;
      auto true_last = GetGraph(true_last_id);
      auto final_graph = GetGraph(final_graph_id_);
      MS_EXCEPTION_IF_NULL(final_graph);
      auto false_last_id = AnfAlgo::GetGraphId(final_graph->output().get());
      auto false_last = GetGraph(false_last_id);
      MS_EXCEPTION_IF_NULL(true_last);
      MS_EXCEPTION_IF_NULL(false_last);
      MS_LOG(INFO) << "The last graph of false branch is " << false_last_id;
      // now only consider the single output
      InsertMultipleAssignToGraph(true_last_id, true_last->output(), false_last->output());
      // insert stream active for loop sink
      auto context_ptr = MsContext::GetInstance();
      MS_EXCEPTION_IF_NULL(context_ptr);
      if (context_ptr->enable_task_sink() && context_ptr->loop_sink_flag() &&
          ConfigManager::GetInstance().iter_num() > 1) {
        // insert active in true graph, another active will be inserted in kernel adjust
        InsertStreamActiveToGraph(true_last_id, kInvalidDistincLabel - 1);
      }
      break;
    }
  }
}

void AscendSession::SwitchCompile(GraphId cond_graph_id, GraphId true_graph_id, GraphId false_graph_id,
                                  const AnfNodePtr &output) {
  if (switches_.find(cond_graph_id) != switches_.end()) {
    MS_LOG(WARNING) << "Condition graph" << cond_graph_id << " has been set before ";
    return;
  }
  switches_[cond_graph_id] = std::pair<GraphId, GraphId>(true_graph_id, false_graph_id);
  condition_output_[cond_graph_id] = output;
  MS_LOG(INFO) << "New switch compile " << cond_graph_id << " " << true_graph_id << " " << false_graph_id;
  // set the type of condition graph
  auto cond_graph_index = ExecOrderOfChildGraph(final_graph_id_, cond_graph_id);
  auto &graph_order_type = GetGraphOrderType(final_graph_id_);
  if (cond_graph_index >= graph_order_type.size()) {
    MS_LOG(EXCEPTION) << "cond_graph_index " << cond_graph_index << " out of range " << graph_order_types_.size();
  }
  graph_order_type[cond_graph_index] = CONDITION_GRAPH;
  // update distinction label of false graph,update before merge to sure the distinction
  if (false_graph_id != kInvalidGraphId) {
    // false graph and condition in graph same stream
    auto condition_graph = GetGraph(cond_graph_id);
    SetStreamDistinctionLabel(GetGraph(false_graph_id), GetDistinctionLabel(condition_graph), true);
    // if false graph is a condition graph and has been switch compiled before,it's false should be updated again
    auto cond_it = switches_.find(false_graph_id);
    while (cond_it != switches_.end() && cond_it->second.second != kInvalidGraphId) {
      cond_graph_id = cond_it->first;
      false_graph_id = cond_it->second.second;
      condition_graph = GetGraph(cond_graph_id);
      SetStreamDistinctionLabel(GetGraph(false_graph_id), GetDistinctionLabel(condition_graph), true);
      cond_it = switches_.find(false_graph_id);
    }
  }
}  // namespace session

void AscendSession::MergeSwitchCompile() {
  auto graph_execute_order = GetGraphOrder(final_graph_id_);
  auto &graph_order_type = GetGraphOrderType(final_graph_id_);
  for (auto switch_compile : switches_) {
    auto cond_graph_id = switch_compile.first;
    auto true_graph_id = switch_compile.second.first;
    auto false_graph_id = switch_compile.second.second;
    MS_LOG(INFO) << "Switch compile: " << cond_graph_id << " " << true_graph_id << " " << false_graph_id;
    auto condition_graph = GetGraph(cond_graph_id);
    auto final_graph = GetGraph(final_graph_id_);
    MS_EXCEPTION_IF_NULL(condition_graph);
    MS_EXCEPTION_IF_NULL(final_graph);
    // insert switch to condition graph
    InsertSwitchToGraph(cond_graph_id, true_graph_id);
    auto cond_graph_index = ExecOrderOfChildGraph(final_graph_id_, cond_graph_id);
    auto prev_graph_id = kInvalidGraphId;
    // if condition graph is the first graph and final graph has assign op,then the final graph is the common graph
    if (cond_graph_index == 0 && !final_graph->execution_order().empty()) {
      prev_graph_id = final_graph_id_;
      // set the distinction label of final graph
      SetStreamDistinctionLabel(final_graph, final_graph_id_, true);
      // if condition graph is not the first graph
    } else if ((cond_graph_index - 1 < graph_execute_order.size()) &&
               (graph_order_type[cond_graph_index - 1] == COMMON_GRAPH)) {
      prev_graph_id = graph_execute_order[cond_graph_index - 1];
    }
    // insert stream active to common graph
    if (prev_graph_id != kInvalidGraphId) {
      InsertStreamActiveToGraph(prev_graph_id, GetDistinctionLabel(condition_graph));
    }
    // if this is a 'if' condition
    auto it = while_condition_graphs_.find(cond_graph_id);
    if (it == while_condition_graphs_.end()) {
      CopyOutputOfIf(false_graph_id);
    } else {
      // if it is a while,insert a stream active to true graph
      GraphId from_graph = it->second;
      InsertStreamActiveToGraph(from_graph, GetDistinctionLabel(condition_graph));
    }
  }
  MS_LOG(INFO) << "Finish!";
}

// insert active to graph
void AscendSession::SetActive(GraphId from, GraphId to) {
  if (while_condition_graphs_.find(to) != while_condition_graphs_.end()) {
    MS_LOG(WARNING) << " to " << to << " has been exits in map,from " << from << ",exist from "
                    << while_condition_graphs_[to];
    return;
  }
  MS_LOG(INFO) << "From " << from << " to " << to;
  auto &graph_order = GetGraphOrder(final_graph_id_);
  auto &graph_type = GetGraphOrderType(final_graph_id_);
  std::vector<GraphId> graph_order_new;
  std::vector<GraphType> graph_type_new;
  for (size_t i = 0; i < graph_order.size(); i++) {
    auto graph_id = graph_order[i];
    graph_order_new.push_back(graph_id);
    graph_type_new.push_back(graph_type[i]);
    if (from == graph_id) {
      graph_order_new.push_back(kInvalidGraphId);
      graph_type_new.push_back(BRANCH_END);
    }
  }
  graph_order = graph_order_new;
  graph_type = graph_type_new;
  // set the graph type of condition graph
  graph_type[ExecOrderOfChildGraph(final_graph_id_, to)] = CONDITION_GRAPH;
  // record the condition graph into while condition set
  while_condition_graphs_[to] = from;
}

void AscendSession::SetChildGraphParameter(const AnfNodePtr &front_anf, const AnfNodePtr &backend_parameter) {
  MS_LOG(INFO) << "Start!";
  MS_EXCEPTION_IF_NULL(backend_parameter);
  MS_EXCEPTION_IF_NULL(front_anf);
  if (!backend_parameter->isa<Parameter>()) {
    MS_LOG(EXCEPTION) << "Backend parameter's type is not a parameter,but is " << backend_parameter->ToString();
  }
  auto from_graph_id = GetGraphIdByNode(front_anf);
  auto from_graph = GetGraph(from_graph_id);
  MS_EXCEPTION_IF_NULL(from_graph);
  auto to_graph_id = AnfAlgo::GetGraphId(backend_parameter.get());
  auto to_graph = GetGraph(to_graph_id);
  auto backend_arg = from_graph->GetBackendAnfByFrontAnf(front_anf);
  MS_EXCEPTION_IF_NULL(to_graph);
  MS_LOG(INFO) << "Set node[" << front_anf->DebugString() << "] of graph[" << from_graph_id << "]to node["
               << backend_parameter->DebugString() << "] of graph[" << AnfAlgo::GetGraphId(backend_parameter.get())
               << "]";
  // a node should not assign to itself
  if (backend_arg.get() == backend_parameter.get()) {
    return;
  }
  // if arg is the the parameter of child graph,it is parameter of final graph too
  if (front_anf->isa<Parameter>()) {
    MS_EXCEPTION_IF_NULL(backend_arg);
    if (!AnfAlgo::OutputAddrExist(backend_arg, 0)) {
      // set parameter's addr in child graph to parameter in final graph
      AnfAlgo::SetOutputAddr(AnfAlgo::GetMutableOutputAddr(backend_parameter, 0), 0, backend_arg.get());
      MS_LOG(INFO) << "Assign mem of node" << backend_parameter->DebugString() << " of graph "
                   << AnfAlgo::GetGraphId(backend_parameter.get()) << " to node" << backend_arg->DebugString()
                   << "of graph " << AnfAlgo::GetGraphId(backend_arg.get());
      return;
    }
    // if a parameter is a weight and not linked to any executable node,device type will be kTypeUnknown,set it's device
    // type same to arg
    if (AnfAlgo::GetOutputDeviceDataType(backend_parameter, 0) == kTypeUnknown) {
      AnfAlgo::SetSelectKernelBuildInfo(AnfAlgo::GetSelectKernelBuildInfo(backend_arg), backend_parameter.get());
    }
    // if front anf is a parameter,we can assign the value back,because backend_parameter won't be change in it's graph
    // unless it's a weight.If backend_parameter is a weight,we should assign the value back.
    AnfAlgo::SetOutputAddr(AnfAlgo::GetMutableOutputAddr(backend_arg, 0), 0, backend_parameter.get());
    return;
  }
  InsertAssignToGraph(from_graph_id, backend_arg, backend_parameter);
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::SetChildGraphParameter(const tensor::TensorPtr &front_tensor, const AnfNodePtr &backend_parameter) {
  MS_LOG(INFO) << "Start!";
  // sync data from host to device
  MS_EXCEPTION_IF_NULL(front_tensor);
  size_t tensor_size = front_tensor->data().nbytes();
  auto addr = AnfAlgo::GetOutputAddr(backend_parameter, 0);
  MS_EXCEPTION_IF_NULL(addr);
  if (!addr->SyncHostToDevice(trans::GetRuntimePaddingShape(backend_parameter, 0), tensor_size,
                              front_tensor->data_type(), front_tensor->data_c(false))) {
    MS_LOG(EXCEPTION) << "Tensor SyncHostToDevice fail!";
  }
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::UpdateGraphOrder(GraphId to_graph_id) {
  MS_LOG(INFO) << "to_graph_id " << to_graph_id;
  auto &graph_order = GetGraphOrder(final_graph_id_);
  auto &graph_type = GetGraphOrderType(final_graph_id_);
  for (size_t i = 0; i < graph_order.size(); i++) {
    if (graph_order[i] == to_graph_id) {
      return;
    }
  }
  // if graph is not in graph order,add it to graph order
  SetStreamDistinctionLabel(GetGraph(to_graph_id), to_graph_id, false);
  graph_order.push_back(to_graph_id);
  graph_type.push_back(COMMON_GRAPH);
  for (size_t i = 0; i < graph_order.size(); i++) {
    MS_LOG(INFO) << "Index " << i << ",graph_id " << graph_order[i] << ",graph_type" << graph_type[i];
  }
}

void AscendSession::SetChildGraphInput(GraphId g, const VectorRef &args) {
  MS_LOG(INFO) << "Set input of graph " << g;
  auto to_graph = GetGraph(g);
  MS_EXCEPTION_IF_NULL(to_graph);
  DumpGraphInputArgs(args);
  UpdateGraphOrder(g);
  std::vector<AnfNodePtr> graph_inputs = to_graph->inputs();
  auto valid_inputs = to_graph->ValidInputs();
  auto real_args = GetRealArgs(to_graph, args);
  size_t input_index = 0;
  for (size_t i = 0; i < real_args.size(); i++) {
    if (input_index >= graph_inputs.size()) {
      MS_LOG(EXCEPTION) << "input_index " << input_index << " out of range size " << graph_inputs.size();
    }
    if (utils::isa<AnfNodePtr>(real_args[i])) {
      // arg is a anf node
      auto real_arg = utils::cast<AnfNodePtr>(real_args[i]);
      auto real_arg_output_num = AnfAlgo::GetOutputTensorNum(real_arg);
      if (!AnfAlgo::CheckPrimitiveType(real_arg, prim::kPrimTupleGetItem) && real_arg_output_num > 1) {
        input_index += real_arg_output_num;
        continue;
      }
      if (valid_inputs[input_index]) {
        SetChildGraphParameter(real_arg, graph_inputs[input_index]);
      } else {
        MS_LOG(DEBUG) << "Invalid input arg" << real_arg->DebugString();
      }
      input_index++;
    } else if (utils::isa<ValuePtr>(args[i])) {
      auto value = utils::cast<ValuePtr>(args[i]);
      MS_EXCEPTION_IF_NULL(value);
      // arg is a tensor
      if (!value->isa<Tensor>()) {
        MS_LOG(EXCEPTION) << "Value Node should be a tensor, unexpected value: " << value->ToString();
      }
      SetChildGraphParameter(value->cast<TensorPtr>(), graph_inputs[input_index]);
      input_index++;
    } else {
      MS_LOG(EXCEPTION) << "Unexpected arg type " << args[i].ToString();
    }
  }
  MS_LOG(INFO) << "Finish!";
}

GraphId AscendSession::GetGraphIdByNode(const AnfNodePtr &front_anf) const {
  for (const auto &graph_item : graphs_) {
    auto graph = graph_item.second;
    MS_EXCEPTION_IF_NULL(graph);
    // if front_anf is a parameter,the backend parameter may have two
    if (graph->GetBackendAnfByFrontAnf(front_anf) != nullptr) {
      return graph_item.first;
    }
  }
  MS_EXCEPTION_IF_NULL(front_anf);
  MS_LOG(DEBUG) << "front_anf " << front_anf->DebugString() << " is not exist in any graph";
  return kInvalidGraphId;
}

void AscendSession::MergeGraphExecOrder() {
  MS_LOG(INFO) << "Start!";
  // insert switch to graph
  MergeSwitchCompile();
  // merge graph order
  auto &graph_order = GetGraphOrder(final_graph_id_);
  auto &graph_type = GetGraphOrderType(final_graph_id_);
  auto final_graph = GetGraph(final_graph_id_);
  MS_EXCEPTION_IF_NULL(final_graph);
  if (graph_order.empty()) {
    MS_LOG(WARNING) << "Graph output is a lonely variable not linked to any op!";
    return;
  }
  // if first graph is common,the final graph has no label,then set the stream of final graph same with the first graph
  SetStreamDistinctionLabel(final_graph, graph_order[0], false);
  std::vector<CNodePtr> final_exec_order = final_graph->execution_order();
  KernelGraphPtr last_graph = nullptr;
  for (size_t i = 0; i < graph_order.size(); i++) {
    auto graph_id = graph_order[i];
    if (graph_type[i] == BRANCH_END || graph_type[i] == BRANCH_START) {
      continue;
    }
    auto child_graph = GetGraph(graph_id);
    last_graph = child_graph;
    MS_EXCEPTION_IF_NULL(child_graph);
    auto exec_order = child_graph->execution_order();
    MS_LOG(INFO) << "Merge graph,graph_id " << graph_id;
    (void)std::copy(exec_order.begin(), exec_order.end(), std::back_inserter(final_exec_order));
    // add all value nodes of child graphs to final graph
    for (auto &value_node : child_graph->graph_value_nodes()) {
      final_graph->AddValueNodeToGraph(value_node);
    }
    // copy ref map to final graph
    auto child_ref_map = child_graph->GetRefMap();
    for (auto &item : child_ref_map) {
      if (final_graph->IsInRefOutputMap(item.first)) {
        MS_LOG(EXCEPTION) << "The ref pair is already in final graph!";
      }
      final_graph->AddRefCorrespondPairs(item.first, item.second);
    }
  }
  // set final_exec_order into final graph
  MS_EXCEPTION_IF_NULL(final_graph);
  DumpGraphExeOrder(final_exec_order);
  final_graph->set_execution_order(final_exec_order);
}

void AscendSession::InsertAssignToGraph(GraphId graph_id, const AnfNodePtr &from, const AnfNodePtr &to) {
  MS_EXCEPTION_IF_NULL(from);
  MS_EXCEPTION_IF_NULL(to);
  if (AnfAlgo::OutputAddrExist(from, 0) && AnfAlgo::OutputAddrExist(to, 0) &&
      AnfAlgo::GetOutputAddr(from, 0) == AnfAlgo::GetOutputAddr(to, 0)) {
    return;
  }
  if (from.get() == to.get()) {
    return;
  }
  MS_LOG(INFO) << "Insert assign to graph " << graph_id << " from " << from->DebugString() << " to "
               << to->DebugString();
  auto graph = graphs_[graph_id];
  MS_EXCEPTION_IF_NULL(graph);
  // config inputs of assign node
  std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>("Assign")), to, from};
  // generate a new cnode
  auto assign_node = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(assign_node);
  assign_node->set_abstract(std::make_shared<abstract::AbstractNone>());
  auto kernel_build_info_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  kernel_build_info_builder->SetKernelType(KernelType::RT_KERNEL);
  AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info_builder->Build(), assign_node.get());
  AnfAlgo::SetStreamDistinctionLabel(GetDistinctionLabel(graph), assign_node.get());
  // append the assign at the end of from graph
  auto exec_order = graph->execution_order();
  exec_order.push_back(assign_node);
  graph->set_execution_order(exec_order);
}

void AscendSession::InsertMultipleAssignToGraph(GraphId graph_id, const AnfNodePtr &from, const AnfNodePtr &to) {
  std::vector<AnfNodePtr> from_outputs = AnfAlgo::GetAllOutput(from, {prim::kPrimTupleGetItem});
  std::vector<AnfNodePtr> to_outputs = AnfAlgo::GetAllOutput(to, {prim::kPrimTupleGetItem});
  MS_LOG(INFO) << "Insert assigns from [" << AnfAlgo::GetGraphId(from.get()) << "] to ["
               << AnfAlgo::GetGraphId(to.get()) << "]";
  if (from_outputs.size() != to_outputs.size()) {
    MS_LOG(INFO) << "From[" << from->DebugString(5) << "] to[" << to->DebugString(5) << "]";
    MS_LOG(EXCEPTION) << "From outputs size[" << from_outputs.size() << "] is not equal to to outputs size["
                      << to_outputs.size() << "]";
  }
  for (size_t i = 0; i < from_outputs.size(); i++) {
    InsertAssignToGraph(graph_id, from_outputs[i], to_outputs[i]);
  }
}

void AscendSession::InsertStreamActiveToGraph(GraphId graph_id, uint32_t actived_stream) {
  MS_LOG(INFO) << "Insert stream_active from " << graph_id << " to " << actived_stream;
  auto from_graph = graphs_[graph_id];
  MS_EXCEPTION_IF_NULL(from_graph);
  std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>("StreamActive"))};
  auto active_node = from_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(active_node);
  active_node->set_abstract(std::make_shared<abstract::AbstractNone>());
  auto kernel_build_info_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  kernel_build_info_builder->SetKernelType(KernelType::RT_KERNEL);
  AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info_builder->Build(), active_node.get());
  // set the active stream id into the attr of active node
  std::vector<uint32_t> active_index_value = {};
  active_index_value.push_back(actived_stream);
  AnfAlgo::SetNodeAttr(kAttrActiveStreamList, MakeValue<std::vector<uint32_t>>(active_index_value), active_node);
  AnfAlgo::SetStreamDistinctionLabel(GetDistinctionLabel(from_graph), active_node.get());
  // append the active node at the end of from graph
  auto exec_order = from_graph->execution_order();
  exec_order.push_back(active_node);
  from_graph->set_execution_order(exec_order);
}

size_t AscendSession::ExecOrderOfChildGraph(GraphId final_graph, GraphId child_graph) {
  auto &graph_order = GetGraphOrder(final_graph);
  for (size_t i = 0; i < graph_order.size(); i++) {
    if (child_graph == graph_order[i]) {
      return i;
    }
  }
  return kInvalidIndex;
}

std::vector<GraphId> &AscendSession::GetGraphOrder(GraphId final_graph_id) {
  auto graph_order_iter = graph_execute_orders_.find(final_graph_id);
  if (graph_order_iter == graph_execute_orders_.end()) {
    MS_LOG(EXCEPTION) << "Final graph" << final_graph_id << "has no child graph";
  }
  return graph_order_iter->second;
}

// get graph order type vector by graph id
std::vector<GraphType> &AscendSession::GetGraphOrderType(GraphId final_graph_id) {
  auto graph_type_iter = graph_order_types_.find(final_graph_id);
  if (graph_type_iter == graph_order_types_.end()) {
    MS_LOG(EXCEPTION) << "Final graph" << final_graph_id << "has no graph_order_types_";
  }
  return graph_type_iter->second;
}
}  // namespace session
}  // namespace mindspore
