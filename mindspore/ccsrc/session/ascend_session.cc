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
#include <map>
#include <tuple>
#include <set>
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
#include "device/ascend/ascend_label_assign.h"
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
  if (is_override || graph->stream_distinction_label() == kInvalidDistincLabel) {
    graph->set_stream_distinction_label(label);
  }
}

std::vector<BaseRef> GetRealArgs(const KernelGraphPtr graph, const VectorRef &args) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> graph_inputs = graph->inputs();
  auto valid_inputs = graph->valid_inputs();
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

void ClearRunOpMemoryResource(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  // clear input parameter memory resource
  for (const auto &input_node : kernel_graph->inputs()) {
    MS_EXCEPTION_IF_NULL(input_node);
    AnfAlgo::SetOutputAddr(nullptr, 0, input_node.get());
  }
  // clear input value node memory resource
  for (const auto &value_node : kernel_graph->graph_value_nodes()) {
    MS_EXCEPTION_IF_NULL(value_node);
    AnfAlgo::SetOutputAddr(nullptr, 0, value_node.get());
  }
  for (const auto &cnode : kernel_graph->execution_order()) {
    MS_EXCEPTION_IF_NULL(cnode);
    // clear output memory resource
    for (size_t index = 0; index < AnfAlgo::GetOutputTensorNum(cnode); ++index) {
      AnfAlgo::SetOutputAddr(nullptr, index, cnode.get());
    }
    // clear workspace memory resource
    auto kernel_mod = AnfAlgo::GetKernelMod(cnode);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    auto workspace_lists = kernel_mod->GetWorkspaceSizeList();
    for (size_t index = 0; index < workspace_lists.size(); ++index) {
      AnfAlgo::SetWorkspaceAddr(nullptr, index, cnode.get());
    }
  }
}
}  // namespace

GraphId AscendSession::CompileGraph(const AnfNodePtrList &lst, const AnfNodePtrList &outputs) {
  MS_LOG(INFO) << "start";
  // construct graph, if successfully, graph_sum_ + 1
  auto graph = ConstructKernelGraph(lst, outputs);
  auto graph_id = graph->graph_id();
  MS_LOG(INFO) << "Compile graph " << graph_id << " success";
  return graph_id;
}

GraphId AscendSession::CompileGraph(NotNull<FuncGraphPtr> func_graph) {
  MS_LOG(INFO) << "start";
  auto graph = ConstructKernelGraph(func_graph);
  // split switch
  SplitSwitch(graph.get());
  // insert goto labels and label_sets
  LinkChildGraphs(graph.get());
  // resource initialize
  InitRuntimeResource();
  // ir fusion
  IRFusion(graph);
  // kernel select
  SelectKernelGraphKernel(*graph);
  // convert model of predict module
  ConvertPredictModel(graph);
  // hardware optimize
  HardwareOptimizeGraphs(graph);
  // adjust kernel
  AdjustKernel(graph);
  // root graph valiate,include genearte execute order and so on
  RootGraphExecutorValidate(graph.get());
  // assign stream
  AssignStream(graph);
  // assign label
  AssignLabel(NOT_NULL(graph));
  // build kernel if node is cnode
  BuildKernel(graph);
  // alloc mem
  MemoryAlloc(graph.get());
  // task generate
  GenerateTaskInfo(graph);
  // load task into device
  LoadTask(graph);
  // return the graph id to backend
  auto graph_id = graph->graph_id();
  MS_LOG(INFO) << "Compile graph " << graph_id << " success";
  return graph_id;
}

void AscendSession::BuildGraph(GraphId graph_id) {
  MS_LOG(INFO) << "start";
  auto graph = GetGraph(graph_id);
  MS_EXCEPTION_IF_NULL(graph);
  // resource initialize
  InitRuntimeResource();
  // multiple graph handle
  if (graph_id == final_graph_id_) {
    if (!graph->executable()) {
      return;
    }
    // insert assigns to child graph
    InsertAllAssigns();
    // insert switch and active to child graph
    MergeSwitchCompile();
    // OptChildGraphs
    auto graph_order = GetGraphOrder(final_graph_id_);
    auto &graph_type = GetGraphOrderType(final_graph_id_);
    for (size_t i = 0; i < graph_order.size(); i++) {
      if (graph_type[i] == BRANCH_END || graph_type[i] == BRANCH_START) {
        continue;
      }
      MS_LOG(INFO) << "Start build child  graph " << graph_order[i];
      auto child_graph = GetGraph(graph_order[i]);
      CompileChildGraph(child_graph);
    }
    // merge child graph
    MergeGraphExecOrder();
  } else {
    auto single_graph = GetGraph(graph_id);
    CompileChildGraph(single_graph);
    // set the distinction label of single graph
    single_graph->set_stream_distinction_label(graph_id);
    single_graph->UpdateExecuteKernelStreamLabel();
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
  // sync the inital const tensor to device
  SyncInitialTenosrToDevice();
  MS_LOG(INFO) << "end";
}

void AscendSession::CompileChildGraph(const KernelGraphPtr &child_graph) {
  MS_EXCEPTION_IF_NULL(child_graph);
  opt::AscendBackendIRFusionOptimization(child_graph);
  // select kernel build info
  SelectKernel(*child_graph);
  // convert kernel Graph to model
  predictmodel::StepConvertGraph(child_graph);
  // optimize graph
  HardwareOptimize(child_graph);
  // assign static memory of parameters
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->AssignStaticMemoryInput(child_graph.get());
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

bool AscendSession::GraphCacheExist(const GraphInfo &graph_info) const {
  if (run_op_graphs_.find(graph_info) != run_op_graphs_.end()) {
    return true;
  }

  return false;
}

void AscendSession::BuildOp(const OpRunInfo &op_run_info, const GraphInfo &graph_info,
                            const std::vector<tensor::TensorPtr> &input_tensors, const std::vector<int> &tensors_mask) {
  MS_LOG(INFO) << "Build op " << op_run_info.op_name << " start !";
  if (GraphCacheExist(graph_info)) {
    MS_LOG(INFO) << "Build op " << op_run_info.op_name << " graph cache has existed !";
    return;
  }

  // construct graph include one op
  auto graph = ConstructSingleOpGraph(op_run_info, input_tensors, tensors_mask);
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
  MS_LOG(INFO) << "Build op " << op_run_info.op_name << " finish !";
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
  ClearRunOpMemoryResource(graph);
  MS_LOG(INFO) << "Run op " << op_run_info.op_name << " finish!";
  return tuple_tensors;
}

// compile graph steps
void AscendSession::SelectKernel(const KernelGraph &kernel_graph) const {
  MS_LOG(INFO) << "Start!";
  size_t raise_precision_count = 0;
  size_t reduce_precision_count = 0;
  for (const auto &cnode : kernel_graph.execution_order()) {
    auto status = device::ascend::SelectKernelInfo(cnode);
    if (status == device::ascend::kStatusRaisePrecision) {
      raise_precision_count++;
    } else if (status == device::ascend::kStatusReducePrecision) {
      reduce_precision_count++;
    }
    MS_LOG(INFO) << "Select ApplyKernel: " << cnode->DebugString();
  }
  if (raise_precision_count > 0) {
    MS_LOG(WARNING) << "There has " << raise_precision_count
                    << " node/nodes used raise precision to selected the kernel!";
  }
  if (reduce_precision_count > 0) {
    MS_LOG(WARNING) << "There has " << reduce_precision_count
                    << " node/nodes used reduce precision to selected the kernel!";
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

void AscendSession::AssignLabel(NotNull<const KernelGraphPtr &> kernel_graph) const {
  MS_LOG(INFO) << "Start!";
  device::ascend::AscendLabelAssign::GetInstance().AssignLabel(kernel_graph);
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
  auto final_graph = NewKernelGraph();
  final_graph_id_ = final_graph->graph_id();
  MS_LOG(INFO) << "Create a new final graph" << final_graph_id_ << " success";
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
      parameter_backend = CreateNewParameterFromParameter(parameter, true, final_graph.get());
      final_graph->FrontBackendlMapAdd(parameter, parameter_backend);
      MS_LOG(INFO) << "New parameter" << parameter->DebugString() << "in final_graph";
    } else {
      // parametr is a parameter of child graph
      auto graph = GetGraph(parameter_belong_graph_id);
      MS_EXCEPTION_IF_NULL(graph);
      MS_LOG(INFO) << "Reuse parameter [" << parameter->DebugString() << "] of child graph ["
                   << parameter_belong_graph_id << "]";
      parameter_backend = graph->GetBackendAnfByFrontAnf(parameter);
      // add parameter in backend to final graph inputs
      auto final_graph_inputs = final_graph->MutableInputs();
      MS_EXCEPTION_IF_NULL(final_graph_inputs);
      final_graph_inputs->push_back(parameter_backend);
    }
    MS_EXCEPTION_IF_NULL(parameter_backend);
    MS_LOG(INFO) << "parameter backend " << parameter_backend->DebugString() << " belong_graph_id "
                 << AnfAlgo::GetGraphId(parameter_backend.get());
  }
  MS_LOG(INFO) << "End final_graph_id " << final_graph_id_;
  return final_graph_id_;
}

AnfNodePtr AscendSession::CreateFakeOutput(GraphId fake_graph_id, const AnfNodePtr &true_output) {
  auto fake_graph = GetGraph(fake_graph_id);
  auto output_item_with_index = AnfAlgo::VisitKernelWithReturnType(true_output, 0);
  auto create_parameter = [&](const AbstractBasePtr &abstract) -> AnfNodePtr {
    auto parameter = fake_graph->NewParameter();
    MS_EXCEPTION_IF_NULL(parameter);
    parameter->set_abstract(abstract);
    auto new_parameter = fake_graph->NewParameter(parameter);
    // Add new parameter to the graph input of fake_graph to sure that all parameters will be allocated memory.
    auto graph_inputs = fake_graph->MutableInputs();
    MS_EXCEPTION_IF_NULL(graph_inputs);
    graph_inputs->push_back(new_parameter);
    return new_parameter;
  };
  auto create_parameter_from_cnode = [&](const AnfNodePtr &cnode, size_t output_idx) -> AnfNodePtr {
    MS_EXCEPTION_IF_NULL(cnode);
    auto abstract = cnode->abstract();
    MS_EXCEPTION_IF_NULL(abstract);
    // create multiple parameters if is a tuple output real kernel
    if (abstract->isa<abstract::AbstractTuple>()) {
      auto tuple_abstract = abstract->cast<abstract::AbstractTuplePtr>();
      MS_EXCEPTION_IF_NULL(tuple_abstract);
      MS_LOG(INFO) << "tuple_size [" << tuple_abstract->size() << "]";
      return create_parameter((*tuple_abstract)[output_idx]);
    }
    return create_parameter(cnode->abstract());
  };
  if (AnfAlgo::CheckPrimitiveType(output_item_with_index.first, prim::kPrimMakeTuple)) {
    std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
    auto make_tuple = output_item_with_index.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(make_tuple);
    for (size_t i = 1; i < make_tuple->inputs().size(); i++) {
      auto input = make_tuple->inputs()[i];
      make_tuple_inputs.push_back(CreateFakeOutput(fake_graph_id, input));
    }
    return fake_graph->NewCNode(make_tuple_inputs);
  }
  return create_parameter_from_cnode(output_item_with_index.first, output_item_with_index.second);
}

void AscendSession::SetFinalGraphOutput(const AnfNodePtr &node) {
  // get the backend anf node related to the output node of front
  auto output_from_graph_id = GetGraphIdByNode(node);
  auto output_from_graph = GetGraph(output_from_graph_id);
  MS_EXCEPTION_IF_NULL(node);
  MS_LOG(INFO) << "Set the output[" << node->DebugString() << "] of graph[" << output_from_graph_id
               << "] to final graph";
  MS_EXCEPTION_IF_NULL(output_from_graph);
  auto final_graph = GetGraph(final_graph_id_);
  MS_EXCEPTION_IF_NULL(final_graph);
  // if output is from final graph,it remarks no child graph exist
  if (final_graph_id_ == output_from_graph_id) {
    MS_LOG(INFO) << "No child graph,output is " << node->DebugString();
    final_graph->set_output(ConstructOutput({node}, final_graph));
    final_graph->set_executable(false);
    return;
  }
  final_graph->set_output(output_from_graph->output());
}

void AscendSession::SetFinalGraphOutput(const ValuePtr &value) {
  auto value_node = NewValueNode(value);
  auto kernel_info = std::make_shared<device::KernelInfo>();
  value_node->set_kernel_info(kernel_info);
  value_node->set_abstract(abstract::FromValue(value));
  auto final_graph = GetGraph(final_graph_id_);
  MS_EXCEPTION_IF_NULL(final_graph);
  final_graph->set_output(final_graph->NewCNode({NewValueNode(prim::kPrimMakeTuple), value_node}));
  final_graph->set_executable(false);
  MS_LOG(INFO) << "Not anf output[" << value->ToString() << "]";
}

void AscendSession::SetFinalGraphOutput(const VectorRef &vec_output) {
  for (auto &output : vec_output) {
    if (utils::isa<AnfNodePtr>(output)) {
      auto output_anf_node = utils::cast<AnfNodePtr>(output);
      SetFinalGraphOutput(output_anf_node);
    } else if (utils::isa<ValuePtr>(output)) {
      auto value = utils::cast<ValuePtr>(output);
      SetFinalGraphOutput(value);
    } else {
      MS_LOG(EXCEPTION) << "Unknown output type:" << output.ToString();
    }
  }
}

void AscendSession::SetFinalGraphOutput(const BaseRef &output) {
  if (utils::isa<AnfNodePtr>(output)) {
    auto output_anf_node = utils::cast<AnfNodePtr>(output);
    SetFinalGraphOutput(output_anf_node);
  } else if (utils::isa<ValuePtr>(output)) {
    auto value = utils::cast<ValuePtr>(output);
    SetFinalGraphOutput(value);
  } else if (utils::isa<VectorRef>(output)) {
    auto vec_output = utils::cast<VectorRef>(output);
    SetFinalGraphOutput(vec_output);
  } else {
    MS_LOG(EXCEPTION) << "Unknown output type:" << output.ToString();
  }
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
  auto cond_output_it = condition_output_.find(condition_graph_id);
  if (cond_output_it == condition_output_.end()) {
    MS_LOG(EXCEPTION) << "Can't find condition graph" << condition_graph_id;
  }
  auto cond_output_kernel =
    AnfAlgo::VisitKernel(condition_graph->GetBackendAnfByFrontAnf(cond_output_it->second), 0).first;
  MS_EXCEPTION_IF_NULL(cond_output_kernel);
  std::vector<AnfNodePtr> inputs = {NewValueNode(switch_primitive), cond_output_kernel, counter_const};
  CNodePtr switch_node = condition_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(switch_node);
  switch_node->set_abstract(std::make_shared<abstract::AbstractNone>());
  AnfAlgo::SetGraphId(condition_graph_id, switch_node.get());
  // set attr: cond_ RT_GREATER
  AnfAlgo::SetNodeAttr(kAttrSwitchCondition, MakeValue<int>(static_cast<int>(RT_GREATER)), switch_node);
  // set attr:data_type
  AnfAlgo::SetNodeAttr(kAttrDataType, MakeValue<int>(static_cast<int>(RT_SWITCH_INT64)), switch_node);
  // set attr:true branch graph id ,which is same to stream distinction label
  AnfAlgo::SetNodeAttr(kAttrTrueBranchStream, MakeValue<uint32_t>(true_graph_id), switch_node);
  // append switch at the end of condition graph
  auto return_node = condition_graph->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  InsertControlDependToGraph(condition_graph_id, return_node->input(1), switch_node);
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
      auto false_last = GetGraph(false_graph_id);
      MS_EXCEPTION_IF_NULL(true_last);
      MS_EXCEPTION_IF_NULL(false_last);
      MS_LOG(INFO) << "The last graph of false branch is " << false_graph_id;
      // create fake output
      auto fake_output_graph = NewKernelGraph();
      graph_execute_order.push_back(fake_output_graph->graph_id());
      graph_order_type.push_back(COMMON_GRAPH);
      fake_output_graph->set_output(CreateFakeOutput(fake_output_graph->graph_id(), final_graph->output()));
      final_graph->set_output(fake_output_graph->output());
      InsertMultipleAssignToGraph(true_last_id, true_last->output(), final_graph->output());
      InsertMultipleAssignToGraph(false_graph_id, false_last->output(), final_graph->output());
      // insert stream active for loop sink
      auto context_ptr = MsContext::GetInstance();
      MS_EXCEPTION_IF_NULL(context_ptr);
      if (context_ptr->enable_task_sink() && context_ptr->loop_sink_flag() &&
          ConfigManager::GetInstance().iter_num() > 1) {
        // insert active in true graph, another active will be inserted in kernel adjust
        InsertStreamActiveToGraph(true_last_id, kSecondStreamSwitchLabel);
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
    SetStreamDistinctionLabel(GetGraph(false_graph_id), condition_graph->stream_distinction_label(), true);
    // if false graph is a condition graph and has been switch compiled before,it's false should be updated again
    auto cond_it = switches_.find(false_graph_id);
    while (cond_it != switches_.end() && cond_it->second.second != kInvalidGraphId) {
      cond_graph_id = cond_it->first;
      false_graph_id = cond_it->second.second;
      condition_graph = GetGraph(cond_graph_id);
      SetStreamDistinctionLabel(GetGraph(false_graph_id), condition_graph->stream_distinction_label(), true);
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
      InsertStreamActiveToGraph(prev_graph_id, condition_graph->stream_distinction_label());
    }
    // if this is a 'if' condition
    auto it = while_condition_graphs_.find(cond_graph_id);
    if (it == while_condition_graphs_.end()) {
      CopyOutputOfIf(false_graph_id);
    } else {
      // if it is a while,insert a stream active to true graph
      GraphId from_graph = it->second;
      InsertStreamActiveToGraph(from_graph, condition_graph->stream_distinction_label());
    }
  }
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::InsertAllAssigns() {
  std::vector<std::pair<AnfNodePtr, AnfNodePtr>> assigns;
  for (auto assign : assigns_) {
    auto front_anf = std::get<0>(assign);
    auto to_graph_id = std::get<1>(assign);
    auto input_idx = std::get<2>(assign);
    auto to_graph = GetGraph(to_graph_id);
    MS_EXCEPTION_IF_NULL(to_graph);
    std::vector<AnfNodePtr> graph_inputs = to_graph->inputs();
    if (input_idx >= graph_inputs.size()) {
      MS_LOG(EXCEPTION) << "input_index " << input_idx << " out of range size " << graph_inputs.size();
    }
    auto backend_parameter = graph_inputs[input_idx];
    assigns.emplace_back(std::pair<AnfNodePtr, AnfNodePtr>(front_anf, backend_parameter));
  }
  // erase the repeat assign
  std::set<std::pair<AnfNodePtr, AnfNodePtr>> inserted_nodes;
  for (auto &assign : assigns) {
    auto front_anf = assign.first;
    auto backend_parameter = assign.second;
    auto from_graph_id = GetGraphIdByNode(front_anf);
    auto from_graph = GetGraph(from_graph_id);
    MS_EXCEPTION_IF_NULL(from_graph);
    auto backend_arg = from_graph->GetBackendAnfByFrontAnf(front_anf);
    if (inserted_nodes.find(assign) == inserted_nodes.end()) {
      InsertAssignToGraph(from_graph_id, backend_arg, backend_parameter);
      (void)inserted_nodes.insert(assign);
    }
  }
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

void AscendSession::SetChildGraphParameter(const AnfNodePtr &front_anf, GraphId to_graph_id, size_t input_idx) {
  MS_LOG(INFO) << "Start!";
  MS_EXCEPTION_IF_NULL(front_anf);
  auto from_graph_id = GetGraphIdByNode(front_anf);
  auto from_graph = GetGraph(from_graph_id);
  MS_EXCEPTION_IF_NULL(from_graph);
  auto to_graph = GetGraph(to_graph_id);
  MS_EXCEPTION_IF_NULL(to_graph);
  std::vector<AnfNodePtr> graph_inputs = to_graph->inputs();
  if (input_idx >= graph_inputs.size()) {
    MS_LOG(EXCEPTION) << "input_index " << input_idx << " out of range size " << graph_inputs.size();
  }
  auto backend_parameter = graph_inputs[input_idx];
  MS_EXCEPTION_IF_NULL(backend_parameter);
  auto backend_arg = from_graph->GetBackendAnfByFrontAnf(front_anf);
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
    MS_LOG(INFO) << "Reuse node [" << backend_arg->DebugString() << "], old node[" << backend_parameter->DebugString()
                 << "] will be replaced.";
    to_graph->ReplaceNode(backend_parameter, backend_arg);
    return;
  }
  MS_LOG(INFO) << "Assign of node" << backend_arg->DebugString() << " of graph " << from_graph_id << " to node"
               << backend_parameter->DebugString() << "of graph " << to_graph_id;
  assigns_.emplace_back(std::tuple<AnfNodePtr, GraphId, size_t>(front_anf, to_graph_id, input_idx));
}

void AscendSession::SetChildGraphParameter(const tensor::TensorPtr &front_tensor, GraphId to_graph_id,
                                           size_t input_idx) {
  MS_LOG(INFO) << "Start!";
  std::pair<GraphId, size_t> graph_input_pair(to_graph_id, input_idx);
  initial_tenosrs_[graph_input_pair] = front_tensor;
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

size_t AscendSession::SetChildGraphInput(const KernelGraphPtr &graph, const AnfNodePtr &node, size_t input_index) {
  auto output_num = AnfAlgo::GetOutputTensorNum(node);
  if (output_num > 1 && !AnfAlgo::CheckPrimitiveType(node, prim::kPrimTupleGetItem)) {
    return input_index + output_num;
  }
  auto valid_inputs = graph->valid_inputs();
  if (valid_inputs[input_index]) {
    SetChildGraphParameter(node, graph->graph_id(), input_index);
  } else {
    MS_LOG(DEBUG) << "Invalid input arg: " << node->DebugString();
  }
  return ++input_index;
}

size_t AscendSession::SetChildGraphInput(const KernelGraphPtr &graph, const ValuePtr &value, size_t input_index) {
  MS_EXCEPTION_IF_NULL(value);
  if (!value->isa<Tensor>()) {
    MS_LOG(EXCEPTION) << "Value Node should be a tensor, unexpected value: " << value->ToString();
  }
  SetChildGraphParameter(value->cast<TensorPtr>(), graph->graph_id(), input_index);
  return ++input_index;
}

size_t AscendSession::SetChildGraphInput(const KernelGraphPtr &graph, const VectorRef &vec_args, size_t input_index) {
  auto index = input_index;
  for (auto &arg : vec_args) {
    if (utils::isa<AnfNodePtr>(arg)) {
      // arg is a anf node
      auto node = utils::cast<AnfNodePtr>(arg);
      index = SetChildGraphInput(graph, node, input_index);
    } else if (utils::isa<ValuePtr>(arg)) {
      // arg is a tensor
      auto value = utils::cast<ValuePtr>(arg);
      index = SetChildGraphInput(graph, value, input_index);
    } else {
      MS_LOG(EXCEPTION) << "Unexpected arg type " << arg.ToString();
    }
  }
  return index;
}

void AscendSession::SetChildGraphInput(GraphId g, const VectorRef &args) {
  MS_LOG(INFO) << "Set input of graph " << g;
  auto to_graph = GetGraph(g);
  MS_EXCEPTION_IF_NULL(to_graph);
  DumpGraphInputArgs(args);
  UpdateGraphOrder(g);
  auto &graph_inputs = to_graph->inputs();
  auto real_args = GetRealArgs(to_graph, args);
  size_t input_index = 0;
  for (size_t i = 0; i < real_args.size(); i++) {
    if (input_index >= graph_inputs.size()) {
      MS_LOG(EXCEPTION) << "input_index " << input_index << " out of range size " << graph_inputs.size();
    }
    auto &real_arg = real_args[i];
    if (utils::isa<AnfNodePtr>(real_arg)) {
      // arg is a anf node
      auto node = utils::cast<AnfNodePtr>(real_arg);
      input_index = SetChildGraphInput(to_graph, node, input_index);
    } else if (utils::isa<ValuePtr>(real_arg)) {
      // arg is a tensor
      auto value = utils::cast<ValuePtr>(real_arg);
      input_index = SetChildGraphInput(to_graph, value, input_index);
    } else if (utils::isa<VectorRef>(real_arg)) {
      // arg is a VectorRef
      auto vec_args = utils::cast<VectorRef>(real_arg);
      input_index = SetChildGraphInput(to_graph, vec_args, input_index);
    } else {
      MS_LOG(EXCEPTION) << "Unexpected arg type " << real_arg.ToString();
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
  // merge graph order
  auto &graph_order = GetGraphOrder(final_graph_id_);
  auto &graph_type = GetGraphOrderType(final_graph_id_);
  auto final_graph = GetGraph(final_graph_id_);
  MS_EXCEPTION_IF_NULL(final_graph);
  if (graph_order.empty()) {
    MS_LOG(WARNING) << "Graph output is a lonely variable not linked to any op!";
    return;
  }
  if (graph_order.size() > 1) {
    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);
    if (!context_ptr->enable_task_sink()) {
      MS_LOG(EXCEPTION) << "Control sink network should run with task-sink mode!";
    }
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
    (void)std::transform(exec_order.begin(), exec_order.end(), std::back_inserter(final_exec_order),
                         [&](CNodePtr node) -> CNodePtr {
                           AnfAlgo::SetStreamDistinctionLabel(child_graph->stream_distinction_label(), node.get());
                           return node;
                         });
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
  assign_node->set_abstract(to->abstract());
  // append the assign at the end of from graph
  InsertDependToGraph(graph_id, assign_node);
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
  auto from_graph = GetGraph(graph_id);
  MS_EXCEPTION_IF_NULL(from_graph);
  std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>("StreamActive"))};
  auto active_node = from_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(active_node);
  active_node->set_abstract(std::make_shared<abstract::AbstractNone>());
  // set the active stream id into the attr of active node
  std::vector<uint32_t> active_index_value = {};
  active_index_value.push_back(actived_stream);
  AnfAlgo::SetNodeAttr(kAttrActiveStreamList, MakeValue<std::vector<uint32_t>>(active_index_value), active_node);
  // append the active node at the end of from graph
  auto return_node = from_graph->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  InsertControlDependToGraph(graph_id, return_node->input(1), active_node);
}

void AscendSession::InsertDependToGraph(GraphId graph_id, const AnfNodePtr &attch_node) {
  MS_LOG(INFO) << "Insert depend at the end of graph, the attach node is " << attch_node->DebugString();
  auto graph = GetGraph(graph_id);
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>("depend"))};
  auto return_node = graph->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  inputs.push_back(return_node->input(1));
  inputs.push_back(attch_node);
  auto depend_node = graph->NewCNode(inputs);
  return_node->set_input(1, depend_node);
}

void AscendSession::InsertControlDependToGraph(GraphId graph_id, const AnfNodePtr &first_node,
                                               const AnfNodePtr &second_node) {
  MS_LOG(INFO) << "Insert control depend at the end of graph, the first node is " << first_node->DebugString()
               << ", the second node is " << second_node->DebugString();
  auto graph = GetGraph(graph_id);
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>("ControlDepend"))};
  inputs.push_back(first_node);
  inputs.push_back(second_node);
  auto control_depend = graph->NewCNode(inputs);
  InsertDependToGraph(graph_id, control_depend);
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

void AscendSession::SyncInitialTenosrToDevice() {
  for (auto &item : initial_tenosrs_) {
    auto to_graph_id = item.first.first;
    auto input_idx = item.first.second;
    auto front_tensor = item.second;
    auto to_graph = GetGraph(to_graph_id);
    MS_EXCEPTION_IF_NULL(to_graph);
    std::vector<AnfNodePtr> graph_inputs = to_graph->inputs();
    if (input_idx >= graph_inputs.size()) {
      MS_LOG(EXCEPTION) << "input_index " << input_idx << " out of range size " << graph_inputs.size();
    }
    auto backend_parameter = graph_inputs[input_idx];
    // sync data from host to device
    MS_EXCEPTION_IF_NULL(front_tensor);
    size_t tensor_size = front_tensor->data().nbytes();
    auto addr = AnfAlgo::GetOutputAddr(backend_parameter, 0);
    MS_EXCEPTION_IF_NULL(addr);
    if (!addr->SyncHostToDevice(trans::GetRuntimePaddingShape(backend_parameter, 0), tensor_size,
                                front_tensor->data_type(), front_tensor->data_c(false))) {
      MS_LOG(EXCEPTION) << "Tensor SyncHostToDevice fail!";
    }
  }
}
}  // namespace session
}  // namespace mindspore
