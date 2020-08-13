/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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
#include "backend/session/ascend_session.h"
#include <algorithm>
#include <map>
#include <tuple>
#include <set>
#include <string>
#include <list>
#include "frontend/operator/ops.h"
#include "ir/tensor.h"
#include "ir/anf.h"
#include "common/trans.h"
#include "runtime/device/kernel_runtime.h"
#include "runtime/device/ascend/kernel_select_ascend.h"
#include "runtime/device/ascend/kernel_build_ascend.h"
#include "runtime/device/ascend/ascend_kernel_runtime.h"
#include "runtime/device/ascend/ascend_device_address.h"
#include "backend/optimizer/ascend/ascend_backend_optimization.h"
#include "backend/optimizer/common/common_backend_optimization.h"
#include "runtime/device/kernel_adjust.h"
#include "runtime/device/ascend/ascend_stream_assign.h"
#include "runtime/device/ascend/ascend_label_assign.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "ir/scalar.h"
#include "debug/anf_ir_dump.h"
#include "debug/anf_ir_utils.h"
#include "debug/draw.h"
#include "utils/ms_utils.h"
#include "backend/optimizer/common/helper.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "utils/config_manager.h"
#include "utils/base_ref_extends.h"
#include "debug/tensor_load.h"

namespace mindspore {
namespace session {
const size_t kInvalidIndex = SIZE_MAX;
constexpr size_t kReturnDataIndex = 1;
namespace {
void DumpGraphExeOrder(const std::vector<CNodePtr> &execution_order, const std::string &tag = "") {
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

  std::stringstream buf;
  buf << "================== execution order ==================\n";
  if (!tag.empty()) {
    buf << tag << "\n";
  }
  buf << "execution_order size: " << execution_order.size() << "\n";
  i = 0;
  for (auto &cnode : execution_order) {
    MS_EXCEPTION_IF_NULL(cnode);
    buf << i << ":\n";
    buf << "\t" << cnode->DebugString() << "\n";
    buf << "\t" << AnfAlgo::GetStreamDistinctionLabel(cnode.get()) << "\n";
    buf << "\t" << AnfAlgo::GetGraphId(cnode.get()) << "\n";
    i++;
  }
  buf << "================== execution order ==================\n";
}

void SetStreamDistinctionLabel(const KernelGraphPtr &graph, uint32_t label, bool is_override) {
  MS_EXCEPTION_IF_NULL(graph);
  if (is_override || graph->stream_distinction_label() == kInvalidDistincLabel) {
    graph->set_stream_distinction_label(label);
  }
}

std::vector<CNodePtr> GetCNodes(const std::vector<AnfNodePtr> &anf_nodes) {
  std::vector<CNodePtr> cnodes = {};
  size_t i = 0;
  for (const auto &anf : anf_nodes) {
    MS_LOG(INFO) << "Apply_list[" << i++ << "] = " << anf->DebugString();
    MS_EXCEPTION_IF_NULL(anf);
    if (anf->isa<CNode>()) {
      cnodes.push_back(anf->cast<CNodePtr>());
    }
  }
  return cnodes;
}

void InsertMakeTupleForOutput(NotNull<KernelGraphPtr> root_graph) {
  auto return_node = root_graph->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  if (return_node->size() <= kReturnDataIndex) {
    return;
  }
  auto make_tuple = root_graph->NewCNode(
    {NewValueNode(std::make_shared<Primitive>(prim::kPrimMakeTuple->name())), root_graph->output()});
  root_graph->set_output(make_tuple);
}
}  // namespace

GraphId AscendSession::CompileGraph(const AnfNodePtrList &lst, const AnfNodePtrList &outputs) {
  MS_LOG(INFO) << "Start";
  // construct graph, if successfully, graph_sum_ + 1
  auto graph = ConstructKernelGraph(lst, outputs);
  auto graph_id = graph->graph_id();
  MS_LOG(INFO) << "Compile graph " << graph_id << " success";
  return graph_id;
}

GraphId AscendSession::CompileGraph(NotNull<FuncGraphPtr> func_graph) {
  MS_LOG(INFO) << "Start";
  std::vector<KernelGraphPtr> all_graphs;
  auto root_graph = ConstructKernelGraph(func_graph, &all_graphs);
  BackendOptimization(all_graphs);
  // empty graph dont entry to backend
  if (root_graph->execution_order().empty()) {
    MS_LOG(INFO) << root_graph->ToString() << " is empty graph.";
    InsertMakeTupleForOutput(NOT_NULL(root_graph));
    root_graph->set_executable(false);
    InitRuntimeResource();
    return root_graph->graph_id();
  }
  // create parameter for multiple branch
  std::set<KernelGraphPtr> memo;
  CreateMultiBranchOutput(NOT_NULL(root_graph), NOT_NULL(&memo));
  memo.clear();
  // insert goto labels and label_sets
  LinkChildGraphs(NOT_NULL(root_graph));
  // resource initialize
  InitRuntimeResource();

  IrFusionPass(NOT_NULL(root_graph), NOT_NULL(&memo));
  memo.clear();

  SelectKernel(NOT_NULL(root_graph));
  memo.clear();

  HardwareOptimize(NOT_NULL(root_graph), NOT_NULL(&memo));
  memo.clear();

  AssignStaticMemory(NOT_NULL(root_graph), NOT_NULL(&memo));
  memo.clear();

  UpdateRefOutputMap(NOT_NULL(root_graph), NOT_NULL(&memo));
  memo.clear();
  // add make_tuple to the output graph
  InsertMakeTupleForOutput(NOT_NULL(root_graph));
  // root root_graph valiate,include genearte execute order and so on
  RootGraphExecutorValidate(NOT_NULL(root_graph));
  // adjust kernel
  AdjustKernel(root_graph);
#if (ENABLE_CPU && (ENABLE_D || ENABLE_GPU))
  // Assign parameter keys.
  AssignParamKey(root_graph);
#endif
  // assign stream
  AssignStream(NOT_NULL(root_graph));
  // insert profiling point
  device::KernelAdjust::GetInstance().Profiling(NOT_NULL(root_graph.get()));
  // build kernel
  BuildKernel(root_graph);
#ifdef ENABLE_DEBUGGER
  if (debugger_) {
    debugger_->PreExecute(root_graph);
  }
#endif
  // alloc mem
  MemoryAlloc(root_graph.get());
  // task generate
  GenerateTaskInfo(root_graph);
  // load task into device
  LoadTask(root_graph);
  DumpAllGraphs(all_graphs);
  // return the root_graph id to backend
  auto graph_id = root_graph->graph_id();
  return graph_id;
}

void AscendSession::SetFinalGraphSummaryFlag(const std::shared_ptr<KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto graph_order = GetGraphOrder(kernel_graph->graph_id());
  for (auto graph_id : graph_order) {
    auto child_graph = GetGraph(graph_id);
    if (child_graph == nullptr) {
      continue;
    }
    if (child_graph->summary_node_exist()) {
      kernel_graph->set_summary_node_exist(true);
      return;
    }
  }
  kernel_graph->set_summary_node_exist(false);
}

void AscendSession::BuildGraph(GraphId graph_id) {
  MS_LOG(INFO) << "Start";
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
    SetFinalGraphSummaryFlag(graph);
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
    SetSummaryNodes(graph.get());
    // merge child graph
    MergeGraphExecOrder();
  } else {
    auto single_graph = GetGraph(graph_id);
    MS_EXCEPTION_IF_NULL(single_graph);
    CompileChildGraph(single_graph);
    // set the distinction label of single graph
    single_graph->set_stream_distinction_label(graph_id);
    single_graph->UpdateExecuteKernelStreamLabel();
  }
  // adjust execution order because  merge child graph and other special operations
  AdjustKernel(graph);
  // Assign streams for control sink and hccl and so on
  AssignStream(NOT_NULL(graph));

  device::KernelAdjust::GetInstance().Profiling(NOT_NULL(graph.get()));
  // build kernel if node is cnode
  BuildKernel(graph);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
#ifdef ENABLE_DEBUGGER
  if (debugger_) {
    debugger_->PreExecute(graph);
  }
#endif
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
  DumpAllGraphs({graph});
  MS_LOG(INFO) << "End";
}

void AscendSession::CompileChildGraph(const KernelGraphPtr &child_graph) {
  MS_EXCEPTION_IF_NULL(child_graph);
  MS_LOG(INFO) << "CompileChildGraph " << child_graph->ToString();
  opt::AscendBackendIRFusionOptimization(child_graph);
  opt::AscendBackendFuseBasicOpt(child_graph, true);
  opt::AscendBackendGraphKernelOpt(child_graph, true);
  child_graph->SetExecOrderByDefault();
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->save_graphs_flag();
  auto save_graphs_path = context_ptr->save_graphs_path();
  if (save_graphs_path.empty()) {
    save_graphs_path = ".";
  }
  if (save_graphs) {
    std::string file_path =
      save_graphs_path + "/" + "select_kernel_before" + "_graph_" + std::to_string(child_graph->graph_id()) + ".ir";
    DumpIR(file_path, child_graph);
  }
  // select kernel build info
  SelectKernel(*child_graph);
  if (save_graphs) {
    std::string file_path =
      save_graphs_path + "/" + "select_kernel_after" + "_graph_" + std::to_string(child_graph->graph_id()) + ".ir";
    DumpIR(file_path, child_graph);
  }
  // optimize graph
  HardwareOptimize(child_graph);
  // assign static memory of parameters
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->AssignStaticMemoryInput(child_graph.get());
  runtime_instance->AssignStaticMemoryValueNode(child_graph.get());
}

void AscendSession::RunGraph(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs,
                             VectorRef *const outputs) {
  MS_LOG(INFO) << "Start";
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
#if (ENABLE_CPU && (ENABLE_D || ENABLE_GPU))
  // Initialize parameter server
  InitPSParamAndOptim(kernel_graph, inputs);
#endif
  {
    py::gil_scoped_release release;
    // run task on device
    ExecTask(kernel_graph);
  }
  // get result from device
  UpdateOutputs(kernel_graph, outputs, inputs);
  // summary
  Summary(kernel_graph.get());
#ifdef ENABLE_DEBUGGER
  // load tensor from device for debugger
  if (debugger_ && debugger_->debugger_enabled()) {
    LoadTensor(kernel_graph);
  }
#endif
  // dump used for debug
  Dump(kernel_graph);
#ifdef ENABLE_DEBUGGER
  // debugger post-execution processing
  if (debugger_) {
    debugger_->PostExecute();
  }
#endif
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
    MS_LOG(EXCEPTION) << "Run task error!";
  }
  MS_LOG(INFO) << "Finish!";
}

bool AscendSession::GraphCacheExist(const GraphInfo &graph_info) const {
  return run_op_graphs_.find(graph_info) != run_op_graphs_.end();
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
  RunOpMemoryAlloc(op_run_info.value, input_tensors, graph.get());
  // load input data to device
  LoadInputData(graph, input_tensors);
  // run op
  RunOpExecTask(graph);
  // get output
  VectorRef outputs;
  if (op_run_info.value != nullptr) {
    std::vector<tensor::TensorPtr> pre_output_tensors;
    TensorValueToTensor(op_run_info.value, &pre_output_tensors);
    for (auto &pre_output : pre_output_tensors) {
      tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(pre_output->data_type(), pre_output->shape());
      tensor->set_device_address(pre_output->device_address());
      tensor->set_dirty(false);
      outputs.emplace_back(tensor);
    }
  } else {
    UpdateOutputs(graph, &outputs, input_tensors);
  }
  // trans output to tuple
  auto output_tensors = TransformBaseRefListToTuple(outputs);
  if (!utils::isa<PyObjectRef>(output_tensors) ||
      !py::isinstance<py::tuple>(utils::cast<PyObjectRef>(output_tensors).object_)) {
    MS_LOG(EXCEPTION) << "The output tensors should be a tuple !";
  }
  py::object tuple_obj = utils::cast<PyObjectRef>(output_tensors).object_;
  py::tuple tuple_tensors = py::cast<py::tuple>(tuple_obj);
  RunOpMemoryClear(graph.get());
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
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->execution_mode() == kGraphMode) {
    if (raise_precision_count > 0) {
      MS_LOG(WARNING) << "There has " << raise_precision_count
                      << " node/nodes used raise precision to selected the kernel!";
    }
    if (reduce_precision_count > 0) {
      MS_LOG(WARNING) << "There has " << reduce_precision_count
                      << " node/nodes used reduce precision to selected the kernel!";
    }
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
  device::ascend::KernelPreBuild(kernel_graph.get());
  MS_LOG(INFO) << "HardwareOptimize start!";
  opt::AscendBackendOptimization(kernel_graph);
  opt::AscendGraphKernelCommonProcess(kernel_graph);
  opt::AscendBackendFuseBasicOpt(kernel_graph, false);
  opt::AscendBackendAddAtomicClean(kernel_graph);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
  MS_LOG(INFO) << "HardwareOptimize Finish!";
}

void AscendSession::AdjustKernel(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  MS_LOG(INFO) << "Start!";
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

void AscendSession::AssignStream(NotNull<KernelGraphPtr> kernel_graph) const {
  MS_LOG(INFO) << "Start!";
  device::ascend::AscendStreamAssign::GetInstance().AssignStream(kernel_graph);
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

void AscendSession::RunOpMemoryAlloc(const ValuePtr &pre_output_value,
                                     const std::vector<tensor::TensorPtr> &input_tensors,
                                     KernelGraph *kernel_graph) const {
  MS_LOG(INFO) << "Start memory alloc!";
  MS_EXCEPTION_IF_NULL(kernel_graph);
  opt::RemoveNopNode(kernel_graph);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->RunOpAssignMemory(pre_output_value, input_tensors, kernel_graph);
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::RunOpMemoryClear(const KernelGraph *kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->RunOpClearMemory(kernel_graph);
}

void AscendSession::GenerateTaskInfo(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  MS_LOG(INFO) << "Start!";
  (void)device::KernelAdjust::GetInstance().StepLoadCtrlInputs(kernel_graph);
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

void AscendSession::DumpAllGraphs(const std::vector<KernelGraphPtr> &all_graphs) {
#ifdef ENABLE_DUMP_IR
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->save_graphs_flag();
  if (!save_graphs) {
    return;
  }
  auto save_graphs_path = context_ptr->save_graphs_path();
  if (save_graphs_path.empty()) {
    save_graphs_path = ".";
  }
  for (auto &graph : all_graphs) {
    MS_EXCEPTION_IF_NULL(graph);
    std::string file_path = save_graphs_path + "/graph_build_" + std::to_string(graph->graph_id()) + ".ir";
    DumpIR(file_path, graph, true);
    DumpIRProto(graph, "vm_build_" + std::to_string(graph->graph_id()));
  }
#endif
}

void AscendSession::LoadTensor(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  MS_LOG(INFO) << "Start!";
  MS_EXCEPTION_IF_NULL(kernel_graph);
#ifdef ENABLE_DEBUGGER
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  DebugServices *debug_services = debugger_->debug_services();
  TensorLoader *tensor_loader = debug_services->tensor_loader();
  // TensorData will be freed up here
  tensor_loader->EmptyTensor();
  uint32_t iter_num = tensor_loader->GetIterNum();
  tensor_loader->set_iter_num(++iter_num);
  (void)runtime_instance->LoadData(kernel_graph.get(), debugger_.get());
  tensor_loader->EmptyPrevTensor();
#endif
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::RecurseSetSummaryNodes(KernelGraph *graph,
                                           std::map<std::string, std::pair<AnfNodePtr, int>> *summary) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(summary);
  // if final graph have no child graph
  auto graph_order_iter = graph_execute_orders_.find(graph->graph_id());
  if (graph_order_iter == graph_execute_orders_.end()) {
    SessionBasic::SetSummaryNodes(graph);
    auto summary_nodes = graph->summary_nodes();
    summary->insert(summary_nodes.begin(), summary_nodes.end());
    return;
  }
  // for every child graph, find summary nodes
  auto graph_order = GetGraphOrder(graph->graph_id());
  for (size_t i = 0; i < graph_order.size(); i++) {
    auto child_graph = GetGraph(graph_order[i]);
    if (child_graph == nullptr) {
      continue;
    }
    SessionBasic::SetSummaryNodes(child_graph.get());
    auto child_graph_summary = child_graph->summary_nodes();
    summary->insert(child_graph_summary.begin(), child_graph_summary.end());
    RecurseSetSummaryNodes(child_graph.get(), summary);
  }
  graph->set_summary_nodes(*summary);
}

void AscendSession::SetSummaryNodes(KernelGraph *graph) {
  MS_LOG(DEBUG) << "Update summary Start";
  MS_EXCEPTION_IF_NULL(graph);
  auto summary_nodes = graph->summary_nodes();
  std::map<std::string, std::pair<AnfNodePtr, int>> summary;
  summary.insert(summary_nodes.begin(), summary_nodes.end());
  RecurseSetSummaryNodes(graph, &summary);
  graph->set_summary_nodes(summary);
  MS_LOG(DEBUG) << "Update summary end size: " << summary.size();
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
      MS_LOG(EXCEPTION) << "Input_index " << input_idx << " out of range size " << graph_inputs.size();
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
  MS_LOG(DEBUG) << "Front_anf " << front_anf->DebugString() << " is not exist in any graph";
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
  AscendControlParser::InsertDependToGraph(NOT_NULL(graph), NOT_NULL(assign_node));
}

const std::vector<GraphId> &AscendSession::GetGraphOrder(GraphId final_graph_id) const {
  auto graph_order_iter = graph_execute_orders_.find(final_graph_id);
  if (graph_order_iter == graph_execute_orders_.end()) {
    MS_LOG(EXCEPTION) << "Final graph" << final_graph_id << "has no child graph";
  }
  return graph_order_iter->second;
}

const std::vector<GraphType> &AscendSession::GetGraphOrderType(GraphId final_graph_id) const {
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
      MS_LOG(EXCEPTION) << "Input_index " << input_idx << " out of range size " << graph_inputs.size();
    }
    auto backend_parameter = graph_inputs[input_idx];
    // sync data from host to device
    MS_EXCEPTION_IF_NULL(front_tensor);
    size_t tensor_size = front_tensor->data().nbytes();
    auto addr = AnfAlgo::GetOutputAddr(backend_parameter, 0);
    MS_EXCEPTION_IF_NULL(addr);
    if (!addr->SyncHostToDevice(trans::GetRuntimePaddingShape(backend_parameter, 0), tensor_size,
                                front_tensor->data_type(), front_tensor->data_c())) {
      MS_LOG(EXCEPTION) << "Tensor SyncHostToDevice fail!";
    }
  }
}

void AscendSession::BackendOptimization(const std::vector<KernelGraphPtr> &all_graphs) {
  MS_LOG(INFO) << "Start BackendCommonOptimization";
  for (auto &graph : all_graphs) {
    opt::BackendCommonOptimization(graph);
  }
  MS_LOG(INFO) << "End.";
}

void AscendSession::LinkChildGraphs(NotNull<KernelGraphPtr> graph) { AscendControlParser::LinkGraph(graph); }

void AscendSession::RootGraphExecutorValidate(NotNull<KernelGraphPtr> graph) {
  AscendControlParser::ExecutorValidate(graph);
}

void AscendSession::CreateMultiBranchOutput(NotNull<KernelGraphPtr> graph, NotNull<std::set<KernelGraphPtr> *> memo) {
  if (memo->find(graph.get()) != memo->end()) {
    return;
  }
  memo->insert(graph.get());
  graph->UpdateChildGraphOrder();
  for (auto &child_graph : graph->child_graph_order()) {
    CreateMultiBranchOutput(NOT_NULL(child_graph), memo);
  }
  std::map<AnfNodePtr, AnfNodePtr> need_replace_list;
  auto node_list = GetCNodes(TopoSort(graph->get_return()));
  for (auto &node : node_list) {
    if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimCall) || AnfAlgo::CheckPrimitiveType(node, prim::kPrimSwitch)) {
      // create a parameter to store the output of multiple branch and set the parameter as the condition graph's output
      auto output_param = graph->TransTupleToMakeTuple(graph->NewParameter(node->abstract()));
      MS_EXCEPTION_IF_NULL(graph->MutableInputs());
      graph->AddChildGraphResult(output_param);

      std::vector<AnfNodePtr> depend_inputs = {
        graph->NewValueNode(NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name()))), output_param, node};
      auto depend = graph->NewCNode(depend_inputs);
      need_replace_list.emplace(node, depend);
      MS_LOG(INFO) << "Create parameter " << output_param->DebugString() << " for call node " << node->DebugString()
                   << ", depend node is " << depend->DebugString();
      // insert assign in order to transfer child graph output to parameter
      auto child_graphs = AnfAlgo::GetCallSwitchKernelGraph(node);
      for (auto &child_graph : child_graphs) {
        MS_EXCEPTION_IF_NULL(child_graph);
        // If graph has no output, the graph is the true graph of while and will call condition graph, no need insert
        // assign from condition to true graph
        if (memo->find(child_graph) != memo->end()) {
          continue;
        }
        if (child_graph->get_output_null()) {
          continue;
        }
        AscendControlParser::InsertMultipleAssignToGraph(NOT_NULL(child_graph), nullptr,
                                                         NOT_NULL(child_graph->output()), NOT_NULL(output_param));
      }
    }
  }
  // searching for nodes' input to replace call by depend(parameter, call)
  for (auto &node : node_list) {
    for (size_t i = 0; i < node->size(); ++i) {
      auto input = node->input(i);
      auto iter = need_replace_list.find(input);
      if (iter != need_replace_list.end()) {
        node->set_input(i, iter->second);
      }
    }
  }
  memo->erase(graph.get());
}

void AscendSession::IrFusionPass(const NotNull<KernelGraphPtr> graph, NotNull<std::set<KernelGraphPtr> *> memo) {
  if (memo->find(graph) != memo->end()) {
    return;
  }
  memo->insert(graph.get());

  opt::AscendBackendIRFusionOptimization(graph);
  opt::AscendBackendFuseBasicOpt(graph, true);
  opt::AscendBackendGraphKernelOpt(graph, true);
  graph->SetExecOrderByDefault();

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->save_graphs_flag();
  auto save_graphs_path = context_ptr->save_graphs_path();
  if (save_graphs) {
    if (save_graphs_path.empty()) {
      save_graphs_path = ".";
    }
    std::string file_path =
      save_graphs_path + "/" + "select_kernel_before" + "_graph_" + std::to_string(graph->graph_id()) + ".ir";
    DumpIR(file_path, graph.get());
  }

  for (auto &child_graph : graph->child_graph_order()) {
    IrFusionPass(NOT_NULL(child_graph), memo);
  }
}

void AscendSession::SelectKernel(NotNull<KernelGraphPtr> root_graph) {
  MS_LOG(INFO) << "Start select kernel.";
  size_t raise_precision_count = 0;
  size_t reduce_precision_count = 0;

  std::set<KernelGraphPtr> memo;
  (void)RecurseSelectKernelInfo(root_graph, NOT_NULL(&memo), &raise_precision_count, &reduce_precision_count);
  memo.clear();

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->execution_mode() == kGraphMode) {
    if (raise_precision_count > 0) {
      MS_LOG(WARNING) << "There are " << raise_precision_count
                      << " node/nodes used raise precision to selected the kernel!";
    }
    if (reduce_precision_count > 0) {
      MS_LOG(WARNING) << "There are " << reduce_precision_count
                      << " node/nodes used reduce precision to selected the kernel!";
    }
  }
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::RecurseSelectKernelInfo(NotNull<KernelGraphPtr> graph,
                                            NotNull<std::set<KernelGraphPtr> *> const memo,
                                            size_t *const raise_precision_count,
                                            size_t *const reduce_precision_count) const {
  if (memo->find(graph) != memo->end()) {
    return;
  }
  memo->insert(graph.get());
  MS_LOG(INFO) << "Start to select kernel info in graph: " << graph->graph_id();

  for (const auto &cnode : graph->execution_order()) {
    if (AnfAlgo::IsCondControlKernel(cnode)) {
      std::vector<KernelGraphPtr> child_graphs;
      if (AnfAlgo::HasNodeAttr(kAttrChildGraph, cnode)) {
        child_graphs = AnfAlgo::GetNodeAttr<std::vector<KernelGraphPtr>>(cnode, kAttrChildGraph);
      }
      for (auto &child_graph : child_graphs) {
        RecurseSelectKernelInfo(NOT_NULL(child_graph), memo, raise_precision_count, reduce_precision_count);
      }
    }

    auto status = device::ascend::SelectKernelInfo(cnode);
    if (status == device::ascend::kStatusRaisePrecision) {
      (*raise_precision_count)++;
    } else if (status == device::ascend::kStatusReducePrecision) {
      (*reduce_precision_count)++;
    }
    MS_LOG(INFO) << "Select ApplyKernel: " << cnode->DebugString();
  }

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->save_graphs_flag();
  auto save_graphs_path = context_ptr->save_graphs_path();
  if (save_graphs) {
    if (save_graphs_path.empty()) {
      save_graphs_path = ".";
    }
    std::string file_path =
      save_graphs_path + "/" + "select_kernel_after" + "_graph_" + std::to_string(graph->graph_id()) + ".ir";
    DumpIR(file_path, graph.get());
  }
  MS_LOG(INFO) << "Finish selecting kernel info in graph: " << graph->graph_id();
}

void AscendSession::HardwareOptimize(NotNull<KernelGraphPtr> graph,
                                     NotNull<std::set<KernelGraphPtr> *> const memo) const {
  if (memo->find(graph) != memo->end()) {
    return;
  }
  memo->insert(graph.get());

  MS_LOG(INFO) << "Start to do HardwareOptimize in graph: " << graph->graph_id();

  HardwareOptimize(graph.get());
  for (auto &child_graph : graph->child_graph_order()) {
    HardwareOptimize(NOT_NULL(child_graph), memo);
  }
  MS_LOG(INFO) << "Finish doing HardwareOptimize in graph: " << graph->graph_id();
}

void AscendSession::AssignStaticMemory(NotNull<KernelGraphPtr> graph,
                                       NotNull<std::set<KernelGraphPtr> *> const memo) const {
  if (memo->find(graph) != memo->end()) {
    return;
  }
  memo->insert(graph.get());

  MS_LOG(INFO) << "Start to assign static memory for parameter in graph: " << graph->graph_id();
  // assign static memory for parameters
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->AssignStaticMemoryInput(graph.get().get());
  runtime_instance->AssignStaticMemoryValueNode(graph.get().get());
  for (auto &child_graph : graph->child_graph_order()) {
    AssignStaticMemory(NOT_NULL(child_graph), memo);
  }
  MS_LOG(INFO) << "Finish assigning static memory for parameter in graph: " << graph->graph_id();
}

void AscendSession::UpdateRefOutputMap(NotNull<KernelGraphPtr> graph,
                                       NotNull<std::set<KernelGraphPtr> *> const memo) const {
  if (memo->find(graph) != memo->end()) {
    return;
  }
  memo->insert(graph.get());

  for (auto &child_graph : graph->child_graph_order()) {
    UpdateRefOutputMap(NOT_NULL(child_graph), memo);
    // copy ref map to final graph
    auto child_ref_map = child_graph->GetRefMap();
    for (auto &item : child_ref_map) {
      if (graph->IsInRefOutputMap(item.first)) {
        MS_LOG(WARNING) << "The ref pair <" << item.first.first->DebugString() << ", " << item.first.second
                        << "> is already in " << graph->ToString();
        continue;
      }
      graph->AddRefCorrespondPairs(item.first, item.second);
    }
  }
}
}  // namespace session
}  // namespace mindspore
