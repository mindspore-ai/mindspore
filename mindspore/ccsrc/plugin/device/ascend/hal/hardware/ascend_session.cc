/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/hal/hardware/ascend_session.h"
#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <list>

#include "utils/hash_set.h"
#include "mindspore/core/ops/core_ops.h"
#include "base/base_ref_utils.h"
#include "ir/tensor.h"
#include "ir/anf.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "runtime/device/kernel_runtime.h"
#include "plugin/device/ascend/hal/device/kernel_select_ascend.h"
#include "plugin/device/ascend/hal/device/kernel_build_ascend.h"
#include "plugin/device/ascend/hal/device/ascend_kernel_runtime.h"
#include "plugin/device/ascend/hal/device/profiling/profiling_manager.h"
#include "plugin/device/ascend/hal/device/ascend_memory_adapter.h"
#include "plugin/device/ascend/optimizer/ascend_backend_optimization.h"
#include "backend/common/optimizer/common_backend_optimization.h"
#include "plugin/device/ascend/hal/device/kernel_adjust.h"
#include "plugin/device/ascend/hal/device/ascend_stream_assign.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "utils/ms_utils.h"
#include "include/common/utils/utils.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "backend/common/optimizer/helper.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "runtime/pynative/op_runtime_info.h"
#include "include/common/utils/config_manager.h"
#ifndef ENABLE_SECURITY
#include "debug/data_dump/dump_json_parser.h"
#include "debug/data_dump/e2e_dump.h"
#include "debug/debugger/debugger_utils.h"
#include "plugin/device/ascend/hal/device/dump/ascend_dump.h"
#endif
#include "backend/common/graph_kernel/adapter/graph_kernel_optimization.h"
#include "plugin/device/ascend/hal/hardware/ascend_auto_monad.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/debug/dump_proto.h"
#include "abstract/utils.h"
#ifdef ENABLE_DEBUGGER
#include "debug/tensor_load.h"
#include "debug/debugger/proto_exporter.h"
#else
#include "debug/debugger/proto_exporter_stub.h"
#endif
#include "common/util/error_manager/error_manager.h"
#include "toolchain/adx_datadump_callback.h"
#include "toolchain/adx_datadump_server.h"
#ifdef ENABLE_DUMP_IR
#include "include/common/debug/rdr/recorder_manager.h"
#include "debug/rdr/graph_recorder.h"
#endif
#ifdef WITH_BACKEND
#include "ps/util.h"
#endif
#include "plugin/device/ascend/hal/device/ascend_device_address.h"
#ifndef ENABLE_SECURITY
#include "plugin/device/ascend/hal/profiler/memory_profiling.h"

using Adx::AdxRegDumpProcessCallBack;
using mindspore::device::ascend::ProfilingManager;
using mindspore::profiler::ascend::MemoryProfiling;
#endif

namespace mindspore::session {
const size_t kLabelNumsThreshold = 1023;
namespace {
#ifndef ENABLE_SECURITY
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
#endif

// Enable device_to_device copy.
bool EnableDeviceCopy() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (common::GetEnv("ENABLE_DEVICE_COPY") != "1") {
    return false;
  }
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) != kGraphMode) {
    return false;
  }
  if (!ms_context->get_param<bool>(MS_CTX_ENABLE_TASK_SINK)) {
    return false;
  }
  if (ms_context->get_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK)) {
    return false;
  }
  return true;
}

// Handle control flow by auto-monad.
void HandleControlFlow(const NotNull<KernelGraphPtr> &graph) {
  MS_LOG(INFO) << "Status record: start handle control flow. graph id: " << graph->graph_id();
  AscendAutoMonad auto_monad(graph);
  auto_monad.Run();
  MS_LOG(INFO) << "Status record: end handle control flow. graph id: " << graph->graph_id();
}

void SetStreamDistinctionLabel(const KernelGraphPtr &graph, uint32_t label, bool is_override) {
  MS_EXCEPTION_IF_NULL(graph);
  if (is_override || graph->stream_distinction_label() == kInvalidDistincLabel) {
    graph->set_stream_distinction_label(label);
  }
}

TensorPtr GetCNodeOutputStubTensor(const KernelWithIndex &kernel_with_index,
                                   const std::map<KernelWithIndex, OutputTensorInfo> &node_output_info,
                                   bool *output_is_weight) {
  MS_EXCEPTION_IF_NULL(output_is_weight);
  const auto &iter = node_output_info.find(kernel_with_index);
  if (iter == node_output_info.end()) {
    MS_EXCEPTION_IF_NULL(kernel_with_index.first);
    MS_LOG(EXCEPTION) << "Can not find output stub tensor of cnode " << kernel_with_index.first->DebugString();
  }
  *output_is_weight = iter->second.is_weight;
  return iter->second.output_stub_tensor;
}

void GenOpOutputStubTensor(const KernelGraphPtr &single_op_graph, const CNodePtr &kernel,
                           const std::map<KernelWithIndex, size_t> &cnode_refcount,
                           std::map<KernelWithIndex, OutputTensorInfo> *op_output_info) {
  MS_EXCEPTION_IF_NULL(single_op_graph);
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(op_output_info);
  OutputTensorInfo output_tensor_info;
  size_t out_idx = 0;
  for (const auto &output : single_op_graph->outputs()) {
    KernelWithIndex kernel_with_index = std::make_pair(kernel, out_idx++);
    if (cnode_refcount.find(kernel_with_index) == cnode_refcount.end()) {
      continue;
    }
    const auto &output_kernel_with_index = common::AnfAlgo::VisitKernel(output, 0);
    const auto &output_node = output_kernel_with_index.first;
    MS_EXCEPTION_IF_NULL(output_node);
    const auto &output_index = output_kernel_with_index.second;
    auto out_abstract = output_node->abstract();
    MS_EXCEPTION_IF_NULL(out_abstract);
    if (out_abstract->isa<abstract::AbstractTuple>()) {
      out_abstract = out_abstract->cast<abstract::AbstractTuplePtr>()->elements()[output_index];
      MS_EXCEPTION_IF_NULL(out_abstract);
    }
    auto tensor_abstract = out_abstract->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor_abstract);
    const auto &infer_type = common::AnfAlgo::GetOutputInferDataType(output_node, output_index);
    tensor::TensorPtr stub_output_tensor =
      std::make_shared<tensor::Tensor>(infer_type, tensor_abstract->shape()->shape(), nullptr);
    MS_EXCEPTION_IF_NULL(stub_output_tensor);
    const auto &output_type = AnfAlgo::GetOutputDeviceDataType(output_node, output_index);
    const auto &output_format = AnfAlgo::GetOutputFormat(output_node, output_index);
    tensor::DeviceInfo device_info;
    device_info.format_ = output_format;
    device_info.data_type_ = TypeIdToType(output_type);
    stub_output_tensor->set_device_info(device_info);
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    device::DeviceAddressPtr device_address = std::make_shared<device::ascend::AscendDeviceAddress>(
      nullptr, 0, output_format, output_type, kAscendDevice, device_id);
    stub_output_tensor->set_device_address(device_address);
    output_tensor_info.output_stub_tensor = stub_output_tensor;
    auto kernel_info = dynamic_cast<const device::KernelInfo *>(output_node->kernel_info());
    MS_EXCEPTION_IF_NULL(kernel_info);
    output_tensor_info.is_weight = !(kernel_info->is_feature_map());
    (*op_output_info)[kernel_with_index] = output_tensor_info;
  }
}

bool NeedMemcpyInDevice(const device::DeviceAddressPtr &src_device_addr,
                        const device::DeviceAddressPtr &dst_device_addr) {
  MS_EXCEPTION_IF_NULL(dst_device_addr);
  if (src_device_addr.get() == nullptr) {
    return false;
  }
  return (src_device_addr->GetDeviceType() == dst_device_addr->GetDeviceType() &&
          src_device_addr->format() == dst_device_addr->format() &&
          src_device_addr->type_id() == dst_device_addr->type_id());
}

bool TensorNeedSync(const std::shared_ptr<KernelGraph> &kernel_graph, const AnfNodePtr &parameter,
                    const tensor::TensorPtr &tensor, uint32_t *memcpy_nums) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(parameter);
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(memcpy_nums);
  if (tensor->NeedSyncHostToDevice()) {
    return true;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_address = AnfAlgo::GetMutableOutputAddr(parameter, 0);
  if (ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER)) {
    return tensor->device_address().get() == nullptr || tensor->device_address() != device_address;
  }
  auto tensor_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
  MS_EXCEPTION_IF_NULL(tensor_address);
  if (tensor_address != device_address) {
    if (!kernel_graph->is_dynamic_shape() && EnableDeviceCopy() && NeedMemcpyInDevice(tensor_address, device_address)) {
      auto status = device_address->AsyncDeviceToDevice(trans::GetRuntimePaddingShape(parameter, 0),
                                                        tensor_address->GetSize(), tensor_address->type_id(),
                                                        tensor_address->GetPtr(), tensor_address->format());
      if (!status) {
        MS_LOG(EXCEPTION) << "SyncDeviceToDevice failed.";
      }
      MS_EXCEPTION_IF_NULL(memcpy_nums);
      (*memcpy_nums)++;
      auto input_param = parameter->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(input_param);
      if (common::AnfAlgo::IsParameterWeight(input_param) || kernel_graph->IsUpdatedParameter(input_param)) {
        tensor->set_device_address(device_address);
      }
      if (kernel_graph->IsUpdatedParameter(input_param)) {
        tensor->SetIsUpdateByDevice();
      }
      return false;
    } else {
      tensor->data_sync(false);
      return true;
    }
  }
  return false;
}

void AddGraphToManager(const NotNull<KernelGraphPtr> &graph, NotNull<FuncGraphManagerPtr> manager,
                       NotNull<std::set<KernelGraphPtr> *> memo) {
  if (memo->find(graph) != memo->end()) {
    return;
  }
  memo->insert(graph.get());
  manager->AddFuncGraph(graph.get(), false);

  for (auto &child_graph : graph->child_graph_order()) {
    AddGraphToManager(NOT_NULL(child_graph.lock()), manager, memo);
  }
}

void CheckControlFlowDynamicShape(const std::vector<KernelGraphPtr> &all_graphs) {
  if (all_graphs.size() <= 1) {
    return;
  }
  for (auto &graph : all_graphs) {
    MS_EXCEPTION_IF_NULL(graph);
    if (graph->is_dynamic_shape()) {
      MS_LOG(EXCEPTION) << "Dynamic shape is not supported with control flow(loop control statements and conditions "
                           "control statements).";
    }
  }
}
}  // namespace

void AscendSession::Init(uint32_t device_id) { InitExecutor(kAscendDevice, device_id); }

void AscendSession::UnifyMindIR(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Status record: start unify mindir. graph id: " << graph->graph_id();
  SessionBasic::UnifyMindIR(graph);
  opt::AscendUnifyMindIR(graph);
  MS_LOG(INFO) << "Status record: end unify mindir. graph id: " << graph->graph_id();
}

void AscendSession::LoadInputData(const std::shared_ptr<KernelGraph> &kernel_graph,
                                  const std::vector<tensor::TensorPtr> &inputs_const) const {
  std::vector<tensor::TensorPtr> inputs(inputs_const);
  uint32_t device_memcpy_nums = 0;
  MS_EXCEPTION_IF_NULL(kernel_graph);
  device::KernelAdjust::GetInstance().LoadDeviceLoopCtrlParameters(kernel_graph);
  auto &input_nodes = kernel_graph->input_nodes();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  if (device::KernelRuntime::UseMemScheduler()) {
    kernel_graph->SetInputTensors(inputs);
    return;
  }
  for (const auto &item : tensor_device_addr_map_) {
    auto output_tensor = item.first;
    output_tensor->set_device_address(item.second);
  }
  if (!tensor_device_addr_map_.empty()) {
    SyncStream();
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto tensor = inputs[i];
    MS_EXCEPTION_IF_NULL(tensor);
    auto input_node = input_nodes[i];
    MS_EXCEPTION_IF_NULL(input_node);
    auto size = LongToSize(tensor->data().nbytes());
    if (!input_node->isa<Parameter>()) {
      continue;
    }
    auto input_param = input_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(input_param);
    if (!input_param->IsUsedByRealKernelInGraph(kernel_graph->graph_id())) {
      tensor->set_sync_status(kNoNeedSync);
      continue;
    } else if (input_param->has_dynamic_shape()) {
      auto tensor_shape = tensor->shape();
      common::AnfAlgo::SetOutputInferTypeAndShape({common::AnfAlgo::GetOutputInferDataType(input_node, 0)},
                                                  {tensor_shape}, input_node.get());
      size = LongToSize(abstract::ShapeSize(tensor_shape)) * abstract::TypeIdSize(tensor->data_type());
    }
    if (AnfAlgo::OutputAddrExist(input_node, 0) &&
        TensorNeedSync(kernel_graph, input_node, tensor, &device_memcpy_nums)) {
      auto device_address = AnfAlgo::GetMutableOutputAddr(input_node, 0);
      MS_EXCEPTION_IF_NULL(device_address);
      if (size != 0 &&
          !device_address->SyncHostToDevice(trans::GetRuntimePaddingShape(input_node, 0), size, tensor->data_type(),
                                            tensor->data_c(), tensor->device_info().host_format_)) {
        MS_LOG(EXCEPTION) << "SyncHostToDevice failed.";
      }
      auto ms_context = MsContext::GetInstance();
      MS_EXCEPTION_IF_NULL(ms_context);
      if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode ||
          common::AnfAlgo::IsParameterWeight(input_param) || kernel_graph->IsUpdatedParameter(input_param)) {
        tensor->set_device_address(device_address);
      }
      if (kernel_graph->IsUpdatedParameter(input_param)) {
        tensor->SetIsUpdateByDevice();
      }
    }
    tensor->set_sync_status(kNoNeedSync);
  }
  if (device_memcpy_nums > 0) {
    auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
    MS_EXCEPTION_IF_NULL(runtime_instance);
    auto compute_stream = runtime_instance->compute_stream();
    auto model_stream = runtime_instance->GetModelStream(kernel_graph->graph_id());
    auto memcpy_event = runtime_instance->CreateDeviceEvent();
    MS_EXCEPTION_IF_NULL(memcpy_event);
    memcpy_event->set_wait_stream(model_stream);
    memcpy_event->set_record_stream(compute_stream);
    memcpy_event->RecordEvent();
    memcpy_event->WaitEvent();
  }
}

GraphId AscendSession::CompileGraphImpl(const AnfNodePtrList &lst, const AnfNodePtrList &outputs) {
  MS_LOG(INFO) << "Status record: start compile graph.";
  // construct graph, if successfully, graph_sum_ + 1
  auto graph = ConstructKernelGraph(lst, outputs, DeviceType::kAscend);
  MS_EXCEPTION_IF_NULL(graph);
  auto graph_id = graph->graph_id();
  MS_LOG(INFO) << "Status record: end compile graph. graph id: " << graph_id;
  return graph_id;
}

GraphId AscendSession::CompileGraphImpl(NotNull<FuncGraphPtr> func_graph) {
  MS_LOG(INFO) << "Status record: start compile graph.";
  std::vector<KernelGraphPtr> all_graphs;
  auto root_graph = ConstructKernelGraph(func_graph, &all_graphs, DeviceType::kAscend);
  MS_EXCEPTION_IF_NULL(root_graph);
  for (const auto &graph : all_graphs) {
    MS_EXCEPTION_IF_NULL(graph);
    graph->set_root_graph_id(root_graph->graph_id());
  }
  UnifyMindIR(root_graph);
  opt::BackendCommonOptimization(root_graph);
  CheckControlFlowDynamicShape(all_graphs);

  // empty graph dont entry to backend
  if (root_graph->execution_order().empty()) {
    MS_LOG(INFO) << root_graph->ToString() << " is empty graph.";
    AnfAlgo::InsertMakeTupleForOutput(NOT_NULL(root_graph));
    root_graph->set_executable(false);
    InitRuntimeResource();
    MS_LOG(INFO) << "Status record: end compile graph. graph id: " << root_graph->graph_id();
    return root_graph->graph_id();
  }

  // Handle control flow by auto-monad.
  HandleControlFlow(NOT_NULL(root_graph));

  std::set<KernelGraphPtr> memo;
  // add all graphs to manager first, so that don't have to make new manager in following passes.
  auto manager = Manage(root_graph, true);
  AddGraphToManager(NOT_NULL(root_graph), NOT_NULL(manager), NOT_NULL(&memo));
  memo.clear();

  // resource initialize
  InitRuntimeResource();

  IrFusionPass(NOT_NULL(root_graph), NOT_NULL(&memo));
  memo.clear();
  SelectKernel(NOT_NULL(root_graph));
  memo.clear();

  HardwareOptimize(NOT_NULL(root_graph), NOT_NULL(&memo));
  memo.clear();
#ifdef ENABLE_DEBUGGER
  // load graphs to debugger.
  if (debugger_ && debugger_->DebuggerBackendEnabled()) {
    LoadGraphsToDbg(NOT_NULL(root_graph), NOT_NULL(&memo));
  }
#endif
  memo.clear();
  UpdateRefOutputMap(NOT_NULL(root_graph), NOT_NULL(&memo));
  memo.clear();
  // add make_tuple to the output graph
  AnfAlgo::InsertMakeTupleForOutput(NOT_NULL(root_graph));
  // root root_graph valiate,include genearte execute order and so on
  RootGraphExecutorValidate(NOT_NULL(root_graph), all_graphs);
#ifdef ENABLE_DUMP_IR
  // dump graph before remove nop nodes
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    DumpIRProto(root_graph, "before_removeNop_" + std::to_string(graph_sum_));
  }
#endif

  // adjust kernel
  AdjustKernel(root_graph);
  // assign stream
  AssignStream(NOT_NULL(root_graph));
#ifndef ENABLE_SECURITY
  // insert profiling point
  device::KernelAdjust::GetInstance().Profiling(NOT_NULL(root_graph.get()));
#endif
  device::KernelAdjust::GetInstance().InsertOverflowCheckOperations(NOT_NULL(root_graph));
  // build kernel
  BuildKernel(root_graph);
#ifndef ENABLE_SECURITY
  SessionBasic::SetSummaryNodes(root_graph.get());
#endif
  // Alloc memory for child graph's inputs
  AssignStaticMemory(NOT_NULL(root_graph), NOT_NULL(&memo));
  memo.clear();
  // Alloc memory for root graph's inputs and node's outputs, workspace
  MemoryAlloc(root_graph.get());
  // generate and load task into device
  Load(root_graph);
  root_graph->SetInputNodes();
  root_graph->SetOptimizerFlag();
  DumpGraphs(all_graphs);
  // Save memory profiling data to proto file
#ifndef ENABLE_SECURITY
  if (MemoryProfiling::GetInstance().IsMemoryProfilingInitialized()) {
    auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
    MS_EXCEPTION_IF_NULL(runtime_instance);
    uint64_t mem_size = runtime_instance->GetMsUsedHbmSize();
    MemoryProfiling::GetInstance().SetDeviceMemSize(mem_size);
    if (MemoryProfiling::GetInstance().NeedSaveMemoryProfiling()) {
      MemoryProfiling::GetInstance().SaveMemoryProfiling();
    }
  }
#endif
  // return the root_graph id to backend
  auto graph_id = root_graph->graph_id();
  MS_LOG(INFO) << "Status record: end compile graph. graph id: " << graph_id;
  return graph_id;
}

#ifndef ENABLE_SECURITY
void AscendSession::SetFinalGraphSummaryFlag(const std::shared_ptr<KernelGraph> &kernel_graph) const {
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
#endif

void AscendSession::BuildGraphImpl(GraphId graph_id) {
  MS_LOG(INFO) << "Start";
  auto graph = GetGraph(graph_id);
  MS_EXCEPTION_IF_NULL(graph);
  // resource initialize
  InitRuntimeResource();
  // multiple graph handle
  if (graph_id == final_graph_id_) {
    MS_LOG(EXCEPTION) << "Unexpected graph id:" << graph_id << ", final_graph_id_:" << final_graph_id_;
  }
  auto single_graph = GetGraph(graph_id);
  MS_EXCEPTION_IF_NULL(single_graph);
  CompileChildGraph(single_graph);
  // set the distinction label of single graph
  single_graph->set_stream_distinction_label(graph_id);
  single_graph->UpdateExecuteKernelStreamLabel();
  // adjust execution order because  merge child graph and other special operations
  AdjustKernel(graph);
  // Assign streams for control sink and hccl and so on
  AssignStream(NOT_NULL(graph));
#ifndef ENABLE_SECURITY
  device::KernelAdjust::GetInstance().Profiling(NOT_NULL(graph.get()));
#endif
  device::KernelAdjust::GetInstance().InsertOverflowCheckOperations(NOT_NULL(graph));
  // build kernel if node is cnode
  BuildKernel(graph);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
#ifdef ENABLE_DEBUGGER
  if (debugger_ && debugger_->partial_memory()) {
    debugger_->PreExecute(graph);
  }
#endif
  if (ms_context->get_param<bool>(MS_CTX_PRECOMPILE_ONLY)) {
    MS_LOG(INFO) << "Precompile only, stop in build kernel step";
  } else {
    // alloc memory, including static memory and dynamic memory
    MemoryAlloc(graph.get());
    if (!device::KernelRuntime::UseMemScheduler()) {
      AnfAlgo::CacheAddrForGraph(graph);
    }
    // generate and load task info to device if it is sink mode
    Load(graph);
  }
  // sync the initial const tensor to device
  SyncInitialTenosrToDevice();
  DumpGraphs({graph});
  MS_LOG(INFO) << "End";
}

void AscendSession::CompileChildGraph(const KernelGraphPtr &child_graph) const {
  MS_EXCEPTION_IF_NULL(child_graph);
  MS_LOG(INFO) << "CompileChildGraph " << child_graph->ToString();
  opt::AscendBackendIRFusionOptimization(child_graph);
  child_graph->SetExecOrderByDefault();
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    std::string file_name = "select_kernel_before_graph_" + std::to_string(child_graph->graph_id()) + ".ir";
    DumpIR(file_name, child_graph);
  }
#endif
  // select kernel build info
  SelectKernel(child_graph);
#ifdef ENABLE_DUMP_IR
  if (context->CanDump(kIntroductory)) {
    std::string file_name = "select_kernel_after_graph_" + std::to_string(child_graph->graph_id()) + ".ir";
    DumpIR(file_name, child_graph);
  }
#endif
  // optimize graph
  HardwareOptimize(child_graph);
  // assign static memory of parameters
  if (!device::KernelRuntime::UseMemScheduler()) {
    auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
    MS_EXCEPTION_IF_NULL(runtime_instance);
    runtime_instance->AssignStaticMemoryInput(*child_graph);
    runtime_instance->AssignStaticMemoryValueNode(*child_graph);
  }
}

bool AscendSession::IsSupportSummary() { return !device::KernelAdjust::NeedLoopSink(); }

// Ascend old runtime.
void AscendSession::PreExecuteGraph(const std::shared_ptr<KernelGraph> &kernel_graph,
                                    const std::vector<tensor::TensorPtr> &inputs, VectorRef *const) {
#ifdef ENABLE_DEBUGGER
  if (debugger_) {
    debugger_->PreExecute(kernel_graph);
  }
#endif
}

// Ascend old runtime.
void AscendSession::PostExecuteGraph(const std::shared_ptr<KernelGraph> &kernel_graph,
                                     const std::vector<tensor::TensorPtr> &, VectorRef *const) {
  // summary
#ifndef ENABLE_SECURITY
  Summary(kernel_graph.get());
#endif
#ifdef ENABLE_DEBUGGER
  // load tensor from device for debugger
  if (debugger_ && debugger_->debugger_enabled()) {
    LoadTensor(kernel_graph);
  }
  // debugger post-execution processing
  if (debugger_) {
    debugger_->PostExecute();
  }
#endif
#ifndef ENABLE_SECURITY
  E2eDump::UpdateIterOldRTDump(kernel_graph.get());
#endif
}

void AscendSession::ExecuteGraph(const std::shared_ptr<KernelGraph> &kernel_graph) { Execute(kernel_graph, true); }

void AscendSession::RunOpHardwareOptimize(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  MS_LOG(INFO) << "HardwareOptimize Start";
  opt::RunOpAscendBackendOptimization(kernel_graph);
  MS_LOG(INFO) << "HardwareOptimize Finish";
}

KernelGraphPtr AscendSession::BuildOpImpl(const BackendOpRunInfoPtr &op_run_info, const GraphInfo &graph_info,
                                          const std::vector<tensor::TensorPtr> &input_tensors,
                                          const std::vector<int64_t> &tensors_mask) {
  auto it = run_op_graphs_.find(graph_info);
  if (it != run_op_graphs_.end()) {
    return it->second;
  }

  const auto &graph = PreBuildOp(op_run_info, input_tensors, tensors_mask);
  MS_EXCEPTION_IF_NULL(graph);
  // init runtime resource
  InitRuntimeResource();
  // build kernel
  RunOpAdjustKernel(graph);
  BuildKernel(graph);
  auto enable_op_graph_cache = MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_OP_GRAPH_CACHE);
  if (enable_op_graph_cache) {
    run_op_graphs_[graph_info] = graph;
  }
  return graph;
}

void AscendSession::BindAddressToTensor(
  const std::map<tensor::TensorPtr, session::KernelWithIndex> &tensor_to_node) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  for (const auto &item : tensor_to_node) {
    auto &tensor = item.first;
    auto &node = item.second.first;
    auto &output_index = item.second.second;
    DeviceAddressPtr address = nullptr;
    if (ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER)) {
      address = AnfAlgo::GetMutableOutputAddr(node, output_index, false);
    } else {
      address = AnfAlgo::GetMutableOutputAddr(node, output_index);
    }
    MS_EXCEPTION_IF_NULL(tensor);
    tensor->set_device_address(address);
  }
}

void StoreCNodePrimitive(const KernelGraphPtr &graph) {
  const auto &nodes = graph->execution_order();
  for (auto &node : nodes) {
    auto primitive = common::AnfAlgo::GetCNodePrimitive(node);
    MS_EXCEPTION_IF_NULL(primitive);
    auto new_primitive = std::make_shared<Primitive>(*primitive);
    node->set_input(kAnfPrimitiveIndex, NewValueNode(new_primitive));
  }
}

void AscendSession::RunOpImpl(const GraphInfo &graph_info, const BackendOpRunInfoPtr &op_run_info,
                              std::vector<tensor::TensorPtr> *input_tensors, VectorRef *outputs,
                              const std::vector<int64_t> &tensors_mask) {
  RunOpImplOrigin(graph_info, op_run_info, input_tensors, outputs, tensors_mask);
}

void AscendSession::RunOpImplOrigin(const GraphInfo &graph_info, const BackendOpRunInfoPtr &op_run_info,
                                    std::vector<tensor::TensorPtr> *input_tensors, VectorRef *outputs,
                                    const std::vector<int64_t> &tensors_mask) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  ProcessInputTensorsForHeterogeneous("Ascend", *input_tensors);
  const auto &graph = BuildOpImpl(op_run_info, graph_info, *input_tensors, tensors_mask);
  EraseValueNodeTensor(tensors_mask, input_tensors);

  // wait for allreduce
  for (auto &tensor : *input_tensors) {
    MS_EXCEPTION_IF_NULL(tensor);
    if (tensor->NeedWaitDevice()) {
      tensor->WaitDevice();
    }
  }

  // malloc mem
  RunOpRemoveNopNode(graph);
  RunOpMemoryAlloc(*input_tensors, graph, op_run_info->is_gradient_out);
  RunOpGenKernelEvent(graph.get());
  AnfAlgo::CacheAddrForGraph(graph);

  // load input data to device
  LoadInputData(graph, *input_tensors);
  // run op
  Execute(graph, false);
  // get output
  std::map<tensor::TensorPtr, session::KernelWithIndex> tensor_to_node;
  UpdateOutputs(graph, outputs, *input_tensors, &tensor_to_node);
  RunOpMemoryClear(graph.get());
}

KernelGraphPtr AscendSession::PreBuildOp(const BackendOpRunInfoPtr &op_run_info,
                                         const std::vector<tensor::TensorPtr> &input_tensors,
                                         const std::vector<int64_t> &tensors_mask) {
  // Construct graph include one op
  auto graph = ConstructSingleOpGraph(op_run_info, input_tensors, tensors_mask, true);
  MS_EXCEPTION_IF_NULL(graph);
  opt::RunOpAscendBackendIRFusionOptimization(graph);
  SelectKernel(graph);
  RunOpHardwareOptimize(graph);
  runtime::OpRuntimeInfo::CacheGraphOpRuntimeInfo(graph);
  return graph;
}

void AscendSession::GetOpInputStubTensors(const CNodePtr &cnode, const std::map<AnfNodePtr, size_t> &parameter_index,
                                          const std::vector<tensor::TensorPtr> &graph_inputs,
                                          const std::map<KernelWithIndex, OutputTensorInfo> &node_output_info,
                                          InputTensorInfo *input_tensor_info) const {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(input_tensor_info);
  const auto input_tensor_num = common::AnfAlgo::GetInputTensorNum(cnode);
  for (size_t i = 1; i <= input_tensor_num; i += 1) {
    const auto &input = cnode->input(i);
    auto kernel_with_index = common::AnfAlgo::VisitKernel(input, 0);
    auto real_input = kernel_with_index.first;
    MS_EXCEPTION_IF_NULL(real_input);
    tensor::TensorPtr tensor = nullptr;
    if (real_input->isa<ValueNode>()) {
      tensor = GetValueNodeOutputTensor(real_input, kernel_with_index.second);
      input_tensor_info->input_tensors_mask.emplace_back(
        GetValueNode(real_input)->isa<StringImm>() ? kValueNodeTensorMask : kParameterDataTensorMask);
    } else if (real_input->isa<Parameter>()) {
      tensor = GetParameterOutputTensor(real_input, parameter_index, graph_inputs);
      auto parameter = real_input->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(parameter);
      input_tensor_info->input_tensors_mask.emplace_back(parameter->has_default() ? kParameterWeightTensorMask
                                                                                  : kParameterDataTensorMask);
    } else if (real_input->isa<CNode>()) {
      bool output_is_weight = false;
      tensor = GetCNodeOutputStubTensor(kernel_with_index, node_output_info, &output_is_weight);
      input_tensor_info->input_tensors_mask.emplace_back(output_is_weight ? kParameterWeightTensorMask
                                                                          : kParameterDataTensorMask);
    } else {
      MS_LOG(EXCEPTION) << "Invalid input node, node = " << real_input->DebugString();
    }
    MS_EXCEPTION_IF_NULL(tensor);
    MS_LOG(DEBUG) << "Get" << i << "th input tensor of " << cnode->fullname_with_scope() << " from "
                  << real_input->fullname_with_scope() << "-" << kernel_with_index.second;
    input_tensor_info->input_tensors.emplace_back(tensor);
  }
}

void AscendSession::BuildOpsInGraph(const GraphId &graph_id, const std::map<AnfNodePtr, size_t> &parameter_index,
                                    const std::vector<tensor::TensorPtr> &graph_inputs,
                                    const std::map<KernelWithIndex, size_t> &cnode_refcount) {
  if (built_graph_id_.find(graph_id) != built_graph_id_.end()) {
    return;
  }
  auto graph = GetGraph(graph_id);
  MS_EXCEPTION_IF_NULL(graph);
  std::map<KernelWithIndex, OutputTensorInfo> op_output_info;
  std::vector<CNodePtr> kernels;
  mindspore::HashMap<KernelGraphPtr, GraphInfo> single_op_graphs;
  // Collect kernels need to be built in single op graphs
  for (const auto &kernel : graph->execution_order()) {
    // Generate fake input tensors, tensor masks and input kernel with index
    InputTensorInfo input_tensor_info;
    GetOpInputStubTensors(kernel, parameter_index, graph_inputs, op_output_info, &input_tensor_info);
    // Get OpRunInfo and GraphInfo
    GraphInfo graph_info;
    BackendOpRunInfoPtr op_run_info = GetSingleOpRunInfo(kernel, graph_info, input_tensor_info, nullptr);
    GetSingleOpGraphInfo(kernel, input_tensor_info, &graph_info, op_run_info);
    op_run_info->base_op_run_info.graph_info = graph_info;
    const auto &single_op_graph_iter = run_op_graphs_.find(graph_info);
    if (single_op_graph_iter != run_op_graphs_.end()) {
      // if graph of same single op exists, the output tensor of current op should be generated
      GenOpOutputStubTensor(single_op_graph_iter->second, kernel, cnode_refcount, &op_output_info);
      continue;
    }
    const auto &single_op_graph =
      PreBuildOp(op_run_info, input_tensor_info.input_tensors, input_tensor_info.input_tensors_mask);
    MS_EXCEPTION_IF_NULL(single_op_graph);
    GenOpOutputStubTensor(single_op_graph, kernel, cnode_refcount, &op_output_info);
    opt::HideNopNode(single_op_graph.get());
    // The graph info could have been changed in PreBuildOp
    GraphInfo new_graph_info;
    GetSingleOpGraphInfo(kernel, input_tensor_info, &new_graph_info, op_run_info);
    single_op_graphs.emplace(single_op_graph, new_graph_info);
    const auto &execution_order = single_op_graph->execution_order();
    std::copy(execution_order.begin(), execution_order.end(), std::back_inserter(kernels));
  }
  InitRuntimeResource();
  // Compile all kernels parallel
  BuildKernel(kernels);
  // Some new kernel may be added after InsertAtomicCleanOps, so collect and build kernels again
  kernels.clear();
  for (const auto &graph_item : single_op_graphs) {
    device::ascend::InsertAtomicCleanOps(graph_item.first);
    const auto &execution_order = graph_item.first->execution_order();
    std::copy(execution_order.begin(), execution_order.end(), std::back_inserter(kernels));
  }
  BuildKernel(kernels);
  // Record single op graphs in run_op_graphs_ so that these graphs can be reused in BuildOpImpl
  for (const auto &graph_item : single_op_graphs) {
    RunOpMemoryClear(graph_item.first.get());
    auto enable_op_graph_cache = MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_OP_GRAPH_CACHE);
    if (enable_op_graph_cache) {
      run_op_graphs_[graph_item.second] = graph_item.first;
    }
    MS_LOG(DEBUG) << "Pre build op finished, graph info: " << graph_item.second;
  }
  built_graph_id_.insert(graph_id);
}

#ifndef ENABLE_SECURITY
void DumpInit(const std::string &device_type, uint32_t device_id) {
  auto &json_parser = DumpJsonParser::GetInstance();
  json_parser.Parse();
  json_parser.CopyDumpJsonToDir(device_id);
  json_parser.CopyHcclJsonToDir(device_id);
  json_parser.CopyMSCfgJsonToDir(device_id);
  if (json_parser.async_dump_enabled()) {
#if !(defined(ENABLE_TEST) || defined(ENABLE_TESTCASES))
    if (device_type == kAscendDevice) {
      // register callback to adx
      if (json_parser.FileFormatIsNpy()) {
        (void)AdxRegDumpProcessCallBack(mindspore::ascend::DumpDataCallBack);
      }
    }
#endif
    if (AdxDataDumpServerInit() != 0) {
      MS_LOG(EXCEPTION) << "Adx data dump server init failed";
    }
  }
}
#endif

void AscendSession::InitRuntimeResource() {
  MS_LOG(INFO) << "Status record: start init runtime resource.";
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  if (!runtime_instance->Init()) {
    MS_LOG(EXCEPTION) << "Kernel runtime init error.";
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<bool>(MS_CTX_ENABLE_HCCL)) {
    // get actual rank id if it's distribution training case.
    rank_id_ = GetRankId();
  }
#ifndef ENABLE_SECURITY
  auto device_type = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  DumpInit(device_type, rank_id_);
#endif
  MS_LOG(INFO) << "Status record: end init runtime resource.";
}

void AscendSession::HardwareOptimize(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  opt::AscendBackendOptimization(kernel_graph);
  FinalOptimize(kernel_graph);
  GraphKernelOptimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
}

void AscendSession::GraphKernelOptimize(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  if (!graphkernel::GraphKernelFlags::GetInstance().IsEnableGraphKernel()) {
    return;
  }
  graphkernel::GraphKernelOptimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
}

void AscendSession::AdjustKernel(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(INFO) << "Status record: start adjust kernel. graph id: " << kernel_graph->graph_id();
  opt::HideNopNode(kernel_graph.get());
  auto execution_order = kernel_graph->execution_order();
  common::AnfAlgo::ReorderExecList(NOT_NULL(&execution_order));
  kernel_graph->set_execution_order(execution_order);
  // Insert CLearZero op
  // prepare for next step from json get atomic info
  BuildKernel(kernel_graph);
  device::ascend::InsertAtomicCleanOps(kernel_graph);
  device::KernelAdjust::GetInstance().InsertDeviceLoopCtrl(kernel_graph);
  device::KernelAdjust::GetInstance().ProcessLoopSink(kernel_graph);
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    DumpIR("after_adjust_kernel.ir", kernel_graph);
  }
#endif
  MS_LOG(INFO) << "Status record: end adjust kernel. graph id: " << kernel_graph->graph_id();
}

void AscendSession::RunOpAdjustKernel(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  MS_LOG(INFO) << "Start!";
  RunOpHideNopNode(kernel_graph);
  // Insert CLearZero op
  // prepare for next step from json get atomic info
  BuildKernel(kernel_graph);
  device::ascend::InsertAtomicCleanOps(kernel_graph);
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::AssignStream(const NotNull<KernelGraphPtr> &kernel_graph) const {
  MS_LOG(INFO) << "Status record: start assign stream, graph id: " << kernel_graph->graph_id();
  device::ascend::AscendStreamAssign::GetInstance().AssignStream(kernel_graph);
  MS_LOG(INFO) << "Status record: end assign stream, graph id: " << kernel_graph->graph_id();
}

void AscendSession::BuildKernel(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(INFO) << "Status record: start build kernel, graph id: " << kernel_graph->graph_id();
  BuildKernel(kernel_graph->execution_order());
  MS_LOG(INFO) << "Status record: end build kernel, graph id: " << kernel_graph->graph_id();
}

void AscendSession::BuildKernel(const std::vector<CNodePtr> &kernels) {
  struct timeval start_time {};
  struct timeval end_time {};
  (void)gettimeofday(&start_time, nullptr);
  auto ret = device::ascend::KernelBuild(kernels);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Kernel build error.";
  }
  (void)gettimeofday(&end_time, nullptr);
  const uint64_t kUSecondInSecond = 1000000;
  uint64_t cost = kUSecondInSecond * static_cast<uint64_t>(end_time.tv_sec - start_time.tv_sec);
  cost += static_cast<uint64_t>(end_time.tv_usec - start_time.tv_usec);
  MS_LOG(INFO) << "KernelBuild run in " << cost << " us.";
}

static CNodePtr GetNextLabelSet(const std::vector<CNodePtr> &kernel_nodes, uint32_t index) {
  size_t node_sizes = kernel_nodes.size();
  if (index >= node_sizes - 1) {
    MS_LOG(EXCEPTION) << "there is no node after this node:" << kernel_nodes[index]->DebugString();
  }
  auto kernel = kernel_nodes[index + 1];
  if (common::AnfAlgo::GetCNodeName(kernel) != kLabelSetOpName) {
    MS_LOG(EXCEPTION) << "the node is not labelset follow labelgoto/labelswitch, index" << index
                      << ", size: " << node_sizes;
  }
  return kernel;
}

static std::vector<CNodePtr> HandleRecursiveCall(const std::vector<CNodePtr> &kernel_cnodes, const uint32_t &back_label,
                                                 uint32_t *index, std::vector<CNodePtr> *back) {
  MS_EXCEPTION_IF_NULL(index);
  MS_EXCEPTION_IF_NULL(back);
  std::vector<CNodePtr> front;
  std::vector<CNodePtr> back_temp;
  bool back_flag = false;
  uint32_t i = *index;
  while (i < kernel_cnodes.size()) {
    if (!back_flag) {
      front.emplace_back(kernel_cnodes[i]);
    } else {
      back->emplace_back(kernel_cnodes[i]);
    }
    if (common::AnfAlgo::HasNodeAttr(kAttrRecursiveEnd, kernel_cnodes[i])) {
      *index = i;
      back->insert(back->cend(), back_temp.cbegin(), back_temp.cend());
      return front;
    }
    if (common::AnfAlgo::HasNodeAttr(kAttrRecursive, kernel_cnodes[i])) {
      back_flag = true;
      if (!common::AnfAlgo::IsLabelIndexInNode(kernel_cnodes[i], back_label)) {
        auto temp = HandleRecursiveCall(kernel_cnodes, back_label, &(++i), &back_temp);
        front.insert(front.end(), temp.begin(), temp.end());
      }
    }
    i++;
  }
  return front;
}

static void UnfoldRecursiveExecOrder(KernelGraph *kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  if (!kernel_graph->recursive_call()) {
    return;
  }
  auto kernel_cnodes = kernel_graph->mem_reuse_exec_order();
  std::vector<CNodePtr> mem_reuse_order;
  mem_reuse_order.reserve(kernel_cnodes.size());
  for (uint32_t i = 0; i < kernel_cnodes.size(); i++) {
    if (!common::AnfAlgo::HasNodeAttr(kAttrRecursiveStart, kernel_cnodes[i])) {
      mem_reuse_order.emplace_back(kernel_cnodes[i]);
      continue;
    }
    auto label_id = common::AnfAlgo::GetNodeAttr<uint32_t>(kernel_cnodes[i], kAttrLabelIndex);
    std::vector<CNodePtr> back;
    auto front = HandleRecursiveCall(kernel_cnodes, label_id, &i, &back);
    mem_reuse_order.insert(mem_reuse_order.cend(), front.cbegin(), front.cend());
    mem_reuse_order.insert(mem_reuse_order.cend(), back.cbegin(), back.cend());
  }
  kernel_graph->set_mem_reuse_exec_order(mem_reuse_order);
}

static void GetSubGraphExecOrder(const KernelGraph *kernel_graph, uint32_t index, const CNodePtr &back_node,
                                 std::vector<CNodePtr> *mem_reuse_order) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(mem_reuse_order);
  auto label_id = common::AnfAlgo::GetNodeAttr<uint32_t>(back_node, kAttrLabelIndex);
  auto kernel_cnodes = kernel_graph->execution_order();
  for (auto i = index; i < kernel_cnodes.size(); i++) {
    mem_reuse_order->emplace_back(kernel_cnodes[i]);
    if (common::AnfAlgo::IsLabelIndexInNode(kernel_cnodes[i], label_id)) {
      return;
    }
  }
}

void InitMemReuseExecOrder(KernelGraph *kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  if (!kernel_graph->subgraph_multi_call()) {
    return;
  }
  mindspore::HashMap<uint32_t, uint32_t> label_id_index_map;
  auto kernel_cnodes = kernel_graph->execution_order();
  std::vector<CNodePtr> mem_reuse_order;
  for (uint32_t i = 0; i < kernel_cnodes.size(); i++) {
    mem_reuse_order.emplace_back(kernel_cnodes[i]);
    if (common::AnfAlgo::CheckPrimitiveType(kernel_cnodes[i], prim::kPrimLabelSwitch) &&
        !common::AnfAlgo::HasNodeAttr(kAttrRecursive, kernel_cnodes[i]) &&
        !common::AnfAlgo::HasNodeAttr(kAttrReturn, kernel_cnodes[i])) {
      auto label_list = common::AnfAlgo::GetNodeAttr<std::vector<uint32_t>>(kernel_cnodes[i], kAttrLabelSwitchList);
      for (auto label_id : label_list) {
        if (label_id_index_map.find(label_id) == label_id_index_map.end()) {
          continue;
        }
        auto back_node = GetNextLabelSet(kernel_cnodes, i);
        GetSubGraphExecOrder(kernel_graph, label_id_index_map[label_id], back_node, &mem_reuse_order);
      }
      continue;
    }
    if (common::AnfAlgo::CheckPrimitiveType(kernel_cnodes[i], prim::kPrimLabelGoto) &&
        !common::AnfAlgo::HasNodeAttr(kAttrRecursive, kernel_cnodes[i]) &&
        !common::AnfAlgo::HasNodeAttr(kAttrReturn, kernel_cnodes[i])) {
      auto label_id = common::AnfAlgo::GetNodeAttr<uint32_t>(kernel_cnodes[i], kAttrLabelIndex);
      if (label_id_index_map.find(label_id) == label_id_index_map.end()) {
        continue;
      }
      auto back_node = GetNextLabelSet(kernel_cnodes, i);
      GetSubGraphExecOrder(kernel_graph, label_id_index_map[label_id], back_node, &mem_reuse_order);
      continue;
    }
    if (common::AnfAlgo::CheckPrimitiveType(kernel_cnodes[i], prim::kPrimLabelSet) &&
        !common::AnfAlgo::HasNodeAttr(kAttrRecursive, kernel_cnodes[i])) {
      auto label_id = common::AnfAlgo::GetNodeAttr<uint32_t>(kernel_cnodes[i], kAttrLabelIndex);
      if (label_id_index_map.find(label_id) != label_id_index_map.end()) {
        MS_LOG(EXCEPTION) << "Two labelsets with same label id.";
      }
      label_id_index_map[label_id] = i;
      continue;
    }
  }
  kernel_graph->set_mem_reuse_exec_order(mem_reuse_order);
  UnfoldRecursiveExecOrder(kernel_graph);
}

void AscendSession::MemoryAlloc(KernelGraph *kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(INFO) << "Status record: start memory alloc. graph id: " << kernel_graph->graph_id();
  InitMemReuseExecOrder(kernel_graph);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->AssignMemory(*kernel_graph);
  device::KernelAdjust::GetInstance().AssignLoopCtrlMemory(*kernel_graph);
  MS_LOG(INFO) << "Status record: end memory alloc. graph id: " << kernel_graph->graph_id()
               << ", Memory Statistics:" << device::ascend::AscendMemAdapter::GetInstance().DevMemStatistics();
  MS_LOG(INFO) << "The dynamic memory pool total size is "
               << device::ascend::AscendMemoryPool::GetInstance().TotalMemStatistics() / kMBToByte
               << "M, total used size is "
               << device::ascend::AscendMemoryPool::GetInstance().TotalUsedMemStatistics() / kMBToByte
               << "M, used peak size is "
               << device::ascend::AscendMemoryPool::GetInstance().UsedMemPeakStatistics() / kMBToByte << "M.";
}

void AscendSession::RunOpMemoryAlloc(const std::vector<tensor::TensorPtr> &input_tensors,
                                     const KernelGraphPtr &kernel_graph, bool is_gradient_out) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->RunOpAssignMemory(input_tensors, *kernel_graph, is_gradient_out);
}

void AscendSession::RunOpMemoryAllocNew(const std::vector<tensor::TensorPtr> &input_tensors,
                                        const std::map<tensor::TensorPtr, session::KernelWithIndex> &tensor_to_node,
                                        const KernelGraph &kernel_graph) const {
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->RunOpAssignMemory(input_tensors, kernel_graph, false, tensor_to_node);
}

void AscendSession::RunOpGenKernelEvent(const KernelGraph *graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  auto kernels = graph->execution_order();
  device::ascend::AscendStreamAssign::GetInstance().AssignStreamForNonTaskSink(kernels);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->GenKernelEvents(*graph);
}

void AscendSession::RunOpMemoryClear(const KernelGraph *kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->RunOpClearMemory(*kernel_graph);
}

void AscendSession::Load(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(INFO) << "Status record: start load task. graph id: " << kernel_graph->graph_id();
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool is_task_sink = context_ptr->get_param<bool>(MS_CTX_ENABLE_TASK_SINK);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  bool ret_ok = runtime_instance->Load(*kernel_graph, is_task_sink);
  if (!ret_ok) {
    MS_LOG(EXCEPTION) << "Load task error!";
  }
  MS_LOG(INFO) << "Status record: end load task. graph id: " << kernel_graph->graph_id();
}

void AscendSession::Execute(const std::shared_ptr<KernelGraph> &kernel_graph, bool is_task) const {
  MS_LOG(DEBUG) << "Start!";
  bool is_task_sink = false;
  if (is_task) {
    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);
    is_task_sink = context_ptr->get_param<bool>(MS_CTX_ENABLE_TASK_SINK);
  }
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  bool ret_ok = runtime_instance->Run(*kernel_graph, is_task_sink);
#ifndef ENABLE_SECURITY
  if (is_task && is_task_sink) {
    Dump(kernel_graph);
  }
#endif
  if (!ret_ok) {
#ifdef ENABLE_DUMP_IR
    mindspore::RDR::TriggerAll();
#endif
    MS_LOG(EXCEPTION) << "run task error!";
  }
  MS_LOG(DEBUG) << "Finish!";
}

#ifndef ENABLE_SECURITY
void AscendSession::Dump(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  MS_LOG(DEBUG) << "Start!";
  MS_EXCEPTION_IF_NULL(kernel_graph);
  E2eDump::DumpRunIter(kernel_graph, rank_id_);
  E2eDump::DumpData(kernel_graph.get(), rank_id_);
  MS_LOG(DEBUG) << "Finish!";
}
#endif

void AscendSession::LoadTensor(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  MS_LOG(INFO) << "Start!";
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  (void)runtime_instance->LoadData(*kernel_graph);
  MS_LOG(INFO) << "Finish!";
}

#ifndef ENABLE_SECURITY
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
  for (unsigned int i : graph_order) {
    auto child_graph = GetGraph(i);
    if (child_graph == nullptr) {
      continue;
    }
    SessionBasic::SetSummaryNodes(child_graph.get());
    auto child_graph_summary = child_graph->summary_nodes();
    summary->insert(child_graph_summary.cbegin(), child_graph_summary.cend());
    RecurseSetSummaryNodes(child_graph.get(), summary);
  }
  graph->set_summary_nodes(*summary);
}

void AscendSession::SetSummaryNodes(KernelGraph *graph) {
  MS_LOG(DEBUG) << "Update summary Start";
  MS_EXCEPTION_IF_NULL(graph);
  auto summary_nodes = graph->summary_nodes();
  std::map<std::string, std::pair<AnfNodePtr, int>> summary;
  summary.insert(summary_nodes.cbegin(), summary_nodes.cend());
  RecurseSetSummaryNodes(graph, &summary);
  graph->set_summary_nodes(summary);
  MS_LOG(DEBUG) << "Update summary end size: " << summary.size();
}
#endif

void AscendSession::MergeGraphExecOrder() const {
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
    if (!context_ptr->get_param<bool>(MS_CTX_ENABLE_TASK_SINK)) {
      MS_LOG(EXCEPTION) << "Control sink network should run with task-sink mode!";
    }
  }
  // if first graph is common,the final graph has no label,then set the stream of final graph same with the first graph
  SetStreamDistinctionLabel(final_graph, graph_order[0], false);
  std::vector<CNodePtr> final_exec_order = final_graph->execution_order();
  for (size_t i = 0; i < graph_order.size(); i++) {
    auto graph_id = graph_order[i];
    if (graph_type[i] == BRANCH_END || graph_type[i] == BRANCH_START) {
      continue;
    }
    auto child_graph = GetGraph(graph_id);
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
    for (const auto &item : child_ref_map) {
      if (final_graph->IsInRefOutputMap(item.first)) {
        MS_LOG(EXCEPTION) << "The ref pair is already in final graph!";
      }
      final_graph->AddRefCorrespondPairs(item.first, item.second);
    }
  }
  // set final_exec_order into final graph
  MS_EXCEPTION_IF_NULL(final_graph);
#ifndef ENABLE_SECURITY
  DumpGraphExeOrder(final_exec_order);
#endif
  final_graph->set_execution_order(final_exec_order);
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
  for (const auto &item : initial_tenosrs_) {
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
    size_t tensor_size = LongToSize(front_tensor->data().nbytes());
    auto addr = AnfAlgo::GetOutputAddr(backend_parameter, 0);
    MS_EXCEPTION_IF_NULL(addr);
    if (!addr->SyncHostToDevice(trans::GetRuntimePaddingShape(backend_parameter, 0), tensor_size,
                                front_tensor->data_type(), front_tensor->data_c(),
                                front_tensor->device_info().host_format_)) {
      MS_LOG(EXCEPTION) << "Tensor SyncHostToDevice fail!";
    }
  }
}

void AscendSession::RootGraphExecutorValidate(const NotNull<KernelGraphPtr> &graph,
                                              const std::vector<KernelGraphPtr> &all_graphs) const {
  AscendAutoMonad auto_monad(graph);
  auto_monad.GenerateExecuteOrder();
  if (graph->label_num() > kLabelNumsThreshold) {
    MS_LOG(EXCEPTION) << "This model with " << all_graphs.size() << " graphs needs " << graph->label_num()
                      << " labels, which out of range of [0, 1024).\n1. Check if front-end composition is correct.\n"
                      << "2. Optimize model expression and reduce the number of graphs and labels.";
  }
}

void AscendSession::IrFusionPass(const NotNull<KernelGraphPtr> &graph, NotNull<std::set<KernelGraphPtr> *> memo) {
  if (memo->find(graph) != memo->end()) {
    return;
  }
  memo->insert(graph.get());
  opt::AscendBackendIRFusionOptimization(graph);
  graph->SetExecOrderByDefault();
  for (auto &child_graph : graph->child_graph_order()) {
    IrFusionPass(NOT_NULL(child_graph.lock()), memo);
  }
}

void AscendSession::SetOperatorInfo(const std::vector<CNodePtr> &nodes) const {
  for (const auto &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    auto status = device::ascend::SelectKernelInfo(node);
    common::AnfAlgo::EraseNodeAttr(kAttrPynativeNextOpName, node);
    common::AnfAlgo::EraseNodeAttr(kAttrPynativeNextIndex, node);
    if (status == device::ascend::kStatusRaisePrecision) {
      raise_precision_count_++;
    } else if (status == device::ascend::kStatusReducePrecision) {
      reduce_precision_count_++;
    }
    MS_LOG(INFO) << "Select ApplyKernel: " << node->DebugString();
  }
}

void AscendSession::RecurseSelectKernelInfo(const KernelGraphPtr &graph, std::set<KernelGraphPtr> *memo) const {
  MS_EXCEPTION_IF_NULL(memo);
  if (memo->find(graph) != memo->end()) {
    return;
  }
  memo->insert(graph);
  MS_LOG(INFO) << "Start to select kernel info in graph: " << graph->graph_id();
  SetOperatorInfo(graph->execution_order());
  MS_LOG(INFO) << "Finish selecting kernel info in graph: " << graph->graph_id();

#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    std::string file_name = "select_kernel_after_graph_" + std::to_string(graph->graph_id()) + ".ir";
    DumpIR(file_name, graph);
  }
#endif

  for (auto &child_graph : graph->child_graph_order()) {
    RecurseSelectKernelInfo(child_graph.lock(), memo);
  }
}

void AscendSession::SelectKernel(const KernelGraphPtr &graph) const {
  MS_LOG(INFO) << "Status record: start select kernel. graph id: " << graph->graph_id();
  raise_precision_count_ = 0;
  reduce_precision_count_ = 0;
  std::set<KernelGraphPtr> memo;
  RecurseSelectKernelInfo(graph, &memo);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode) {
    if (raise_precision_count_ > 0) {
      MS_LOG(WARNING) << "There are " << raise_precision_count_
                      << " node/nodes used raise precision to selected the kernel!";
    }
    if (reduce_precision_count_ > 0) {
      MS_LOG(WARNING) << "There are " << reduce_precision_count_
                      << " node/nodes used reduce precision to selected the kernel!";
    }
  }
  MS_LOG(INFO) << "Status record: end select kernel. graph id: " << graph->graph_id();
}

void AscendSession::HardwareOptimize(NotNull<KernelGraphPtr> graph, NotNull<std::set<KernelGraphPtr> *> memo) const {
  if (memo->find(graph) != memo->end()) {
    return;
  }
  memo->insert(graph.get());
  HardwareOptimize(graph.get());
  for (auto &child_graph : graph->child_graph_order()) {
    HardwareOptimize(NOT_NULL(child_graph.lock()), memo);
  }
}

#ifdef ENABLE_DEBUGGER
// Load graphs and their children for Ascend old runtime.
void AscendSession::LoadGraphsToDbg(const NotNull<KernelGraphPtr> &graph,
                                    NotNull<std::set<KernelGraphPtr> *> memo) const {
  if (memo->find(graph) != memo->end()) {
    return;
  }
  memo->insert(graph.get());

  MS_LOG(INFO) << "Start to do LoadGraphsToDbg in graph: " << graph->graph_id();

  MS_EXCEPTION_IF_NULL(debugger_);
  debugger_->LoadGraphs(graph);
  MS_LOG(INFO) << "graph_sum_: " << graph_sum_;
  for (auto &child_graph : graph->child_graph_order()) {
    LoadGraphsToDbg(NOT_NULL(child_graph.lock()), memo);
  }
  MS_LOG(INFO) << "Finish doing LoadGraphsToDbg in graph: " << graph->graph_id();
}
#endif

void AscendSession::AssignStaticMemory(NotNull<KernelGraphPtr> graph, NotNull<std::set<KernelGraphPtr> *> memo) const {
  if (memo->find(graph) != memo->end()) {
    return;
  }
  memo->insert(graph.get());
  MS_LOG(INFO) << "Status record: start assign static memory for parameter in graph. graph id: " << graph->graph_id();
  // assign static memory for parameters
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->ClearGlobalIdleMem();
  runtime_instance->AssignStaticMemoryInput(*graph.get());
  runtime_instance->AssignStaticMemoryValueNode(*graph.get());
  for (auto &child_graph : graph->child_graph_order()) {
    AssignStaticMemory(NOT_NULL(child_graph.lock()), memo);
  }
  MS_LOG(INFO) << "Status record: end assign static memory for parameter in graph. graph id: " << graph->graph_id();
}

void AscendSession::UpdateRefOutputMap(NotNull<KernelGraphPtr> graph, NotNull<std::set<KernelGraphPtr> *> memo) const {
  if (memo->find(graph) != memo->end()) {
    return;
  }
  memo->insert(graph.get());

  for (auto &child_graph : graph->child_graph_order()) {
    std::shared_ptr<KernelGraph> child_graph_ptr = child_graph.lock();
    MS_EXCEPTION_IF_NULL(child_graph_ptr);
    UpdateRefOutputMap(NOT_NULL(child_graph_ptr), memo);
    // copy ref map to final graph
    auto child_ref_map = child_graph_ptr->GetRefMap();
    for (const auto &item : child_ref_map) {
      if (graph->IsInRefOutputMap(item.first)) {
        MS_LOG(DEBUG) << "The ref pair <" << item.first.first->DebugString() << ", " << item.first.second
                      << "> is already in " << graph->ToString();
        continue;
      }
      graph->AddRefCorrespondPairs(item.first, item.second);
    }
  }
}

void AscendSession::SyncStream() const {
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  auto ret = runtime_instance->SyncStream();
  if (!ret) {
    MS_LOG(EXCEPTION) << "Sync stream error!";
  }
}

void AscendSession::SetThreadContext() { ErrorManager::GetInstance().GenWorkStreamIdDefault(); }

void AscendSession::UpdateOutputTensors(const VectorRef *outputs,
                                        const std::map<tensor::TensorPtr, session::KernelWithIndex> &tensor_to_node,
                                        std::map<DeviceAddressPtr, DeviceAddressPtr> *) {
  if (device::KernelRuntime::UseMemScheduler()) {
    return;
  }
  MS_EXCEPTION_IF_NULL(outputs);
  tensor_device_addr_map_.clear();
  for (const auto &item : *outputs) {
    if (utils::isa<VectorRefPtr>(item)) {
      const auto &vector_ref = utils::cast<VectorRef>(item);
      std::map<DeviceAddressPtr, DeviceAddressPtr> new_to_old_device_address;
      UpdateOutputTensors(&vector_ref, tensor_to_node, &new_to_old_device_address);
    } else if (utils::isa<tensor::TensorPtr>(item)) {
      const auto &tensor = utils::cast<tensor::TensorPtr>(item);
      MS_EXCEPTION_IF_NULL(tensor);
      const auto &iter = tensor_to_node.find(tensor);
      if (iter != tensor_to_node.end()) {
        const auto &node = iter->second.first;
        size_t output_index = iter->second.second;
        if (!AnfAlgo::OutputAddrExist(node, output_index, true)) {
          continue;
        }
        const auto &address = AnfAlgo::GetMutableOutputAddr(node, output_index);
        tensor->set_device_address(address);
        if (EnableDeviceCopy() && tensor->NeedSyncDeviceToHostImmediately()) {
          auto dst_device_address = AssignExtraMemForGraphOutput(tensor, node, output_index);
          MS_EXCEPTION_IF_NULL(dst_device_address);
          if (!dst_device_address->AsyncDeviceToDevice(trans::GetRuntimePaddingShape(node, output_index),
                                                       address->GetSize(), address->type_id(), address->GetPtr(),
                                                       address->format())) {
            MS_LOG(EXCEPTION) << "SyncDeviceToDevice failed!";
          }
          tensor->set_sync_status(kNoNeedSync);
          tensor_device_addr_map_[tensor] = dst_device_address;
        }

        if (common::AnfAlgo::IsDynamicShape(node)) {
          const auto &updated_shape = common::AnfAlgo::GetOutputInferShape(node, output_index);
          (void)tensor->set_shape(updated_shape);
        }
      }
      if (tensor->NeedSyncDeviceToHostImmediately()) {
        tensor->data_sync(false);
        tensor->set_device_address(nullptr);
        tensor->set_sync_status(kNeedSyncHostToDevice);
      }
    }
  }
}
DeviceAddressPtr AscendSession::AssignExtraMemForGraphOutput(const tensor::TensorPtr &tensor, const AnfNodePtr &node,
                                                             size_t index) const {
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(node);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  return runtime_instance->AssignExtraStaticMem(tensor, node, index);
}
}  // namespace mindspore::session
