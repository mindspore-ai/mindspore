/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "runtime/hardware/ascend/ascend_device_context.h"
#include <algorithm>
#include <set>
#include <unordered_map>
#include "acl/acl_rt.h"
#include "runtime/dev.h"
#include "backend/optimizer/ascend/ascend_backend_optimization.h"
#include "backend/optimizer/graph_kernel/adapter/graph_kernel_optimization.h"
#include "utils/context/graph_kernel_flags.h"
#include "utils/utils.h"
#include "runtime/device/ascend/kernel_select_ascend.h"
#include "runtime/device/kernel_adjust.h"
#include "runtime/device/ascend/ascend_stream_assign.h"
#include "runtime/device/ascend/kernel_build_ascend.h"
#include "runtime/hardware/ascend/ascend_graph_optimization.h"
#include "backend/kernel_compiler/ascend_kernel_mod.h"
#include "backend/kernel_compiler/aicpu/aicpu_kernel_load.h"
#include "runtime/device/ascend/ascend_bucket.h"
#include "common/util/error_manager/error_manager.h"
#include "runtime/device/ascend/ascend_memory_adapter.h"

#ifndef ENABLE_SECURITY
#include "debug/data_dump/dump_json_parser.h"
#include "toolchain/adx_datadump_server.h"
#include "toolchain/adx_datadump_callback.h"
#include "debug/anf_ir_dump.h"
#include "debug/dump_proto.h"
#include "debug/data_dump/e2e_dump.h"
#include "debug/debugger/debugger_utils.h"
#endif
#ifdef ENABLE_DEBUGGER
#include "debug/tensor_load.h"
#include "debug/debugger/proto_exporter.h"
#else
#include "debug/debugger/proto_exporter_stub.h"
#endif
#ifdef ENABLE_DUMP_IR
#include "debug/rdr/running_data_recorder.h"
#include "debug/rdr/recorder_manager.h"
#include "debug/rdr/graph_recorder.h"
#endif

#ifndef ENABLE_SECURITY
#include "profiler/device/ascend/memory_profiling.h"
#include "runtime/device/ascend/profiling/profiling_manager.h"
#include "utils/anf_utils.h"

using Adx::AdxRegDumpProcessCallBack;
using mindspore::device::ascend::ProfilingManager;
using mindspore::profiler::ascend::MemoryProfiling;
#endif

namespace mindspore {
namespace device {
namespace ascend {
using KernelGraph = mindspore::session::KernelGraph;
const char kMsVm[] = "vm";
constexpr size_t kAtomicCleanInputSize = 2;
constexpr auto kUnknowErrorString = "Unknown error occurred";
constexpr auto kAscend910 = "ascend910";
namespace {
CNodePtr GetNextLabelSet(const std::vector<CNodePtr> &kernel_nodes, uint32_t index) {
  size_t node_sizes = kernel_nodes.size();
  if (index >= node_sizes - 1) {
    MS_LOG(EXCEPTION) << "there is no node after this node:" << kernel_nodes[index]->DebugString();
  }
  auto kernel = kernel_nodes[index + 1];
  if (AnfAlgo::GetCNodeName(kernel) != kLabelSetOpName) {
    MS_LOG(EXCEPTION) << "the node is not labelset follow labelgoto/labelswitch, node: "
                      << kernel_nodes[index]->DebugString();
  }
  return kernel;
}

std::vector<CNodePtr> HandleRecursiveCall(const std::vector<CNodePtr> &kernel_cnodes, const uint32_t &back_label,
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
    if (AnfAlgo::HasNodeAttr(kAttrRecursiveEnd, kernel_cnodes[i])) {
      *index = i;
      back->insert(back->end(), back_temp.begin(), back_temp.end());
      return front;
    }
    if (AnfAlgo::HasNodeAttr(kAttrRecursive, kernel_cnodes[i])) {
      back_flag = true;
      if (!AnfAlgo::IsLabelIndexInNode(kernel_cnodes[i], back_label)) {
        auto temp = HandleRecursiveCall(kernel_cnodes, back_label, &(++i), &back_temp);
        front.insert(front.end(), temp.begin(), temp.end());
      }
    }
    i++;
  }
  return front;
}

void UnfoldRecursiveExecOrder(KernelGraph *kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  if (!kernel_graph->recursive_call()) {
    return;
  }
  auto kernel_cnodes = kernel_graph->mem_reuse_exec_order();
  std::vector<CNodePtr> mem_reuse_order;
  mem_reuse_order.reserve(kernel_cnodes.size());
  for (uint32_t i = 0; i < kernel_cnodes.size(); i++) {
    if (!AnfAlgo::HasNodeAttr(kAttrRecursiveStart, kernel_cnodes[i])) {
      mem_reuse_order.emplace_back(kernel_cnodes[i]);
      continue;
    }
    auto label_id = AnfAlgo::GetNodeAttr<uint32_t>(kernel_cnodes[i], kAttrLabelIndex);
    std::vector<CNodePtr> back;
    auto front = HandleRecursiveCall(kernel_cnodes, label_id, &i, &back);
    mem_reuse_order.insert(mem_reuse_order.end(), front.begin(), front.end());
    mem_reuse_order.insert(mem_reuse_order.end(), back.begin(), back.end());
  }
  kernel_graph->set_mem_reuse_exec_order(mem_reuse_order);
}

void GetSubGraphExecOrder(const KernelGraph *kernel_graph, uint32_t index, const CNodePtr &back_node,
                          std::vector<CNodePtr> *mem_reuse_order) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(mem_reuse_order);
  auto label_id = AnfAlgo::GetNodeAttr<uint32_t>(back_node, kAttrLabelIndex);
  auto kernel_cnodes = kernel_graph->execution_order();
  for (auto i = index; i < kernel_cnodes.size(); i++) {
    mem_reuse_order->emplace_back(kernel_cnodes[i]);
    if (AnfAlgo::IsLabelIndexInNode(kernel_cnodes[i], label_id)) {
      return;
    }
  }
}

void InitMemReuseExecOrder(KernelGraph *kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  if (!kernel_graph->subgraph_multi_call()) {
    return;
  }
  std::unordered_map<uint32_t, uint32_t> label_id_index_map;
  auto kernel_cnodes = kernel_graph->execution_order();
  std::vector<CNodePtr> mem_reuse_order;
  for (uint32_t i = 0; i < kernel_cnodes.size(); i++) {
    mem_reuse_order.emplace_back(kernel_cnodes[i]);
    if (AnfAlgo::CheckPrimitiveType(kernel_cnodes[i], prim::kPrimLabelSwitch) &&
        !AnfAlgo::HasNodeAttr(kAttrRecursive, kernel_cnodes[i]) &&
        !AnfAlgo::HasNodeAttr(kAttrReturn, kernel_cnodes[i])) {
      auto label_list = AnfAlgo::GetNodeAttr<std::vector<uint32_t>>(kernel_cnodes[i], kAttrLabelSwitchList);
      for (auto label_id : label_list) {
        if (label_id_index_map.find(label_id) == label_id_index_map.end()) {
          continue;
        }
        auto back_node = GetNextLabelSet(kernel_cnodes, i);
        GetSubGraphExecOrder(kernel_graph, label_id_index_map[label_id], back_node, &mem_reuse_order);
      }
      continue;
    }
    if (AnfAlgo::CheckPrimitiveType(kernel_cnodes[i], prim::kPrimLabelGoto) &&
        !AnfAlgo::HasNodeAttr(kAttrRecursive, kernel_cnodes[i]) &&
        !AnfAlgo::HasNodeAttr(kAttrReturn, kernel_cnodes[i])) {
      auto label_id = AnfAlgo::GetNodeAttr<uint32_t>(kernel_cnodes[i], kAttrLabelIndex);
      if (label_id_index_map.find(label_id) == label_id_index_map.end()) {
        continue;
      }
      auto back_node = GetNextLabelSet(kernel_cnodes, i);
      GetSubGraphExecOrder(kernel_graph, label_id_index_map[label_id], back_node, &mem_reuse_order);
      continue;
    }
    if (AnfAlgo::CheckPrimitiveType(kernel_cnodes[i], prim::kPrimLabelSet) &&
        !AnfAlgo::HasNodeAttr(kAttrRecursive, kernel_cnodes[i])) {
      auto label_id = AnfAlgo::GetNodeAttr<uint32_t>(kernel_cnodes[i], kAttrLabelIndex);
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
}  // namespace
#ifndef ENABLE_SECURITY
void DumpInit(uint32_t device_id) {
  auto &json_parser = DumpJsonParser::GetInstance();
  json_parser.Parse();
  json_parser.CopyDumpJsonToDir(device_id);
  json_parser.CopyHcclJsonToDir(device_id);
  json_parser.CopyMSCfgJsonToDir(device_id);
  if (json_parser.async_dump_enabled()) {
#ifdef ENABLE_D
    // register callback to adx
    if (json_parser.FileFormatIsNpy()) {
      AdxRegDumpProcessCallBack(DumpDataCallBack);
    }
#endif
    if (AdxDataDumpServerInit() != 0) {
      MS_LOG(EXCEPTION) << "Adx data dump server init failed";
    }
  }
}
#endif

void AscendDeviceContext::Initialize() {
  MS_LOG(INFO) << "Status record: Enter Initialize...";
  if (initialized_) {
    MS_EXCEPTION_IF_NULL(runtime_instance_);
    runtime_instance_->SetContext();
    return;
  }

  MS_LOG(INFO) << "Status record: Initialize start...";
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  runtime_instance_ = dynamic_cast<AscendKernelRuntime *>(
    device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id));
  MS_EXCEPTION_IF_NULL(runtime_instance_);
  if (!runtime_instance_->Init()) {
    MS_LOG(EXCEPTION) << "Kernel runtime init error.";
  }
  mem_manager_ = runtime_instance_->GetMemoryManager();
  MS_EXCEPTION_IF_NULL(mem_manager_);

  auto env_rank_id = common::GetEnv("RANK_ID");
  if (ms_context->get_param<bool>(MS_CTX_ENABLE_HCCL) && !env_rank_id.empty()) {
    // get actual rank id if it's distribution training case.
    rank_id_ = GetRankId();
  }
#ifndef ENABLE_SECURITY
  DumpInit(rank_id_);
#endif
  compute_stream_ = runtime_instance_->compute_stream();
  communication_stream_ = runtime_instance_->communication_stream();

  initialized_ = true;
  MS_LOG(INFO) << "Status record: Initialize success.";
}

bool AscendDeviceContext::IsGraphMode() {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  return context->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode;
}

void AscendDeviceContext::Destroy() {
  MS_LOG(INFO) << "Status record: Enter Destroy...";
  if (!initialized_) {
    return;
  }
  MS_LOG(INFO) << "Status record: Destroy start...";
  rank_id_ = 0;
  if (runtime_instance_) {
    // TODO(lzlang): Destroy runtime instance after fully support MindRT, otherwise runtime will be destructed
    // repeatedly. runtime_instance_->ReleaseDeviceRes();
    runtime_instance_ = nullptr;
  }
  AscendGraphOptimization::GetInstance().Clear();
  initialized_ = false;
  MS_LOG(INFO) << "Status record: Destroy success.";
}

std::vector<GraphSegmentPtr> AscendDeviceContext::PartitionGraph(
  const FuncGraphPtr &func_graph, const std::vector<GraphSegmentPtr> &default_partition_segments) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  // TODO(lzlang): delete
  if (!(common::GetEnv("ENABLE_ASCEND_KERNEL_MINDRT") == "1" || common::kEnableAscendKernelByKernel)) {
    std::string backend = context_ptr->backend_policy();
    auto task_sink = context_ptr->get_param<bool>(MS_CTX_ENABLE_TASK_SINK);
    if (func_graph->ContainMultiTarget() || !task_sink) {
      context_ptr->set_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK, false);
      context_ptr->set_param<bool>(MS_CTX_ENABLE_LOOP_SINK, false);
    } else if (context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE) != kPynativeMode) {
      std::string device_target = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
      auto manager = func_graph->manager();
      MS_EXCEPTION_IF_NULL(manager);
      auto graphs = manager->func_graphs();
      bool exist_while =
        std::any_of(graphs.cbegin(), graphs.cend(), [](const FuncGraphPtr &fg) { return fg->recursive(); });
      if (device_target == kAscendDevice && backend != kMsVm && !exist_while) {
        MS_LOG(INFO) << "Run graph mode with multigraph sink.";
        context_ptr->set_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK, true);
      } else {
        MS_LOG(INFO) << "Run graph mode with vm.";
        context_ptr->set_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK, false);
        context_ptr->set_param<bool>(MS_CTX_ENABLE_LOOP_SINK, false);
      }
    }
  }
  if (context_ptr->get_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK)) {
    return std::vector<GraphSegmentPtr>();
  }
  return default_partition_segments;
}

void AscendDeviceContext::UnifyMindIR(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  AscendGraphOptimization::GetInstance().UnifyMindIR(graph);
}

void AscendDeviceContext::OptimizeGraph(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  AscendGraphOptimization::GetInstance().OptimizeGraph(graph);
}

void AscendDeviceContext::SetOperatorInfo(const std::vector<CNodePtr> &nodes) const {
  AscendGraphOptimization::GetInstance().SetOperatorInfo(nodes);
}

void AscendDeviceContext::CreateKernel(const std::vector<CNodePtr> &nodes) const {
  MS_LOG(INFO) << "Status record: start create kernel.";
  struct timeval start_time, end_time;
  (void)gettimeofday(&start_time, nullptr);
  auto ret = device::ascend::KernelBuild(nodes);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Kernel build error.";
  }
  (void)gettimeofday(&end_time, nullptr);
  const uint64_t kUSecondInSecond = 1000000;
  uint64_t cost = kUSecondInSecond * static_cast<uint64_t>(end_time.tv_sec - start_time.tv_sec);
  cost += static_cast<uint64_t>(end_time.tv_usec - start_time.tv_usec);
  MS_LOG(INFO) << "CreateKernel finish run in " << cost << " us.";
  MS_LOG(INFO) << "Status record: end create kernel.";
}

void AscendDeviceContext::UpdateExecOrder(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<CNodePtr> new_orders;
  auto nodes = graph->execution_order();
  for (const auto &node : nodes) {
    if (node_atomics_.find(node) != node_atomics_.end()) {
      auto atomics = node_atomics_[node];
      (void)std::copy(atomics.begin(), atomics.end(), std::back_inserter(new_orders));
    }
    new_orders.push_back(node);
  }
  graph->set_execution_order(new_orders);
  node_atomics_.clear();
}

void AscendDeviceContext::GenKernelEvents(const NotNull<KernelGraphPtr> &root_graph) const {
  MS_LOG(INFO) << "Start GenKernelEvents for graph " << root_graph->graph_id();
  MS_EXCEPTION_IF_NULL(runtime_instance_);
  runtime_instance_->GenKernelEvents(*root_graph.get());
  MS_LOG(INFO) << "Finish!";
}

void AscendDeviceContext::PreprocessBeforeRunGraph(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Status record: start preprocess before run graph. graph id: " << graph->graph_id();
  if (graph->is_executing_sink()) {
    device::ascend::InsertAtomicCleanOps(graph->execution_order(), &node_atomics_);
    UpdateExecOrder(graph);
    device::KernelAdjust::GetInstance().InsertDeviceLoopCtrl(graph);
    device::KernelAdjust::GetInstance().ProcessLoopSink(graph);
    AscendStreamAssign::GetInstance().AssignStream(NOT_NULL(graph));
#ifndef ENABLE_SECURITY
    // Insert profiling point, this function must be executed after assign stream.
    device::KernelAdjust::GetInstance().Profiling(NOT_NULL(graph.get()));
#endif
    CreateKernel(graph->execution_order());
    AllocateGraphMemory(NOT_NULL(graph));
    LoadModel(NOT_NULL(graph));
    AssignOutputNopNodeDeviceAddress(graph);
  } else {
    PreprocessBeforeRunSingleOpGraph(graph);
    AscendStreamAssign::GetInstance().AssignStream(NOT_NULL(graph));
    GenKernelEvents(NOT_NULL(graph));
  }

  MS_LOG(INFO) << "Status record: end preprocess before run graph. graph id: " << graph->graph_id();
}

void AscendDeviceContext::AssignOutputNopNodeDeviceAddress(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  auto outputs = AnfAlgo::GetAllOutput(graph->output(), {prim::kPrimTupleGetItem});
  for (auto output : outputs) {
    if (!output->isa<CNode>() || !AnfUtils::IsRealKernel(output)) {
      continue;
    }

    if (!opt::IsNopNode(output)) {
      continue;
    }

    if (!AnfAlgo::IsNeedSkipNopOpAddr(output)) {
      continue;
    }

    size_t input_num = AnfAlgo::GetInputTensorNum(output);
    if (input_num != 1) {
      MS_LOG(WARNING) << "The input number of nop node :" << output->fullname_with_scope() << " is " << input_num
                      << ", not equal 1";
      continue;
    }

    auto real_input_index = AnfAlgo::GetRealInputIndex(output, 0);
    auto pre_node_out_device_address = AnfAlgo::GetPrevNodeOutputAddr(output, real_input_index);
    MS_EXCEPTION_IF_NULL(pre_node_out_device_address);
    auto ptr = pre_node_out_device_address->GetPtr();
    auto size = pre_node_out_device_address->GetSize();
    std::string output_format = AnfAlgo::GetOutputFormat(output, 0);
    auto output_type = AnfAlgo::GetOutputDeviceDataType(output, 0);
    auto device_address = CreateDeviceAddress(const_cast<void *>(ptr), size, output_format, output_type);
    device_address->set_is_ptr_persisted(true);
    device_address->set_host_shape(trans::GetRuntimePaddingShape(output, 0));
    AnfAlgo::SetOutputAddr(device_address, 0, output.get());
    AnfAlgo::SetNodeAttr(kAttrSkipNopOpAddr, MakeValue(false), output);
    MS_LOG(INFO) << "Assign device address to output nop node " << output->fullname_with_scope();
  }
}

void AscendDeviceContext::AllocateGraphMemory(const NotNull<KernelGraphPtr> &root_graph) const {
  MS_LOG(INFO) << "Status record: start memory alloc. graph id: " << root_graph->graph_id();
  MS_EXCEPTION_IF_NULL(runtime_instance_);
  runtime_instance_->ClearGlobalIdleMem();
  memo_.clear();
  mem_manager_->ResetDynamicMemory();
  AssignInputMemory(root_graph, NOT_NULL(&memo_));
  device::KernelAdjust::GetInstance().AssignLoopCtrlMemory(*root_graph.get());
  InitMemReuseExecOrder(root_graph.get().get());
  runtime_instance_->AssignStaticMemoryOutput(*root_graph.get());
  runtime_instance_->AssignDynamicMemory(*root_graph.get());
  runtime_instance_->UpdateRefNodeOutputMem(*root_graph.get());

  MS_LOG(INFO) << "Status record: end memory alloc. graph id: " << root_graph->graph_id()
               << ", Memory Statistics: " << device::ascend::AscendMemAdapter::GetInstance().DevMemStatistics();
  MS_LOG(INFO) << "The dynamic memory pool total size is: "
               << device::ascend::AscendMemoryPool::GetInstance().TotalMemStatistics() / kMBToByte
               << "M, total used size is "
               << device::ascend::AscendMemoryPool::GetInstance().TotalUsedMemStatistics() / kMBToByte
               << "M, used peak size is "
               << device::ascend::AscendMemoryPool::GetInstance().UsedMemPeakStatistics() / kMBToByte << "M.";

#ifndef ENABLE_SECURITY
  if (MemoryProfiling::GetInstance().IsMemoryProfilingInitialized()) {
    uint64_t mem_size = runtime_instance_->GetMsUsedHbmSize();
    MemoryProfiling::GetInstance().SetDeviceMemSize(mem_size);
    if (MemoryProfiling::GetInstance().NeedSaveMemoryProfiling()) {
      MemoryProfiling::GetInstance().SaveMemoryProfiling();
    }
  }
#endif
}

void AscendDeviceContext::AssignInputMemory(const NotNull<KernelGraphPtr> &graph,
                                            NotNull<std::set<KernelGraphPtr> *> const memo) const {
  if (memo->find(graph) != memo->end()) {
    return;
  }
  memo->insert(graph.get());

  MS_LOG(INFO) << "Start to assign static memory for Parameter and Value node in graph: " << graph->graph_id();
  runtime_instance_->AssignStaticMemoryInput(*graph.get());
  runtime_instance_->AssignStaticMemoryValueNode(*graph.get());
  for (auto &child_graph : graph->child_graph_order()) {
    AssignInputMemory(NOT_NULL(child_graph.lock()), memo);
  }
  MS_LOG(INFO) << "Finish assigning static memory for Parameter and Value node in graph: " << graph->graph_id();
}

void AscendDeviceContext::LoadModel(const NotNull<KernelGraphPtr> &root_graph) const {
  MS_LOG(INFO) << "Start LoadModel for graph " << root_graph->graph_id();
  MS_EXCEPTION_IF_NULL(runtime_instance_);
  bool ret_ok = runtime_instance_->Load(*root_graph.get(), true);
  if (!ret_ok) {
    MS_LOG(EXCEPTION) << "Load task error!";
  }
  MS_LOG(INFO) << "Finish!";
}

bool AscendDeviceContext::AllocateMemory(DeviceAddress *const &address, size_t size) const {
  MS_EXCEPTION_IF_NULL(address);
  MS_EXCEPTION_IF_NULL(runtime_instance_);
  runtime_instance_->SetContext();
  auto device_ptr = mem_manager_->MallocMemFromMemPool(size, address->from_persistent_mem_);
  if (!device_ptr) {
    return false;
  }
  address->ptr_ = device_ptr;
  address->size_ = size;
  address->from_mem_pool_ = true;
  return true;
}

void AscendDeviceContext::FreeMemory(DeviceAddress *const &address) const {
  MS_EXCEPTION_IF_NULL(address);
  MS_EXCEPTION_IF_NULL(address->ptr_);
  if (!address->from_mem_pool()) {
    return;
  }
  mem_manager_->FreeMemFromMemPool(address->ptr_);
  address->ptr_ = nullptr;
}

bool AscendDeviceContext::AllocateContinuousMemory(const std::vector<DeviceAddressPtr> &addr_list, size_t total_size,
                                                   const std::vector<size_t> &size_list) const {
  MS_EXCEPTION_IF_NULL(runtime_instance_);
  runtime_instance_->SetContext();
  return mem_manager_->MallocContinuousMemFromMemPool(addr_list, total_size, size_list);
}

void *AscendDeviceContext::AllocateMemory(size_t size) const {
  MS_EXCEPTION_IF_NULL(runtime_instance_);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  runtime_instance_->SetContext();
  return mem_manager_->MallocMemFromMemPool(size, false);
}

void AscendDeviceContext::FreeMemory(void *const ptr) const {
  MS_EXCEPTION_IF_NULL(ptr);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  mem_manager_->FreeMemFromMemPool(ptr);
}

bool AscendDeviceContext::ExecuteGraph(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  const uint64_t kUSecondInSecond = 1000000;
  bool ret = false;
  if (graph->is_executing_sink()) {
#if defined(_WIN32) || defined(_WIN64)
    auto start_time = std::chrono::steady_clock::now();
#else
    struct timeval start_time {};
    struct timeval end_time {};
    (void)gettimeofday(&start_time, nullptr);
#endif
    MS_EXCEPTION_IF_NULL(runtime_instance_);
    {
      std::lock_guard<std::mutex> locker(launch_mutex_);
      ret = runtime_instance_->RunTask(*graph);
    }
#if defined(_WIN32) || defined(_WIN64)
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::ratio<1, kUSecondInSecond>> cost = end_time - start_time;
    MS_LOG(INFO) << "Call MS Run Success in " << cost.count() << " us";
#else
    (void)gettimeofday(&end_time, nullptr);
    uint64_t cost = kUSecondInSecond * static_cast<uint64_t>(end_time.tv_sec - start_time.tv_sec);
    cost += static_cast<uint64_t>(end_time.tv_usec - start_time.tv_usec);
    MS_LOG(INFO) << "Call MS Run Success in " << cost << " us";
#endif
  } else {
    MS_LOG(EXCEPTION) << graph->ToString() << " does not sink, should launch kernels";
  }
  return ret;
}

bool AscendDeviceContext::LaunchGraph(const KernelGraphPtr &graph) const {
  MS_LOG(INFO) << "Status record: start launch graph. graph id: " << graph->graph_id();
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(runtime_instance_);
  runtime_instance_->SetContext();
  device::KernelAdjust::GetInstance().LoadDeviceLoopCtrlParameters(graph);
  auto ret = ExecuteGraph(graph);
  if (!ret) {
    MS_LOG(ERROR) << "run task error!";
    ReportErrorMessage();
    return ret;
  }
  ReportWarningMessage();
  MS_LOG(INFO) << "Status record: end launch graph. graph id: " << graph->graph_id();
  return ret;
}

void AscendDeviceContext::ReportErrorMessage() const {
  const string &error_message = ErrorManager::GetInstance().GetErrorMessage();
  if (!error_message.empty() && error_message.find(kUnknowErrorString) == string::npos) {
    MS_LOG(ERROR) << "Ascend error occurred, error message:\n" << error_message;
  }
}

void AscendDeviceContext::ReportWarningMessage() const {
  const string &warning_message = ErrorManager::GetInstance().GetWarningMessage();
  if (!warning_message.empty()) {
    MS_LOG(WARNING) << "Ascend warning message:\n" << warning_message;
  }
}

bool AscendDeviceContext::SyncStream(size_t stream_id) const {
  MS_EXCEPTION_IF_NULL(runtime_instance_);
  return runtime_instance_->SyncStream();
}

bool AscendDeviceContext::IsExecutingSink(const KernelGraphPtr &graph) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  return ms_context->get_param<bool>(MS_CTX_ENABLE_TASK_SINK) && IsGraphMode();
}

bool AscendDeviceContext::IsLoopCountSink(const KernelGraphPtr &graph) const {
  return device::KernelAdjust::NeedLoopSink() && IsGraphMode();
}

// kernel by kernel mode interface
void AscendDeviceContext::OptimizeSingleOpGraph(const KernelGraphPtr &graph) const {
  AscendGraphOptimization::GetInstance().OptimizeSingleOpGraph(graph);
}

void AscendDeviceContext::PreprocessBeforeRunSingleOpGraph(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  const auto &nodes = graph->execution_order();

  for (const auto &node : nodes) {
    // Remove placeholder
    auto op_name = AnfAlgo::GetCNodeName(node);
    static const std::set<std::string> place_holder_nodes = {kDynamicRNNOpName, kDynamicGRUV2OpName};
    auto iter = place_holder_nodes.find(op_name);
    if (iter != place_holder_nodes.end()) {
      auto none_index = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(node, kAttrPlaceHolderIndex);
      // Remove seq_length
      auto input_num = AnfAlgo::GetInputTensorNum(node);
      std::vector<AnfNodePtr> new_inputs = {AnfAlgo::GetCNodePrimitiveNode(node)};
      for (size_t i = 0; i < input_num; ++i) {
        auto item = std::find(none_index.begin(), none_index.end(), i);
        if (item == none_index.end()) {
          auto input_node = AnfAlgo::GetInputNode(node, i);
          new_inputs.emplace_back(input_node);
        }
      }
      node->set_inputs(new_inputs);
    }

    // Save the nop_op that needs to be memcpy
    if (op_name == prim::kPrimTranspose->name() && AnfAlgo::HasNodeAttr(kAttrNopOp, node)) {
      nop_op_to_memcpy_.insert(node);
    }
  }

  device::ascend::InsertAtomicCleanOps(nodes, &node_atomics_persistent_cache_);
  std::vector<CNodePtr> atomic_nodes;
  for (const auto &node : nodes) {
    auto iter = node_atomics_persistent_cache_.find(node);
    if (iter != node_atomics_persistent_cache_.end()) {
      const auto &atomics = iter->second;
      std::copy(atomics.begin(), atomics.end(), std::back_inserter(atomic_nodes));
    }
  }

  CreateKernel(atomic_nodes);

  if (!mindspore::kernel::AicpuOpKernelLoad::GetInstance().LaunchAicpuKernelSo()) {
    MS_LOG(EXCEPTION) << "Cust aicpu kernel so load failed.";
  }
}

void AscendDeviceContext::UpdateDynamicShape(const CNodePtr &kernel) const {}

std::shared_ptr<Bucket> AscendDeviceContext::CreateBucket(uint32_t bucket_id, uint32_t bucket_size) const {
  auto bucket = std::make_shared<AscendBucket>(bucket_id, bucket_size);
  MS_EXCEPTION_IF_NULL(bucket);

  bucket->Init({compute_stream_}, {communication_stream_});
  return bucket;
}

bool AscendDeviceContext::PySyncRuning() const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if ((ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) &&
      ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE) && !SyncStream()) {
    return false;
  }
  return true;
}

bool AscendDeviceContext::MemoryCopyAsync(const CNodePtr &node, const vector<AddressPtr> &inputs,
                                          const vector<AddressPtr> &outputs) const {
  MS_LOG(DEBUG) << "Launch MemoryCopyAsync instead for kernel " << node->fullname_with_scope();
  if (inputs.size() != 1 || outputs.size() != 1) {
    MS_LOG(ERROR) << "Kernel " << node->fullname_with_scope() << " input output size should be 1 but"
                  << " input size is:" << inputs.size() << " output size is:" << outputs.size();
    return false;
  }

  aclError status = aclrtMemcpyAsync(outputs[0]->addr, outputs[0]->size, inputs[0]->addr, inputs[0]->size,
                                     ACL_MEMCPY_DEVICE_TO_DEVICE, GetKernelStream(node));
  if (status != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "MemCpyAsync op aclrtMemcpyAsync failed, ret:" << status;
    return false;
  }
  return true;
}

bool AscendDeviceContext::LaunchCustomFunc(const AnfNodePtr &kernel) const {
  MS_EXCEPTION_IF_NULL(kernel);
  auto custom_func = AnfUtils::GetCustomFunc(kernel);
  BindDeviceToCurrentThread();
  custom_func(nullptr);
  return true;
}

void *AscendDeviceContext::GetKernelStream(const CNodePtr &node) const {
  auto kernel_mod = AnfAlgo::GetKernelMod(node);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
    return compute_stream_;
  } else {
    auto stream = kernel_mod->GetStream();
    if (stream == nullptr) {
      stream = compute_stream_;
      MS_LOG(INFO) << "Assign default compute stream for node " << node->fullname_with_scope();
    }
    return stream;
  }
}

bool AscendDeviceContext::GetKernelRealInputs(const CNodePtr &kernel, const vector<AddressPtr> &inputs,
                                              std::vector<AddressPtr> *real_inputs) const {
  auto input_num = AnfAlgo::GetInputTensorNum(kernel);
  if (input_num != inputs.size()) {
    MS_LOG(ERROR) << "Input num is " << input_num << " but input address num is " << inputs.size();
    return false;
  }

  for (size_t i = 0; i < input_num; ++i) {
    auto real_index = AnfAlgo::GetRealInputIndex(kernel, i);
    if (real_index >= input_num) {
      MS_LOG(ERROR) << "Total input num is " << input_num << " but get real_index " << real_index;
      return false;
    }
    real_inputs->push_back(inputs[real_index]);
  }
  return true;
}

bool AscendDeviceContext::LaunchKernel(const CNodePtr &kernel, const vector<AddressPtr> &inputs,
                                       const vector<AddressPtr> &workspace, const vector<AddressPtr> &outputs,
                                       bool is_dynamic_shape) const {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_LOG(DEBUG) << "Launch kernel: " << kernel->fullname_with_scope();
  BindDeviceToCurrentThread();

  auto event_funcs = runtime_instance_->GetKernelEventFuncs(kernel);

  std::vector<AddressPtr> real_inputs;
  bool ret = GetKernelRealInputs(kernel, inputs, &real_inputs);
  if (!ret) {
    MS_LOG(ERROR) << "Get real input fail for kernel " << kernel->fullname_with_scope();
    return false;
  }
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);

  // start launch
  std::lock_guard<std::mutex> locker(launch_mutex_);

  // launch pre events
  MS_LOG(DEBUG) << "Launch pre-events for kernel " << kernel->fullname_with_scope();
  for (auto &pre_event_func : event_funcs.first) {
    pre_event_func();
  }

  // launch atomic clean
  if (!LaunchAtomicClean(kernel, workspace, outputs)) {
    MS_LOG(ERROR) << "Launch AtomicClean failed, pre kernel full name: " << kernel->fullname_with_scope();
    return false;
  }

  // launch kernel
  if (nop_op_to_memcpy_.find(kernel) != nop_op_to_memcpy_.end()) {
    MemoryCopyAsync(kernel, real_inputs, outputs);
  } else {
    MS_LOG(DEBUG) << "Launch kernel " << kernel->fullname_with_scope();
    if (is_dynamic_shape) {
      kernel::AscendKernelMod *ascend_kernel = dynamic_cast<kernel::AscendKernelMod *>(kernel_mod);
      MS_EXCEPTION_IF_NULL(ascend_kernel);
      ascend_kernel->InitDynamicKernel(kernel, GetKernelStream(kernel));
      auto dynamic_kernel = ascend_kernel->DynamicKernel();
      MS_EXCEPTION_IF_NULL(dynamic_kernel);
      dynamic_kernel->InferShape();
      dynamic_kernel->UpdateArgs();
      dynamic_kernel->Execute();
      dynamic_kernel->PostExecute();
    } else {
      ret = kernel_mod->Launch(real_inputs, workspace, outputs, GetKernelStream(kernel));
      if (!ret) {
        MS_LOG(ERROR) << "Launch kernel failed, kernel full name: " << kernel->fullname_with_scope();
        return false;
      }
    }
  }

  // launch post event
  MS_LOG(DEBUG) << "Launch post-events for kernel " << kernel->fullname_with_scope();
  for (auto &post_event_func : event_funcs.second) {
    post_event_func();
  }

  return PySyncRuning();
}

void AscendDeviceContext::BindDeviceToCurrentThread() const {
  if (initialized_) {
    runtime_instance_->SetContext();
  }
}

bool AscendDeviceContext::LaunchAtomicClean(const CNodePtr &node, const std::vector<AddressPtr> &workspace,
                                            const std::vector<AddressPtr> &outputs) const {
  auto iter = node_atomics_persistent_cache_.find(node);
  if (iter == node_atomics_persistent_cache_.end()) {
    return true;
  }
  MS_LOG(DEBUG) << "Launch atomic clean for kernel " << node->fullname_with_scope();
  auto atomic_node = iter->second.at(0);
  vector<AddressPtr> atomic_inputs;
  // The output addr need to clean
  MS_EXCEPTION_IF_NULL(atomic_node);
  if (atomic_node->inputs().size() != kAtomicCleanInputSize) {
    MS_LOG(EXCEPTION) << "Atomic Addr clean Node Input nodes not equal 2.";
  }
  if (AnfAlgo::HasNodeAttr(kAttrAtomicOutputIndexs, node)) {
    auto clean_output_indexes = AnfAlgo::GetNodeAttr<std::vector<size_t>>(node, kAttrAtomicOutputIndexs);
    for (auto output_index : clean_output_indexes) {
      if (output_index >= outputs.size()) {
        MS_LOG(EXCEPTION) << "Invalid output_index:" << output_index << " except less than " << outputs.size();
      }
      atomic_inputs.push_back(outputs[output_index]);
    }
  }

  // The workspace addr need to clean
  if (AnfAlgo::HasNodeAttr(kAttrAtomicWorkspaceIndexs, node)) {
    auto clean_workspace_indexes = AnfAlgo::GetNodeAttr<std::vector<size_t>>(node, kAttrAtomicWorkspaceIndexs);
    for (auto workspace_index : clean_workspace_indexes) {
      if (workspace_index >= workspace.size()) {
        MS_LOG(EXCEPTION) << "Invalid workspace_index:" << workspace_index << " except less than " << workspace.size();
      }
      atomic_inputs.push_back(workspace[workspace_index]);
    }
  }
  // Launch Atomic Node
  auto kernel_mod = AnfAlgo::GetKernelMod(atomic_node);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  return kernel_mod->Launch(atomic_inputs, {}, {}, GetKernelStream(atomic_node));
}

MS_REGISTER_DEVICE(kAscendDevice, AscendDeviceContext);
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
