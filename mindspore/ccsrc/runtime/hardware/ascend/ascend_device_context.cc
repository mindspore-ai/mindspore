/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "backend/optimizer/ascend/ascend_backend_optimization.h"
#include "backend/optimizer/graph_kernel/graph_kernel_optimization.h"
#include "backend/session/ascend_auto_monad.h"
#include "utils/context/graph_kernel_flags.h"
#include "runtime/device/ascend/kernel_select_ascend.h"
#include "runtime/device/kernel_adjust.h"
#include "runtime/device/ascend/ascend_stream_assign.h"
#include "runtime/device/ascend/kernel_build_ascend.h"
#include "runtime/hardware/ascend/ascend_graph_optimization.h"

#ifndef ENABLE_SECURITY
#include "debug/data_dump/dump_json_parser.h"
#include "toolchain/adx_datadump_server.h"
#include "debug/anf_ir_dump.h"
#include "debug/dump_proto.h"
#include "debug/data_dump/e2e_dump.h"
#endif

namespace mindspore {
namespace device {
namespace ascend {
using KernelGraph = mindspore::session::KernelGraph;

#ifndef ENABLE_SECURITY
void DumpInit(uint32_t device_id) {
  auto &json_parser = DumpJsonParser::GetInstance();
  json_parser.Parse();
  json_parser.CopyDumpJsonToDir(device_id);
  json_parser.CopyHcclJsonToDir(device_id);
  json_parser.CopyMSCfgJsonToDir(device_id);
  if (json_parser.async_dump_enabled()) {
    if (AdxDataDumpServerInit() != 0) {
      MS_LOG(EXCEPTION) << "Adx data dump server init failed";
    }
  }
}

void DumpSetup(const KernelGraphPtr &graph) {
  MS_LOG(DEBUG) << "Start!";
  MS_EXCEPTION_IF_NULL(graph);
  E2eDump::DumpSetup(graph.get());
  MS_LOG(DEBUG) << "Finish!";
}

void Dump(const KernelGraphPtr &graph, uint32_t rank_id) {
  MS_LOG(DEBUG) << "Start!";
  MS_EXCEPTION_IF_NULL(graph);
  E2eDump::DumpData(graph.get(), rank_id);
  MS_LOG(DEBUG) << "Finish!";
}
#endif

void AscendDeviceContext::Initialize() {
  MS_LOG(INFO) << "Status record: Enter Initialize...";
  if (initialized_) {
    MS_EXCEPTION_IF_NULL(runtime_instance_);
    runtime_instance_->SetCurrentContext();
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
  initialized_ = true;
  MS_LOG(INFO) << "Status record: Initialize success.";
}

void AscendDeviceContext::Destroy() {
  MS_LOG(INFO) << "Status record: Enter Destroy...";
  if (!initialized_) {
    return;
  }
  MS_LOG(INFO) << "Status record: Destroy start...";
  rank_id_ = 0;
  if (runtime_instance_ != nullptr) {
    runtime_instance_->ReleaseDeviceRes();
    runtime_instance_ = nullptr;
  }
  initialized_ = false;
  MS_LOG(INFO) << "Status record: Destroy success.";
}

std::vector<GraphSegmentPtr> AscendDeviceContext::PartitionGraph(
  const FuncGraphPtr &func_graph, const std::vector<GraphSegmentPtr> &default_partition_segments) {
  return std::vector<GraphSegmentPtr>();
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
  MS_LOG(INFO) << "CreateKernel Start...";
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
  MS_LOG(INFO) << "CreateKernel finish run in  " << PRIu64 << " us " << cost;
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

void AscendDeviceContext::PreprocessBeforeRunGraph(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "PreprocessBeforeRunGraph Start for graph " << graph->graph_id();
  device::ascend::InsertAtomicCleanOps(graph->execution_order(), &node_atomics_);
  if (graph->is_executing_sink()) {
    UpdateExecOrder(graph);
    device::KernelAdjust::GetInstance().InsertDeviceLoopCtrl(graph);
    device::KernelAdjust::GetInstance().ProcessLoopSink(graph);
    AscendStreamAssign::GetInstance().AssignStream(NOT_NULL(graph));
    CreateKernel(graph->execution_order());
    AllocateGraphMemory(NOT_NULL(graph));
    LoadModel(NOT_NULL(graph));
    MS_LOG(INFO) << "PreprocessBeforeRunGraph success.";
    return;
  }
  MS_LOG(INFO) << "PreprocessBeforeRunGraph success.";
}

void AscendDeviceContext::AllocateGraphMemory(const NotNull<KernelGraphPtr> &root_graph) const {
  MS_EXCEPTION_IF_NULL(runtime_instance_);
  runtime_instance_->ClearGlobalIdleMem();
  memo_.clear();
  AssignInputMemory(root_graph, NOT_NULL(&memo_));
  device::KernelAdjust::GetInstance().AssignLoopCtrlMemory(*root_graph.get());
  runtime_instance_->AssignStaticMemoryOutput(*root_graph.get());
  mem_manager_->ResetDynamicMemory();
  runtime_instance_->AssignDynamicMemory(*root_graph.get());
  runtime_instance_->UpdateRefNodeOutputMem(*root_graph.get());
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
  runtime_instance_->SetCurrentContext();
  auto device_ptr = mem_manager_->MallocMemFromMemPool(size);
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
  runtime_instance_->SetCurrentContext();
  return mem_manager_->MallocContinuousMemFromMemPool(addr_list, total_size, size_list);
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
#ifndef ENABLE_SECURITY
    DumpSetup(graph);
#endif
    {
      std::lock_guard<std::mutex> locker(launch_mutex_);
      ret = runtime_instance_->RunTask(*graph);
    }
#ifndef ENABLE_SECURITY
    Dump(graph, GetRankID());
#endif
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
  runtime_instance_->SetCurrentContext();
  device::KernelAdjust::GetInstance().LoadDeviceLoopCtrlParameters(graph);
  auto ret = ExecuteGraph(graph);
  MS_LOG(INFO) << "Status record: end launch graph. graph id: " << graph->graph_id();
  return ret;
}

bool AscendDeviceContext::SyncStream(size_t stream_id) const {
  MS_EXCEPTION_IF_NULL(runtime_instance_);
  return runtime_instance_->SyncStream();
}
bool AscendDeviceContext::IsExecutingSink(const KernelGraphPtr &graph) const { return true; }
bool AscendDeviceContext::IsLoopCountSink(const KernelGraphPtr &graph) const { return true; }

// kernel by kernel mode interface
void AscendDeviceContext::OptimizeSingleOpGraph(const KernelGraphPtr &graph) const {
  MS_LOG(ERROR) << "!!! Ascend with MindRT not support kernel by kernel mode. !!! ";
}

void AscendDeviceContext::PreprocessBeforeRunSingleOpGraph(const KernelGraphPtr &graph) const {
  MS_LOG(ERROR) << "!!! Ascend with MindRT not support kernel by kernel mode. !!! ";
}

void AscendDeviceContext::UpdateDynamicShape(const CNodePtr &kernel) const {
  MS_LOG(ERROR) << "!!! Ascend with MindRT not support function UpdateDynamicShape. !!! ";
}

std::shared_ptr<Bucket> AscendDeviceContext::CreateBucket(uint32_t bucket_id, uint32_t bucket_size) const {
  MS_LOG(ERROR) << "!!! Ascend with MindRT not support function CreateBucket. !!! ";
  return DeviceContext::CreateBucket(bucket_id, bucket_size);
}

bool AscendDeviceContext::LaunchKernel(const CNodePtr &kernel, const vector<AddressPtr> &inputs,
                                       const vector<AddressPtr> &workspace, const vector<AddressPtr> &outputs,
                                       bool is_dynamic_shape) const {
  MS_LOG(ERROR) << "!!! Ascend with MindRT not support kernel by kernel mode. !!! ";
  return true;
}

bool AscendDeviceContext::BindDeviceToCurrentThread() const {
  runtime_instance_->SetCurrentContext();
  return true;
}

MS_REGISTER_DEVICE(kAscendDevice, AscendDeviceContext);
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
