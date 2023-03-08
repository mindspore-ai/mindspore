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

#include "plugin/device/ascend/hal/hardware/ascend_graph_executor.h"
#include <unordered_map>
#include <algorithm>
#include "include/common/utils/utils.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include "include/backend/kernel_graph.h"
#include "plugin/device/ascend/hal/device/kernel_build_ascend.h"
#include "plugin/device/ascend/hal/device/kernel_adjust.h"
#include "plugin/device/ascend/hal/device/ascend_stream_assign.h"
#include "plugin/device/ascend/hal/device/ascend_memory_adapter.h"
#ifndef ENABLE_SECURITY
#include "plugin/device/ascend/hal/profiler/memory_profiling.h"
using mindspore::profiler::ascend::MemoryProfiling;
#endif

namespace mindspore {
namespace device {
namespace ascend {
namespace {
CNodePtr GetNextLabelSet(const std::vector<CNodePtr> &kernel_nodes, uint32_t index) {
  size_t node_sizes = kernel_nodes.size();
  if (index >= node_sizes - 1) {
    MS_LOG(EXCEPTION) << "there is no node after this node:" << kernel_nodes[index]->DebugString();
  }
  auto kernel = kernel_nodes[index + 1];
  if (common::AnfAlgo::GetCNodeName(kernel) != kLabelSetOpName) {
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
      (void)front.emplace_back(kernel_cnodes[i]);
    } else {
      (void)back->emplace_back(kernel_cnodes[i]);
    }
    if (common::AnfAlgo::HasNodeAttr(kAttrRecursiveEnd, kernel_cnodes[i])) {
      *index = i;
      (void)back->insert(back->end(), back_temp.begin(), back_temp.end());
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

void UnfoldRecursiveExecOrder(KernelGraph *kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  if (!kernel_graph->recursive_call()) {
    return;
  }
  auto kernel_cnodes = kernel_graph->mem_reuse_exec_order();
  std::vector<CNodePtr> mem_reuse_order;
  mem_reuse_order.reserve(kernel_cnodes.size());
  for (uint32_t i = 0; i < kernel_cnodes.size(); i++) {
    if (!common::AnfAlgo::HasNodeAttr(kAttrRecursiveStart, kernel_cnodes[i])) {
      (void)mem_reuse_order.emplace_back(kernel_cnodes[i]);
      continue;
    }
    auto label_id = common::AnfAlgo::GetNodeAttr<uint32_t>(kernel_cnodes[i], kAttrLabelIndex);
    std::vector<CNodePtr> back;
    auto index = i;
    auto front = HandleRecursiveCall(kernel_cnodes, label_id, &index, &back);
    (void)mem_reuse_order.insert(mem_reuse_order.end(), front.begin(), front.end());
    (void)mem_reuse_order.insert(mem_reuse_order.end(), back.begin(), back.end());
  }
  kernel_graph->set_mem_reuse_exec_order(mem_reuse_order);
}

void GetSubGraphExecOrder(const KernelGraph *kernel_graph, uint32_t index, const CNodePtr &back_node,
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
  std::unordered_map<uint32_t, uint32_t> label_id_index_map;
  auto kernel_cnodes = kernel_graph->execution_order();
  std::vector<CNodePtr> mem_reuse_order;
  for (uint32_t i = 0; i < kernel_cnodes.size(); i++) {
    (void)mem_reuse_order.emplace_back(kernel_cnodes[i]);
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

void EnableGraphInputZeroCopy(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  // Zero copy is only enabled for PyNative and Subgraph sink.
  if ((!graph->has_flag(kFlagPyNativeRunInGraph) && !graph->has_flag(kFlagEnableZeroCopyInGraph)) ||
      !graph->is_graph_run_mode()) {
    return;
  }
  const auto &input_nodes = graph->input_nodes();
  for (const auto &input : input_nodes) {
    MS_EXCEPTION_IF_NULL(input);
    if (AnfAlgo::OutputAddrExist(input, 0)) {
      auto input_address = AnfAlgo::GetMutableOutputAddr(input, 0);
      MS_EXCEPTION_IF_NULL(input_address);
      input_address->set_is_ptr_persisted(false);
      MS_LOG(INFO) << "Enable zero copy for input " << input->DebugString();
    }
  }
}

void EnableGraphOutputZeroCopy(const KernelGraphPtr &graph) {
  MS_LOG(DEBUG) << "EnableGraphOutputZeroCopy start";
  MS_EXCEPTION_IF_NULL(graph);
  if ((!graph->has_flag(kFlagEnableZeroCopyInGraph)) || !graph->is_graph_run_mode()) {
    MS_LOG(DEBUG) << "EnableGraphOutputZeroCopy start return";
    return;
  }
  // Zero copy is only enabled for subgraph sink.
  auto outputs = common::AnfAlgo::GetAllOutputWithIndex(graph->output());
  for (const auto &output : outputs) {
    const auto &node_with_index = common::AnfAlgo::FetchRealNodeSkipMonadControl(output);
    const auto &node = node_with_index.first;
    const auto &index = node_with_index.second;
    MS_EXCEPTION_IF_NULL(node);
    MS_LOG(DEBUG) << "EnableGraphOutputZeroCopy check node:" << node->DebugString();
    if (node->isa<CNode>() && AnfAlgo::OutputAddrExist(node, index)) {
      auto device_address = AnfAlgo::GetMutableOutputAddr(node, index, false);
      MS_EXCEPTION_IF_NULL(device_address);
      device_address->set_is_ptr_persisted(false);
      MS_LOG(DEBUG) << "Disable ptr persisted in output node:" << node->DebugString() << " index:" << index
                    << " address:" << device_address << " for graph:" << graph->ToString();
    }
  }
}
}  // namespace

void AscendGraphExecutor::Initialize() {
  res_manager_ = dynamic_cast<AscendDeviceResManager *>(device_context_->device_res_manager_.get());
  MS_EXCEPTION_IF_NULL(res_manager_);
  runtime_instance_ = res_manager_->runtime_instance_;
  MS_EXCEPTION_IF_NULL(runtime_instance_);
  mem_manager_ = res_manager_->mem_manager_;
  MS_EXCEPTION_IF_NULL(mem_manager_);
}

void AscendGraphExecutor::Destroy() {
  mem_manager_ = nullptr;
  runtime_instance_ = nullptr;
  res_manager_ = nullptr;
}

bool AscendGraphExecutor::RunGraph(const FuncGraphPtr &graph, const std::vector<tensor::Tensor> &inputs,
                                   std::vector<tensor::Tensor> *outputs,
                                   const std::map<string, string> &compile_options) {
  MS_EXCEPTION_IF_NULL(graph);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(INFO) << "Status record: start launch graph. graph id: " << kernel_graph->graph_id()
               << ", options:" << compile_options;
  PROF_START(launch_graph);
  MS_EXCEPTION_IF_NULL(runtime_instance_);
  runtime_instance_->SetContext();
  device::KernelAdjust::GetInstance().LoadDeviceLoopCtrlParameters(kernel_graph);
  auto ret = ExecuteGraph(kernel_graph);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Run task for graph:" << kernel_graph->ToString()
                      << " error! The details refer to 'Ascend Error Message'.";
  }
  if (auto warning_message = ErrorManagerAdapter::GetWarningMessage(true); !warning_message.empty()) {
    MS_LOG(WARNING) << warning_message;
  }
  PROF_END(launch_graph);
  MS_LOG(INFO) << "Status record: end launch graph. graph id: " << kernel_graph->graph_id();
  return ret;
}

void AscendGraphExecutor::PreprocessBeforeRun(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  device::ascend::InsertAtomicCleanOps(graph->execution_order(), &node_atomics_);
  UpdateExecOrder(graph);
  device::KernelAdjust::GetInstance().ProcessLoopSink(graph);
  AscendStreamAssign::GetInstance().AssignStream(NOT_NULL(graph));
#ifndef ENABLE_SECURITY
  // Insert profiling point, this function must be executed after assign stream.
  device::KernelAdjust::GetInstance().Profiling(NOT_NULL(graph.get()));
#endif
  device_context_->kernel_executor_->CreateKernel(graph->execution_order());
  AllocateGraphMemory(NOT_NULL(graph));
  LoadModel(NOT_NULL(graph));
  AssignOutputNopNodeDeviceAddress(graph, device_context_);
  EnableGraphInputZeroCopy(graph);
  EnableGraphOutputZeroCopy(graph);
}

void AscendGraphExecutor::UpdateExecOrder(const KernelGraphPtr &graph) const {
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

void AscendGraphExecutor::AllocateGraphMemory(const NotNull<KernelGraphPtr> &root_graph) const {
  MS_LOG(INFO) << "Status record: start memory alloc. graph id: " << root_graph->graph_id();
  PROF_START(graph_memory_alloc);
  MS_EXCEPTION_IF_NULL(runtime_instance_);
  runtime_instance_->ClearGlobalIdleMem();
  std::set<KernelGraphPtr> memo;
  memo.clear();
  mem_manager_->ResetDynamicMemory();
  AssignInputMemory(root_graph, NOT_NULL(&memo));
  device::KernelAdjust::GetInstance().AssignLoopCtrlMemory(*root_graph.get());
  InitMemReuseExecOrder(root_graph.get().get());
  runtime_instance_->SetReuseCommunicationAddress(*root_graph.get());
  runtime_instance_->AssignStaticMemoryOutput(*root_graph.get());
  runtime_instance_->AssignDynamicMemory(*root_graph.get());
  runtime_instance_->UpdateRefNodeOutputMem(*root_graph.get());

  PROF_END(graph_memory_alloc);
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

void AscendGraphExecutor::AssignInputMemory(const NotNull<KernelGraphPtr> &graph,
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

void AscendGraphExecutor::LoadModel(const NotNull<KernelGraphPtr> &root_graph) const {
  MS_LOG(INFO) << "Status record: start load model. graph id: " << root_graph->graph_id();
  PROF_START(load_model);
  MS_EXCEPTION_IF_NULL(runtime_instance_);
  bool ret_ok = runtime_instance_->Load(*root_graph.get(), true);
  if (!ret_ok) {
    MS_LOG(EXCEPTION) << "Load task error!";
  }
  PROF_END(load_model);
  MS_LOG(INFO) << "Status record: end load model. graph id: " << root_graph->graph_id();
}

bool AscendGraphExecutor::ExecuteGraph(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  const uint64_t kUSecondInSecond = 1000000;
  bool ret = false;
  if (graph->is_graph_run_mode()) {
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
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
