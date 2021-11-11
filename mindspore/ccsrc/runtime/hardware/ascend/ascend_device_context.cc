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

namespace mindspore {
namespace device {
namespace ascend {
using KernelGraph = mindspore::session::KernelGraph;
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
  E2eDump::DumpRunIter(graph, rank_id);
  E2eDump::DumpData(graph.get(), rank_id);
  MS_LOG(DEBUG) << "Finish!";
}
#endif

void AscendDeviceContext::DumpAllGraphs(const std::vector<KernelGraphPtr> &all_graphs) const {
#ifdef ENABLE_DUMP_IR
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG);
  auto &json_parser = DumpJsonParser::GetInstance();
  json_parser.Parse();
  if (!save_graphs && !json_parser.e2e_dump_enabled() && !json_parser.async_dump_enabled() &&
      !mindspore::RecorderManager::Instance().RdrEnable()) {
    return;
  }
  for (auto &graph : all_graphs) {
    MS_EXCEPTION_IF_NULL(graph);
    std::string name = "graph_build." + std::to_string(graph->graph_id());
    DumpGraphParams dump_params = {true, static_cast<int>(kWholeStack)};
    (void)mindspore::RDR::RecordAnfGraph(SUBMODULE_ID, name, graph, dump_params, ".ir;.pb");
    if (save_graphs) {
      std::string file_name = "graph_build_" + std::to_string(graph->graph_id()) + ".ir";
      DumpIR(file_name, graph, true, kWholeStack);
      DumpIRProto(graph, "vm_build_" + std::to_string(graph->graph_id()));
      DumpIR("trace_code_graph", graph, true, kWholeStack);
    }
    std::string final_graph = "trace_code_graph_" + std::to_string(graph->graph_id());
    if (json_parser.e2e_dump_enabled() || json_parser.async_dump_enabled()) {
      std::string root_dir = json_parser.path() + "/rank_" + std::to_string(rank_id_);
      std::string target_dir = root_dir + "/graphs";
      std::string ir_file_path = target_dir + "/" + "ms_output_" + final_graph + ".ir";
      DumpIRProtoWithSrcInfo(graph, final_graph, target_dir, kDebugWholeStack);
      DumpIR("trace_code_graph", graph, true, kWholeStack, ir_file_path);
      DumpGraphExeOrder("ms_execution_order_graph_" + std::to_string(graph->graph_id()) + ".csv", root_dir,
                        graph->execution_order());
    }
  }
#endif
}

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
  if (runtime_instance_) {
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
  AssignOutputNopNodeDeviceAddress(graph);
  MS_LOG(INFO) << "PreprocessBeforeRunGraph success.";
}

void AscendDeviceContext::AssignOutputNopNodeDeviceAddress(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  auto outputs = AnfAlgo::GetAllOutput(graph->output(), {prim::kPrimTupleGetItem});
  for (auto output : outputs) {
    if (!output->isa<CNode>() || !AnfAlgo::IsRealKernel(output)) {
      continue;
    }

    if (!opt::IsNopNode(output)) {
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
    AnfAlgo::SetOutputAddr(device_address, 0, output.get());

    AnfAlgo::SetNodeAttr(kAttrSkipNopOpAddr, MakeValue(false), output);
    MS_LOG(INFO) << "Assign device address to output nop node " << output->fullname_with_scope();
  }
}

void AscendDeviceContext::AllocateGraphMemory(const NotNull<KernelGraphPtr> &root_graph) const {
  MS_EXCEPTION_IF_NULL(runtime_instance_);
  runtime_instance_->ClearGlobalIdleMem();
  memo_.clear();
  AssignInputMemory(root_graph, NOT_NULL(&memo_));
  device::KernelAdjust::GetInstance().AssignLoopCtrlMemory(*root_graph.get());
  InitMemReuseExecOrder(root_graph.get().get());
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
  runtime_instance_->SetContext();
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
  runtime_instance_->SetContext();
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
#ifndef ENABLE_SECURITY
    DumpSetup(graph);
#endif
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
  runtime_instance_->SetContext();
  return true;
}

MS_REGISTER_DEVICE(kAscendDevice, AscendDeviceContext);
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
