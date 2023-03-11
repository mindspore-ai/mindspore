/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "include/backend/debug/debugger/debugger.h"
#include <dirent.h>
#include <tuple>
#include <vector>
#include <algorithm>
#include <iostream>
#include <map>
#include <regex>
#include "include/backend/debug/data_dump/dump_json_parser.h"
#include "backend/common/session/session_basic.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "runtime/device/kernel_runtime.h"
#include "include/backend/debug/data_dump/e2e_dump.h"
#include "include/common/utils/config_manager.h"
#include "include/common/debug/env_config_parser.h"
#include "include/common/utils/comm_manager.h"
#include "runtime/hardware/device_context_manager.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/debug/anf_dump_utils.h"
#include "runtime/graph_scheduler/device_tensor_store.h"
#ifdef ENABLE_DEBUGGER
#include "debug/debugger/proto_exporter.h"
#endif
#include "include/backend/debug/debugger/proto_exporter.h"
#include "debug/debugger/debugger_utils.h"
#include "debug/debugger/grpc_client.h"
#include "debug/debug_services.h"
#include "runtime/device/ms_device_shape_transfer.h"

using debugger::Chunk;
using debugger::EventReply;
using debugger::GraphProto;
using debugger::ModelProto;
using debugger::Statistics;
using debugger::TensorProto;
using debugger::WatchCondition;
using debugger::WatchCondition_Condition_inf;
using debugger::WatchCondition_Condition_nan;
using debugger::WatchCondition_Parameter;
using debugger::WatchNode;
using debugger::WatchpointHit;
using mindspore::runtime::DeviceTensorStore;

namespace mindspore {

static constexpr auto g_chunk_size = 1024 * 1024 * 3;
static constexpr int32_t heartbeat_period_second = 30;

std::shared_ptr<Debugger> Debugger::GetInstance() {
  std::lock_guard<std::mutex> i_lock(instance_lock_);
  if (debugger_ == nullptr) {
    debugger_ = std::shared_ptr<Debugger>(new (std::nothrow) Debugger());
  }
  return debugger_;
}

Debugger::Debugger()
    : grpc_client_(nullptr),
      debug_services_(nullptr),
      heartbeat_thread_(nullptr),
      device_id_(0),
      device_target_(""),
      num_step_(0),
      debugger_enabled_(false),
      suspended_at_last_kernel_(false),
      run_level_(""),
      node_name_(""),
      cur_name_(""),
      training_done_(false),
      send_metadata_done_(false),
      received_new_graph_(false),
      is_dataset_graph_(false),
      partial_memory_(false),
      initial_suspend_(true),
      enable_heartbeat_(false),
      not_dataset_graph_sum_(0),
      ascend_kernel_by_kernel_(false),
      enable_debugger_called_(false),
      version_("") {
  CheckDebuggerEnabledParam();
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::string device_target = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  MS_LOG(INFO) << "Debugger got device_target: " << device_target;
  if (!CheckDebuggerEnabled()) {
    return;
  } else if (device_target == kCPUDevice) {
    MS_LOG(WARNING) << "Not enabling debugger. Debugger does not support CPU.";
  } else {
    // configure partial memory reuse
    partial_memory_ = CheckDebuggerPartialMemoryEnabled();

    // switch memory reuse on or off
    EnvConfigParser::GetInstance().SetSysMemreuse(partial_memory_);
    // print some message about memory reuse to user
    if (partial_memory_) {
      MS_LOG(WARNING)
        << "Partial Memory Reuse is enabled. Note: 1. Please only set watchpoints before running the first "
           "step. 2. Tensor values are only available for nodes that are watched by any watchpoint.";
    } else {
      MS_LOG(WARNING)
        << "Memory Reuse is disabled. Set environment variable MS_DEBUGGER_PARTIAL_MEM=1 to reduce memory "
           "usage for large models.";
    }
  }
}

void Debugger::Init(const uint32_t device_id, const std::string device_target) {
  // access lock for public method
  std::lock_guard<std::mutex> a_lock(access_lock_);
  // save device_id
  MS_LOG(INFO) << "Debugger got device_id: " << device_id;
  device_id_ = device_id;
  MS_LOG(INFO) << "Debugger got device_target: " << device_target;
  device_target_ = device_target;
  version_ = MSVERSION;
}

bool IsTypeDebuggerSupported(TypeId type) {
  if (type < TypeId::kNumberTypeEnd && type > TypeId::kNumberTypeBegin && type != kNumberTypeComplex64) {
    return true;
  }
  MS_LOG(INFO) << "Debugger does not support type: " << TypeIdLabel(type);
  return false;
}

void Debugger::EnableDebugger() {
  // reset some of the class members
  num_step_ = 0;
  debugger_enabled_ = false;
  enable_heartbeat_ = false;
  partial_memory_ = false;
  grpc_client_ = nullptr;
  debug_services_ = nullptr;
  heartbeat_thread_ = nullptr;
  enable_debugger_called_ = true;

  // see if dump using debugger backend is enabled
  bool dump_enabled = CheckDebuggerDumpEnabled();
  MS_LOG(INFO) << "dump using debugger backend = " << dump_enabled;

  // check if debugger enabled
  debugger_enabled_ = CheckDebuggerEnabled();
  MS_LOG(INFO) << "debugger_enabled_ = " << debugger_enabled_;

  if (!debugger_enabled_ && !dump_enabled) {
    MS_LOG(INFO) << "Not enabling debugger. Set environment variable ENABLE_MS_DEBUGGER=1 to enable debugger.";
    return;
  }

  if (debugger_enabled_) {
    // configure grpc host
    std::string env_host_str = common::GetEnv("MS_DEBUGGER_HOST");
    std::string host;
    if (!env_host_str.empty()) {
      if (CheckIp(env_host_str)) {
        MS_LOG(INFO) << "Getenv MS_DEBUGGER_HOST: " << env_host_str;
        host = env_host_str;
      } else {
        debugger_enabled_ = false;
        MS_EXCEPTION(ValueError) << "Environment variable MS_DEBUGGER_HOST isn't a valid IP address. "
                                    "Please set environment variable MS_DEBUGGER_HOST=x.x.x.x to a valid IP";
      }
    } else {
      MS_LOG(INFO) << "Environment variable MS_DEBUGGER_HOST doesn't exist. Using default debugger host: localhost";
      host = "localhost";
    }
    // configure grpc port
    std::string env_port_str = common::GetEnv("MS_DEBUGGER_PORT");
    std::string port;
    if (!env_port_str.empty()) {
      if (CheckPort(env_port_str)) {
        MS_LOG(INFO) << "Getenv MS_DEBUGGER_PORT: " << env_port_str;
        port = env_port_str;
      } else {
        debugger_enabled_ = false;
        MS_EXCEPTION(ValueError) << "Environment variable MS_DEBUGGER_PORT is not valid. Custom port ranging from 1 to "
                                    "65535";
      }
    } else {
      port = "50051";
      if (!CheckPort(port)) {
        MS_EXCEPTION(ValueError) << "Default MS_DEBUGGER_PORT is not valid. Custom port ranging from 1 to 65535";
      }
      MS_LOG(INFO) << "Environment variable MS_DEBUGGER_PORT doesn't exist. Using default debugger port: 50051";
    }
    // initialize grpc client
    grpc_client_ = std::make_unique<GrpcClient>(host, port);
    // initialize sending heartbeat
    heartbeat_thread_ = std::make_unique<std::thread>([this]() { SendHeartbeat(heartbeat_period_second); });
  }
  debug_services_ = std::make_unique<DebugServices>();
}

void Debugger::CheckDatasetSinkMode(const KernelGraphPtr &graph_ptr) {
  bool sink_mode =
    ConfigManager::GetInstance().dataset_mode() == DatasetMode::DS_SINK_MODE || graph_ptr->IsDatasetGraph();
  if (CheckDebuggerDumpEnabled() && sink_mode && device_target_ == kGPUDevice) {
    MS_EXCEPTION(NotSupportError)
      << "e2e_dump is not supported on GPU with dataset_sink_mode=True. Please set dataset_sink_mode=False";
  }

  if (CheckDebuggerEnabled() && sink_mode) {
    MS_EXCEPTION(NotSupportError)
      << "Debugger is not supported with dataset_sink_mode=True. Please set dataset_sink_mode=False";
  }
}

bool Debugger::CheckDebuggerDumpEnabled() const {
  // see if dump is enabled
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  if (device_target_ == kGPUDevice) {
    return dump_json_parser.e2e_dump_enabled();
  } else if (device_target_ == kAscendDevice) {
    return dump_json_parser.async_dump_enabled() || dump_json_parser.e2e_dump_enabled();
  }
  return false;
}

bool Debugger::CheckDebuggerEnabled() const {
  // get env variables to configure debugger
  std::string env_enable_str = common::GetEnv("ENABLE_MS_DEBUGGER");
  if (!env_enable_str.empty()) {
    (void)std::transform(env_enable_str.begin(), env_enable_str.end(), env_enable_str.begin(), ::tolower);
    if ((env_enable_str == "1" || env_enable_str == "true") && device_target_ != kCPUDevice) {
      return true;
    }
  }
  return false;
}

void Debugger::CheckDebuggerEnabledParam() const {
  // check the value of env variable ENABLE_MS_DEBUGGER
  std::string env_enable_str = common::GetEnv("ENABLE_MS_DEBUGGER");
  if (!env_enable_str.empty()) {
    (void)std::transform(env_enable_str.begin(), env_enable_str.end(), env_enable_str.begin(), ::tolower);
    if (env_enable_str != "0" && env_enable_str != "1" && env_enable_str != "false" && env_enable_str != "true") {
      MS_LOG(WARNING) << "Env variable ENABLE_MS_DEBUGGER should be True/False/1/0 (case insensitive), but get: "
                      << env_enable_str;
    }
  }
}

bool Debugger::CheckDebuggerPartialMemoryEnabled() const {
  std::string env_partial_mem_str = common::GetEnv("MS_DEBUGGER_PARTIAL_MEM");
  if (!env_partial_mem_str.empty()) {
    MS_LOG(INFO) << "Getenv MS_DEBUGGER_PARTIAL_MEM: " << env_partial_mem_str;
    if (env_partial_mem_str == "1") {
      return true;
    }
  }
  return false;
}

/*
 * Feature group: Dump, Online debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT
 * Description: Returns true if online debugger or dump is enabled.
 */
bool Debugger::DebuggerBackendEnabled() const { return CheckDebuggerDumpEnabled() || CheckDebuggerEnabled(); }

void Debugger::Reset() {
  // access lock for public method
  std::lock_guard<std::mutex> a_lock(access_lock_);
  // reset components
  if (heartbeat_thread_ && heartbeat_thread_->joinable()) {
    SetEnableHeartbeat(false);
    heartbeat_thread_->join();
    MS_LOG(INFO) << "Join Heartbeat thread.";
  }
  heartbeat_thread_ = nullptr;
  device_id_ = 0;
  device_target_ = "";
  num_step_ = 0;
  debugger_enabled_ = false;
  is_dataset_graph_ = false;
  partial_memory_ = false;
  graph_ptr_ = nullptr;
  grpc_client_ = nullptr;
  debug_services_ = nullptr;
  graph_proto_list_.clear();
  graph_ptr_list_.clear();
  graph_ptr_step_vec_.clear();
  executed_graph_ptr_set_.clear();
  parameters_mindRT_.clear();
  visited_root_graph_ids_.clear();
  MS_LOG(INFO) << "Release Debugger resource.";
}

/*
 * Feature group: Dump, Online debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: MindRT.
 * Description: Sets root_graph_id for all the graphs in the compiled graph list. Sets cur_root_graph_id_ and
 * prev_root_graph_id_ and calls PreExecute function for all the graphs.
 */
void Debugger::PreExecuteGraphDebugger(const std::vector<KernelGraphPtr> &graphs,
                                       const std::vector<AnfNodePtr> &origin_parameters_order) {
  // MindRTBackend for GPU and Ascend
  if (device_target_ == kCPUDevice) {
    return;
  }
  // Store graphs that are run in one step.
  graph_ptr_step_vec_ = graphs;
  parameters_mindRT_ = origin_parameters_order;
  prev_root_graph_id_ = cur_root_graph_id_;
  // set first run graph as the root graph
  cur_root_graph_id_ = graph_ptr_step_vec_[0]->graph_id();
  MS_LOG(DEBUG) << "Current root graph id: " << cur_root_graph_id_ << " prev_root_graph_id_: " << prev_root_graph_id_
                << " for step: " << num_step_ << ".";
  MS_LOG(DEBUG) << "Set root graph for all the subgraphs:";
  for (size_t graph_index = 0; graph_index < graphs.size(); ++graph_index) {
    const auto &graph = graphs[graph_index];
    // set root graph id for GPU mindrt runtime.
    MS_LOG(INFO) << "Set root graph for graph: " << graph->graph_id() << " to: " << cur_root_graph_id_ << ".";
    graph->set_root_graph_id(cur_root_graph_id_);
    if (debugger_) {
      debugger_->PreExecute(graph);
    }
  }
}

/*
 * Feature group: Dump.
 * Target device group: Ascend.
 * Runtime category: Old runtime, MindRT.
 * Description: When async dump is enabled and dataset_sink_mode is true, graph_iter_num_map_ stores the number of
 * iterations per epoch for each running graph.
 */
void Debugger::UpdateGraphIterMap(uint32_t graph_id, int32_t iter_num) {
  if (graph_iter_num_map_.find(graph_id) == graph_iter_num_map_.end()) {
    graph_iter_num_map_[graph_id] = iter_num;
  }
}

/*
 * Feature group: Dump, Online debugger.
 * Target device group: Ascend.
 * Runtime category: Old runtime.
 * Description: For Ascend old runtime, this function sets the current and previous root graph id.
 */
void Debugger::SetCurrentAndPrevRootGraph(uint32_t root_graph_id) {
  // for GPU and ascend MindRT root graphs are set in PreExecuteGraphDebugger.
  if (device_target_ != kAscendDevice || MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
    return;
  }
  prev_root_graph_id_ = cur_root_graph_id_;
  cur_root_graph_id_ = root_graph_id;
  MS_LOG(DEBUG) << "Current root graph id: " << cur_root_graph_id_ << " prev_root_graph_id_: " << prev_root_graph_id_
                << " for step: " << num_step_ << ".";
}

/*
 * Feature group: Dump, Online debugger.
 * Target device group: GPU.
 * Runtime category: Old runtime.
 * Description: In the case of GPU old runtime and when we have multiple subgraphs, we use the first run graph id to
 * update the step number.
 */
void Debugger::StoreRunGraphIdList(uint32_t graph_id) {
  // collect rungrap_ids to update step number in multigraph case for GPU old runtime
  if (rungraph_id_list_.size() > 0) {
    rungraph_id_list_.push_back(graph_id);
  } else {
    if (std::find(rungraph_id_list_.begin(), rungraph_id_list_.end(), graph_id) == rungraph_id_list_.end()) {
      rungraph_id_list_.push_back(graph_id);
    }
  }
}

/*
 * Feature group: Dump, Online debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Sets previous and current root_graph_id for Ascend old runtime, sends graphs to online debugger when
 * debugger_enabled_ is true.
 */
void Debugger::PreExecute(const KernelGraphPtr &graph_ptr) {
  MS_EXCEPTION_IF_NULL(graph_ptr);
  // access lock for public method
  std::lock_guard<std::mutex> a_lock(access_lock_);
  if (!MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
    // Checking dataset_sink_mode for mindRT is done in debug_actor
    CheckDatasetSinkMode(graph_ptr);
  }
  auto graph_id = graph_ptr->graph_id();
  MS_LOG(DEBUG) << "PreExecute for graph: " << graph_id << " in step: " << num_step_ << ".";
  StoreRunGraphIdList(graph_id);
  SetCurrentAndPrevRootGraph(graph_ptr->root_graph_id());
  // multiple graphs
  if (graph_proto_list_.size() > 1) {
    // there are more than one graphs are not dataset_graph
    if (not_dataset_graph_sum_ > 0) {
      SendMultiGraphsAndClear(graph_ptr);
    }
  } else if (graph_proto_list_.size() == 1) {
    // single graph, and not the initial step
    if (device_target_ == kGPUDevice && !MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_MINDRT) &&
        num_step_ != 0) {
      if (debugger_enabled_ && !(run_level_ == "node" && suspended_at_last_kernel_)) {
        CommandLoop();
      }
      debug_services_->ResetLoadedTensors();
    }
    // In single graph case, reset graph_ptr_ to be nullptr when debugger receives a new graph
    if (received_new_graph_) {
      graph_ptr_ = nullptr;
      CheckGraphPtr(graph_ptr);
    }
  } else if (debugger_enabled_ && graph_id == rungraph_id_list_.front() && device_target_ == kGPUDevice &&
             !MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
    // Multiple graph, and not the initial step,
    // stop only when receive the first sub run graph for each step for old runtime
    // if we have stopped for the last kernel before, no need to stop again
    if (Common::GetDebugTerminate()) {
      return;
    }
    if (!(run_level_ == "node" && suspended_at_last_kernel_)) {
      CommandLoop();
    }
    debug_services_->ResetLoadedTensors();
  }
  // resets for the new graph
  suspended_at_last_kernel_ = false;
}

/*
 * Feature group: Online debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Sends all the subgraphs to online debugger when debugger_enabled_ is true.
 */
void Debugger::SendMultiGraphsAndClear(const KernelGraphPtr &graph_ptr) {
  // only try to enable debugger if they are not all dataset graphs
  if (!enable_debugger_called_) {
    EnableDebugger();
  }
  if (debugger_enabled_) {
    // only send compiled graphs once at the initial step.
    auto dbg_graph_ptr = graph_ptr_;
    // use current graph ptr to load parameters
    graph_ptr_ = graph_ptr;
    LoadParametersAndConst();
    // revert graph ptr to original value
    graph_ptr_ = dbg_graph_ptr;

    SendMultiGraphsAndSuspend(graph_proto_list_);

    graph_proto_list_.clear();
    received_new_graph_ = false;
  }
}

/*
 * Feature group: Dump.
 * Target device group: Ascend, GPU.
 * Runtime category: MindRT.
 * Description: Returns the rank_id for GPU and Ascend kernel-bykernel mindRT.
 */
uint32_t Debugger::GetRankID() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::string device_target = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  uint32_t device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto &device_context =
    device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({device_target, device_id});
  uint32_t rank_id = 0;
  auto deprecated_kernel_executor =
    dynamic_cast<device::DeprecatedKernelExecutor *>(device_context->kernel_executor_.get());
  if (deprecated_kernel_executor != nullptr) {
    rank_id = deprecated_kernel_executor->GetRankID();
  }
  return rank_id;
}

/*
 * Feature group: Dump.
 * Target device group: Ascend, GPU.
 * Runtime category: MindRT.
 * Description: When dump is enabled, this function: 1) Dumps parameters for the current root_graph_id to the
 * root_graph's directory. 2) Dumps constant data once for each graph. 3) Dumps graph run history for each graph.
 */
void Debugger::DumpParamsAndConstAndHistory() {
  if (!CheckDebuggerDumpEnabled()) {
    return;
  }
  LoadParametersAllGraphs();
  E2eDump::DumpParametersData(GetRankID(), debugger_.get());
  // Whether constant data was already dumped for the current root graph.
  bool cur_root_graph_checked = std::find(visited_root_graph_ids_.begin(), visited_root_graph_ids_.end(),
                                          cur_root_graph_id_) != visited_root_graph_ids_.end();
  for (auto graph : graph_ptr_step_vec_) {
    if (!cur_root_graph_checked) {
      LoadConstsForGraph(graph);
      // Dump constant data for GPU.
      E2eDump::DumpConstantData(graph.get(), GetRankID(), debugger_.get());
      // Dump constant data for Ascend.
      DumpConstantDataAscend(graph);
    }
  }
  for (auto kernel_graph = executed_graph_ptr_set_.cbegin(); kernel_graph != executed_graph_ptr_set_.cend();
       ++kernel_graph) {
    // Dump graph run hisotry for each graph.
    if (Debugger::GetInstance()->GetAscendKernelByKernelFlag() &&
        (*kernel_graph)->graph_id() != (*kernel_graph)->root_graph_id()) {
      MS_LOG(INFO) << "current graph graph_id = " << (*kernel_graph)->graph_id() << " is not root graph.";
    } else {
      E2eDump::DumpRunIter(*kernel_graph, GetRankID());
    }
  }
  if (!cur_root_graph_checked) {
    visited_root_graph_ids_.push_back(cur_root_graph_id_);
  }
}

void Debugger::DumpConstantDataAscend(const KernelGraphPtr &graph) {
  if (device_target_ != kAscendDevice) {
    return;
  }
  auto &json_parser = DumpJsonParser::GetInstance();
  if (json_parser.e2e_dump_enabled() || json_parser.async_dump_enabled()) {
    // Dump constant data for ascend mindRT, for old runtime constant data is dumped in session_basic.
    uint32_t rank_id = GetRankID();
    std::string cst_file_dir = GenerateDumpPath(graph->root_graph_id(), rank_id, true);
    DumpConstantInfo(graph, cst_file_dir);
  }
}

/*
 * Feature group: Dump.
 * Target device group: Ascend, GPU.
 * Runtime category: MindRT.
 * Description: Dumps a single node for given graph_id.
 */
void Debugger::DumpSingleNode(const CNodePtr &node, uint32_t graph_id) const {
  if (debugger_ && debugger_->DebuggerBackendEnabled()) {
    uint32_t rank_id = GetRankID();
    (void)E2eDump::DumpSingleNodeData(node, graph_id, rank_id, debugger_.get());
  }
}

/*
 * Feature group: Dump.
 * Target device group: GPU.
 * Runtime category: MindRT.
 * Description: This function is used for new GPU runtime using MindRTBackend, on Ascend platform, graphs are saved in
 * session_basic.
 */
void Debugger::DumpInGraphCompiler(const KernelGraphPtr &kernel_graph) {
  if (device_target_ == kAscendDevice) {
    return;
  }
  auto &json_parser = DumpJsonParser::GetInstance();
  if (json_parser.e2e_dump_enabled()) {
    uint32_t rank_id = GetRankID();
    kernel_graph->set_root_graph_id(kernel_graph->graph_id());
    std::string final_graph = "trace_code_graph_" + std::to_string(kernel_graph->graph_id());
    std::string root_dir = json_parser.path() + "/rank_" + std::to_string(rank_id);
    std::string target_dir = root_dir + "/graphs";
    std::string ir_file_path = target_dir + "/" + "ms_output_" + final_graph + ".ir";
    DumpIRProtoWithSrcInfo(kernel_graph, final_graph, target_dir, kDebugWholeStack);
    DumpIR("trace_code_graph", kernel_graph, true, kWholeStack, ir_file_path);
    DumpGraphExeOrder("ms_execution_order_graph_" + std::to_string(kernel_graph->graph_id()) + ".csv", root_dir,
                      kernel_graph->execution_order());
  }
}

/*
 * Feature group: Dump, Online debugger.
 * Target device group: Ascend, GPU and CPU.
 * Runtime category: MindRT.
 * Description: Load and dump parameters and constant data, call postExecute and update dump iter.
 */
void Debugger::PostExecuteGraphDebugger() {
  // On CPU, update dump iteration， Parameters and consts are not dumped here
  if (device_target_ == kCPUDevice) {
    DumpJsonParser::GetInstance().UpdateDumpIter();
    return;
  }
  DumpParamsAndConstAndHistory();
  // debug used for dump
  if (CheckDebuggerDumpEnabled() && !debugger_enabled()) {
    ClearCurrentData();
  }
  if (debugger_) {
    debugger_->PostExecute();
  }
  E2eDump::UpdateIterMindRTDump();
  executed_graph_ptr_set_.clear();
}

/*
 * Feature group: Online debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Send hit watchpoints, update the step number and reset loaded tensors.
 */
void Debugger::PostExecute() {
  // access lock for public method
  std::lock_guard<std::mutex> a_lock(access_lock_);
  if (Common::GetDebugTerminate()) {
    return;
  }
  if (debugger_ && debugger_->DebuggerBackendEnabled()) {
    // analyze tensor data and send the watchpoints been hit
    if (debugger_enabled_ && !is_dataset_graph_) {
      SendWatchpoints(CheckWatchpoints());
      // no need to suspend at each graph for GPU old runtime, suspension happens in preExecute
      if (device_target_ == kAscendDevice) {
        CommandLoop();
      } else if (device_target_ == kGPUDevice && MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
        if (!(run_level_ == "node" && suspended_at_last_kernel_)) {
          CommandLoop();
        }
      }
      if (device_target_ != kGPUDevice) {
        num_step_++;
      }
    }
    // Only keep parameters in th current map
    // GPU ResetLoadedTensors for old runtime happens in preExecute
    if ((device_target_ == kGPUDevice && MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_MINDRT)) ||
        device_target_ == kAscendDevice) {
      if (debug_services_ != nullptr) {
        debug_services_->ResetLoadedTensors();
      } else {
        MS_LOG(DEBUG) << "debug_services_ is nullptr";
      }
    }
  }
}

bool Debugger::ReadNodeDataRequired(const CNodePtr &kernel) const {
  if (debugger_enabled_ && !is_dataset_graph_) {
    auto is_watchpoint = debug_services_->IsWatchPoint(cur_name_, kernel);
    // if node has a watchpoint on it, is next_to node, or continue_to node then read the kernel tensor data
    if (is_watchpoint || (run_level_ == "node" && (node_name_ == "" || node_name_ == cur_name_))) {
      return true;
    }
  }
  return false;
}

/*
 * Feature group: Online debugger.
 * Target device group: GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Check and send watchpoint hit for a single node, suspend if a watchpoint is hit or we are continuing
 * in node level.
 */
void Debugger::PostExecuteNode(const CNodePtr &kernel, bool last_kernel) {
  // access lock for public method
  std::lock_guard<std::mutex> a_lock(access_lock_);
  if (Common::GetDebugTerminate()) {
    return;
  }
  if (debugger_enabled_ && !is_dataset_graph_) {
    auto is_watchpoint = debug_services_->IsWatchPoint(cur_name_, kernel);

    // if kernel is watchpoint,and get hit. suspend.
    bool hit_empty_flag = true;
    if (is_watchpoint) {
      auto hits = CheckWatchpoints(cur_name_, kernel);
      if (!hits.empty()) {
        SendWatchpoints(hits);
        CommandLoop();

        hit_empty_flag = false;
      }
    }
    if (hit_empty_flag && run_level_ == "node" && (node_name_ == "" || node_name_ == cur_name_)) {
      // if kernel is not watchpoint and is next_to or continue_to node, suspend
      // sets a bool to be checked in preExecute to avoid double stopping at last kernel in the last graph
      if (last_kernel) {
        suspended_at_last_kernel_ = true;
      }
      CommandLoop();
    }
    return;
  }
}

/*
 * Feature group: Dump, Online debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Get graph proto and add it to graph proto list and add loaded graph pointers to a list.
 */
void Debugger::LoadGraphs(const KernelGraphPtr &graph_ptr) {
  MS_EXCEPTION_IF_NULL(graph_ptr);
  if (graph_ptr_ != graph_ptr) {
    MS_LOG(INFO) << "LoadGraphs Debugger got new graph: " << graph_ptr->graph_id();
    received_new_graph_ = true;
    // save new graph_ptr
    graph_ptr_ = graph_ptr;
    CheckDatasetGraph();
    if (!is_dataset_graph_) {
      // get proto for new graph_ptr
      auto graph_proto = GetGraphProto(graph_ptr);
      // add new graph proto to graph_proto_list_
      graph_proto_list_.push_back(graph_proto);
      graph_ptr_list_.push_back(graph_ptr);
      not_dataset_graph_sum_++;
    }
    // reset is_dataset_graph to be false
    is_dataset_graph_ = false;
  }
}

// In single graph cases, check single graph ptr
void Debugger::CheckGraphPtr(const KernelGraphPtr &graph_ptr) {
  MS_EXCEPTION_IF_NULL(graph_ptr);
  if (graph_ptr_ != graph_ptr) {
    MS_LOG(INFO) << "CheckGraphPtr Debugger got new graph: " << graph_ptr->graph_id();
    // save new graph_ptr
    graph_ptr_ = graph_ptr;
    if (!is_dataset_graph_) {
      // only try to enable debugger if it is not a dataset graph
      if (!enable_debugger_called_) {
        EnableDebugger();
      }
      if (debugger_enabled_) {
        LoadParametersAndConst();
        // get graph proto and send to MindInsight
        auto graph_proto = graph_proto_list_.front();
        SendGraphAndSuspend(graph_proto);
        graph_proto_list_.clear();
        received_new_graph_ = false;
      }
    }
  }
}

void Debugger::CheckDatasetGraph() {
  // print parameter node names
  MS_EXCEPTION_IF_NULL(graph_ptr_);
  const auto &params = graph_ptr_->inputs();
  for (const auto &param : params) {
    MS_LOG(INFO) << "param: " << GetKernelNodeName(param);
  }
  // check if there is GetNext or InitDataSetQueue node
  const auto &nodes = graph_ptr_->execution_order();
  for (const auto &node : nodes) {
    auto node_name = common::AnfAlgo::GetCNodeName(node);
    MS_LOG(INFO) << "node: " << GetKernelNodeName(node);
    if (node_name == "GetNext" || node_name == "InitDataSetQueue") {
      MS_LOG(INFO) << "Not enabling debugger for graph " << graph_ptr_->graph_id() << ": found dataset graph node "
                   << node_name;
      is_dataset_graph_ = true;
      return;
    }
  }
  is_dataset_graph_ = false;
}

GraphProto Debugger::GetGraphProto(const KernelGraphPtr &graph_ptr) const {
  // convert kernel graph to debugger modelproto
  ModelProto model = GetDebuggerFuncGraphProto(graph_ptr);
  return model.graph();
}

/*
 * Feature group: Online debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Send debugger backend heartbeat to online debugger every few seconds.
 */
void Debugger::SendHeartbeat(int32_t period) {
  int num_heartbeat_fail = 0;
  const int max_num_heartbeat_fail = 5;
  const int retry_milliseconds = 500;

  Heartbeat heartbeat;
  heartbeat.set_message("Debugger is alive");
  heartbeat.set_period(heartbeat_period_second);

  SetEnableHeartbeat(CheckDebuggerEnabled());
  while (enable_heartbeat_) {
    MS_EXCEPTION_IF_NULL(grpc_client_);
    EventReply reply = grpc_client_->SendHeartbeat(heartbeat);
    if (reply.status() != EventReply::OK) {
      MS_LOG(ERROR) << "Error: SendHeartbeat failed";
      num_heartbeat_fail++;
      if (num_heartbeat_fail >= max_num_heartbeat_fail) {
        MS_LOG(ERROR) << "Maximum number of failure for SendHeartbeat reached : exiting training session.";
        SetEnableHeartbeat(false);
        break;
      } else {
        MS_LOG(ERROR) << "Number of consecutive SendHeartbeat fail:" << num_heartbeat_fail;
        std::this_thread::sleep_for(std::chrono::milliseconds(retry_milliseconds));
      }
    } else {
      int recheck_period_ms = 200;
      for (int i = 0; i < (period * 1000 / recheck_period_ms); i++) {
        if (enable_heartbeat_) {
          std::this_thread::sleep_for(std::chrono::milliseconds(recheck_period_ms));
        } else {
          break;
        }
      }
    }
  }
}

void Debugger::SendGraphAndSuspend(const GraphProto &graph_proto) {
  if (!CheckSendMetadata()) {
    return;
  }
  // send graph to MindInsight server
  MS_EXCEPTION_IF_NULL(grpc_client_);
  EventReply reply = grpc_client_->SendGraph(graph_proto);
  if (reply.status() != EventReply::OK) {
    MS_LOG(ERROR) << "Error: SendGraph failed";
  }
  // enter command loop, wait and process commands
  CommandLoop();
}

bool Debugger::SendMetadata(bool version_check) {
  // prepare metadata
  MS_EXCEPTION_IF_NULL(graph_ptr_);
  std::string device_name = std::to_string(device_id_) + ":" + std::to_string(graph_ptr_->graph_id());
  Metadata metadata;
  metadata.set_device_name(device_name);
  metadata.set_cur_step(num_step_);
  metadata.set_backend(device_target_);
  metadata.set_cur_node(cur_name_);
  metadata.set_training_done(training_done_);
  metadata.set_ms_version(version_);
  MS_LOG(INFO) << "Is training done?" << training_done_;
  // set graph number to not_dataset_graph_sum_
  metadata.set_graph_num(not_dataset_graph_sum_);

  MS_EXCEPTION_IF_NULL(grpc_client_);
  EventReply reply_metadata = grpc_client_->SendMetadata(metadata);

  bool ret = false;
  if (reply_metadata.status() == EventReply::OK) {
    if (version_check) {
      // get type of the command in meta data reply, it should be version matched
      DebuggerCommand cmd = GetCommand(reply_metadata);
      if (cmd != DebuggerCommand::kVersionMatchedCMD) {
        MS_LOG(ERROR) << "MindInsight version is too old, Mindspore version is " << version_;
        Exit();
      } else {
        if (GetMiVersionMatched(reply_metadata)) {
          MS_LOG(INFO) << "MindSpore version is " << version_ << " matches MindInsight version.";
          ret = true;
        } else {
          MS_LOG(ERROR) << "MindSpore version " << version_ << ", did not match MindInsight version.";
          CommandLoop();
        }
      }
    } else {
      // version check is done before so we can just return true here
      ret = true;
    }
  } else {
    MS_LOG(ERROR) << "Error: SendMetadata failed";
  }

  return ret;
}

void Debugger::SendMultiGraphsAndSuspend(const std::list<GraphProto> &graph_proto_list) {
  if (!CheckSendMetadata()) {
    return;
  }
  MS_EXCEPTION_IF_NULL(grpc_client_);
  // send multiple graphs to mindinght server
  // split graph into chunks if one graph is larger than chunk size
  std::list<Chunk> chunked_graph_proto_list;
  Chunk chunk;
  for (auto graph : graph_proto_list) {
    std::string str = graph.SerializeAsString();
    auto graph_size = graph.ByteSize();
    if (graph_size > g_chunk_size) {
      auto sub_graph_str = grpc_client_->ChunkString(str, graph_size);

      for (unsigned int i = 0; i < sub_graph_str.size(); i++) {
        chunk.set_buffer(sub_graph_str[i]);
        if (i < sub_graph_str.size() - 1) {
          chunk.set_finished(false);
        } else {
          chunk.set_finished(true);
        }
        chunked_graph_proto_list.push_back(chunk);
      }
    } else {
      chunk.set_buffer(str);
      chunk.set_finished(true);
      chunked_graph_proto_list.push_back(chunk);
    }
  }
  EventReply reply = grpc_client_->SendMultiGraphs(chunked_graph_proto_list);
  if (reply.status() != EventReply::OK) {
    MS_LOG(ERROR) << "Error: SendGraph failed";
  }
  // enter command loop, wait and process commands
  CommandLoop();
}

bool Debugger::CheckSendMetadata() {
  if (!send_metadata_done_) {
    if (!SendMetadata(true)) {
      return false;
    }
    send_metadata_done_ = true;
  }
  return true;
}

void Debugger::CommandLoop() {
  // prepare metadata
  MS_EXCEPTION_IF_NULL(graph_ptr_);
  std::string device_name = std::to_string(device_id_) + ":" + std::to_string(cur_root_graph_id_);
  Metadata metadata;

  metadata.set_device_name(device_name);
  metadata.set_cur_step(num_step_);
  metadata.set_backend(device_target_);
  metadata.set_cur_node(cur_name_);
  metadata.set_training_done(training_done_);

  // loop exit flag
  bool run = false;
  int num_wait_fail = 0;
  const int max_num_wait_fail = 5;

  while (!run) {
    // wait for command
    MS_EXCEPTION_IF_NULL(grpc_client_);
    EventReply reply = grpc_client_->WaitForCommand(metadata);
    if (reply.status() != EventReply::OK) {
      MS_LOG(ERROR) << "Error: WaitForCommand failed";
      num_wait_fail++;
      if (num_wait_fail > max_num_wait_fail) {
        MS_LOG(ERROR) << "Maximum number of WaitForCommand retry reached: exiting training session.";
        MS_LOG(ERROR) << "Failed to connect to MindInsight debugger server. Please check the config "
                         "of debugger host and port.";
        Exit();
        run = true;
      } else {
        MS_LOG(ERROR) << "Number of consecutive WaitForCommand fail:" << num_wait_fail << "; Retry after "
                      << num_wait_fail << "s";
        std::this_thread::sleep_for(std::chrono::seconds(num_wait_fail));
      }
      continue;
    }

    // get type of the command in reply
    DebuggerCommand cmd = GetCommand(reply);
    if (cmd == DebuggerCommand::kUnknownCMD) {
      MS_LOG(DEBUG) << "Debug: debugger received unknown command";
      continue;
    }

    MS_LOG(INFO) << "received command: ";
    switch (cmd) {
      case DebuggerCommand::kUnknownCMD:
        MS_LOG(INFO) << "UnknownCMD";
        break;
      case DebuggerCommand::kExitCMD:
        MS_LOG(INFO) << "ExitCMD";
        Exit(true);
        // Used for debugger termination
        run = true;
        break;
      case DebuggerCommand::kRunCMD:
        ProcessRunCMD(reply);
        if (GetRunLevel(reply) != "recheck") {
          // exit loop
          run = true;
        }
        break;
      case DebuggerCommand::kSetCMD:
        ProcessKSetCMD(reply);
        break;
      case DebuggerCommand::kViewCMD:
        ProcessKViewCMD(reply);
        break;
      case DebuggerCommand::kVersionMatchedCMD:
        MS_LOG(ERROR) << "Received unexpected Version Matched CMD from MindInsight.";
        Exit();
        break;
      default:
        MS_LOG(ERROR) << "Received unknown CMD from MindInsight";
        Exit();
        break;
    }
  }
}

void Debugger::ProcessRunCMD(const EventReply &reply) {
  MS_LOG(INFO) << "RunCMD";
  if (GetRunLevel(reply) == "recheck") {
    MS_LOG(INFO) << "rechecking all watchpoints";
    SendWatchpoints(CheckWatchpoints("", nullptr, true));
  } else {
    // no longer the initial suspension.
    initial_suspend_ = false;
    // print run cmd content
    // get run_level and node_name
    run_level_ = GetRunLevel(reply);
    node_name_ = GetNodeName(reply);

    MS_LOG(INFO) << "run_level: " << run_level_;
    MS_LOG(INFO) << "node_name_: " << node_name_;
  }
}

void Debugger::ProcessKSetCMD(const EventReply &reply) {
  MS_LOG(INFO) << "SetCMD";
  MS_LOG(INFO) << "id: " << GetWatchpointID(reply);
  MS_LOG(INFO) << "delete: " << GetWatchpointDelete(reply);
  if (GetWatchpointDelete(reply)) {
    MS_LOG(INFO) << "Deleting watchpoint";
    RemoveWatchpoint(GetWatchpointID(reply));
  } else {
    MS_LOG(INFO) << "Setting watchpoint";
    MS_LOG(INFO) << "condition: " << GetWatchcondition(reply).condition();
    ProtoVector<WatchNode> recieved_nodes = GetWatchnodes(reply);
    for (const auto &node : recieved_nodes) {
      MS_LOG(INFO) << "node name: " << node.node_name();
      MS_LOG(INFO) << "node type: " << node.node_type();
    }
    ProtoVector<WatchCondition_Parameter> parameters = GetParameters(reply);
    for (const auto &parameter : parameters) {
      MS_LOG(INFO) << "parameter name: " << parameter.name();
      MS_LOG(INFO) << "parameter is disabled: " << parameter.disabled();
      MS_LOG(INFO) << "parameter value: " << parameter.value();
    }
    SetWatchpoint(GetWatchnodes(reply), GetWatchcondition(reply), GetWatchpointID(reply), GetParameters(reply));
  }
}

void Debugger::ProcessKViewCMD(const EventReply &reply) {
  MS_LOG(INFO) << "ViewCMD";
  // print view cmd content
  ProtoVector<TensorProto> received_tensors = GetTensors(reply);
  for (auto received_tensor : received_tensors) {
    MS_LOG(INFO) << "tensor node name: " << received_tensor.node_name();
    MS_LOG(INFO) << "tensor slot: " << received_tensor.slot();
    MS_LOG(INFO) << "tensor finished: " << std::boolalpha << received_tensor.finished() << std::noboolalpha;
    MS_LOG(INFO) << "tensor iter: " << received_tensor.iter();
    MS_LOG(INFO) << "tensor truncate: " << std::boolalpha << received_tensor.truncate() << std::noboolalpha;
  }

  switch (reply.view_cmd().level()) {
    case debugger::ViewCMD_Level::ViewCMD_Level_base:
      MS_LOG(INFO) << "Tensor base request.";
      ViewBaseLevel(reply);
      break;

    case debugger::ViewCMD_Level::ViewCMD_Level_statistics:
      MS_LOG(INFO) << "Tensor statistics request.";
      ViewStatLevel(reply);
      break;

    case debugger::ViewCMD_Level::ViewCMD_Level_value:
      MS_LOG(INFO) << "Tensor value request.";
      ViewValueLevel(reply);
      break;
    default:
      MS_LOG(DEBUG) << "Debug: Unknown tensor info level";
      break;
  }
}

void Debugger::ViewValueLevel(const EventReply &reply) {
  MS_LOG(INFO) << "Sending tensors";
  std::list<TensorProto> tensors = LoadTensors(GetTensors(reply));
  // print view cmd reply
  for (auto tensor = tensors.cbegin(); tensor != tensors.cend(); ++tensor) {
    MS_LOG(INFO) << "tensor node name: " << tensor->node_name();
    MS_LOG(INFO) << "tensor slot: " << tensor->slot();
    MS_LOG(INFO) << "tensor finished: " << std::boolalpha << tensor->finished() << std::noboolalpha;
    MS_LOG(INFO) << "tensor iter: " << tensor->iter();
    MS_LOG(INFO) << "tensor truncate: " << std::boolalpha << tensor->truncate() << std::noboolalpha;
    MS_LOG(INFO) << "tensor dims: ";
    for (auto dim = tensor->dims().cbegin(); dim != tensor->dims().cend(); dim++) {
      MS_LOG(INFO) << *dim << ",";
    }
    MS_LOG(INFO) << "tensor dtype: " << tensor->data_type();
  }
  MS_EXCEPTION_IF_NULL(grpc_client_);
  EventReply send_tensors_reply = grpc_client_->SendTensors(tensors);
  if (send_tensors_reply.status() != debugger::EventReply::OK) {
    MS_LOG(ERROR) << "Error: SendTensors failed";
  }
}

void Debugger::ViewStatLevel(const EventReply &reply) {
  std::list<TensorSummary> tensor_stats_list = LoadTensorsStat(GetTensors(reply));
  EventReply send_tensors_stat_reply = grpc_client_->SendTensorStats(tensor_stats_list);
  if (send_tensors_stat_reply.status() != debugger::EventReply::OK) {
    MS_LOG(ERROR) << "Error: SendTensorsStats failed.";
  }
}

void Debugger::ViewBaseLevel(const EventReply &reply) {
  std::list<TensorBase> tensor_base_list = LoadTensorsBase(GetTensors(reply));
  EventReply send_tensor_base_reply = grpc_client_->SendTensorBase(tensor_base_list);
  if (send_tensor_base_reply.status() != debugger::EventReply::OK) {
    MS_LOG(ERROR) << "Error: SendTensorsBase failed.";
  }
}

void AddTensorProtoInfo(TensorProto *tensor_item, const TensorProto &tensor) {
  tensor_item->set_node_name(tensor.node_name());
  tensor_item->set_slot(tensor.slot());
  tensor_item->set_iter(tensor.iter());
  tensor_item->set_truncate(tensor.truncate());
  tensor_item->clear_tensor_content();
  tensor_item->clear_data_type();
  tensor_item->clear_dims();
}

void AddTensorStatInfo(const DebugServices::TensorStat &tensor_stat,
                       std::list<TensorSummary> *const tensor_summary_list) {
  if (tensor_summary_list == nullptr) {
    MS_LOG(DEBUG) << "tensor_summary_list is nullptr.";
    return;
  }
  TensorSummary tensor_summary_item;
  TensorBase *tensor_base = tensor_summary_item.mutable_tensor_base();
  tensor_base->set_data_type(tensor_stat.dtype);
  tensor_base->set_data_size(static_cast<int64_t>(tensor_stat.data_size));
  for (auto elem : tensor_stat.shape) {
    tensor_base->add_shape(elem);
  }

  Statistics *tensor_statistics = tensor_summary_item.mutable_statistics();
  tensor_statistics->set_is_bool(tensor_stat.is_bool);
  tensor_statistics->set_max_value(static_cast<float>(tensor_stat.max_value));
  tensor_statistics->set_min_value(static_cast<float>(tensor_stat.min_value));
  tensor_statistics->set_avg_value(static_cast<float>(tensor_stat.avg_value));
  tensor_statistics->set_count(tensor_stat.count);
  tensor_statistics->set_neg_zero_count(tensor_stat.neg_zero_count);
  tensor_statistics->set_pos_zero_count(tensor_stat.pos_zero_count);
  tensor_statistics->set_nan_count(tensor_stat.nan_count);
  tensor_statistics->set_neg_inf_count(tensor_stat.neg_inf_count);
  tensor_statistics->set_pos_inf_count(tensor_stat.pos_inf_count);
  tensor_statistics->set_zero_count(tensor_stat.zero_count);

  tensor_summary_list->push_back(tensor_summary_item);
}

void Debugger::SetWatchpoint(const ProtoVector<WatchNode> &nodes, const WatchCondition &condition, const int32_t id,
                             const ProtoVector<WatchCondition_Parameter> &parameters) {
  std::vector<std::tuple<std::string, bool>> check_node_list;
  std::vector<DebugServices::parameter_t> parameter_list;

  std::transform(nodes.begin(), nodes.end(), std::back_inserter(check_node_list),
                 [](const WatchNode &node) -> std::tuple<std::string, bool> {
                   return make_tuple(node.node_name(), node.node_type() == "scope");
                 });

  std::transform(
    parameters.begin(), parameters.end(), std::back_inserter(parameter_list),
    [](const WatchCondition_Parameter &parameter) -> DebugServices::parameter_t {
      return DebugServices::parameter_t{parameter.name(), parameter.disabled(), parameter.value(), parameter.hit()};
    });
  debug_services_->AddWatchpoint(id, static_cast<int>(condition.condition()), condition.value(), check_node_list,
                                 parameter_list);
}

void Debugger::RemoveWatchpoint(const int32_t id) { debug_services_->RemoveWatchpoint(id); }

std::list<TensorProto> Debugger::LoadTensors(const ProtoVector<TensorProto> &tensors) const {
  std::vector<std::string> name;
  std::vector<std::string> ret_name;
  std::vector<const char *> data_ptr;
  std::vector<ssize_t> data_size;
  std::vector<unsigned int> dtype;
  std::vector<std::vector<int64_t>> shape;

  std::transform(tensors.begin(), tensors.end(), std::back_inserter(name), GetTensorFullName);

  // ret_name will contain tensor names that are found in TensorLoader
  // items in ret_name will be in the same order with tensors if found
  debug_services_->ReadNodesTensors(name, &ret_name, &data_ptr, &data_size, &dtype, &shape);
  std::list<TensorProto> tensor_list;
  size_t result_index = 0;

  for (auto tensor : tensors) {
    ssize_t size_iter = 0;
    if (result_index >= ret_name.size() || ret_name[result_index] != GetTensorFullName(tensor)) {
      TensorProto tensor_item;
      tensor_item.set_finished(true);
      AddTensorProtoInfo(&tensor_item, tensor);
      tensor_list.push_back(tensor_item);
      continue;
    }
    ssize_t tensor_size = data_size[result_index];
    while (size_iter < tensor_size) {
      ssize_t chunk_size = g_chunk_size;
      TensorProto tensor_item;
      tensor_item.set_finished(false);
      if (tensor_size - size_iter <= g_chunk_size) {
        chunk_size = tensor_size - size_iter;
        tensor_item.set_finished(true);
      }
      AddTensorProtoInfo(&tensor_item, tensor);
      // return empty tensor if didn't find the requested tensor

      tensor_item.set_tensor_content(data_ptr[result_index] + size_iter, chunk_size);

      tensor_item.set_data_type(static_cast<debugger::DataType>(dtype[result_index]));
      for (auto &elem : shape[result_index]) {
        tensor_item.add_dims(elem);
      }
      // add tensor to result list and increment result_index to check next item in ret_name
      tensor_list.push_back(tensor_item);
      if (size_iter > INT_MAX - g_chunk_size) {
        MS_EXCEPTION(ValueError) << size_iter << " + " << g_chunk_size << " would lead to integer overflow!";
      }
      size_iter += g_chunk_size;
    }
    result_index++;
  }
  return tensor_list;
}

std::list<TensorBase> Debugger::LoadTensorsBase(const ProtoVector<TensorProto> &tensors) const {
  std::list<TensorBase> tensor_base_list;
  std::vector<std::string> name;
  std::transform(tensors.begin(), tensors.end(), std::back_inserter(name), GetTensorFullName);
  std::vector<std::tuple<std::string, std::shared_ptr<TensorData>>> result_list;
  debug_services_->SearchNodesTensors(name, &result_list);
  for (auto result : result_list) {
    auto tensor = std::get<1>(result);
    if (!tensor || ((cur_root_graph_id_ != tensor->GetRootGraphId()) &&
                    MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_MINDRT))) {
      // tensor was not found or tensor's graph was not executed in the current step, creating empty tensor base.
      TensorBase tensor_base_item;
      tensor_base_item.set_data_size(0);
      tensor_base_item.set_data_type(0);
      tensor_base_item.add_shape(0);
      tensor_base_list.push_back(tensor_base_item);
      continue;
    }
    // tensor was found creating tensor base object.
    TensorBase tensor_base_item;
    tensor_base_item.set_data_size(static_cast<int64_t>(tensor->GetByteSize()));
    tensor_base_item.set_data_type(static_cast<int32_t>(tensor->GetType()));
    for (auto elem : tensor->GetShape()) {
      tensor_base_item.add_shape(elem);
    }
    tensor_base_list.push_back(tensor_base_item);
  }
  return tensor_base_list;
}

std::list<TensorSummary> Debugger::LoadTensorsStat(const ProtoVector<TensorProto> &tensors) const {
  std::list<TensorSummary> tensor_summary_list;
  std::vector<std::string> name;
  std::transform(tensors.begin(), tensors.end(), std::back_inserter(name), GetTensorFullName);
  std::vector<std::tuple<std::string, std::shared_ptr<TensorData>>> result_list;
  debug_services_->SearchNodesTensors(name, &result_list);
  for (auto result : result_list) {
    auto tensor = std::get<1>(result);
    if (!tensor || ((cur_root_graph_id_ != tensor->GetRootGraphId()) &&
                    MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_MINDRT))) {
      // tensor was not found or tensor's graph was not executed in the current step, creating empty tensor summary.
      DebugServices::TensorStat tensor_stat;
      AddTensorStatInfo(tensor_stat, &tensor_summary_list);
      continue;
    }
    // tensor was found creating tensor summary object.
    DebugServices::TensorStat tensor_stat = DebugServices::GetTensorStatistics(tensor);
    AddTensorStatInfo(tensor_stat, &tensor_summary_list);
  }
  return tensor_summary_list;
}

std::shared_ptr<TensorData> Debugger::GetTensor(const std::string &tensor_name) const {
  return debug_services_->GetTensor(tensor_name);
}

void Debugger::Exit(bool exit_success) {
  // debugger will notify main thread to exit because main thread can only exit at step boundary.
  MS_LOG(INFO) << "Exit Debugger";
  SetEnableHeartbeat(false);
  Common::DebugTerminate(true, exit_success);
}

std::list<WatchpointHit> Debugger::CheckWatchpoints(const std::string &watchnode, const CNodePtr &kernel,
                                                    bool recheck) {
  std::vector<std::string> name;
  std::vector<std::string> slot;
  std::vector<int> condition;
  std::vector<unsigned int> watchpoint_id;
  std::vector<std::vector<DebugServices::parameter_t>> parameters;
  std::vector<int32_t> error_codes;
  std::vector<std::shared_ptr<TensorData>> tensor_list;
  if (watchnode.empty()) {
    tensor_list = debug_services_->GetTensor();
  } else {
    tensor_list = debug_services_->GetNodeTensor(kernel);
  }
  DebugServices::ProcessedNPYFiles processed_npy_files;
  MS_LOG(INFO) << "checkwatchpoints call for step " << num_step_;
  debug_services_->CheckWatchpoints(&name, &slot, &condition, &watchpoint_id, &parameters, &error_codes,
                                    &processed_npy_files, &tensor_list, initial_suspend_, watchnode.empty(), recheck);
  std::list<WatchpointHit> hits;
  for (unsigned int i = 0; i < name.size(); i++) {
    WatchpointHit hit;
    std::vector<DebugServices::parameter_t> &parameter = parameters[i];
    hit.set_id(watchpoint_id[i]);
    hit.set_error_code(error_codes[i]);
    // here TensorProto act as a tensor indicator, not sending tensor content
    TensorProto *tensor_item = hit.mutable_tensor();
    tensor_item->set_node_name(name[i]);
    tensor_item->set_slot(slot[i]);
    tensor_item->set_finished(true);

    WatchCondition *condition_item = hit.mutable_watch_condition();
    condition_item->set_condition(debugger::WatchCondition_Condition(condition[i]));
    for (const auto &p : parameter) {
      auto x = condition_item->mutable_params()->Add();
      x->set_name(p.name);
      x->set_disabled(p.disabled);
      x->set_value(p.value);
      x->set_hit(p.hit);
      x->set_actual_value(p.actual_value);
    }
    hits.push_back(hit);
  }
  return hits;
}

void Debugger::SendWatchpoints(const std::list<WatchpointHit> &points) {
  // send info about watchpoint
  if (!points.empty()) {
    MS_EXCEPTION_IF_NULL(grpc_client_);
    EventReply reply = grpc_client_->SendWatchpointHits(points);
    if (reply.status() != EventReply::OK) {
      MS_LOG(ERROR) << "Error: SendWatchpointHits failed";
    }
  }
}

bool Debugger::DumpTensorToFile(const std::string &filepath, const std::string &tensor_name, size_t slot) const {
  if (debug_services_ == nullptr) {
    MS_LOG(INFO) << "The debug_services_ is nullptr.";
    return false;
  }
  return debug_services_.get()->DumpTensorToFile(filepath, tensor_name, slot);
}

bool Debugger::LoadNewTensor(const std::shared_ptr<TensorData> &tensor, bool keep_prev) {
  if (debug_services_ == nullptr) {
    debug_services_ = std::make_unique<DebugServices>();
  }
  return debug_services_.get()->LoadNewTensor(tensor, keep_prev);
}

bool Debugger::debugger_enabled() const { return debugger_enabled_; }

bool Debugger::partial_memory() const { return partial_memory_; }

void Debugger::SetEnableHeartbeat(bool enabled) { enable_heartbeat_ = enabled; }

void Debugger::SetCurNode(const std::string &cur_name) {
  // access lock for public method
  std::lock_guard<std::mutex> a_lock(access_lock_);
  cur_name_ = cur_name;
}

std::string Debugger::run_level() const { return run_level_; }

void Debugger::SetTrainingDone(bool training_done) { training_done_ = training_done; }

bool Debugger::CheckPort(const std::string &port) const {
  int num = 0;
  const int min_port_num = 1;
  const int max_port_num = 65535;
  const int decimal = 10;
  if (port[0] == '0' && port[1] != '\0') {
    return false;
  }
  size_t i = 0;
  while (port[i] != '\0') {
    if (port[i] < '0' || port[i] > '9') {
      return false;
    }
    num = num * decimal + (port[i] - '0');
    if (num > max_port_num) {
      return false;
    }
    i++;
  }
  if (num < min_port_num) {
    return false;
  }
  return true;
}

bool Debugger::CheckIp(const std::string &host) const {
  std::regex reg_ip(
    "(25[0-4]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[1-9])"
    "[.](25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])"
    "[.](25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])"
    "[.](25[0-4]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[1-9])");
  std::smatch smat;
  std::string host_str = host;
  return std::regex_match(host_str, smat, reg_ip);
}

uint32_t Debugger::GetFirstRunGraphId() const { return rungraph_id_list_.front(); }

/*
 * Feature group: Dump.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Load a single parameter or value node.
 */
void Debugger::LoadSingleAnfnode(const AnfNodePtr &anf_node, const size_t output_index, uint32_t root_graph_id) {
  MS_EXCEPTION_IF_NULL(anf_node);
  if (!anf_node->isa<Parameter>() && !anf_node->isa<ValueNode>()) {
    return;
  }
  // When MindRT is used, only ValueNodes and ParameterWeights can be loaded from device to host
  if (MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
    if (!anf_node->isa<ValueNode>() &&
        !(anf_node->isa<Parameter>() && common::AnfAlgo::IsParameterWeight(anf_node->cast<ParameterPtr>()))) {
      return;
    }
  }
  // for parameters and value nodes, set its execution order to be 0;
  int exec_order = 0;
  std::string node_name = GetKernelNodeName(anf_node);
  GetFileKernelName(NOT_NULL(&node_name));
  // check if output adde exists, if not, return;
  if (!AnfAlgo::OutputAddrExist(anf_node, output_index)) {
    return;
  }
  auto addr = AnfAlgo::GetOutputAddr(anf_node, output_index);
  MS_EXCEPTION_IF_NULL(addr);
  auto type = common::AnfAlgo::GetOutputInferDataType(anf_node, output_index);
  if (!IsTypeDebuggerSupported(type)) {
    return;
  }
  auto format = kOpFormat_DEFAULT;
  string tensor_name = node_name + ':' + "0";
  ShapeVector int_shapes = trans::GetRuntimePaddingShape(anf_node, output_index);
  bool keep_prev;
  if (anf_node->isa<Parameter>()) {
    keep_prev = true;
    debug_services_->MoveTensorCurrentToPrev(tensor_name);
  } else {
    keep_prev = false;
  }
  bool ret =
    addr->LoadMemToHost(tensor_name, exec_order, format, int_shapes, type, 0, keep_prev, root_graph_id, false, true);
  if (!ret) {
    MS_LOG(ERROR) << "LoadMemToHost:"
                  << ", tensor_name:" << tensor_name << ", host_format:" << format << ".!";
  }
}

void Debugger::LoadSingleParameterMindRT(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto root_graph_id = cur_root_graph_id_;
  // This function is only  for loading parameters mindRT.
  std::string node_name = GetKernelNodeName(node);
  GetFileKernelName(NOT_NULL(&node_name));
  TypeId type;
  TypeId device_type;
  ShapeVector int_shapes;
  auto device_addr = GetParameterInfo(node, NOT_NULL(&int_shapes), NOT_NULL(&type), NOT_NULL(&device_type));
  if (device_addr == nullptr || device_addr->GetPtr() == nullptr) {
    MS_LOG(DEBUG) << "Skip node: " << node_name << ". Parameter data is not available for mindRT.";
    return;
  }
  if (!IsTypeDebuggerSupported(type)) {
    return;
  }
  auto format = kOpFormat_DEFAULT;
  string tensor_name = node_name + ':' + "0";
  if (debug_services_ != nullptr) {
    debug_services_->MoveTensorCurrentToPrev(tensor_name);
  }
  // Keep_prev is True for parameters.
  // force update for parameters.
  bool ret = device_addr->LoadMemToHost(tensor_name, 0, format, int_shapes, type, 0, true, root_graph_id, true, true);
  if (!ret) {
    MS_LOG(ERROR) << "LoadMemToHost:"
                  << ", tensor_name:" << tensor_name << ", host_format:" << format << ".!";
  }
}

/*
 * Feature group: Dump, Online debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Load all the parameters and value nodes for the last loaded graph.
 */
void Debugger::LoadParametersAndConst() {
  if (!(debugger_enabled_ || CheckDebuggerDumpEnabled())) {
    return;
  }
  MS_EXCEPTION_IF_NULL(graph_ptr_);
  // load parameters
  MS_LOG(INFO) << "Start to load Parameters for graph " << graph_ptr_->graph_id() << ".";
  auto root_graph_id = graph_ptr_->root_graph_id();
  const auto &parameters = graph_ptr_->inputs();
  for (auto &item : parameters) {
    LoadSingleAnfnode(item, kParameterOutputIndex, root_graph_id);
  }
  // load value nodes
  // get all constant values from the graph
  MS_LOG(INFO) << "Start to load value nodes for graph " << graph_ptr_->graph_id() << ".";
  const auto value_nodes = graph_ptr_->graph_value_nodes();
  for (auto &item : value_nodes) {
    LoadSingleAnfnode(item, kValueNodeOutputIndex, root_graph_id);
  }
}

/*
 * Feature group: Dump, Online debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Load all the parameters and value nodes for the given graph.
 */
void Debugger::LoadParametersAndConst(const KernelGraphPtr &graph) {
  if (!(debugger_enabled_ || CheckDebuggerDumpEnabled())) {
    return;
  }
  MS_EXCEPTION_IF_NULL(graph);
  // load parameters
  MS_LOG(INFO) << "Start to load Parameters for graph " << graph->graph_id() << ".";
  auto root_graph_id = graph->root_graph_id();
  const auto &parameters = graph->inputs();
  for (auto &item : parameters) {
    LoadSingleAnfnode(item, kParameterOutputIndex, root_graph_id);
  }
  // load value nodes
  // get all constant values from the graph
  MS_LOG(INFO) << "Start to load value nodes for graph " << graph->graph_id() << ".";
  const auto value_nodes = graph->graph_value_nodes();
  for (auto &item : value_nodes) {
    LoadSingleAnfnode(item, kValueNodeOutputIndex, root_graph_id);
  }
}

/*
 * Feature group: Dump.
 * Target device group: GPU.
 * Runtime category: MindRT.
 * Description: This function is for loading parameters' data from device to host into tensor_list_map_ for GPU dump.
 * Ascend does not use tensor_map_list_ for dump so it is not needed for ascend dump.
 */
void Debugger::LoadParametersAllGraphs() {
  if (!(device_target_ == kGPUDevice && CheckDebuggerDumpEnabled())) {
    return;
  }
  for (auto &node : parameters_mindRT_) {
    LoadSingleParameterMindRT(node);
  }
}

/*
 * Feature group: Dump.
 * Target device group: GPU.
 * Runtime category: MindRT.
 * Description: This function is for loading constant data from device to host into tensor_list_map_ for GPU dump.
 * Ascend does not use tensor_map_list_ for dump so it is not needed for ascend dump.
 */
void Debugger::LoadConstsForGraph(const KernelGraphPtr &graph) {
  if (!(device_target_ == kGPUDevice && CheckDebuggerDumpEnabled())) {
    return;
  }
  // load value nodes
  // get all constant values from the graph
  MS_LOG(INFO) << "Start to load value nodes for graph " << graph->graph_id() << ".";
  auto root_graph_id = graph->root_graph_id();
  const auto value_nodes = graph->graph_value_nodes();
  for (auto &item : value_nodes) {
    LoadSingleAnfnode(item, kValueNodeOutputIndex, root_graph_id);
  }
}

/*
 * Feature group: Online debugger.
 * Target device group: Ascend.
 * Runtime category: Old runtime, MindRT.
 * Description: Load all the kernels for the last loaded graph.
 */
void Debugger::LoadGraphOutputs() {
  if (!(debugger_enabled() && device_target_ == kAscendDevice)) {
    return;
  }
  MS_EXCEPTION_IF_NULL(graph_ptr_);
  const auto &apply_kernels = graph_ptr_->execution_order();
  auto root_graph_id = graph_ptr_->root_graph_id();
  // for kernels, execution order starts from 1
  int exec_order = 1;
  for (const auto &node : apply_kernels) {
    MS_EXCEPTION_IF_NULL(node);
    std::string kernel_name = GetKernelNodeName(node);
    auto output_size = AnfAlgo::GetOutputTensorNum(node);
    if (partial_memory_) {
      if (!debug_services_->IsWatchPoint(kernel_name, node)) {
        continue;
      }
    }
    for (size_t j = 0; j < output_size; ++j) {
      if (!AnfAlgo::OutputAddrExist(node, j)) {
        MS_LOG(INFO) << "Cannot find output addr for slot " << j << " for " << kernel_name;
        continue;
      }
      auto addr = AnfAlgo::GetOutputAddr(node, j);
      MS_EXCEPTION_IF_NULL(addr);
      auto type = common::AnfAlgo::GetOutputInferDataType(node, j);
      if (!IsTypeDebuggerSupported(type)) {
        continue;
      }
      auto format = kOpFormat_DEFAULT;
      string tensor_name = kernel_name + ':' + std::to_string(j);
      ShapeVector int_shapes = trans::GetRuntimePaddingShape(node, j);
      auto ret =
        addr->LoadMemToHost(tensor_name, exec_order, format, int_shapes, type, j, false, root_graph_id, false, true);
      if (!ret) {
        MS_LOG(ERROR) << "LoadMemToHost:"
                      << ", tensor_name:" << tensor_name << ", host_format:" << format << ".!";
      }
    }
    exec_order = exec_order + 1;
  }
}

/*
 * Feature group: Online debugger.
 * Target device group: GPU.
 * Runtime category: Old runtime.
 * Description: Update step number if we are processing the first graph (to support multigraph).
 */
void Debugger::UpdateStepNum(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(debugger_);
  if (device_target_ == kGPUDevice && (debugger_enabled_ || device::KernelRuntime::DumpDataEnabledIteration()) &&
      (graph->graph_id() == debugger_->GetFirstRunGraphId())) {
    // access lock for public method
    std::lock_guard<std::mutex> a_lock(access_lock_);
    ++num_step_;
  }
}

/*
 * Feature group: Online debugger.
 * Target device group: GPU.
 * Runtime category: MindRT.
 * Description: Update step number when DebugActor::DebugOnStepEnd is called at the end of each step.
 */
void Debugger::UpdateStepNumGPU() {
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  if (device_target_ == kGPUDevice && (debugger_enabled_ || dump_json_parser.DumpEnabledForIter())) {
    // access lock for public method
    std::lock_guard<std::mutex> a_lock(access_lock_);
    ++num_step_;
    MS_LOG(DEBUG) << "Update step for GPU, current step: " << num_step_;
  }
}

void Debugger::ClearCurrentData() {
  if ((device_target_ == kGPUDevice) && (debugger_enabled_ || device::KernelRuntime::DumpDataEnabledIteration())) {
    if (debug_services_) {
      debug_services_->EmptyCurrentTensor();
    } else {
      MS_LOG(WARNING) << "debug_services_ is nullptr";
    }
  }
}

bool Debugger::TensorExistsInCurrent(const std::string &tensor_name) {
  if (debug_services_ != nullptr) {
    return debug_services_->TensorExistsInCurrent(tensor_name);
  }
  return false;
}
}  // namespace mindspore
