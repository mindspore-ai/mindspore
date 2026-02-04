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

#include "tools/data_dump/debugger/debugger.h"

#include <dirent.h>

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <regex>
#include <string>
#include <tuple>
#include <vector>

#include "include/backend/common/kernel_graph/session_basic.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/utils/anfalgo.h"
#include "include/utils/comm_manager.h"
#include "utils/config_manager.h"
#include "include/utils/env_config_parser.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_dump_utils.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "tools/data_dump/dump_json_parser.h"
#include "tools/data_dump/e2e_dump.h"
#ifdef ENABLE_DEBUGGER
#include "tools/data_dump/debugger/debugger_proto_exporter.h"
#endif
#include "proto/debug_graph.pb.h"
#include "tools/data_dump/debug_services.h"
#include "tools/data_dump/debugger/debugger_utils.h"
#include "tools/data_dump/debugger/proto_exporter.h"
#include "tools/data_dump/utils.h"

using debugger::GraphProto;
using debugger::ModelProto;
using debugger::Statistics;
using debugger::TensorProto;

namespace mindspore {
namespace {
bool DumpDataEnabledIteration() {
  // Returns true if e2e dump is enabled and current iteration must be dumped.
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  if (!dump_json_parser.e2e_dump_enabled()) {
    return false;
  }

  auto cur_iter = dump_json_parser.cur_dump_iter();
  if (dump_json_parser.IsDumpIter(cur_iter)) {
    return true;
  }
  return false;
}
}  // namespace

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
    : debug_services_(nullptr),
      device_id_(0),
      device_target_(""),
      is_dataset_graph_(false),
      not_dataset_graph_sum_(0),
      ascend_kernel_by_kernel_(false),
      enable_debugger_called_(false),
      version_("") {}

void Debugger::Init(const uint32_t device_id, const std::string device_target) {
  // access lock for public method
  std::lock_guard<std::mutex> a_lock(access_lock_);
  // save device_id
  MS_VLOG(VL_DUMP) << "Debugger got device_id: " << device_id;
  device_id_ = device_id;
  MS_VLOG(VL_DUMP) << "Debugger got device_target: " << device_target;
  device_target_ = device_target;
  version_ = MSVERSION;
}

bool IsTypeDebuggerSupported(TypeId type) {
  if (type < TypeId::kNumberTypeEnd && type > TypeId::kNumberTypeBegin && type != kNumberTypeComplex64) {
    return true;
  }
  MS_VLOG(VL_DUMP) << "Debugger does not support type: " << TypeIdLabel(type);
  return false;
}

void Debugger::CheckDatasetSinkMode(const KernelGraphPtr &graph_ptr) {
  bool sink_mode =
    ConfigManager::GetInstance().dataset_mode() == DatasetMode::DS_SINK_MODE || graph_ptr->IsDatasetGraph();
  if (CheckDebuggerDumpEnabled() && sink_mode && device_target_ == kGPUDevice) {
    MS_EXCEPTION(NotSupportError)
      << "e2e_dump is not supported on GPU with dataset_sink_mode=True. Please set dataset_sink_mode=False";
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

/*
 * Feature group: Dump, Online debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT
 * Description: Returns true if online debugger or dump is enabled.
 */
bool Debugger::DebuggerBackendEnabled() const { return CheckDebuggerDumpEnabled(); }

void Debugger::Reset() {
  // access lock for public method
  std::lock_guard<std::mutex> a_lock(access_lock_);
  device_id_ = 0;
  device_target_ = "";
  is_dataset_graph_ = false;
  graph_ptr_ = nullptr;
  debug_services_ = nullptr;
  graph_ptr_list_.clear();
  graph_ptr_step_vec_.clear();
  executed_graph_ptr_set_.clear();
  parameters_mindRT_.clear();
  visited_root_graph_ids_.clear();
  MS_VLOG(VL_DUMP) << "Release Debugger resource.";
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
                << ".";
  MS_LOG(DEBUG) << "Set root graph for all the subgraphs:";
  for (size_t graph_index = 0; graph_index < graphs.size(); ++graph_index) {
    const auto &graph = graphs[graph_index];
    // set root graph id for GPU mindrt runtime.
    MS_VLOG(VL_DUMP) << "Set root graph for graph: " << graph->graph_id() << " to: " << cur_root_graph_id_ << ".";
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
                << ".";
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
  MS_LOG(DEBUG) << "PreExecute for graph: " << graph_id << ".";
  StoreRunGraphIdList(graph_id);
  SetCurrentAndPrevRootGraph(graph_ptr->root_graph_id());
  if (debug_services_ == nullptr) {
    debug_services_ = std::make_unique<DebugServices>();
  }
  debug_services_->ResetLoadedTensors();
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
  E2eDump::DumpParametersData(datadump::GetRankID(), debugger_.get());
  // Whether constant data was already dumped for the current root graph.
  bool cur_root_graph_checked = std::find(visited_root_graph_ids_.begin(), visited_root_graph_ids_.end(),
                                          cur_root_graph_id_) != visited_root_graph_ids_.end();
  for (auto graph : graph_ptr_step_vec_) {
    if (!cur_root_graph_checked) {
      LoadConstsForGraph(graph);
      // Dump constant data for GPU.
      E2eDump::DumpConstantData(graph.get(), datadump::GetRankID(), debugger_.get());
    }
  }
  for (auto kernel_graph = executed_graph_ptr_set_.cbegin(); kernel_graph != executed_graph_ptr_set_.cend();
       ++kernel_graph) {
    auto debugger = Debugger::GetInstance();
    MS_EXCEPTION_IF_NULL(debugger);
    // Dump graph run hisotry for each graph.
    if (debugger->GetAscendKernelByKernelFlag() && (*kernel_graph)->graph_id() != (*kernel_graph)->root_graph_id()) {
      MS_VLOG(VL_DUMP) << "current graph graph_id = " << (*kernel_graph)->graph_id() << " is not root graph.";
    } else {
      E2eDump::DumpRunIter(*kernel_graph, datadump::GetRankID());
    }
  }
  if (!cur_root_graph_checked) {
    visited_root_graph_ids_.push_back(cur_root_graph_id_);
  }
}

/*
 * Feature group: Dump.
 * Target device group: Ascend, GPU.
 * Runtime category: MindRT.
 * Description: Dumps a single node for given graph_id.
 */
void Debugger::DumpSingleNode(const CNodePtr &node, uint32_t graph_id, const DeviceContext *device_context) const {
  if (debugger_ && debugger_->DebuggerBackendEnabled()) {
    uint32_t rank_id = datadump::GetRankID();
    (void)E2eDump::DumpSingleNodeData(node, graph_id, rank_id, debugger_.get(), device_context);
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
    uint32_t rank_id = datadump::GetRankID();
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
 * Target device group: GPU and CPU.
 * Runtime category: MindRT.
 * Description: Load and dump parameters and constant data, call postExecute and update dump iter.
 */
void Debugger::PostExecuteGraphDebugger() {
  if (device_target_ == kAscendDevice) {
    MS_LOG(DEBUG) << "On Ascend, parameters and constant data is not dumped here.";
    return;
  }
  // On CPU, update dump iteration， Parameters and consts are not dumped here
  if (device_target_ == kCPUDevice) {
    DumpJsonParser::GetInstance().UpdateDumpIter();
    return;
  }
  DumpParamsAndConstAndHistory();
  // debug used for dump
  if (CheckDebuggerDumpEnabled()) {
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

  if (debugger_ && debugger_->DebuggerBackendEnabled()) {
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

/*
 * Feature group: Dump, Online debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Get graph proto and add it to graph proto list and add loaded graph pointers to a list.
 */
void Debugger::LoadGraphs(const KernelGraphPtr &graph_ptr) {
  MS_EXCEPTION_IF_NULL(graph_ptr);
  if (graph_ptr_ != graph_ptr) {
    MS_VLOG(VL_DUMP) << "LoadGraphs Debugger got new graph: " << graph_ptr->graph_id();
    // save new graph_ptr
    graph_ptr_ = graph_ptr;
    CheckDatasetGraph();
    if (!is_dataset_graph_) {
      // get proto for new graph_ptr
      // add new graph proto to graph_proto_list_
      graph_ptr_list_.push_back(graph_ptr);
      not_dataset_graph_sum_++;
    }
    // reset is_dataset_graph to be false
    is_dataset_graph_ = false;
  }
}

void Debugger::CheckDatasetGraph() {
  // print parameter node names
  MS_EXCEPTION_IF_NULL(graph_ptr_);
  const auto &params = graph_ptr_->inputs();
  for (const auto &param : params) {
    MS_VLOG(VL_DUMP) << "param: " << GetKernelNodeName(param);
  }
  // check if there is GetNext or InitDataSetQueue node
  const auto &nodes = graph_ptr_->execution_order();
  for (const auto &node : nodes) {
    auto node_name = common::AnfAlgo::GetCNodeName(node);
    MS_VLOG(VL_DUMP) << "node: " << GetKernelNodeName(node);
    if (node_name == "GetNext" || node_name == "InitDataSetQueue") {
      MS_VLOG(VL_DUMP) << "Not enabling debugger for graph " << graph_ptr_->graph_id() << ": found dataset graph node "
                       << node_name;
      is_dataset_graph_ = true;
      return;
    }
  }
  is_dataset_graph_ = false;
}

std::unique_ptr<GraphProto> Debugger::GetGraphProto(const KernelGraphPtr &graph_ptr) const {
  // convert kernel graph to debugger modelproto
  ModelProto model = GetDebuggerFuncGraphProto(graph_ptr);
  return std::make_unique<GraphProto>(model.graph());
}

std::shared_ptr<TensorData> Debugger::GetTensor(const std::string &tensor_name) const {
  return debug_services_->GetTensor(tensor_name);
}

bool Debugger::DumpTensorToFile(const std::string &filepath, const std::string &tensor_name, size_t slot) const {
  if (debug_services_ == nullptr) {
    MS_VLOG(VL_DUMP) << "The debug_services_ is nullptr.";
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
  std::string node_name = GetKernelNodeName(anf_node);
  GetFileKernelName(NOT_NULL(&node_name));
  // check if output adde exists, if not, return;
  if (!AnfAlgo::OutputAddrExist(anf_node, output_index)) {
    return;
  }
  auto type = common::AnfAlgo::GetOutputInferDataType(anf_node, output_index);
  if (!IsTypeDebuggerSupported(type)) {
    return;
  }
  auto format = kOpFormat_DEFAULT;
  string tensor_name = node_name + ':' + "0";
  ShapeVector int_shapes = AnfAlgo::GetRuntimePaddingShape(anf_node, output_index);
  bool ret = LoadMemToHost(AnfAlgo::GetOutputKernelTensor(anf_node, output_index).get(), tensor_name, format,
                           int_shapes, type, 0, false, root_graph_id, false, true);
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
  auto kernel_tensor = GetParameterInfo(node, NOT_NULL(&int_shapes), NOT_NULL(&type), NOT_NULL(&device_type));
  if (kernel_tensor == nullptr || kernel_tensor->device_address() == nullptr ||
      kernel_tensor->device_address()->GetPtr() == nullptr) {
    MS_LOG(DEBUG) << "Skip node: " << node_name << ". Parameter data is not available for mindRT.";
    return;
  }
  if (!IsTypeDebuggerSupported(type)) {
    return;
  }
  auto format = kOpFormat_DEFAULT;
  string tensor_name = node_name + ':' + "0";
  bool ret =
    LoadMemToHost(kernel_tensor.get(), tensor_name, format, int_shapes, type, 0, false, root_graph_id, true, true);
  if (!ret) {
    MS_LOG(ERROR) << "LoadMemToHost:"
                  << ", tensor_name:" << tensor_name << ", host_format:" << format << ".!";
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
  MS_VLOG(VL_DUMP) << "Start to load value nodes for graph " << graph->graph_id() << ".";
  auto root_graph_id = graph->root_graph_id();
  const auto value_nodes = graph->graph_value_nodes();
  for (auto &item : value_nodes) {
    LoadSingleAnfnode(item, kValueNodeOutputIndex, root_graph_id);
  }
}

void Debugger::ClearCurrentData() {
  if (DumpDataEnabledIteration()) {
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

void DebuggerReset() {
  auto debugger = Debugger::GetInstance();
  MS_EXCEPTION_IF_NULL(debugger);
  debugger->Reset();
}
void DebuggerInit(const uint32_t device_id, const std::string &device_target) {
  auto debugger = Debugger::GetInstance();
  MS_EXCEPTION_IF_NULL(debugger);
  debugger->Init(device_id, device_target);
}
void DumpInGraphCompiler(const KernelGraphPtr &kernel_graph) {
  auto debugger = Debugger::GetInstance();
  MS_EXCEPTION_IF_NULL(debugger);
  debugger->DumpInGraphCompiler(kernel_graph);
}
bool DebuggerBackendEnabled() {
  auto debugger = Debugger::GetInstance();
  MS_EXCEPTION_IF_NULL(debugger);
  return debugger->DebuggerBackendEnabled();
}
void DebuggerLoadGraphs(const KernelGraphPtr &graph_ptr) {
  auto debugger = Debugger::GetInstance();
  MS_EXCEPTION_IF_NULL(debugger);
  debugger->LoadGraphs(graph_ptr);
}
}  // namespace mindspore
