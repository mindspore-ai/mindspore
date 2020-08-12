/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include <fstream>
#include <tuple>
#include <vector>
#include <algorithm>
#include "debug/debugger/debugger.h"
#include "pipeline/jit/pipeline.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "runtime/device/kernel_runtime_manager.h"

using debugger::EventReply;
using debugger::GraphProto;
using debugger::ModelProto;
using debugger::TensorProto;
using debugger::WatchCondition;
using debugger::WatchCondition_Condition_inf;
using debugger::WatchCondition_Condition_nan;
using debugger::WatchNode;
using debugger::WatchpointHit;

namespace mindspore {

DebuggerPtr Debugger::debugger_ = nullptr;
std::mutex Debugger::instance_lock_;

Debugger::Debugger()
    : grpc_client_(nullptr),
      debug_services_(nullptr),
      device_id_(0),
      device_target_(""),
      num_step_(0),
      debugger_enabled_(false),
      run_level_(""),
      node_name_(""),
      cur_name_(""),
      is_dataset_graph_(false),
      partial_memory_(false) {}

void Debugger::Init(const uint32_t device_id, const std::string device_target) {
  // access lock for public method
  std::lock_guard<std::mutex> a_lock(access_lock_);
  // save device_id
  MS_LOG(INFO) << "Debugger got device_id: " << device_id;
  device_id_ = device_id;
  MS_LOG(INFO) << "Debugger got device_target: " << device_target;
  device_target_ = device_target;
}

void Debugger::EnableDebugger() {
  // reset some of the class members
  num_step_ = 0;
  debugger_enabled_ = false;
  partial_memory_ = false;
  grpc_client_ = nullptr;
  debug_services_ = nullptr;

  // see if dump is enabled
  bool dump_enabled = false;
  if (device_target_ == kGPUDevice) {
    auto runtime_instance = device::KernelRuntimeManager::Instance().GetSingleKernelRuntime(kGPUDevice, device_id_);
    MS_EXCEPTION_IF_NULL(runtime_instance);
    dump_enabled = runtime_instance->DumpDataEnabled();
  }

  // get env variables to configure debugger
  const char *env_enable_str = std::getenv("ENABLE_MS_DEBUGGER");
  if (env_enable_str != nullptr) {
    MS_LOG(INFO) << "Getenv ENABLE_MS_DEBUGGER: " << env_enable_str;
    if (std::strcmp(env_enable_str, "1") == 0) {
      debugger_enabled_ = true;
    }
  }

  if (!debugger_enabled_ && !dump_enabled) {
    MS_LOG(WARNING) << "Not enabling debugger. Set environment variable ENABLE_MS_DEBUGGER=1 to enable debugger.";
    return;
  }

  // configure grpc host
  const char *env_host_str = std::getenv("MS_DEBUGGER_HOST");
  std::string host;
  if (env_host_str != nullptr) {
    MS_LOG(INFO) << "Getenv MS_DEBUGGER_HOST: " << env_host_str;
    host = std::string(env_host_str);
  } else {
    MS_LOG(WARNING) << "Environment variable MS_DEBUGGER_HOST doesn't exist. Using default debugger host: localhost";
    host = "localhost";
  }
  // configure grpc port
  const char *env_port_str = std::getenv("MS_DEBUGGER_PORT");
  std::string port;
  if (env_port_str != nullptr) {
    MS_LOG(INFO) << "Getenv MS_DEBUGGER_PORT: " << env_port_str;
    port = std::string(env_port_str);
  } else {
    MS_LOG(WARNING) << "Environment variable MS_DEBUGGER_PORT doesn't exist. Using default debugger port: 50051";
    port = "50051";
  }

  // configure partial memory reuse
  const char *env_partial_mem_str = std::getenv("MS_DEBUGGER_PARTIAL_MEM");
  if (env_partial_mem_str != nullptr) {
    MS_LOG(INFO) << "Getenv MS_DEBUGGER_PARTIAL_MEM: " << env_partial_mem_str;
    if (std::strcmp(env_partial_mem_str, "1") == 0) {
      partial_memory_ = true;
    }
  }
  // switch memory reuse on or off
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  context_ptr->set_enable_mem_reuse(partial_memory_);
  // print some message about memory reuse to user
  if (partial_memory_) {
    MS_LOG(WARNING) << "Partial Memory Reuse is enabled. Note: 1. Please only set watchpoints before running the first "
                       "step. 2. Tensor values are only available for nodes that are watched by any watchpoint.";
  } else {
    MS_LOG(WARNING) << "Memory Reuse is disabled. Set environment variable MS_DEBUGGER_PARTIAL_MEM=1 to reduce memory "
                       "usage for large models.";
  }

  // initialize grpc client
  if (debugger_enabled_) {
    grpc_client_ = std::make_unique<GrpcClient>(host, port);
  }

  debug_services_ = std::make_unique<DebugServices>();
}

void Debugger::Reset() {
  // access lock for public method
  std::lock_guard<std::mutex> a_lock(access_lock_);
  // reset components
  device_id_ = 0;
  device_target_ = "";
  num_step_ = 0;
  debugger_enabled_ = false;
  is_dataset_graph_ = false;
  partial_memory_ = false;
  graph_ptr_ = nullptr;
  grpc_client_ = nullptr;
  debug_services_ = nullptr;
}

void Debugger::PreExecute(const KernelGraphPtr &graph_ptr) {
  // access lock for public method
  std::lock_guard<std::mutex> a_lock(access_lock_);
  // check and save graph_ptr, suspend if graph is new
  CheckGraphPtr(graph_ptr);
}

void Debugger::PostExecute() {
  // access lock for public method
  std::lock_guard<std::mutex> a_lock(access_lock_);
  // analyze tensor data and send the watchpoints been hit
  if (run_level_ == "node") {
    MS_LOG(INFO) << "Debugger is in node level mode ";
    return;
  }
  if (debugger_enabled_ && !is_dataset_graph_) {
    if (device_target_ != kGPUDevice) {
      num_step_++;
      MS_LOG(INFO) << "Debugger suspend at end of step; number of steps executed: " << num_step_;
      SendWatchpointsAndSuspend(CheckWatchpoints());
    } else {
      CommandLoop();
    }
  }
}

bool Debugger::ReadNodeDataRequired() {
  if (debugger_enabled_ && !is_dataset_graph_) {
    auto watchpoint_table = debug_services_->GetWatchpointTable();
    auto is_watchpoint = debug_services_->IsWatchPoint(cur_name_, watchpoint_table);
    // if node has a watchpoint on it, is next_to node, or continue_to node then read the kernel tensor data
    if (is_watchpoint || (run_level_ == "node" && (node_name_ == "" || node_name_ == cur_name_))) {
      return true;
    }
  }
  return false;
}

void Debugger::PostExecuteNode() {
  // access lock for public method
  std::lock_guard<std::mutex> a_lock(access_lock_);
  if (debugger_enabled_ && !is_dataset_graph_) {
    auto watchpoint_table = debug_services_->GetWatchpointTable();
    auto is_watchpoint = debug_services_->IsWatchPoint(cur_name_, watchpoint_table);
    // if kernel is watchpoint,and get hit. suspend.
    if (is_watchpoint) {
      auto hits = CheckSingleWatchpoint(cur_name_);
      if (!hits.empty()) {
        SendWatchpointsAndSuspend(hits);
      }
    }
    // if kernel is not watchpoint and is next_to or continue_to node, suspend.
    if (run_level_ == "node" && (node_name_ == "" || node_name_ == cur_name_)) {
      CommandLoop();
    }
    return;
  }
}

void Debugger::PostDebugOp() {
  // access lock for public method
  std::lock_guard<std::mutex> a_lock(access_lock_);
  // suspend if debugger is enabled
  if (debugger_enabled_ && !is_dataset_graph_) {
    MS_LOG(INFO) << "Debugger suspend at debug_op";
    CommandLoop();
  }
}

void Debugger::CheckGraphPtr(const KernelGraphPtr &graph_ptr) {
  if (graph_ptr_ != graph_ptr) {
    MS_LOG(INFO) << "Debugger got new graph: " << graph_ptr->graph_id();
    // save new graph_ptr
    graph_ptr_ = graph_ptr;
    // check if it is dataset graph
    CheckDatasetGraph();
    if (!is_dataset_graph_) {
      // only try to enable debugger if it is not a dataset graph
      EnableDebugger();
      if (debugger_enabled_) {
        // get graph proto and send to mindinsight
        SendGraphAndSuspend(GetGraphProto());
      }
    }
  }
}

void Debugger::CheckDatasetGraph() {
  // print parameter node names
  const auto &params = graph_ptr_->inputs();
  for (const auto &param : params) {
    MS_LOG(INFO) << "param: " << param->fullname_with_scope();
  }
  // check if there is GetNext or InitDataSetQueue node
  const auto &nodes = graph_ptr_->execution_order();
  for (const auto &node : nodes) {
    auto node_name = AnfAlgo::GetCNodeName(node);
    MS_LOG(INFO) << "node: " << node->fullname_with_scope();
    if (node_name == "GetNext" || node_name == "InitDataSetQueue") {
      MS_LOG(WARNING) << "Not enabling debugger for graph " << graph_ptr_->graph_id() << ": found dataset graph node "
                      << node_name;
      is_dataset_graph_ = true;
      return;
    }
  }
  is_dataset_graph_ = false;
}

GraphProto Debugger::GetGraphProto() const {
  // convert kernel graph to debugger modelproto
  ModelProto model = GetDebuggerFuncGraphProto(graph_ptr_);
  return model.graph();
}

void Debugger::SendGraphAndSuspend(const GraphProto &graph_proto) {
  // prepare metadata
  std::string device_name = std::to_string(device_id_) + ":" + std::to_string(graph_ptr_->graph_id());
  Metadata metadata;
  metadata.set_device_name(device_name);
  metadata.set_cur_step(num_step_);
  metadata.set_backend(device_target_);
  metadata.set_cur_node(cur_name_);
  EventReply reply_metadata = grpc_client_->SendMetadata(metadata);
  if (reply_metadata.status() != reply_metadata.OK) {
    MS_LOG(ERROR) << "Error: SendMetadata failed";
  }
  // send graph to mindinght server
  EventReply reply = grpc_client_->SendGraph(graph_proto);
  if (reply.status() != reply.OK) {
    MS_LOG(ERROR) << "Error: SendGraph failed";
  }
  // enter command loop, wait and process commands
  CommandLoop();
}

void Debugger::CommandLoop() {
  // prepare metadata
  std::string device_name = std::to_string(device_id_) + ":" + std::to_string(graph_ptr_->graph_id());
  Metadata metadata;

  metadata.set_device_name(device_name);
  metadata.set_cur_step(num_step_);
  metadata.set_backend(device_target_);
  metadata.set_cur_node(cur_name_);

  // loop exit flag
  bool run = false;
  int num_wait_fail = 0;
  const int max_num_wait_fail = 5;

  while (!run) {
    // wait for command
    EventReply reply = grpc_client_->WaitForCommand(metadata);
    if (reply.status() != reply.OK) {
      MS_LOG(ERROR) << "Error: WaitForCommand failed";
      num_wait_fail++;
      if (num_wait_fail > max_num_wait_fail) {
        MS_LOG(ERROR) << "Maximum number of WaitForCommand retry reached: exiting training session";
        Exit();
      }
      MS_LOG(ERROR) << "Number of consecutive WaitForCommand fail:" << num_wait_fail << "; Retry after "
                    << num_wait_fail << "s";
      std::this_thread::sleep_for(std::chrono::milliseconds(1000 * num_wait_fail));
      continue;
    }

    // get type of the command in reply
    DebuggerCommand cmd = GetCommand(reply);
    if (cmd == DebuggerCommand::kUnknownCMD) {
      MS_LOG(ERROR) << "Error: debugger recieved unknown command";
      continue;
    }

    MS_LOG(INFO) << "recieved command: ";
    switch (cmd) {
      case DebuggerCommand::kUnknownCMD:
        MS_LOG(INFO) << "UnknownCMD";
        break;
      case DebuggerCommand::kExitCMD:
        MS_LOG(INFO) << "ExitCMD";
        Exit();
        break;
      case DebuggerCommand::kRunCMD:
        MS_LOG(INFO) << "RunCMD";
        {
          // print run cmd content
          // get run_level and node_name
          run_level_ = GetRunLevel(reply);
          node_name_ = GetNodeName(reply);

          MS_LOG(INFO) << "run_level: " << run_level_;
          MS_LOG(INFO) << "node_name_: " << node_name_;
        }

        // exit loop
        run = true;
        break;
      case DebuggerCommand::kSetCMD:
        MS_LOG(INFO) << "SetCMD";
        {
          // print set cmd content
          ProtoVector<WatchNode> recieved_nodes = GetWatchnodes(reply);
          for (auto node : recieved_nodes) {
            MS_LOG(INFO) << "node name: " << node.node_name();
            MS_LOG(INFO) << "node type: " << node.node_type();
          }
          MS_LOG(INFO) << "condition: " << GetWatchcondition(reply).condition();
          MS_LOG(INFO) << "id: " << GetWatchpointID(reply);
          MS_LOG(INFO) << "delete: " << GetWatchpointDelete(reply);
        }
        MS_LOG(INFO) << "Setting watchpoint";
        if (GetWatchpointDelete(reply)) {
          RemoveWatchpoint(GetWatchpointID(reply));
        } else {
          SetWatchpoint(GetWatchnodes(reply), GetWatchcondition(reply), GetWatchpointID(reply));
        }
        break;
      case DebuggerCommand::kViewCMD:
        MS_LOG(INFO) << "ViewCMD";
        {
          // print view cmd content
          ProtoVector<TensorProto> received_tensors = GetTensors(reply);
          for (auto tensor : received_tensors) {
            MS_LOG(INFO) << "tensor node name: " << tensor.node_name();
            MS_LOG(INFO) << "tensor slot: " << tensor.slot();
            MS_LOG(INFO) << "tensor finished: " << std::boolalpha << tensor.finished() << std::noboolalpha;
            MS_LOG(INFO) << "tensor iter: " << tensor.iter();
            MS_LOG(INFO) << "tensor truncate: " << std::boolalpha << tensor.truncate() << std::noboolalpha;
          }
        }
        MS_LOG(INFO) << "Sending tensors";
        std::list<TensorProto> tensors = LoadTensors(GetTensors(reply));
        {
          // print view cmd reply
          for (auto tensor : tensors) {
            MS_LOG(INFO) << "tensor node name: " << tensor.node_name();
            MS_LOG(INFO) << "tensor slot: " << tensor.slot();
            MS_LOG(INFO) << "tensor finished: " << std::boolalpha << tensor.finished() << std::noboolalpha;
            MS_LOG(INFO) << "tensor iter: " << tensor.iter();
            MS_LOG(INFO) << "tensor truncate: " << std::boolalpha << tensor.truncate() << std::noboolalpha;
            MS_LOG(INFO) << "tensor dims: ";
            for (auto dim : tensor.dims()) {
              MS_LOG(INFO) << dim << ",";
            }
            MS_LOG(INFO) << "tensor dtype: " << tensor.data_type();
          }
        }
        EventReply send_tensors_reply = grpc_client_->SendTensors(tensors);
        if (send_tensors_reply.status() != send_tensors_reply.OK) {
          MS_LOG(ERROR) << "Error: SendTensors failed";
        }
        break;
    }
  }
}

void Debugger::SetWatchpoint(const ProtoVector<WatchNode> &nodes, const WatchCondition &condition, const int32_t id) {
  std::vector<std::tuple<std::string, bool>> check_node_list;
  std::transform(nodes.begin(), nodes.end(), std::back_inserter(check_node_list),
                 [](WatchNode node) -> std::tuple<std::string, bool> {
                   return make_tuple(node.node_name(), node.node_type() == "scope");
                 });
  debug_services_->AddWatchpoint(id, condition.condition(), check_node_list);
}

void Debugger::RemoveWatchpoint(const int32_t id) { debug_services_->RemoveWatchpoint(id); }

std::list<TensorProto> Debugger::LoadTensors(const ProtoVector<TensorProto> &tensors) const {
  std::vector<std::string> name;
  std::vector<std::string> ret_name;
  std::vector<char *> data_ptr;
  std::vector<unsigned int> data_size;
  std::vector<TypePtr> dtype;
  std::vector<std::vector<int>> shape;

  std::transform(tensors.begin(), tensors.end(), std::back_inserter(name), GetTensorFullName);

  // ret_name will contain tensor names that are found in TensorLoader
  // items in ret_name will be in the same order with tensors if found
  debug_services_->ReadNodesTensors(name, &ret_name, &data_ptr, &data_size, &dtype, &shape);

  std::list<TensorProto> tensor_list;
  unsigned int result_index = 0;
  for (auto tensor : tensors) {
    TensorProto tensor_item;
    tensor_item.set_node_name(tensor.node_name());
    tensor_item.set_slot(tensor.slot());
    tensor_item.set_iter(tensor.iter());
    tensor_item.set_truncate(tensor.truncate());
    tensor_item.clear_tensor_content();
    tensor_item.clear_data_type();
    tensor_item.clear_dims();
    // always set finished to true before big tensor splitting is supported
    tensor_item.set_finished(true);

    // return empty tensor if didn't find the requested tensor
    if (result_index >= ret_name.size() || ret_name[result_index] != GetTensorFullName(tensor)) {
      tensor_list.push_back(tensor_item);
      continue;
    }

    tensor_item.set_tensor_content(data_ptr[result_index], data_size[result_index]);
    tensor_item.set_data_type(GetDebuggerNumberDataType(dtype[result_index]));
    for (auto &elem : shape[result_index]) {
      tensor_item.add_dims(elem);
    }

    // add tensor to result list and increment result_index to check next item in ret_name
    tensor_list.push_back(tensor_item);
    result_index++;
  }
  return tensor_list;
}

void Debugger::Exit() {
  // clear resource before exit
  pipeline::ClearResAtexit();
  std::exit(EXIT_FAILURE);
}

std::list<WatchpointHit> Debugger::CheckWatchpoints() const {
  std::vector<std::string> name;
  std::vector<std::string> slot;
  std::vector<char *> data_ptr;
  std::vector<unsigned int> data_size;
  std::vector<int> condition;
  std::vector<unsigned int> watchpoint_id;

  debug_services_->CheckWatchpoints(&name, &slot, &data_ptr, &data_size, &condition, &watchpoint_id);
  std::list<WatchpointHit> hits;
  for (unsigned int i = 0; i < name.size(); i++) {
    WatchpointHit hit;
    hit.set_id(watchpoint_id[i]);

    // here TensorProto act as a tensor indicator, not sending tensor content
    TensorProto *tensor_item = hit.mutable_tensor();
    tensor_item->set_node_name(name[i]);
    tensor_item->set_slot(slot[i]);
    tensor_item->set_finished(true);

    WatchCondition *condition_item = hit.mutable_watch_condition();
    condition_item->set_condition(debugger::WatchCondition_Condition(condition[i]));

    hits.push_back(hit);
  }
  return hits;
}

std::list<WatchpointHit> Debugger::CheckSingleWatchpoint(std::string watchnode) const {
  auto tensor_loader = debug_services_->tensor_loader();
  auto tensors = tensor_loader->GetNodeTensorMap(watchnode);
  std::list<WatchpointHit> hits;
  for (std::vector<std::shared_ptr<TensorData>>::iterator it = tensors.begin(); it != tensors.end(); ++it) {
    auto cur_tensor = *it;
    std::string name = "";
    std::string slot = "";
    char *data_ptr = nullptr;
    unsigned int data_size = 0;
    int condition = -1;
    unsigned int watchpoint_id = -1;
    WatchpointHit hit;
    debug_services_->CheckSingleWatchpoint(cur_tensor, &name, &slot, &data_ptr, &data_size, &condition, &watchpoint_id);
    if (name != "") {
      hit.set_id(watchpoint_id);
      // here TensorProto act as a tensor indicator, not sending tensor content
      TensorProto *tensor_item = hit.mutable_tensor();
      tensor_item->set_node_name(name);
      tensor_item->set_slot(slot);
      tensor_item->set_finished(true);
      WatchCondition *condition_item = hit.mutable_watch_condition();
      condition_item->set_condition(debugger::WatchCondition_Condition(condition));
      hits.push_back(hit);
    }
  }
  return hits;
}

void Debugger::SendWatchpointsAndSuspend(const std::list<WatchpointHit> &points) {
  // send info about watchpoint
  if (!points.empty()) {
    EventReply reply = grpc_client_->SendWatchpointHits(points);
    if (reply.status() != reply.OK) {
      MS_LOG(ERROR) << "Error: SendWatchpointHits failed";
    }
  }
  // enter command loop
  CommandLoop();
}

DebugServices *Debugger::debug_services() const { return debug_services_.get(); }

bool Debugger::debugger_enabled() const { return debugger_enabled_; }

DebuggerCommand GetCommand(const EventReply &reply) {
  DebuggerCommand cmd = DebuggerCommand::kUnknownCMD;
  switch (reply.cmd_case()) {
    case debugger::EventReply::CmdCase::kExit:
      cmd = DebuggerCommand::kExitCMD;
      break;
    case debugger::EventReply::CmdCase::kRunCmd:
      cmd = DebuggerCommand::kRunCMD;
      break;
    case debugger::EventReply::CmdCase::kSetCmd:
      cmd = DebuggerCommand::kSetCMD;
      break;
    case debugger::EventReply::CmdCase::kViewCmd:
      cmd = DebuggerCommand::kViewCMD;
      break;
    default:
      MS_LOG(ERROR) << "Error: UnknownCMD";
      break;
  }
  return cmd;
}

ProtoVector<WatchNode> GetWatchnodes(const EventReply &reply) {
  if (!reply.has_set_cmd()) {
    MS_LOG(ERROR) << "Error: Not SetCMD, can not get WatchNodes. Returning default value: ProtoVector<WatchNode>().";
    return ProtoVector<WatchNode>();
  }
  return reply.set_cmd().watch_nodes();
}

std::string GetRunLevel(const EventReply &reply) {
  if (!reply.has_run_cmd()) {
    MS_LOG(ERROR) << "Error: Not RunCMD, can not get RunLevel. Returning default value: "
                     "";
    return "";
  }
  return reply.run_cmd().run_level();
}

std::string GetNodeName(const EventReply &reply) {
  if (!reply.has_run_cmd()) {
    MS_LOG(ERROR) << "Error: Not RunCMD, can not get NodeName. Returning default value: "
                     "";
    return "";
  }
  return reply.run_cmd().node_name();
}

WatchCondition GetWatchcondition(const EventReply &reply) {
  if (!reply.has_set_cmd() || !reply.set_cmd().has_watch_condition()) {
    MS_LOG(ERROR) << "Error: Can not get WatchCondition from command. Returning default value: WatchCondition().";
    return WatchCondition();
  }
  return reply.set_cmd().watch_condition();
}

int32_t GetWatchpointID(const EventReply &reply) {
  if (!reply.has_set_cmd()) {
    MS_LOG(ERROR) << "Error: Not SetCMD, can not get Watchpoint ID. Returning default value: 0.";
    return 0;
  }
  return reply.set_cmd().id();
}

bool GetWatchpointDelete(const EventReply &reply) {
  if (!reply.has_set_cmd()) {
    MS_LOG(ERROR) << "Error: Not SetCMD, can not get Watchpoint delete flag. Returning default value: false.";
    return false;
  }
  return reply.set_cmd().delete_();
}

ProtoVector<TensorProto> GetTensors(const EventReply &reply) {
  if (!reply.has_view_cmd()) {
    MS_LOG(ERROR) << "Error: Not ViewCMD, can not get Tensors. Returning default value: ProtoVector<TensorProto>().";
    return ProtoVector<TensorProto>();
  }
  return reply.view_cmd().tensors();
}

std::string GetTensorFullName(const TensorProto &tensor) {
  string node_name = tensor.node_name();
  if (tensor.truncate()) {
    // scopes in node name are seperated by '/'
    // use the name without scope if truncate is true
    std::size_t found = node_name.find_last_of("/");
    node_name = node_name.substr(found + 1);
  }
  return node_name + ":" + tensor.slot() + (tensor.iter() == "" ? "" : ":" + tensor.iter());
}

bool Debugger::partial_memory() { return partial_memory_; }

void Debugger::SetCurNode(std::string cur_name) {
  // access lock for public method
  std::lock_guard<std::mutex> a_lock(access_lock_);
  cur_name_ = cur_name;
}

std::string Debugger::run_level() const { return run_level_; }

void Debugger::SetStepNum(int32_t cur_num_step) {
  // access lock for public method
  std::lock_guard<std::mutex> a_lock(access_lock_);
  num_step_ = cur_num_step;
}

int32_t Debugger::step_num() const { return num_step_; }

}  // namespace mindspore
