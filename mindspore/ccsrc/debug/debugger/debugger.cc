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
#include "pipeline/pipeline.h"
#include "session/anf_runtime_algorithm.h"

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
      num_step_(0),
      debugger_enabled_(false),
      is_dataset_graph_(false) {}

void Debugger::Init(const uint32_t device_id) {
  // access lock for public method
  std::lock_guard<std::mutex> a_lock(access_lock_);
  // save device_id
  MS_LOG(INFO) << "Debugger got device_id: " << device_id;
  device_id_ = device_id;
}

void Debugger::EnableDebugger() {
  // reset some of the class members
  num_step_ = 0;
  debugger_enabled_ = false;
  grpc_client_ = nullptr;
  debug_services_ = nullptr;

  // get env variables to configure debugger
  const char *env_enable_str = std::getenv("ENABLE_MS_DEBUGGER");
  if (env_enable_str != nullptr) {
    MS_LOG(INFO) << "Getenv ENABLE_MS_DEBUGGER: " << env_enable_str;
    if (std::strcmp(env_enable_str, "1") == 0) {
      debugger_enabled_ = true;
    }
  }
  if (!debugger_enabled_) {
    MS_LOG(WARNING) << "Not enabling debugger. Set environment variable ENABLE_MS_DEBUGGER=1 to enable debugger.";
    return;
  }
  // configure host
  const char *env_host_str = std::getenv("MS_DEBUGGER_HOST");
  std::string host;
  if (env_host_str != nullptr) {
    MS_LOG(INFO) << "Getenv MS_DEBUGGER_HOST: " << env_host_str;
    host = std::string(env_host_str);
  } else {
    MS_LOG(WARNING) << "Environment variable MS_DEBUGGER_HOST doesn't exist. Using default debugger host: localhost";
    host = "localhost";
  }
  // configure port
  const char *env_port_str = std::getenv("MS_DEBUGGER_PORT");
  std::string port;
  if (env_port_str != nullptr) {
    MS_LOG(INFO) << "Getenv MS_DEBUGGER_PORT: " << env_port_str;
    port = std::string(env_port_str);
  } else {
    MS_LOG(WARNING) << "Environment variable MS_DEBUGGER_PORT doesn't exist. Using default debugger port: 50051";
    port = "50051";
  }

  // initialize grpc client
  grpc_client_ = std::make_unique<GrpcClient>(host, port);
  debug_services_ = std::make_unique<DebugServices>();
}

void Debugger::Reset() {
  // access lock for public method
  std::lock_guard<std::mutex> a_lock(access_lock_);
  // reset components
  device_id_ = 0;
  num_step_ = 0;
  debugger_enabled_ = false;
  is_dataset_graph_ = false;
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
  if (debugger_enabled_ && !is_dataset_graph_) {
    num_step_++;
    MS_LOG(INFO) << "Debugger suspend at end of step; number of steps executed: " << num_step_;
    SendWatchpointsAndSuspend(CheckWatchpoints());
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

GraphProto Debugger::GetGraphProto() {
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
          WatchCondition recieved_condition = GetWatchcondition(reply);
          MS_LOG(INFO) << "condition: " << recieved_condition.condition();
          int32_t id = GetWatchpointID(reply);
          MS_LOG(INFO) << "id: " << id;
          bool delete_ = GetWatchpointDelete(reply);
          MS_LOG(INFO) << "delete: " << delete_;
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
          }
        }
        MS_LOG(INFO) << "Sending tensors";
        std::list<TensorProto> tensors = LoadTensors(GetTensors(reply));
        {
          for (auto tensor : tensors) {
            MS_LOG(INFO) << "tensor node name: " << tensor.node_name();
            MS_LOG(INFO) << "tensor slot: " << tensor.slot();
            MS_LOG(INFO) << "tensor finished: " << std::boolalpha << tensor.finished() << std::noboolalpha;
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

DebuggerCommand Debugger::GetCommand(const EventReply &reply) {
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

ProtoVector<WatchNode> Debugger::GetWatchnodes(const EventReply &reply) {
  if (!reply.has_set_cmd()) {
    MS_LOG(ERROR) << "Error: Not SetCMD, can not get WatchNodes. Returning default value: ProtoVector<WatchNode>().";
    return ProtoVector<WatchNode>();
  }
  return reply.set_cmd().watch_nodes();
}

WatchCondition Debugger::GetWatchcondition(const EventReply &reply) {
  if (!reply.has_set_cmd() || !reply.set_cmd().has_watch_condition()) {
    MS_LOG(ERROR) << "Error: Can not get WatchCondition from command. Returning default value: WatchCondition().";
    return WatchCondition();
  }
  return reply.set_cmd().watch_condition();
}

int32_t Debugger::GetWatchpointID(const EventReply &reply) {
  if (!reply.has_set_cmd()) {
    MS_LOG(ERROR) << "Error: Not SetCMD, can not get Watchpoint ID. Returning default value: 0.";
    return 0;
  }
  return reply.set_cmd().id();
}

bool Debugger::GetWatchpointDelete(const EventReply &reply) {
  if (!reply.has_set_cmd()) {
    MS_LOG(ERROR) << "Error: Not SetCMD, can not get Watchpoint delete flag. Returning default value: false.";
    return false;
  }
  return reply.set_cmd().delete_();
}

ProtoVector<TensorProto> Debugger::GetTensors(const EventReply &reply) {
  if (!reply.has_view_cmd()) {
    MS_LOG(ERROR) << "Error: Not ViewCMD, can not get Tensors. Returning default value: ProtoVector<TensorProto>().";
    return ProtoVector<TensorProto>();
  }
  return reply.view_cmd().tensors();
}

void Debugger::SetWatchpoint(const ProtoVector<WatchNode> &nodes, const WatchCondition &condition, const int32_t id) {
  std::vector<std::tuple<std::string, bool>> check_node_list;
  std::transform(nodes.begin(), nodes.end(), std::back_inserter(check_node_list),
                 [](WatchNode node) -> std::tuple<std::string, bool> {
                   return make_tuple(node.node_name(), node.node_type() == "scope");
                 });

  debug_services_->add_watchpoint(id, condition.condition(), check_node_list);
}

void Debugger::RemoveWatchpoint(const int32_t id) { debug_services_->remove_watchpoint(id); }

std::list<TensorProto> Debugger::LoadTensors(const ProtoVector<TensorProto> &tensors) {
  std::vector<std::string> name;
  std::vector<std::string> ret_name;
  std::vector<char *> data_ptr;
  std::vector<unsigned int> data_size;
  std::vector<TypePtr> dtype;
  std::vector<std::vector<int>> shape;

  std::transform(tensors.begin(), tensors.end(), std::back_inserter(name),
                 [](TensorProto tensor) -> std::string { return tensor.node_name() + ":" + tensor.slot(); });

  debug_services_->read_nodes_tensors(name, &ret_name, &data_ptr, &data_size, &dtype, &shape);

  std::list<TensorProto> tensor_list;
  unsigned int result_index = 0;
  TensorProto tensor_item;

  for (auto tensor : tensors) {
    tensor_item.set_node_name(tensor.node_name());
    tensor_item.set_slot(tensor.slot());
    tensor_item.set_finished(true);

    // return empty tensor if didn't find the requested tensor
    if (result_index >= ret_name.size() || ret_name[result_index] != tensor.node_name() + ":" + tensor.slot()) {
      tensor_list.push_back(tensor_item);
      continue;
    }

    tensor_item.set_tensor_content(data_ptr[result_index], data_size[result_index]);
    tensor_item.set_data_type(GetDebuggerNumberDataType(dtype[result_index]));
    tensor_item.clear_dims();
    for (auto &elem : shape[result_index]) {
      tensor_item.add_dims(elem);
    }

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

std::list<WatchpointHit> Debugger::CheckWatchpoints() {
  std::vector<std::string> name;
  std::vector<std::string> slot;
  std::vector<char *> data_ptr;
  std::vector<unsigned int> data_size;
  std::vector<int> condition;
  std::vector<unsigned int> watchpoint_id;

  debug_services_->check_watchpoints(&name, &slot, &data_ptr, &data_size, &condition, &watchpoint_id);

  std::list<WatchpointHit> points;

  for (unsigned int i = 0; i < name.size(); i++) {
    TensorProto *tensor_item;
    tensor_item = new TensorProto();
    tensor_item->set_node_name(name[i]);
    tensor_item->set_slot(slot[i]);
    tensor_item->set_tensor_content(data_ptr[i], data_size[i]);

    // finished in TensorProto will always be true before we implement big tensor splitting
    tensor_item->set_finished(true);

    WatchCondition *condition_item;
    condition_item = new WatchCondition();
    condition_item->set_condition(debugger::WatchCondition_Condition(condition[i]));

    WatchpointHit point;
    point.set_allocated_tensor(tensor_item);
    point.set_allocated_watch_condition(condition_item);
    point.set_id(watchpoint_id[i]);

    points.push_back(point);
  }

  return points;
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

DebugServices *Debugger::get_debug_services() { return debug_services_.get(); }

bool Debugger::debugger_enabled() { return debugger_enabled_; }

}  // namespace mindspore
