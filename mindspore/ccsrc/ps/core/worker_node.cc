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

#include "ps/core/worker_node.h"

namespace mindspore {
namespace ps {
namespace core {
WorkerNode::~WorkerNode() {
  MS_LOG(INFO) << "Stop worker node!";
  if (!is_already_stopped_.load()) {
    is_ready_ = true;
    is_timeout_ = true;
    client_to_scheduler_->Stop();
    if (!connected_nodes_.empty()) {
      for (auto &connected_node : connected_nodes_) {
        connected_node.second->Stop();
      }
    }
    client_to_scheduler_->StopEventBase();
    if (worker_thread_->joinable()) {
      worker_thread_->join();
    }
    if (heart_beat_thread_->joinable()) {
      heart_beat_thread_->join();
    }
    is_already_stopped_ = true;
  }
}
bool WorkerNode::Start(const uint32_t &timeout) {
  MS_LOG(INFO) << "Starting worker node!";
  Initialize();
  Register();
  Heartbeat(client_to_scheduler_);

  if (!WaitForStart(timeout)) {
    MS_LOG(ERROR) << "Start Worker node timeout!";
    return false;
  }
  MS_LOG(INFO) << "The node is ready to fetch servers!";

  if (!is_timeout_.load()) {
    FetchServers(client_to_scheduler_);
    MS_LOG(INFO) << "Fetch servers successful!";
  }
  MS_LOG(INFO) << "The Worker node has successfully started.";
  return true;
}

void WorkerNode::Register() {
  MessageMeta message_meta;
  message_meta.set_cmd(NodeCommand::REGISTER);

  RegisterMessage register_message;
  register_message.set_node_id(node_info_.node_id_);
  register_message.set_role(node_info_.node_role_);

  CommMessage comm_message;
  *comm_message.mutable_pb_meta() = {message_meta};
  comm_message.set_data(register_message.SerializeAsString());
  if (!SendMessageSync(client_to_scheduler_, comm_message)) {
    MS_LOG(EXCEPTION) << "Worker node register timeout!";
  }
  MS_LOG(INFO) << "The worker node id:" << node_info_.node_id_
               << "is registering to scheduler, the request id is:" << message_meta.request_id();
}

void WorkerNode::ProcessRegisterResp(const CommMessage &message) {
  RegisterRespMessage register_resp_message;
  register_resp_message.ParseFromString(message.data());
  if (register_resp_message.node_id() != node_info_.node_id_) {
    MS_LOG(EXCEPTION) << "The node id received:" << register_resp_message.node_id()
                      << " is not match the current node id:" << node_info_.node_id_;
  }

  node_info_.rank_id_ = register_resp_message.rank_id();

  MS_LOG(INFO) << "The client node id is:" << node_info_.node_id_ << ", and the rank id is:" << node_info_.rank_id_;
}

void WorkerNode::Initialize() {
  is_already_stopped_ = false;
  node_info_.node_id_ = CommUtil::GenerateUUID();
  node_info_.node_role_ = NodeRole::WORKER;
  MS_LOG(INFO) << "The node role is:" << CommUtil::NodeRoleToString(node_info_.node_role_)
               << ", the node id is:" << node_info_.node_id_;
  InitClientToScheduler();
}

void WorkerNode::InitClientToScheduler() {
  std::string scheduler_host = ClusterConfig::scheduler_host();
  uint16_t scheduler_port = ClusterConfig::scheduler_port();
  client_to_scheduler_ = std::make_shared<TcpClient>(scheduler_host, scheduler_port);
  client_to_scheduler_->SetMessageCallback([&](const TcpClient &client, const CommMessage &message) {
    switch (message.pb_meta().cmd()) {
      case NodeCommand::HEARTBEAT:
        ProcessHeartbeatResp(message);
        break;
      case NodeCommand::REGISTER:
        ProcessRegisterResp(message);
        break;
      case NodeCommand::FETCH_SERVER:
        ProcessFetchServersResp(message);
        break;
      case NodeCommand::FINISH:
        MS_LOG(INFO) << "The Node id:" << node_info_.node_id_ << " receive a finish message response!";
        break;
      default:
        MS_LOG(EXCEPTION) << "The cmd:" << message.pb_meta().cmd() << " is not supported!";
    }
    NotifyMessageArrival(message);
  });

  client_to_scheduler_->Init();
  worker_thread_ = std::make_unique<std::thread>([&]() {
    MS_LOG(INFO) << "The worker node start a tcp client!";
    client_to_scheduler_->Start();
  });
  worker_thread_->detach();
}

bool WorkerNode::Stop() {
  MS_LOG(INFO) << "Stop worker node!";
  if (!is_already_stopped_.load()) {
    is_ready_ = true;
    is_timeout_ = true;
    client_to_scheduler_->Stop();
    if (!connected_nodes_.empty()) {
      for (auto &connected_node : connected_nodes_) {
        connected_node.second->Stop();
      }
    }
    client_to_scheduler_->StopEventBase();
    if (worker_thread_->joinable()) {
      worker_thread_->join();
    }
    if (heart_beat_thread_->joinable()) {
      heart_beat_thread_->join();
    }
    is_already_stopped_ = true;
  }
  return true;
}

bool WorkerNode::Finish(const uint32_t &timeout) {
  std::lock_guard<std::mutex> lock(finish_mutex_);
  if (is_already_finished_) {
    MS_LOG(INFO) << "Worker node already finish!";
    return true;
  }
  MS_LOG(INFO) << "Finish worker node!";
  is_already_finished_ = true;
  return Disconnect(client_to_scheduler_, timeout);
}

bool WorkerNode::BroadcastToServers(const std::string &message) {
  uint64_t request_id = ++next_request_id_;
  message_tracker_[request_id] = std::make_pair(nodes_address_.size(), 0);
  for (auto it = nodes_address_.begin(); it != nodes_address_.end(); ++it) {
    MessageMeta message_meta;
    message_meta.set_cmd(NodeCommand::SEND_DATA);
    message_meta.set_request_id(request_id);

    CommMessage comm_message;
    *comm_message.mutable_pb_meta() = {message_meta};
    comm_message.set_data(message);
    auto client = GetOrCreateTcpClient((*it).first.second);
    client->SendMessage(comm_message);
  }
  return Wait(request_id);
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
