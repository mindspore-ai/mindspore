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

#include "ps/core/abstract_node.h"

namespace mindspore {
namespace ps {
namespace core {
void AbstractNode::Register(const std::shared_ptr<TcpClient> &client) {
  MessageMeta message_meta;
  message_meta.set_cmd(NodeCommand::REGISTER);

  RegisterMessage register_message;
  register_message.set_node_id(node_info_.node_id_);
  register_message.set_role(node_info_.node_role_);
  register_message.set_ip(node_info_.ip_);
  register_message.set_port(node_info_.port_);

  CommMessage comm_message;
  *comm_message.mutable_pb_meta() = {message_meta};
  comm_message.set_data(register_message.SerializeAsString());
  comm_message.set_user_cmd("");
  if (!SendMessageSync(client, comm_message)) {
    MS_LOG(EXCEPTION) << "The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                      << " the node id:" << node_info_.node_id_ << " register timeout!";
  }

  MS_LOG(INFO) << "The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
               << " the node id:" << node_info_.node_id_ << "is registering to scheduler!";
}

void AbstractNode::ProcessRegisterResp(const CommMessage &message) {
  RegisterRespMessage register_resp_message;
  register_resp_message.ParseFromString(message.data());
  if (register_resp_message.node_id() != node_info_.node_id_) {
    MS_LOG(EXCEPTION) << "The node id received:" << register_resp_message.node_id()
                      << " is not match the current node id:" << node_info_.node_id_;
  }

  node_info_.rank_id_ = register_resp_message.rank_id();

  MS_LOG(INFO) << "The node id is:" << node_info_.node_id_ << ", and the rank id is:" << node_info_.rank_id_;
}

bool AbstractNode::Broadcast(const enum NodeRole &node_role, const CommMessage &message, const uint32_t &timeout) {
  if (node_role != NodeRole::SERVER) {
    MS_LOG(EXCEPTION) << "Currently only supports broadcast to server nodes";
  }

  CommMessage &comm_message = const_cast<CommMessage &>(message);
  uint64_t request_id = ++next_request_id_;
  message_tracker_[request_id] = std::make_pair(nodes_address_.size(), 0);

  for (auto it = nodes_address_.begin(); it != nodes_address_.end(); ++it) {
    MessageMeta message_meta;
    message_meta.set_cmd(NodeCommand::SEND_DATA);
    message_meta.set_request_id(request_id);
    message_meta.set_rank_id(node_info_.rank_id_);
    message_meta.set_role(node_info_.node_role_);

    *comm_message.mutable_pb_meta() = {message_meta};
    auto client = GetOrCreateTcpClient((*it).first.second);
    client->SendMessage(comm_message);
  }
  MS_LOG(DEBUG) << "The node role is:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                << ", the node id is:" << node_info_.node_id_ << " send the request id is:" << request_id;
  return Wait(request_id, timeout);
}

void AbstractNode::set_event_callback(const OnNodeEventMessage &on_node_event_message) {
  on_node_event_message_ = on_node_event_message;
}

bool AbstractNode::Send(const enum NodeRole &node_role, const uint32_t &rank_id, const CommMessage &message,
                        const uint32_t &timeout) {
  if (!CommUtil::ValidateRankId(node_role, rank_id)) {
    MS_LOG(EXCEPTION) << "The node role or rank_id is illegal!";
  }

  CommMessage &comm_message = const_cast<CommMessage &>(message);

  MessageMeta message_meta;
  message_meta.set_cmd(NodeCommand::SEND_DATA);
  message_meta.set_rank_id(node_info_.rank_id_);
  message_meta.set_role(node_info_.node_role_);

  *comm_message.mutable_pb_meta() = {message_meta};
  auto client = GetOrCreateTcpClient(rank_id);
  return SendMessageSync(client, comm_message, timeout);
}

bool AbstractNode::Send(const NodeRole &node_role, const std::vector<uint32_t> &rank_ids,
                        const std::vector<CommMessage> &data, const uint32_t &timeout) {
  uint64_t request_id = ++next_request_id_;
  message_tracker_[request_id] = std::make_pair(data.size(), 0);

  if (rank_ids.size() != data.size()) {
    MS_LOG(EXCEPTION) << "The number of rank ids is not equal to the number of data!";
  }
  for (size_t it = 0; it < rank_ids.size(); ++it) {
    if (!CommUtil::ValidateRankId(node_role, rank_ids.at(it))) {
      MS_LOG(EXCEPTION) << "The node role or rank_id is illegal!";
    }

    MessageMeta message_meta;
    message_meta.set_cmd(NodeCommand::SEND_DATA);
    message_meta.set_request_id(request_id);
    message_meta.set_rank_id(node_info_.rank_id_);
    message_meta.set_role(node_info_.node_role_);

    CommMessage &comm_message = const_cast<CommMessage &>(data.at(it));
    *comm_message.mutable_pb_meta() = {message_meta};

    auto client = GetOrCreateTcpClient(rank_ids.at(it));
    client->SendMessage(comm_message);
  }
  MS_LOG(DEBUG) << "The node role is:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                << ", the node id is:" << node_info_.node_id_ << " send the request id is:" << request_id;
  return Wait(request_id, timeout);
}

bool AbstractNode::Send(const enum NodeRole &node_role, const uint32_t &rank_id, const CommMessage &message,
                        CommMessage *output, const uint32_t &timeout) {
  MS_EXCEPTION_IF_NULL(output);
  if (!CommUtil::ValidateRankId(node_role, rank_id)) {
    MS_LOG(EXCEPTION) << "The node role or rank_id is illegal!";
  }

  CommMessage &comm_message = const_cast<CommMessage &>(message);

  uint64_t request_id = ++next_request_id_;
  message_tracker_[request_id] = std::make_pair(1, 0);
  set_message_callback(request_id, [&]() {
    receive_messages_mutex_.lock();
    auto res = receive_messages_[request_id];
    *output = res[rank_id];
    receive_messages_.erase(request_id);
    receive_messages_mutex_.unlock();
  });

  MessageMeta message_meta;
  message_meta.set_cmd(NodeCommand::SEND_DATA);
  message_meta.set_request_id(request_id);
  message_meta.set_rank_id(node_info_.rank_id_);
  message_meta.set_role(node_info_.node_role_);

  *comm_message.mutable_pb_meta() = {message_meta};
  auto client = GetOrCreateTcpClient(rank_id);
  client->SendMessage(comm_message);
  MS_LOG(DEBUG) << "The node role is:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                << ", the node id is:" << node_info_.node_id_ << " send the request id is:" << request_id;
  return Wait(request_id, timeout);
}

bool AbstractNode::Send(const NodeRole &node_role, const std::vector<uint32_t> &rank_ids,
                        const std::vector<CommMessage> &data, std::vector<CommMessage> *output,
                        const uint32_t &timeout) {
  MS_EXCEPTION_IF_NULL(output);
  uint64_t request_id = ++next_request_id_;
  message_tracker_[request_id] = std::make_pair(data.size(), 0);

  if (rank_ids.size() != data.size()) {
    MS_LOG(EXCEPTION) << "The number of rank ids, data, comm_message_resp should be equal!";
  }

  size_t len = rank_ids.size();

  set_message_callback(request_id, [&]() {
    receive_messages_mutex_.lock();
    auto res = receive_messages_[request_id];
    for (size_t it = 0; it < len; ++it) {
      (*output).push_back(res[rank_ids.at(it)]);
    }
    receive_messages_.erase(request_id);
    receive_messages_mutex_.unlock();
  });

  for (size_t it = 0; it < len; ++it) {
    if (!CommUtil::ValidateRankId(node_role, rank_ids.at(it))) {
      MS_LOG(EXCEPTION) << "The node role or rank_id is illegal!";
    }

    MessageMeta message_meta;
    message_meta.set_cmd(NodeCommand::SEND_DATA);
    message_meta.set_request_id(request_id);
    message_meta.set_rank_id(node_info_.rank_id_);
    message_meta.set_role(node_info_.node_role_);

    CommMessage &comm_message = const_cast<CommMessage &>(data.at(it));
    *comm_message.mutable_pb_meta() = {message_meta};

    auto client = GetOrCreateTcpClient(rank_ids.at(it));
    client->SendMessage(comm_message);
  }
  MS_LOG(DEBUG) << "The node role is:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                << ", the node id is:" << node_info_.node_id_ << " send the request id is:" << request_id;
  return Wait(request_id, timeout);
}

bool AbstractNode::Wait(uint64_t request_id, const uint32_t &timeout) {
  std::unique_lock<std::mutex> lock(message_tracker_mutex_);
  bool res = message_tracker_cond_.wait_for(lock, std::chrono::seconds(timeout), [&] {
    bool ret = message_tracker_[request_id].first == message_tracker_[request_id].second;
    return ret;
  });
  message_tracker_.erase(request_id);
  return res;
}

uint64_t AbstractNode::CollectiveSendAsync(const enum NodeRole &node_role, const uint32_t &rank_id,
                                           const CommMessage &message) {
  if (!CommUtil::ValidateRankId(node_role, rank_id)) {
    MS_LOG(EXCEPTION) << "The node role or rank_id is illegal!";
  }

  CommMessage &comm_message = const_cast<CommMessage &>(message);

  MessageMeta message_meta;
  message_meta.set_cmd(NodeCommand::COLLECTIVE_SEND_DATA);
  message_meta.set_rank_id(node_info_.rank_id_);
  message_meta.set_role(node_info_.node_role_);

  *comm_message.mutable_pb_meta() = {message_meta};
  auto client = GetOrCreateTcpClient(rank_id);
  return SendMessageAsync(client, comm_message);
}

std::pair<uint32_t, uint64_t> AbstractNode::CollectiveReceiveAsync(const enum NodeRole &node_role,
                                                                   const uint32_t &rank_id, CommMessage *output) {
  if (!CommUtil::ValidateRankId(node_role, rank_id)) {
    MS_LOG(EXCEPTION) << "The node role or rank_id is illegal!";
  }

  uint64_t rank_request_id = NextExpectedRankRequestId(rank_id);
  if (received_data_.count(std::make_pair(rank_id, rank_request_id)) > 0) {
    *output = received_data_[std::make_pair(rank_id, rank_request_id)];
    received_data_.erase(std::make_pair(rank_id, rank_request_id));
  } else {
    set_receive_callback(rank_id, rank_request_id, [=]() {
      receive_callbacks_mutex_.lock();
      *output = received_data_[std::make_pair(rank_id, rank_request_id)];
      received_data_.erase(std::make_pair(rank_id, rank_request_id));
      receive_callbacks_mutex_.unlock();
    });
  }
  return std::make_pair(rank_id, rank_request_id);
}

bool AbstractNode::CollectiveWait(std::pair<uint32_t, uint64_t> request_id, const uint32_t &timeout) {
  std::unique_lock<std::mutex> lock(receive_callbacks_mutex_);
  bool res = receive_cond_.wait_for(lock, std::chrono::seconds(timeout), [&] {
    if (actual_rank_request_ids_.count(request_id.first) &&
        (actual_rank_request_ids_[request_id.first] >= request_id.second)) {
      return true;
    } else {
      return false;
    }
  });
  return res;
}

void AbstractNode::StartHeartbeatTimer(const std::shared_ptr<TcpClient> &client) {
  MS_LOG(INFO) << "The node role: " << CommUtil::NodeRoleToString(node_info_.node_role_)
               << ", the node id:" << node_info_.node_id_ << ", the node rank id:" << node_info_.rank_id_
               << " begin send heartbeat to the scheduler!";
  heart_beat_thread_ = std::make_unique<std::thread>([&]() {
    while (!is_finish_.load()) {
      if (!Heartbeat(client)) {
        MS_LOG(ERROR) << "The node role is:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                      << ", the node id is:" << node_info_.node_id_ << " Send heartbeat timeout!";
        if (!CheckSchedulerTimeout() && on_node_event_message_) {
          MS_LOG(ERROR) << "The node role is:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                        << ", the node id is:" << node_info_.node_id_ << " exited due to scheduler timeout!";
          is_finish_ = true;
          wait_finish_cond_.notify_all();
          on_node_event_message_(NodeEvent::SCHEDULER_TIMEOUT);
        }
      } else {
        UpdateSchedulerTime();
      }
      std::this_thread::sleep_for(std::chrono::seconds(ClusterConfig::heartbeat_interval()));
    }
  });
}

bool AbstractNode::Heartbeat(const std::shared_ptr<TcpClient> &client, bool is_node_finish) {
  MessageMeta meta;
  meta.set_cmd(NodeCommand::HEARTBEAT);

  HeartbeatMessage heartbeat_message;
  heartbeat_message.set_node_id(node_info_.node_id_);
  heartbeat_message.set_is_node_finish(is_node_finish);

  CommMessage message;
  *message.mutable_pb_meta() = {meta};
  message.set_data(heartbeat_message.SerializeAsString());
  if (!SendMessageSync(client, message)) {
    MS_LOG(ERROR) << "The node id:" << node_info_.node_id_ << " Send heartbeat timeout!";
  }
  return true;
}

void AbstractNode::UpdateSchedulerTime() {
  struct timeval current_time {};
  (void)gettimeofday(&current_time, nullptr);
  scheduler_time_ = current_time;
  MS_LOG(DEBUG) << "The node role: " << CommUtil::NodeRoleToString(node_info_.node_role_)
                << ", the node id:" << node_info_.node_id_ << ", the node rank id:" << node_info_.rank_id_
                << " update scheduler time, the current time is: " << current_time.tv_sec;
}

bool AbstractNode::CheckSchedulerTimeout() const {
  struct timeval current_time {};
  (void)gettimeofday(&current_time, nullptr);
  if (scheduler_time_.tv_sec + ClusterConfig::scheduler_timeout() < current_time.tv_sec) {
    return true;
  }
  return false;
}

void AbstractNode::ProcessHeartbeatResp(const CommMessage &message) {
  HeartbeatRespMessage heartbeat_resp_message;
  heartbeat_resp_message.ParseFromString(message.data());

  is_ready_ = heartbeat_resp_message.is_cluster_ready();
  if (is_ready_.load()) {
    wait_start_cond_.notify_all();
    MS_LOG(DEBUG) << "The node id:" << node_info_.node_id_ << " is ready!";
  }
  if (heartbeat_resp_message.is_cluster_finish()) {
    Heartbeat(client_to_scheduler_, true);
    is_finish_ = true;
    wait_finish_cond_.notify_all();
    MS_LOG(DEBUG) << "The node id:" << node_info_.node_id_ << " is finish!";
  }
  is_timeout_ = heartbeat_resp_message.is_cluster_timeout();
  if (is_timeout_ && on_node_event_message_) {
    is_ready_ = true;
    wait_start_cond_.notify_all();
    on_node_event_message_(NodeEvent::CLUSTER_TIMEOUT);
  }

  if (heartbeat_resp_message.is_node_timeout() && on_node_event_message_) {
    on_node_event_message_(NodeEvent::NODE_TIMEOUT);
  }
}

void AbstractNode::FetchServers(const std::shared_ptr<TcpClient> &client) {
  MessageMeta meta;
  meta.set_cmd(NodeCommand::FETCH_SERVER);

  CommMessage message;
  *message.mutable_pb_meta() = {meta};
  if (!SendMessageSync(client, message)) {
    MS_LOG(EXCEPTION) << "Fetch servers address timeout!";
  }
}

void AbstractNode::ProcessFetchServersResp(const CommMessage &message) {
  FetchServersRespMessage fetch_servers_resp_message;
  fetch_servers_resp_message.ParseFromString(message.data());

  for (const auto &it : fetch_servers_resp_message.servers_meta()) {
    nodes_address_[std::make_pair(NodeRole::SERVER, it.rank_id())] = std::make_pair(it.ip(), it.port());
  }

  MS_LOG(DEBUG) << "The all server host size is:" << nodes_address_.size();
}

bool AbstractNode::Disconnect(const std::shared_ptr<TcpClient> &client, const uint32_t &timeout) {
  MessageMeta meta;
  meta.set_cmd(NodeCommand::FINISH);

  FinishMessage finish_message;
  finish_message.set_node_id(node_info_.node_id_);

  CommMessage message;
  *message.mutable_pb_meta() = {meta};
  message.set_data(finish_message.SerializeAsString());
  if (!SendMessageSync(client, message)) {
    MS_LOG(ERROR) << "The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                  << " the node id:" << node_info_.node_id_ << " send Finish Message timeout!";
  }
  return WaitForDisconnect(timeout);
}

bool AbstractNode::WaitForDisconnect(const uint32_t &timeout) {
  std::unique_lock<std::mutex> lock(wait_finish_mutex_);
  bool res = wait_finish_cond_.wait_for(lock, std::chrono::seconds(timeout), [&] {
    if (is_finish_.load()) {
      MS_LOG(INFO) << "The node id:" << node_info_.node_id_ << " is success finish!";
    }
    return is_finish_.load();
  });
  return res;
}

bool AbstractNode::InitClientToScheduler() {
  std::string scheduler_host = ClusterConfig::scheduler_host();
  uint16_t scheduler_port = ClusterConfig::scheduler_port();
  client_to_scheduler_ = std::make_shared<TcpClient>(scheduler_host, scheduler_port);
  client_to_scheduler_->SetMessageCallback([&](const TcpClient &client, const CommMessage &message) {
    if (handlers_.count(message.pb_meta().cmd()) == 0) {
      MS_LOG(EXCEPTION) << "The cmd:" << message.pb_meta().cmd() << " is not supported!";
    }
    if (handlers_[message.pb_meta().cmd()] != nullptr) {
      const auto &handler_ptr = handlers_[message.pb_meta().cmd()];
      (this->*handler_ptr)(message);
    }
    NotifyMessageArrival(message);
  });

  client_to_scheduler_->Init();
  client_to_scheduler_thread_ = std::make_unique<std::thread>([&]() {
    MS_LOG(INFO) << "The node start a tcp client!";
    client_to_scheduler_->Start();
  });

  client_to_scheduler_->set_disconnected_callback([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(ClusterConfig::connect_interval()));
    client_to_scheduler_->Init();
  });
  return client_to_scheduler_->WaitConnected();
}

const std::shared_ptr<TcpClient> &AbstractNode::GetOrCreateTcpClient(const int &rank_id) {
  std::lock_guard<std::mutex> lock(client_mutex_);
  if (connected_nodes_.find(rank_id) != connected_nodes_.end()) {
    return connected_nodes_[rank_id];
  } else {
    if (nodes_address_.find(std::make_pair(NodeRole::SERVER, rank_id)) == nodes_address_.end()) {
      MS_LOG(EXCEPTION) << "Worker node Fetch servers failed!";
    }
    std::string ip = nodes_address_[std::make_pair(NodeRole::SERVER, rank_id)].first;
    uint16_t port = nodes_address_[std::make_pair(NodeRole::SERVER, rank_id)].second;
    auto client = std::make_shared<TcpClient>(ip, port);
    client->SetMessageCallback([&](const TcpClient &client, const CommMessage &message) {
      switch (message.pb_meta().cmd()) {
        case NodeCommand::SEND_DATA:
          ProcessSendDataResp(message);
          RunMessageCallback(message.pb_meta().request_id());
          break;
        case NodeCommand::COLLECTIVE_SEND_DATA:
          MS_LOG(INFO) << "The Node id:" << node_info_.node_id_ << " receive a collective_send_data message response!";
          break;
        default:
          MS_LOG(EXCEPTION) << "The cmd:" << message.pb_meta().cmd() << " is not supported!";
      }
      NotifyMessageArrival(message);
    });
    client->Init();
    connected_nodes_[rank_id] = client;
    return connected_nodes_[rank_id];
  }
}

bool AbstractNode::SendMessageSync(const std::shared_ptr<TcpClient> &client, const CommMessage &message,
                                   const uint32_t &timeout) {
  uint64_t request_id = ++next_request_id_;
  message_tracker_[request_id] = std::make_pair(1, 0);
  const_cast<CommMessage &>(message).mutable_pb_meta()->set_request_id(request_id);
  client->SendMessage(message);
  MS_LOG(DEBUG) << "The node role is:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                << ", the node id is:" << node_info_.node_id_ << " send the request id is:" << request_id;
  return Wait(request_id, timeout);
}

uint64_t AbstractNode::SendMessageAsync(const std::shared_ptr<TcpClient> &client, const CommMessage &message) {
  uint64_t request_id = ++next_request_id_;
  message_tracker_[request_id] = std::make_pair(1, 0);
  const_cast<CommMessage &>(message).mutable_pb_meta()->set_request_id(request_id);
  client->SendMessage(message);
  MS_LOG(DEBUG) << "The node role is:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                << ", the node id is:" << node_info_.node_id_ << " send the request id is:" << request_id;
  return request_id;
}

void AbstractNode::ProcessSendDataResp(const CommMessage &message) {
  std::lock_guard<std::mutex> lock(receive_messages_mutex_);
  const MessageMeta &message_meta = message.pb_meta();
  const uint32_t &rank_id = message_meta.rank_id();
  const uint64_t request_id = message_meta.request_id();
  MS_LOG(DEBUG) << "The node role is:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                << ", the node id is:" << node_info_.node_id_ << " send the request id is:" << request_id;
  auto it = receive_messages_.find(request_id);
  if (it != receive_messages_.end()) {
    it->second[rank_id] = message;
  } else {
    std::unordered_map<uint32_t, CommMessage> res;
    res.insert(std::make_pair(rank_id, message));
    receive_messages_[request_id] = res;
  }
}

void AbstractNode::RunMessageCallback(const uint64_t &request_id) {
  message_callbacks_mutex_.lock();
  // When receiving a message's response, Then compare with the desired number of responses,
  // If they are equal, then call the callback function
  if (message_tracker_[request_id].first == message_tracker_[request_id].second + 1) {
    auto it = message_callbacks_.find(request_id);
    if (it != message_callbacks_.end()) {
      message_callbacks_mutex_.unlock();

      if (it->second) {
        it->second();
      }

      message_callbacks_mutex_.lock();
      message_callbacks_.erase(it);
    }
  }
  message_callbacks_mutex_.unlock();
}

void AbstractNode::set_message_callback(const uint64_t &request_id, const MessageCallback &callback) {
  if (!callback) {
    return;
  }
  std::lock_guard<std::mutex> lock(message_callbacks_mutex_);
  message_callbacks_[request_id] = callback;
}

void AbstractNode::NotifyMessageArrival(const CommMessage &message) {
  std::lock_guard<std::mutex> lock(message_tracker_mutex_);
  const MessageMeta &message_meta = message.pb_meta();
  uint64_t request_id = message_meta.request_id();

  message_tracker_[request_id].second++;
  message_tracker_cond_.notify_all();
}

void AbstractNode::set_receive_callback(const uint32_t &rank_id, const uint64_t &request_id,
                                        const MessageCallback &callback) {
  if (!callback) {
    return;
  }
  std::lock_guard<std::mutex> lock(receive_callbacks_mutex_);
  receive_callbacks_[std::make_pair(rank_id, request_id)] = callback;
}

void AbstractNode::RunReceiveCallback(const CommMessage &message) {
  receive_callbacks_mutex_.lock();
  uint32_t rank_id = message.pb_meta().rank_id();
  // When receiving a collective message, Then generate rank request id,compare with the desired rank request id,
  // If they are equal, then call the callback function
  uint64_t rank_request_id = NextActualRankRequestId(rank_id);
  received_data_[std::make_pair(rank_id, rank_request_id)] = message;
  auto it = receive_callbacks_.find(std::make_pair(rank_id, rank_request_id));
  if (it != receive_callbacks_.end()) {
    receive_callbacks_mutex_.unlock();

    if (it->second) {
      it->second();
    }

    receive_callbacks_mutex_.lock();
    receive_cond_.notify_all();
    receive_callbacks_.erase(it);
  }
  receive_callbacks_mutex_.unlock();
}

uint64_t AbstractNode::NextExpectedRankRequestId(const uint32_t &rank_id) {
  std::lock_guard<std::mutex> lock(rank_request_ids_mutex);
  uint64_t rank_request_id = 1;
  if (expected_rank_request_ids_.count(rank_id)) {
    rank_request_id = ++expected_rank_request_ids_[rank_id];
    expected_rank_request_ids_[rank_id] = rank_request_id;
  } else {
    expected_rank_request_ids_[rank_id] = rank_request_id;
  }
  return rank_request_id;
}

uint64_t AbstractNode::NextActualRankRequestId(const uint32_t &rank_id) {
  std::lock_guard<std::mutex> lock(rank_request_ids_mutex);
  uint64_t rank_request_id = 1;
  if (actual_rank_request_ids_.count(rank_id)) {
    rank_request_id = ++actual_rank_request_ids_[rank_id];
    actual_rank_request_ids_[rank_id] = rank_request_id;
  } else {
    actual_rank_request_ids_[rank_id] = rank_request_id;
  }
  return rank_request_id;
}

void AbstractNode::InitCommandHandler() {
  handlers_[NodeCommand::HEARTBEAT] = &AbstractNode::ProcessHeartbeatResp;
  handlers_[NodeCommand::REGISTER] = &AbstractNode::ProcessRegisterResp;
  handlers_[NodeCommand::FETCH_SERVER] = &AbstractNode::ProcessFetchServersResp;
  handlers_[NodeCommand::FINISH] = nullptr;
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
