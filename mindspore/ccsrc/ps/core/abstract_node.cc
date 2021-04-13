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
  MS_EXCEPTION_IF_NULL(client);
  auto message_meta = std::make_shared<MessageMeta>();
  message_meta->set_cmd(NodeCommand::REGISTER);

  RegisterMessage register_message;
  register_message.set_node_id(node_info_.node_id_);
  register_message.set_role(node_info_.node_role_);
  register_message.set_ip(node_info_.ip_);
  register_message.set_port(node_info_.port_);

  if (!SendMessageSync(client, message_meta, Protos::PROTOBUF, register_message.SerializeAsString().data(),
                       register_message.ByteSizeLong())) {
    MS_LOG(EXCEPTION) << "The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                      << " the node id:" << node_info_.node_id_ << " register timeout!";
  }

  MS_LOG(INFO) << "The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
               << " the node id:" << node_info_.node_id_ << "is registering to scheduler!";
}

void AbstractNode::ProcessRegisterResp(std::shared_ptr<MessageMeta> meta, const void *data, size_t size) {
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  RegisterRespMessage register_resp_message;
  register_resp_message.ParseFromArray(data, size);
  if (register_resp_message.node_id() != node_info_.node_id_) {
    MS_LOG(EXCEPTION) << "The node id received:" << register_resp_message.node_id()
                      << " is not match the current node id:" << node_info_.node_id_;
  }

  if (register_resp_message.rank_id() < 0) {
    MS_LOG(EXCEPTION) << "The rank id is wrong.";
  }
  node_info_.rank_id_ = register_resp_message.rank_id();

  MS_LOG(INFO) << "The node id is:" << node_info_.node_id_ << ", and the rank id is:" << node_info_.rank_id_
               << " registered scheduler success!";
}

bool AbstractNode::Broadcast(const enum NodeRole &node_role, const DataPtr &message, size_t size, int command,
                             const uint32_t &timeout) {
  MS_EXCEPTION_IF_NULL(message);
  if (node_role != NodeRole::SERVER) {
    MS_LOG(EXCEPTION) << "Currently only supports broadcast to server nodes";
  }

  uint64_t request_id = AddMessageTrack(nodes_address_.size());

  for (auto it = nodes_address_.begin(); it != nodes_address_.end(); ++it) {
    auto message_meta = std::make_shared<MessageMeta>();
    message_meta->set_cmd(NodeCommand::SEND_DATA);
    message_meta->set_request_id(request_id);
    message_meta->set_rank_id(node_info_.rank_id_);
    message_meta->set_role(node_info_.node_role_);
    message_meta->set_user_cmd(command);

    auto client = GetOrCreateTcpClient((*it).first.second);
    client->SendMessage(message_meta, Protos::RAW, message.get(), size);
  }
  MS_LOG(DEBUG) << "The node role is:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                << ", the node id is:" << node_info_.node_id_ << " send the request id is:" << request_id;
  return Wait(request_id, timeout);
}

void AbstractNode::set_event_callback(const OnNodeEventMessage &on_node_event_message) {
  on_node_event_message_ = on_node_event_message;
}

bool AbstractNode::Send(const enum NodeRole &node_role, const uint32_t &rank_id, const DataPtr &data, size_t len,
                        int command, const uint32_t &timeout) {
  MS_EXCEPTION_IF_NULL(data);
  if (!CommUtil::ValidateRankId(node_role, rank_id)) {
    MS_LOG(EXCEPTION) << "The node role or rank_id is illegal!";
  }

  auto message_meta = std::make_shared<MessageMeta>();
  message_meta->set_cmd(NodeCommand::SEND_DATA);
  message_meta->set_rank_id(node_info_.rank_id_);
  message_meta->set_role(node_info_.node_role_);
  message_meta->set_user_cmd(command);

  auto client = GetOrCreateTcpClient(rank_id);
  return SendMessageSync(client, message_meta, Protos::RAW, data.get(), len, timeout);
}

bool AbstractNode::Send(const NodeRole &node_role, const std::vector<uint32_t> &rank_ids,
                        const std::vector<DataPtr> &data, const std::vector<size_t> &lens, int command,
                        const uint32_t &timeout) {
  uint64_t request_id = AddMessageTrack(data.size());

  if (rank_ids.size() != data.size() || rank_ids.size() != lens.size()) {
    MS_LOG(EXCEPTION) << "The number of rank ids, data and lens are not equal!";
  }
  for (size_t it = 0; it < rank_ids.size(); ++it) {
    if (!CommUtil::ValidateRankId(node_role, rank_ids.at(it))) {
      MS_LOG(EXCEPTION) << "The node role or rank_id is illegal!";
    }

    auto message_meta = std::make_shared<MessageMeta>();
    message_meta->set_cmd(NodeCommand::SEND_DATA);
    message_meta->set_request_id(request_id);
    message_meta->set_rank_id(node_info_.rank_id_);
    message_meta->set_role(node_info_.node_role_);
    message_meta->set_user_cmd(command);

    auto send = data.at(it);
    auto len = lens.at(it);
    auto client = GetOrCreateTcpClient(rank_ids.at(it));
    client->SendMessage(message_meta, Protos::RAW, send.get(), len);
  }
  MS_LOG(DEBUG) << "The node role is:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                << ", the node id is:" << node_info_.node_id_ << " send the request id is:" << request_id;
  return Wait(request_id, timeout);
}

bool AbstractNode::Send(const enum NodeRole &node_role, const uint32_t &rank_id, const DataPtr &message, size_t len,
                        int command, VectorPtr *output, const uint32_t &timeout) {
  MS_EXCEPTION_IF_NULL(message);
  MS_EXCEPTION_IF_NULL(output);
  if (!CommUtil::ValidateRankId(node_role, rank_id)) {
    MS_LOG(EXCEPTION) << "The node role or rank_id is illegal!";
  }

  uint64_t request_id = AddMessageTrack(1);
  set_message_callback(request_id, [&]() {
    receive_messages_mutex_.lock();
    auto res = receive_messages_[request_id];
    *output = res[rank_id];
    receive_messages_.erase(request_id);
    receive_messages_mutex_.unlock();
  });

  auto message_meta = std::make_shared<MessageMeta>();
  message_meta->set_cmd(NodeCommand::SEND_DATA);
  message_meta->set_request_id(request_id);
  message_meta->set_rank_id(node_info_.rank_id_);
  message_meta->set_role(node_info_.node_role_);
  message_meta->set_user_cmd(command);

  auto client = GetOrCreateTcpClient(rank_id);
  client->SendMessage(message_meta, Protos::RAW, message.get(), len);
  MS_LOG(DEBUG) << "The node role is:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                << ", the node id is:" << node_info_.node_id_ << " send the request id is:" << request_id;
  return Wait(request_id, timeout);
}

bool AbstractNode::Send(const NodeRole &node_role, const std::vector<uint32_t> &rank_ids,
                        const std::vector<DataPtr> &data, const std::vector<size_t> &data_lens, int command,
                        std::vector<VectorPtr> *output, const uint32_t &timeout) {
  MS_EXCEPTION_IF_NULL(output);
  uint64_t request_id = AddMessageTrack(data.size());

  if (rank_ids.size() != data.size()) {
    MS_LOG(EXCEPTION) << "The number of rank ids, data, comm_message_resp should be equal!";
  }

  size_t size = rank_ids.size();

  set_message_callback(request_id, [&]() {
    receive_messages_mutex_.lock();
    auto res = receive_messages_[request_id];
    for (size_t it = 0; it < size; ++it) {
      (*output).push_back(res[rank_ids.at(it)]);
    }
    receive_messages_.erase(request_id);
    receive_messages_mutex_.unlock();
  });

  for (size_t it = 0; it < size; ++it) {
    if (!CommUtil::ValidateRankId(node_role, rank_ids.at(it))) {
      MS_LOG(EXCEPTION) << "The node role or rank_id is illegal!";
    }

    auto message_meta = std::make_shared<MessageMeta>();
    message_meta->set_cmd(NodeCommand::SEND_DATA);
    message_meta->set_request_id(request_id);
    message_meta->set_rank_id(node_info_.rank_id_);
    message_meta->set_role(node_info_.node_role_);
    message_meta->set_user_cmd(command);

    auto send = data.at(it);
    auto len = data_lens.at(it);

    auto client = GetOrCreateTcpClient(rank_ids.at(it));
    client->SendMessage(message_meta, Protos::RAW, send.get(), len);
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

uint64_t AbstractNode::CollectiveSendAsync(const enum NodeRole &node_role, const uint32_t &rank_id, const void *data,
                                           size_t size) {
  MS_EXCEPTION_IF_NULL(data);
  if (!CommUtil::ValidateRankId(node_role, rank_id)) {
    MS_LOG(EXCEPTION) << "The node role or rank_id is illegal!";
  }

  std::shared_ptr<MessageMeta> message_meta = std::make_shared<MessageMeta>();
  message_meta->set_cmd(NodeCommand::COLLECTIVE_SEND_DATA);
  message_meta->set_rank_id(node_info_.rank_id_);
  message_meta->set_role(node_info_.node_role_);

  auto client = GetOrCreateTcpClient(rank_id);
  return SendMessageAsync(client, message_meta, Protos::RAW, data, size);
}

std::pair<uint32_t, uint64_t> AbstractNode::CollectiveReceiveAsync(const enum NodeRole &node_role,
                                                                   const uint32_t &rank_id, VectorPtr *output) {
  MS_EXCEPTION_IF_NULL(output);
  if (!CommUtil::ValidateRankId(node_role, rank_id)) {
    MS_LOG(EXCEPTION) << "The node role or rank_id is illegal!";
  }

  receive_callbacks_mutex_.lock();
  uint64_t rank_request_id = NextExpectedRankRequestId(rank_id);
  receive_messages_done_[std::make_pair(rank_id, rank_request_id)] = false;
  if (received_data_.count(std::make_pair(rank_id, rank_request_id)) > 0) {
    auto res = received_data_[std::make_pair(rank_id, rank_request_id)];
    *output = res;
    received_data_.erase(std::make_pair(rank_id, rank_request_id));
    receive_messages_done_[std::make_pair(rank_id, rank_request_id)] = true;
    MS_LOG(DEBUG) << "Receive data from rank id:" << rank_id << ", the rank request id is:" << rank_request_id;
  } else {
    receive_callbacks_[std::make_pair(rank_id, rank_request_id)] = [=]() mutable {
      receive_callbacks_mutex_.lock();
      auto res = received_data_[std::make_pair(rank_id, rank_request_id)];
      *output = res;
      received_data_.erase(std::make_pair(rank_id, rank_request_id));
      receive_messages_done_[std::make_pair(rank_id, rank_request_id)] = true;
      MS_LOG(DEBUG) << "Receive data from rank id:" << rank_id << ", the rank request id is:" << rank_request_id;
      receive_callbacks_mutex_.unlock();
    };
  }
  receive_callbacks_mutex_.unlock();
  return std::make_pair(rank_id, rank_request_id);
}

bool AbstractNode::CollectiveWait(std::pair<uint32_t, uint64_t> request_id, const uint32_t &timeout) {
  std::unique_lock<std::mutex> lock(receive_callbacks_mutex_);
  bool res =
    receive_cond_.wait_for(lock, std::chrono::seconds(timeout), [&] { return receive_messages_done_[request_id]; });
  return res;
}

void AbstractNode::StartHeartbeatTimer(const std::shared_ptr<TcpClient> &client) {
  MS_LOG(INFO) << "The node role: " << CommUtil::NodeRoleToString(node_info_.node_role_)
               << ", the node id:" << node_info_.node_id_ << ", the node rank id:" << node_info_.rank_id_
               << " begin send heartbeat to the scheduler!";
  heart_beat_thread_ = std::make_unique<std::thread>([&]() {
    while (!is_finish_.load()) {
      if (!Heartbeat(client)) {
        MS_LOG(WARNING) << "The node role is:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                        << ", the node id is:" << node_info_.node_id_ << " Send heartbeat timeout!";
        if (CheckSchedulerTimeout() && on_node_event_message_) {
          MS_LOG(WARNING) << "The node role is:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                          << ", the node id is:" << node_info_.node_id_ << " exited due to scheduler timeout!";
          is_finish_ = true;
          wait_finish_cond_.notify_all();
          on_node_event_message_(NodeEvent::SCHEDULER_TIMEOUT);
        }
      } else {
        UpdateSchedulerTime();
      }
      std::this_thread::sleep_for(std::chrono::seconds(ClusterMetadata::instance()->heartbeat_interval()));
    }
  });
  heart_beat_thread_->detach();
}

bool AbstractNode::Heartbeat(const std::shared_ptr<TcpClient> &client, bool is_node_finish) {
  auto meta = std::make_shared<MessageMeta>();
  meta->set_cmd(NodeCommand::HEARTBEAT);

  HeartbeatMessage heartbeat_message;
  heartbeat_message.set_node_id(node_info_.node_id_);
  heartbeat_message.set_is_node_finish(is_node_finish);

  if (!SendMessageSync(client, meta, Protos::PROTOBUF, heartbeat_message.SerializeAsString().data(),
                       heartbeat_message.ByteSizeLong())) {
    MS_LOG(WARNING) << "The node id:" << node_info_.node_id_ << " Send heartbeat timeout!";
    return false;
  }
  return true;
}

void AbstractNode::UpdateSchedulerTime() {
  struct timeval current_time {};
  (void)gettimeofday(&current_time, nullptr);
  scheduler_time_ = current_time;
  MS_LOG(DEBUG) << "Update scheduler time, the current time is: " << current_time.tv_sec;
}

bool AbstractNode::CheckSchedulerTimeout() const {
  struct timeval current_time {};
  (void)gettimeofday(&current_time, nullptr);
  if (scheduler_time_.tv_sec + ClusterMetadata::instance()->scheduler_timeout() < current_time.tv_sec) {
    return true;
  }
  return false;
}

void AbstractNode::ProcessHeartbeatResp(std::shared_ptr<MessageMeta> meta, const void *data, size_t size) {
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  HeartbeatRespMessage heartbeat_resp_message;
  heartbeat_resp_message.ParseFromArray(data, size);

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
  auto meta = std::make_shared<MessageMeta>();
  meta->set_cmd(NodeCommand::FETCH_SERVER);

  FetchServersMessage fetch_servers;
  fetch_servers.set_node_id(node_info_.node_id_);
  if (!SendMessageSync(client, meta, Protos::PROTOBUF, fetch_servers.SerializeAsString().data(),
                       fetch_servers.ByteSizeLong())) {
    MS_LOG(EXCEPTION) << "Fetch servers address timeout!";
  }
}

void AbstractNode::ProcessFetchServersResp(std::shared_ptr<MessageMeta> meta, const void *data, size_t size) {
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  FetchServersRespMessage fetch_servers_resp_message;
  fetch_servers_resp_message.ParseFromArray(data, size);

  for (const auto &it : fetch_servers_resp_message.servers_meta()) {
    nodes_address_[std::make_pair(NodeRole::SERVER, it.rank_id())] = std::make_pair(it.ip(), it.port());
  }

  MS_LOG(DEBUG) << "The all server host size is:" << nodes_address_.size();
}

bool AbstractNode::Disconnect(const std::shared_ptr<TcpClient> &client, const uint32_t &timeout) {
  auto meta = std::make_shared<MessageMeta>();
  meta->set_cmd(NodeCommand::FINISH);

  std::string finish_message = node_info_.node_id_;

  if (!SendMessageSync(client, meta, Protos::RAW, finish_message.data(), finish_message.length())) {
    MS_LOG(WARNING) << "The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
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
  std::string scheduler_host = ClusterMetadata::instance()->scheduler_host();
  uint16_t scheduler_port = ClusterMetadata::instance()->scheduler_port();
  client_to_scheduler_ = std::make_shared<TcpClient>(scheduler_host, scheduler_port);
  client_to_scheduler_->SetMessageCallback(
    [&](std::shared_ptr<MessageMeta> meta, const Protos &protos, const void *data, size_t size) {
      try {
        if (handlers_.count(meta->cmd()) == 0) {
          MS_LOG(EXCEPTION) << "The cmd:" << meta->cmd() << " is not supported!";
        }
        if (handlers_[meta->cmd()] != nullptr) {
          const auto &handler_ptr = handlers_[meta->cmd()];
          (this->*handler_ptr)(meta, data, size);
        }
        NotifyMessageArrival(meta);
      } catch (const std::exception &e) {
        MsException::Instance().SetException();
      }
    });

  client_to_scheduler_->Init();
  client_to_scheduler_thread_ = std::make_unique<std::thread>([&]() {
    MS_LOG(INFO) << "The node start a tcp client!";
    client_to_scheduler_->Start();
  });
  client_to_scheduler_thread_->detach();

  client_to_scheduler_->set_disconnected_callback([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(ClusterMetadata::instance()->connect_interval()));
    if (is_ready_.load() == false) {
      client_to_scheduler_->Init();
    }
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
    client->SetMessageCallback([&](std::shared_ptr<MessageMeta> meta, const Protos &protos, const void *data,
                                   size_t size) {
      switch (meta->cmd()) {
        case NodeCommand::SEND_DATA:
          ProcessSendDataResp(meta, protos, data, size);
          RunMessageCallback(meta->request_id());
          break;
        case NodeCommand::COLLECTIVE_SEND_DATA:
          MS_LOG(DEBUG) << "The Node id:" << node_info_.node_id_ << " receive a collective_send_data message response!";
          break;
        default:
          MS_LOG(EXCEPTION) << "The cmd:" << meta->cmd() << " is not supported!";
      }
      NotifyMessageArrival(meta);
    });
    client->Init();
    connected_nodes_[rank_id] = client;
    return connected_nodes_[rank_id];
  }
}

bool AbstractNode::SendMessageSync(const std::shared_ptr<TcpClient> &client, const CommMessage &message,
                                   const uint32_t &timeout) {
  uint64_t request_id = AddMessageTrack(1);
  const_cast<CommMessage &>(message).mutable_pb_meta()->set_request_id(request_id);
  client->SendMessage(message);
  MS_LOG(DEBUG) << "The node role is:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                << ", the node id is:" << node_info_.node_id_ << " send the request id is:" << request_id;
  return Wait(request_id, timeout);
}

uint64_t AbstractNode::SendMessageAsync(const std::shared_ptr<TcpClient> &client, std::shared_ptr<MessageMeta> meta,
                                        const Protos &protos, const void *data, size_t size) {
  MS_EXCEPTION_IF_NULL(client);
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  uint64_t request_id = AddMessageTrack(1);
  meta->set_request_id(request_id);
  client->SendMessage(meta, protos, data, size);
  MS_LOG(DEBUG) << "The node role is:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                << ", the node id is:" << node_info_.node_id_ << " send the request id is:" << request_id;
  return request_id;
}

bool AbstractNode::SendMessageSync(const std::shared_ptr<TcpClient> &client, std::shared_ptr<MessageMeta> meta,
                                   const Protos &protos, const void *data, size_t size, const uint32_t &timeout) {
  MS_EXCEPTION_IF_NULL(client);
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  uint64_t request_id = AddMessageTrack(1);
  meta->set_request_id(request_id);
  client->SendMessage(meta, protos, data, size);
  MS_LOG(DEBUG) << "The node role is:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                << ", the node id is:" << node_info_.node_id_ << " send the request id is:" << request_id;
  return Wait(request_id, timeout);
}

void AbstractNode::ProcessSendDataResp(std::shared_ptr<MessageMeta> meta, const Protos &protos, const void *data,
                                       size_t size) {
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  std::lock_guard<std::mutex> lock(receive_messages_mutex_);
  const uint32_t &rank_id = meta->rank_id();
  const uint64_t request_id = meta->request_id();
  MS_LOG(DEBUG) << "The node role is:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                << ", the node id is:" << node_info_.node_id_ << " send the request id is:" << request_id;
  auto it = receive_messages_.find(request_id);
  VectorPtr received_data = std::make_shared<std::vector<unsigned char>>(size, 0);
  if (size > 0) {
    size_t dest_size = size;
    size_t src_size = size;
    auto ret = memcpy_s(received_data.get()->data(), dest_size, data, src_size);
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "The memcpy_s error, errorno(" << ret << ")";
    }
  }
  if (it != receive_messages_.end()) {
    it->second[rank_id] = received_data;
  } else {
    std::unordered_map<uint32_t, VectorPtr> res;
    res.insert(std::make_pair(rank_id, received_data));
    receive_messages_[request_id] = res;
  }
}

void AbstractNode::RunMessageCallback(const uint64_t &request_id) {
  message_callbacks_mutex_.lock();
  // When receiving a message's response, Then compare with the desired number of responses,
  // If they are equal, then call the callback function
  if (CheckMessageTrack(request_id)) {
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

void AbstractNode::NotifyMessageArrival(std::shared_ptr<MessageMeta> meta) {
  std::lock_guard<std::mutex> lock(message_tracker_mutex_);
  uint64_t request_id = meta->request_id();

  message_tracker_[request_id].second++;
  message_tracker_cond_.notify_all();
}

void AbstractNode::RunReceiveCallback(std::shared_ptr<MessageMeta> meta, const Protos &protos, const void *data,
                                      size_t size) {
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  receive_callbacks_mutex_.lock();
  uint32_t rank_id = meta->rank_id();
  // When receiving a collective message, Then generate rank request id,compare with the desired rank request id,
  // If they are equal, then call the callback function
  uint64_t rank_request_id = NextActualRankRequestId(rank_id);
  std::shared_ptr<std::vector<unsigned char>> received_data = std::make_shared<std::vector<unsigned char>>(size, 0);
  size_t dest_size = size;
  size_t src_size = size;
  int ret = memcpy_s(received_data->data(), dest_size, data, src_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "The memcpy_s error, errorno(" << ret << ")";
  }
  received_data_[std::make_pair(rank_id, rank_request_id)] = received_data;
  MS_LOG(DEBUG) << "Run Receive data callback,the rank id:" << rank_id << ", the rank request id is:" << rank_request_id
                << ", the send request id is:" << meta->request_id() << " the size is:" << size;
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

uint64_t AbstractNode::AddMessageTrack(const uint32_t &expected_response) {
  std::lock_guard<std::mutex> lock(message_tracker_mutex_);
  uint64_t request_id = ++next_request_id_;
  message_tracker_[request_id] = std::make_pair(expected_response, 0);
  return request_id;
}

bool AbstractNode::CheckMessageTrack(const uint64_t &request_id) {
  std::lock_guard<std::mutex> lock(message_tracker_mutex_);
  return message_tracker_[request_id].first == message_tracker_[request_id].second + 1;
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
