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

#include "ps/core/node.h"

namespace mindspore {
namespace ps {
namespace core {
std::string Node::node_id() const { return node_info_.node_id_; }

uint32_t Node::rank_id() const { return node_info_.rank_id_; }

NodeRole Node::role() const { return node_info_.node_role_; }

bool Node::Wait(uint64_t request_id, const uint32_t &timeout) {
  std::unique_lock<std::mutex> lock(message_tracker_mutex_);
  bool res = message_tracker_cond_.wait_for(lock, std::chrono::seconds(timeout), [&] {
    bool ret = message_tracker_[request_id].first == message_tracker_[request_id].second;
    return ret;
  });
  message_tracker_.erase(request_id);
  return res;
}

bool Node::Send(const enum NodeRole &node_role, const uint32_t &rank_id, const std::string &message,
                const uint32_t &timeout) {
  if (!CommUtil::ValidateRankId(node_role, rank_id)) {
    MS_LOG(EXCEPTION) << "The node role or rank_id is illegal!";
  }

  MessageMeta message_meta;
  message_meta.set_cmd(NodeCommand::SEND_DATA);

  CommMessage comm_message;
  *comm_message.mutable_pb_meta() = {message_meta};
  comm_message.set_data(message);
  auto client = GetOrCreateTcpClient(rank_id);
  return SendMessageSync(client, comm_message);
}

bool Node::Send(const NodeRole &node_role, const std::vector<uint32_t> &rank_ids, const std::vector<std::string> &data,
                const uint32_t &timeout) {
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

    CommMessage comm_message;
    *comm_message.mutable_pb_meta() = {message_meta};
    comm_message.set_data(data.at(it));

    auto client = GetOrCreateTcpClient(rank_ids.at(it));
    client->SendMessage(comm_message);
  }
  return Wait(request_id, timeout);
}

bool Node::Send(const enum NodeRole &node_role, const uint32_t &rank_id, const std::string &message,
                CommMessage *comm_message_resp, const uint32_t &timeout) {
  MS_EXCEPTION_IF_NULL(comm_message_resp);
  if (!CommUtil::ValidateRankId(node_role, rank_id)) {
    MS_LOG(EXCEPTION) << "The node role or rank_id is illegal!";
  }

  uint64_t request_id = ++next_request_id_;
  message_tracker_[request_id] = std::make_pair(1, 0);
  set_message_callback(request_id, [&]() {
    receive_messages_mutex_.lock();
    auto res = receive_messages_[request_id];
    *comm_message_resp = res[rank_id];
    receive_messages_.erase(request_id);
    receive_messages_mutex_.unlock();
  });

  MessageMeta message_meta;
  message_meta.set_cmd(NodeCommand::SEND_DATA);
  message_meta.set_request_id(request_id);
  message_meta.set_rank_id(node_info_.rank_id_);
  message_meta.set_role(node_info_.node_role_);

  CommMessage comm_message;
  *comm_message.mutable_pb_meta() = {message_meta};
  comm_message.set_data(message);
  auto client = GetOrCreateTcpClient(rank_id);
  client->SendMessage(comm_message);
  return Wait(request_id, timeout);
}

bool Node::Send(const NodeRole &node_role, const std::vector<uint32_t> &rank_ids, const std::vector<std::string> &data,
                std::vector<CommMessage *> *comm_message_resp, const uint32_t &timeout) {
  MS_EXCEPTION_IF_NULL(comm_message_resp);
  uint64_t request_id = ++next_request_id_;
  message_tracker_[request_id] = std::make_pair(data.size(), 0);

  if (rank_ids.size() != data.size() || rank_ids.size() != (*comm_message_resp).size()) {
    MS_LOG(EXCEPTION) << "The number of rank ids, data, comm_message_resp should be equal!";
  }

  size_t len = rank_ids.size();

  set_message_callback(request_id, [&]() {
    receive_messages_mutex_.lock();
    auto res = receive_messages_[request_id];
    for (size_t it = 0; it < len; ++it) {
      comm_message_resp->at(it) = &res[rank_ids.at(it)];
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

    CommMessage comm_message;
    *comm_message.mutable_pb_meta() = {message_meta};
    comm_message.set_data(data.at(it));

    auto client = GetOrCreateTcpClient(rank_ids.at(it));
    client->SendMessage(comm_message);
  }
  return Wait(request_id, timeout);
}

bool Node::WaitForStart(const uint32_t &timeout) {
  std::unique_lock<std::mutex> lock(wait_start_mutex_);
  bool res = wait_start_cond_.wait_for(lock, std::chrono::seconds(timeout), [&] {
    bool res = is_ready_.load();
    if (res) {
      MS_LOG(INFO) << "The node id:" << node_info_.node_id_ << " is success start!";
    }
    return res;
  });
  return res;
}

bool Node::SendMessageSync(const std::shared_ptr<TcpClient> &client, const CommMessage &message,
                           const uint32_t &timeout) {
  uint64_t request_id = ++next_request_id_;
  message_tracker_[request_id] = std::make_pair(1, 0);
  const_cast<CommMessage &>(message).mutable_pb_meta()->set_request_id(request_id);
  client->SendMessage(message);
  return Wait(request_id, timeout);
}

void Node::SendMessageAsync(const std::shared_ptr<TcpClient> &client, const CommMessage &message) {
  uint64_t request_id = ++next_request_id_;
  const_cast<CommMessage &>(message).mutable_pb_meta()->set_request_id(request_id);
  client->SendMessage(message);
}

const std::shared_ptr<TcpClient> &Node::GetOrCreateTcpClient(const int &rank_id) {
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

void Node::ProcessSendDataResp(const CommMessage &message) {
  std::lock_guard<std::mutex> lock(receive_messages_mutex_);
  const MessageMeta &message_meta = message.pb_meta();
  const uint32_t &rank_id = message_meta.rank_id();
  const uint64_t request_id = message_meta.request_id();
  auto it = receive_messages_.find(request_id);
  if (it != receive_messages_.end()) {
    it->second.insert(std::make_pair(rank_id, message));
  } else {
    std::unordered_map<uint32_t, CommMessage> res;
    res.insert(std::make_pair(rank_id, message));
    receive_messages_[request_id] = res;
  }
}

void Node::RunMessageCallback(const uint64_t &request_id) {
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

void Node::set_message_callback(const uint64_t &request_id, const MessageCallback &message_callback) {
  if (!message_callback) {
    return;
  }
  std::lock_guard<std::mutex> lock(message_callbacks_mutex_);
  message_callbacks_[request_id] = message_callback;
}

void Node::NotifyMessageArrival(const CommMessage &message) {
  std::lock_guard<std::mutex> lock(message_tracker_mutex_);
  const MessageMeta &message_meta = message.pb_meta();
  uint64_t request_id = message_meta.request_id();

  message_tracker_[request_id].second++;
  message_tracker_cond_.notify_all();
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
