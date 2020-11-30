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
void Node::Heartbeat(const std::shared_ptr<TcpClient> &client) {
  MS_LOG(INFO) << "The node role: " << CommUtil::NodeRoleToString(node_info_.node_role_)
               << ", the node id:" << node_info_.node_id_ << ", the node rank id:" << node_info_.rank_id_
               << " begin send heartbeat to the scheduler!";
  heart_beat_thread_ = std::make_unique<std::thread>([&]() {
    while (!is_finish_.load()) {
      std::this_thread::sleep_for(std::chrono::seconds(ClusterConfig::heartbeat_interval()));
      MessageMeta meta;
      meta.set_cmd(NodeCommand::HEARTBEAT);

      HeartbeatMessage heartbeat_message;
      heartbeat_message.set_node_id(node_info_.node_id_);

      CommMessage message;
      *message.mutable_pb_meta() = {meta};
      message.set_data(heartbeat_message.SerializeAsString());
      SendMessageAsync(client, message);
    }
  });
  heart_beat_thread_->detach();
}

void Node::ProcessHeartbeatResp(const CommMessage &message) {
  HeartbeatRespMessage heartbeat_resp_message;
  heartbeat_resp_message.ParseFromString(message.data());
  is_ready_ = heartbeat_resp_message.is_cluster_ready();
  if (is_ready_.load()) {
    wait_start_cond_.notify_all();
  }
  is_finish_ = heartbeat_resp_message.is_cluster_finish();
  if (is_finish_.load()) {
    wait_finish_cond_.notify_all();
  }
  is_timeout_ = heartbeat_resp_message.is_cluster_timeout();
  if (is_timeout_ && on_node_event_message_) {
    on_node_event_message_(NodeEvent::NODE_TIMEOUT);
  }
}

void Node::FetchServers(const std::shared_ptr<TcpClient> &client) {
  MessageMeta meta;
  meta.set_cmd(NodeCommand::FETCH_SERVER);

  CommMessage message;
  *message.mutable_pb_meta() = {meta};
  SendMessageSync(client, message);
}

void Node::ProcessFetchServersResp(const CommMessage &message) {
  FetchServersRespMessage fetch_servers_resp_message;
  fetch_servers_resp_message.ParseFromString(message.data());

  for (const auto &it : fetch_servers_resp_message.servers_meta()) {
    server_rank_ids_[it.rank_id()] = std::make_pair(it.ip(), it.port());
  }

  MS_LOG(DEBUG) << "The all server host size is:" << server_rank_ids_.size();
}

std::string Node::node_id() const { return node_info_.node_id_; }

uint32_t Node::rank_id() const { return node_info_.rank_id_; }

void Node::set_callback(const OnNodeEventMessage &on_node_event_message) {
  on_node_event_message_ = on_node_event_message;
}

void Node::Wait(uint64_t request_id) {
  std::unique_lock<std::mutex> lock(message_mutex_);
  message_tracker_cond_.wait(lock, [&] {
    bool ret = message_tracker_[request_id].first == message_tracker_[request_id].second;
    if (ret) {
      MS_LOG(DEBUG) << "Message tracker remove request id:" << request_id;
      message_tracker_.erase(request_id);
    }
    return ret;
  });
}

void Node::Disconnect(const std::shared_ptr<TcpClient> &client) {
  MessageMeta meta;
  meta.set_cmd(NodeCommand::FINISH);

  FinishMessage finish_message;
  finish_message.set_node_id(node_info_.node_id_);

  CommMessage message;
  *message.mutable_pb_meta() = {meta};
  message.set_data(finish_message.SerializeAsString());
  SendMessageSync(client, message);
  WaitForDisconnect();
}

void Node::WaitForStart() {
  std::unique_lock<std::mutex> lock(wait_start_mutex_);
  wait_start_cond_.wait(lock, [&] {
    if (is_ready_.load()) {
      MS_LOG(DEBUG) << "The node id:" << node_info_.node_id_ << " is success start!";
    }
    return is_ready_.load();
  });
}

void Node::WaitForDisconnect() {
  std::unique_lock<std::mutex> lock(wait_finish_mutex_);
  wait_finish_cond_.wait(lock, [&] {
    if (is_finish_.load()) {
      MS_LOG(DEBUG) << "The node id:" << node_info_.node_id_ << " is success finish!";
    }
    return is_finish_.load();
  });
}

void Node::SendMessageSync(const std::shared_ptr<TcpClient> &client, const CommMessage &message) {
  uint64_t request_id = ++next_request_id_;
  message_tracker_[request_id] = std::make_pair(1, 0);
  const_cast<CommMessage &>(message).mutable_pb_meta()->set_request_id(request_id);
  client->SendMessage(message);
  Wait(request_id);
}

void Node::SendMessageAsync(const std::shared_ptr<TcpClient> &client, const CommMessage &message) {
  uint64_t request_id = ++next_request_id_;
  const_cast<CommMessage &>(message).mutable_pb_meta()->set_request_id(request_id);
  client->SendMessage(message);
}

void Node::NotifyMessageArrival(const CommMessage &message) {
  const MessageMeta &message_meta = message.pb_meta();
  uint64_t request_id = message_meta.request_id();

  message_tracker_[request_id].second++;
  message_tracker_cond_.notify_all();
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
