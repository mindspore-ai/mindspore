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

#include "cluster/rpc/core/node.h"

namespace mindspore {
namespace ps {
namespace core {
std::string Node::node_id() const { return node_info_.node_id_; }

uint32_t Node::rank_id() const { return node_info_.rank_id_; }

NodeRole Node::role() const { return node_info_.node_role_; }

bool Node::Wait(uint64_t request_id, const uint32_t &timeout) {
  std::unique_lock<std::mutex> tracker_lock(message_tracker_mutex_);
  bool res = message_tracker_cond_.wait_for(tracker_lock, std::chrono::seconds(timeout), [&] {
    if (message_tracker_.count(request_id)) {
      bool ret = message_tracker_[request_id].first == message_tracker_[request_id].second;
      return ret;
    }
    return false;
  });
  (void)message_tracker_.erase(request_id);
  tracker_lock.unlock();

  std::unique_lock<std::mutex> msgs_lock(receive_messages_mutex_);
  // The messages should be already copied before message_tracker_cond_ is notified.
  if (receive_messages_.count(request_id) != 0) {
    (void)receive_messages_.erase(request_id);
  }
  msgs_lock.unlock();
  return res;
}

uint64_t Node::AddMessageTrack(const uint32_t &expected_response) {
  std::lock_guard<std::mutex> lock(message_tracker_mutex_);
  uint64_t request_id = ++next_request_id_;
  message_tracker_[request_id] = std::make_pair(expected_response, 0);
  return request_id;
}

bool Node::CheckMessageTrack(const uint64_t &request_id) {
  std::lock_guard<std::mutex> lock(message_tracker_mutex_);
  if (message_tracker_.count(request_id)) {
    return message_tracker_[request_id].first == message_tracker_[request_id].second + 1;
  }
  MS_LOG(INFO) << "The message tracker is not contain the id:" << request_id;
  return false;
}

void Node::NotifyMessageArrival(const std::shared_ptr<MessageMeta> &meta) {
  std::lock_guard<std::mutex> lock(message_tracker_mutex_);
  uint64_t request_id = meta->request_id();
  if (message_tracker_.count(request_id)) {
    message_tracker_[request_id].second++;
    message_tracker_cond_.notify_all();
  }
}

void Node::set_message_callback(const uint64_t &request_id, const MessageCallback &callback) {
  if (!callback) {
    return;
  }
  std::lock_guard<std::mutex> lock(message_callbacks_mutex_);
  message_callbacks_[request_id] = callback;
}

void Node::ProcessSendDataResp(const std::shared_ptr<MessageMeta> &meta, const Protos &, const void *data,
                               size_t size) {
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  std::lock_guard<std::mutex> lock(receive_messages_mutex_);
  const uint32_t &rank_id = meta->rank_id();
  const uint64_t request_id = meta->request_id();
  MS_LOG(DEBUG) << "The node role is:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                << ", the node id is:" << node_info_.node_id_ << " send the request id is:" << request_id;
  if (meta->role() == NodeRole::SERVER) {
    auto it = receive_messages_.find(request_id);
    VectorPtr received_data = std::make_shared<std::vector<unsigned char>>(size, 0);
    if (size > 0) {
      size_t dest_size = size;
      size_t src_size = size;
      errno_t ret = Memcpy(received_data.get()->data(), dest_size, data, src_size);
      if (ret != EOK) {
        MS_LOG(EXCEPTION) << "The Memcpy error, errorno(" << ret << ")";
      }
    }
    if (it != receive_messages_.end()) {
      it->second[rank_id] = received_data;
    } else {
      std::unordered_map<uint32_t, VectorPtr> res;
      (void)res.insert(std::make_pair(rank_id, received_data));
      receive_messages_[request_id] = res;
    }
  } else {
    auto it = workder_receive_messages_.find(request_id);
    VectorPtr received_data = std::make_shared<std::vector<unsigned char>>(size, 0);
    if (size > 0) {
      size_t dest_size = size;
      size_t src_size = size;
      errno_t ret = Memcpy(received_data.get()->data(), dest_size, data, src_size);
      if (ret != EOK) {
        MS_LOG(EXCEPTION) << "The Memcpy error, errorno(" << ret << ")";
      }
    }
    if (it != workder_receive_messages_.end()) {
      it->second[rank_id] = received_data;
    } else {
      std::unordered_map<uint32_t, VectorPtr> res;
      (void)res.insert(std::make_pair(rank_id, received_data));
      workder_receive_messages_[request_id] = res;
    }
  }
}

void Node::RunMessageCallback(const uint64_t &request_id) {
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
      (void)message_callbacks_.erase(it);
    }
  }
  message_callbacks_mutex_.unlock();
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
