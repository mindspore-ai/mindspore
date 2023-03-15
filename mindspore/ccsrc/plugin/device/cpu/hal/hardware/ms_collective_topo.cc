/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/hal/hardware/ms_collective_topo.h"
#include <string>
#include <memory>
#include <utility>
#include "include/backend/distributed/rpc/tcp/constants.h"

namespace mindspore {
namespace device {
namespace cpu {
bool TopologyNode::Initialize() {
  // Initialize the rank id.
  MS_EXCEPTION_IF_NULL(cgn_);
  rank_id_ = cgn_->rank_id();

  // Initialize the tcp server.
  tcp_server_ = std::make_unique<distributed::rpc::TCPServer>();
  RETURN_IF_FALSE_WITH_LOG(tcp_server_->Initialize(), "Failed to initialize the tcp server.");
  tcp_server_->SetMessageHandler(std::bind(&TopologyNode::HandleMessage, this, std::placeholders::_1));

  // Put the address of this topo node into meta server node.
  auto ip = tcp_server_->GetIP();
  auto port = tcp_server_->GetPort();
  auto rank_name = "RNAK_ID_" + std::to_string(rank_id_);
  auto address = ip + ":" + std::to_string(port);
  (void)cgn_->PutMetadata(rank_name, address);

  // Get the address of the topo node of the next rank from meta server node and create an tcp connection to it.
  // A thread is used because all the addresses of other rank are registered asynchronously into the meta server.
  distributed::rpc::TCPClient *tcp_client = new distributed::rpc::TCPClient();
  RETURN_IF_FALSE_WITH_LOG(tcp_client->Initialize(), "Failed to initialize the tcp client to the next rank.");

  size_t next_rank_id = (rank_id_ + 1) % this->total_node_num_;
  tcp_clients_[next_rank_id] = tcp_client;

  // Because all the topo node address metadata are registered into the metadata server asynchronously, a separate
  // thread is needed to fetch these metadata.
  init_thread_ = std::thread([this, next_rank_id]() {
    size_t retry = 60;
    while (retry-- > 0) {
      // Lookup the address from meta server node.
      auto next_rank_name = "RNAK_ID_" + std::to_string(next_rank_id);
      std::string next_rank_addr = this->cgn_->GetMetadata(next_rank_name);
      if (next_rank_addr.length() > 0) {
        if (this->tcp_clients_[next_rank_id]->Connect(next_rank_addr)) {
          this->node_addresses_[next_rank_id] = next_rank_addr;
          this->initialized_ = true;
          break;
        }
      }
      MS_LOG(INFO) << "Retry to get the address of next rank : " << next_rank_name;
      static const uint32_t interval = 3;
      (void)sleep(interval);
    }
  });
  return true;
}

bool TopologyNode::Initialized() {
  init_thread_.join();
  return initialized_;
}

bool TopologyNode::Finalize() {
  // Destroy the tcp server.
  MS_EXCEPTION_IF_NULL(tcp_server_);
  tcp_server_->Finalize();
  tcp_server_.reset();

  // Destroy the tcp clients.
  for (auto iter = tcp_clients_.begin(); iter != tcp_clients_.end(); iter++) {
    auto &client = iter->second;
    if (client != nullptr) {
      client->Finalize();
      delete client;
      client = nullptr;
    }
  }

  // Destroy the received message queues.
  for (auto iter = received_messages_.begin(); iter != received_messages_.end(); iter++) {
    auto &queue = iter->second;
    if (queue != nullptr) {
      delete queue;
      queue = nullptr;
    }
  }
  return true;
}

bool TopologyNode::SendAsync(size_t rank_id, const void *data, size_t size) {
  if (tcp_clients_.find(rank_id) == tcp_clients_.end()) {
    MS_LOG(ERROR) << "Cann not find tcp client for rank id: " << rank_id << ", local rank: " << rank_id_;
    return false;
  }
  auto &tcp_client = tcp_clients_[rank_id];
  MS_EXCEPTION_IF_NULL(tcp_client);

  std::unique_ptr<MessageBase> message = std::make_unique<MessageBase>();
  MS_EXCEPTION_IF_NULL(message);

  message->name = std::to_string(rank_id_);
  message->to = AID("", node_addresses_[rank_id]);
  message->body.reserve(size);
  (void)message->body.append(static_cast<const char *>(data), size);

  tcp_client->SendAsync(std::move(message));
  return true;
}

bool TopologyNode::WaitForSend(size_t rank_id) {
  // Wait for all the pending data to be sent to the destination of specified rank id.
  if (tcp_clients_.find(rank_id) == tcp_clients_.end()) {
    MS_LOG(ERROR) << "Can not find tcp client for rank id: " << rank_id << ", local rank: " << rank_id_;
    return false;
  }
  if (node_addresses_.find(rank_id) == node_addresses_.end()) {
    MS_LOG(ERROR) << "Can not find the address for rank id: " << rank_id << ", local rank: " << rank_id_;
  }
  auto &tcp_client = tcp_clients_[rank_id];
  MS_EXCEPTION_IF_NULL(tcp_client);

  return tcp_client->Flush(node_addresses_[rank_id]);
}

bool TopologyNode::Receive(size_t rank_id, MessageBase **message, size_t timeout) {
  std::unique_lock<std::mutex> lock(cond_mutex_);
  bool rt = cond_var_.wait_for(lock, std::chrono::seconds(timeout), [this, rank_id] {
    return this->received_messages_.find(rank_id) != this->received_messages_.end() &&
           this->received_messages_[rank_id] != nullptr && this->received_messages_[rank_id]->size() > 0;
  });
  if (rt) {
    auto queue = this->received_messages_[rank_id];
    MS_EXCEPTION_IF_NULL(queue);
    auto recv_msg = queue->front();
    queue->pop();

    MS_EXCEPTION_IF_NULL(message);
    MS_EXCEPTION_IF_NULL(recv_msg);
    *message = recv_msg;
  } else {
    MS_LOG(ERROR) << "Failed to receive message from rank: " << rank_id << ", local rank: " << rank_id_;
  }
  return rt;
}

size_t TopologyNode::rank_id() const { return rank_id_; }

size_t TopologyNode::rank_size() const { return total_node_num_; }

MessageBase *const TopologyNode::HandleMessage(MessageBase *const message) {
  MS_EXCEPTION_IF_NULL(message);
  auto rank_id = std::stoi(message->name);

  std::lock_guard<std::mutex> lock(cond_mutex_);
  std::queue<MessageBase *> *queue = nullptr;
  if (received_messages_.find(rank_id) == received_messages_.end()) {
    queue = new std::queue<MessageBase *>();
    received_messages_[rank_id] = queue;
  }
  MS_EXCEPTION_IF_NULL(queue);
  queue->push(message);
  cond_var_.notify_all();
  return distributed::rpc::NULL_MSG;
}
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
