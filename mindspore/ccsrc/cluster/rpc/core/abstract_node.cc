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

#include "include/cluster/rpc/abstract_node.h"

#include "include/utils/common.h"

namespace mindspore {
namespace ps {
namespace core {
AbstractNode::~AbstractNode() {
  try {
    if (client_to_scheduler_ != nullptr) {
      client_to_scheduler_->Stop();
    }
    if (client_to_scheduler_thread_ != nullptr && client_to_scheduler_thread_->joinable()) {
      client_to_scheduler_thread_->join();
    }
    if (heart_beat_thread_ != nullptr && heart_beat_thread_->joinable()) {
      heart_beat_thread_->join();
    }
    if (server_ != nullptr) {
      server_->Stop();
    }
    if (server_thread_ != nullptr && server_thread_->joinable()) {
      server_thread_->join();
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "AbstractNode destructor run failed, error message: " << e.what();
  } catch (...) {
    MS_LOG(ERROR) << "AbstractNode destructor run failed, unknown error occurred.";
  }
}

uint64_t AbstractNode::CollectiveSendAsync(const NodeRole &node_role, const uint32_t &rank_id, const void *data,
                                           size_t size) {
  MS_EXCEPTION_IF_NULL(data);
  if (!CommUtil::ValidateRankId(node_role, rank_id, worker_num_, server_num_)) {
    MS_LOG(ERROR) << "The node role or rank_id is illegal, the worker num:" << worker_num_
                  << ", the server num:" << server_num_ << ", the rank id:" << rank_id;
    return 0;
  }

  std::shared_ptr<MessageMeta> message_meta = std::make_shared<MessageMeta>();
  MS_EXCEPTION_IF_NULL(message_meta);
  message_meta->set_cmd(NodeCommand::COLLECTIVE_SEND_DATA);
  message_meta->set_rank_id(node_info_.rank_id_);
  message_meta->set_role(node_info_.node_role_);

  auto client = GetOrCreateTcpClient(rank_id, node_role);
  MS_EXCEPTION_IF_NULL(client);
  return SendCollectiveMeta(client, message_meta, Protos::RAW, data, size);
}

std::pair<uint32_t, uint64_t> AbstractNode::CollectiveReceiveAsync(const NodeRole &node_role, const uint32_t &rank_id,
                                                                   VectorPtr *output) {
  MS_EXCEPTION_IF_NULL(output);
  if (!CommUtil::ValidateRankId(node_role, rank_id, worker_num_, server_num_)) {
    MS_LOG(ERROR) << "The node role or rank_id is illegal, the worker num:" << worker_num_
                  << ", the server num:" << server_num_ << ", the rank id:" << rank_id;
    return std::make_pair(0, 0);
  }

  receive_callbacks_mutex_.lock();
  uint64_t rank_request_id = NextExpectedRankRequestId(rank_id);
  auto pair_data = std::make_pair(rank_id, rank_request_id);
  receive_messages_done_[pair_data] = false;
  if (received_data_.count(pair_data) > 0) {
    auto res = received_data_[pair_data];
    MS_EXCEPTION_IF_NULL(res);
    *output = res;
    (void)received_data_.erase(pair_data);
    receive_messages_done_[pair_data] = true;
    MS_LOG(DEBUG) << "Receive data from rank id:" << rank_id << ", the rank request id is:" << rank_request_id;
  } else {
    receive_callbacks_[pair_data] = [=]() mutable {
      auto res_output = received_data_[std::make_pair(rank_id, rank_request_id)];
      MS_EXCEPTION_IF_NULL(res_output);
      if (*output != nullptr) {
        MS_LOG(WARNING) << "The output is not empty.";
      }
      *output = res_output;
      received_data_.erase(std::make_pair(rank_id, rank_request_id));
      receive_messages_done_[std::make_pair(rank_id, rank_request_id)] = true;
      MS_LOG(DEBUG) << "Receive data from rank id:" << rank_id << ", the rank request id is:" << rank_request_id;
    };
  }
  receive_callbacks_mutex_.unlock();
  return std::make_pair(rank_id, rank_request_id);
}

bool AbstractNode::CollectiveWait(const std::pair<uint32_t, uint64_t> &request_id, const uint32_t &timeout) {
  std::unique_lock<std::mutex> lock(receive_callbacks_mutex_);
  bool res =
    receive_cond_.wait_for(lock, std::chrono::seconds(timeout), [&] { return receive_messages_done_[request_id]; });
  if (receive_messages_done_.count(request_id) != 0) {
    (void)receive_messages_done_.erase(request_id);
  }
  return res;
}

const std::shared_ptr<TcpClient> &AbstractNode::GetOrCreateTcpClient(const uint32_t &rank_id, const NodeRole &role) {
  std::lock_guard<std::mutex> lock(client_mutex_);
  auto key = std::make_pair(role, rank_id);
  if (connected_nodes_.find(key) != connected_nodes_.end()) {
    return connected_nodes_[key];
  } else {
    if (nodes_address_.find(key) == nodes_address_.end()) {
      MS_LOG(EXCEPTION) << "Worker receive nodes info from scheduler failed. Role: " << role << ", rank: " << rank_id;
    }
    if (config_ == nullptr) {
      MS_LOG(EXCEPTION) << "The config is empty.";
    }

    MS_LOG(INFO) << "Create tcp client for role: " << role << ", rank: " << rank_id;
    std::string ip = nodes_address_[key].first;
    uint16_t port = nodes_address_[key].second;
    auto client = std::make_shared<TcpClient>(ip, port, role);
    MS_EXCEPTION_IF_NULL(client);
    client->SetMessageCallback([&](const std::shared_ptr<MessageMeta> &meta, const Protos &protos, const void *data,
                                   size_t size) {
      switch (meta->cmd()) {
        case NodeCommand::SEND_DATA:
          ProcessSendDataResp(meta, protos, data, size);
          RunMessageCallback(meta->request_id());
          break;
        case NodeCommand::COLLECTIVE_SEND_DATA:
          MS_LOG(DEBUG) << "The Node id:" << node_info_.node_id_ << " receive a collective_send_data message response!";
          break;
        case NodeCommand::SEND_EVENT:
          MS_LOG(DEBUG) << "The Node id:" << node_info_.node_id_ << " receive a send_event command message response!";
          break;
        default:
          MS_LOG(EXCEPTION) << "The cmd:" << meta->cmd() << " is not supported!";
      }
      NotifyMessageArrival(meta);
    });
    client->Init();
    connected_nodes_[key] = client;
    return connected_nodes_[key];
  }
}

uint64_t AbstractNode::SendCollectiveMeta(const std::shared_ptr<TcpClient> &client,
                                          const std::shared_ptr<MessageMeta> &meta, const Protos &protos,
                                          const void *data, size_t size) {
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

void AbstractNode::ProcessCollectiveSendData(const std::shared_ptr<TcpConnection> &conn,
                                             const std::shared_ptr<MessageMeta> &meta, const Protos &protos,
                                             const void *data, size_t size) {
  MS_EXCEPTION_IF_NULL(conn);
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  if (!server_->SendMessage(conn, meta, Protos::RAW, data, size)) {
    MS_LOG(WARNING) << "Server response message failed.";
  }
  RunReceiveCallback(meta, protos, data, size);
}

void AbstractNode::NotifyMessageArrival(const std::shared_ptr<MessageMeta> &meta) {
  MS_EXCEPTION_IF_NULL(meta);
  std::lock_guard<std::mutex> lock(message_tracker_mutex_);
  uint64_t request_id = meta->request_id();
  if (message_tracker_.count(request_id)) {
    message_tracker_[request_id].second++;
  } else {
    MS_LOG(WARNING) << "The requset id:" << request_id << " is removed.";
  }
  message_tracker_cond_.notify_all();
}

void AbstractNode::RunReceiveCallback(const std::shared_ptr<MessageMeta> &meta, const Protos &, const void *data,
                                      size_t size) {
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  std::shared_ptr<std::vector<unsigned char>> received_data = std::make_shared<std::vector<unsigned char>>(size, 0);
  size_t dest_size = size;
  size_t src_size = size;
  int ret = Memcpy(received_data->data(), dest_size, data, src_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "The Memcpy error, errorno(" << ret << ")";
  }
  receive_callbacks_mutex_.lock();
  uint32_t rank_id = meta->rank_id();
  // When receiving a collective message, Then generate rank request id,compare with the desired rank request id,
  // If they are equal, then call the callback function
  uint64_t rank_request_id = NextActualRankRequestId(rank_id);
  received_data_[std::make_pair(rank_id, rank_request_id)] = received_data;
  MS_LOG(DEBUG) << "Run Receive data callback,the rank id:" << rank_id << ", the rank request id is:" << rank_request_id
                << ", the send request id is:" << meta->request_id() << " the size is:" << size;
  auto it = receive_callbacks_.find(std::make_pair(rank_id, rank_request_id));
  if (it != receive_callbacks_.end()) {
    if (receive_messages_done_.count(std::make_pair(rank_id, rank_request_id)) != 0) {
      if (it->second) {
        it->second();
      }
    }
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
  handlers_[NodeCommand::FINISH] = nullptr;
  handlers_[NodeCommand::SCALE_OUT_DONE] = nullptr;
  handlers_[NodeCommand::SCALE_IN_DONE] = nullptr;
  handlers_[NodeCommand::SEND_EVENT] = nullptr;
  RegisterInitCollectCommResphandler();
}

void AbstractNode::InitServerHandler() {
  server_handler_[NodeCommand::COLLECTIVE_SEND_DATA] = &AbstractNode::ProcessCollectiveSendData;
}

void AbstractNode::InitNodeInfo(const NodeRole &role) {
  MS_EXCEPTION_IF_NULL(config_);
  MS_EXCEPTION_IF_NULL(server_);
  if (PSContext::instance()->node_id().empty() && config_->Exists(kNodeId)) {
    node_info_.node_id_ = config_->Get(kNodeId, "");
  } else {
    node_info_.node_id_ = PSContext::instance()->node_id();
  }

  if (node_info_.node_id_.empty()) {
    node_info_.node_id_ = CommUtil::GenerateUUID();
  }
  node_info_.node_role_ = role;
  node_info_.ip_ = server_->BoundIp();
  node_info_.port_ = server_->BoundPort();

  MS_LOG(INFO) << "The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
               << " is generate uuid is:" << node_info_.node_id_ << ", the ip:" << server_->BoundIp()
               << ", the port:" << server_->BoundPort();
}

void AbstractNode::InitNodeNum() {
  worker_num_ = PSContext::instance()->cluster_config().initial_worker_num;
  server_num_ = PSContext::instance()->cluster_config().initial_server_num;
  scheduler_ip_ = PSContext::instance()->cluster_config().scheduler_host;
  scheduler_port_ = PSContext::instance()->cluster_config().scheduler_port;
  MS_LOG(INFO) << "The worker num:" << worker_num_ << ", the server num:" << server_num_
               << ", the scheduler ip:" << scheduler_ip_ << ", the scheduler port:" << scheduler_port_;
}

void AbstractNode::CreateTcpServer(const std::pair<uint32_t, uint32_t> &port_range) {
  MS_EXCEPTION_IF_NULL(config_);
  std::string interface;
  std::string server_ip = common::GetEnv("MS_WORKER_IP");
  if (server_ip.empty()) {
    MS_LOG(INFO) << "'MS_WORKER_IP' env is not set, so get first available network interface.";
    CommUtil::GetAvailableInterfaceAndIP(&interface, &server_ip);
  }

  server_ = std::make_shared<TcpServer>(server_ip, 0, config_.get(), port_range);
  MS_EXCEPTION_IF_NULL(server_);
  server_->SetMessageCallback([&](const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<MessageMeta> &meta,
                                  const Protos &protos, const void *data, size_t size) {
    MS_EXCEPTION_IF_NULL(conn);
    MS_EXCEPTION_IF_NULL(meta);
    MS_EXCEPTION_IF_NULL(data);
    MS_LOG(DEBUG) << "Receive message cmd " << meta->cmd() << ", size is " << size;
    const auto &handler_pair = server_handler_.find(meta->cmd());
    if (handler_pair == server_handler_.end()) {
      MS_LOG(EXCEPTION) << "The cmd:" << meta->cmd() << " is not supported!";
    }
    (this->*(handler_pair->second))(conn, meta, protos, data, size);
  });

  server_->Init();
  server_thread_ = std::make_unique<std::thread>([this]() {
    MS_LOG(INFO) << "The worker node or server node start a tcp server!";
    this->server_->Start();
  });
  MS_EXCEPTION_IF_NULL(server_thread_);
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
