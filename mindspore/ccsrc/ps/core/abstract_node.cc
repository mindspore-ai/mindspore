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
#include "ps/core/node_recovery.h"
#include "ps/core/communicator/tcp_communicator.h"
#include "ps/core/communicator/http_communicator.h"

namespace mindspore {
namespace ps {
namespace core {
void AbstractNode::Register(const std::shared_ptr<TcpClient> &client) {
  MS_EXCEPTION_IF_NULL(client);
  auto message_meta = std::make_shared<MessageMeta>();
  MS_EXCEPTION_IF_NULL(message_meta);
  message_meta->set_cmd(NodeCommand::REGISTER);
  message_meta->set_rank_id(node_info_.rank_id_);

  RegisterMessage register_message;
  register_message.set_node_id(node_info_.node_id_);
  register_message.set_role(node_info_.node_role_);
  register_message.set_ip(node_info_.ip_);
  register_message.set_port(node_info_.port_);

  MS_LOG(INFO) << "The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
               << " the node id:" << node_info_.node_id_ << " begin to register to the scheduler!";

  if (!SendMessageSync(client, message_meta, Protos::PROTOBUF, register_message.SerializeAsString().data(),
                       register_message.ByteSizeLong())) {
    MS_LOG(EXCEPTION) << "The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                      << " the node id:" << node_info_.node_id_ << " register timeout!";
  }
}

void AbstractNode::ProcessRegisterResp(const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size) {
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  RegisterRespMessage register_resp_message;
  CHECK_RETURN_TYPE(register_resp_message.ParseFromArray(data, SizeToInt(size)));
  if (register_resp_message.node_id() != node_info_.node_id_) {
    MS_LOG(EXCEPTION) << "The node id received:" << register_resp_message.node_id()
                      << " is not match the current node id:" << node_info_.node_id_;
  }

  // Receive the Register message, indicating that the scheduler is alive, so update the time point at which the
  // scheduler is alive
  UpdateSchedulerTime();

  MS_LOG(INFO) << "The node id is:" << node_info_.node_id_ << " registered scheduler success!";
}

bool AbstractNode::Broadcast(const NodeRole &node_role, const DataPtr &message, size_t size, int command,
                             const uint32_t &timeout) {
  MS_EXCEPTION_IF_NULL(message);
  if (node_role != NodeRole::SERVER) {
    MS_LOG(EXCEPTION) << "Currently only supports broadcast to server nodes";
  }

  uint64_t request_id = AddMessageTrack(nodes_address_.size());

  for (auto it = nodes_address_.begin(); it != nodes_address_.end(); ++it) {
    auto message_meta = std::make_shared<MessageMeta>();
    MS_EXCEPTION_IF_NULL(message_meta);
    message_meta->set_cmd(NodeCommand::SEND_DATA);
    message_meta->set_request_id(request_id);
    message_meta->set_rank_id(node_info_.rank_id_);
    message_meta->set_role(node_info_.node_role_);
    message_meta->set_user_cmd(command);

    auto client = GetOrCreateTcpClient((*it).first.second);
    if (!client->SendMessage(message_meta, Protos::RAW, message.get(), size)) {
      MS_LOG(WARNING) << "Client send message failed.";
    }
  }
  MS_LOG(DEBUG) << "The node role is:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                << ", the node id is:" << node_info_.node_id_ << " send the request id is:" << request_id;
  return Wait(request_id, timeout);
}

void AbstractNode::set_ready_for_scale_out() {
  MS_LOG(INFO) << "[Scale out]: begin to set ready for scale out.";
  Register(client_to_scheduler_);
  std::lock_guard<std::mutex> lock(client_mutex_);
  connected_nodes_.clear();
}

void AbstractNode::set_ready_for_scale_in() {
  MS_LOG(INFO) << "[Scale in]: begin to set ready for scale in.";
  if (!is_current_node_scale_in_) {
    Register(client_to_scheduler_);
    std::lock_guard<std::mutex> lock(client_mutex_);
    connected_nodes_.clear();
  }
}

void AbstractNode::set_scale_out_done() {
  MS_LOG(INFO) << "[Scale out]: begin to set scale out done.";
  auto message_meta = std::make_shared<MessageMeta>();
  MS_EXCEPTION_IF_NULL(message_meta);
  message_meta->set_cmd(NodeCommand::SCALE_OUT_DONE);

  ScaleOutDoneMessage scale_out_done_message;
  scale_out_done_message.set_node_id(node_info_.node_id_);

  if (!SendMessageSync(client_to_scheduler_, message_meta, Protos::PROTOBUF,
                       scale_out_done_message.SerializeAsString().data(), scale_out_done_message.ByteSizeLong())) {
    MS_LOG(WARNING) << "The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                    << " the node id:" << node_info_.node_id_ << " scale_out_done timeout!";
    return;
  }

  MS_LOG(INFO) << "The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
               << " the node id:" << node_info_.node_id_ << "is send scale_out_done to scheduler successful!";
}

void AbstractNode::set_scale_in_done() {
  MS_LOG(INFO) << "[Scale in]: begin to set scale in done.";
  auto message_meta = std::make_shared<MessageMeta>();
  MS_EXCEPTION_IF_NULL(message_meta);
  message_meta->set_cmd(NodeCommand::SCALE_IN_DONE);

  ScaleInDoneMessage scale_in_done_message;
  scale_in_done_message.set_node_id(node_info_.node_id_);

  if (!SendMessageSync(client_to_scheduler_, message_meta, Protos::PROTOBUF,
                       scale_in_done_message.SerializeAsString().data(), scale_in_done_message.ByteSizeLong())) {
    MS_LOG(WARNING) << "The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                    << " the node id:" << node_info_.node_id_ << " scale_in_done timeout!";
    return;
  }

  MS_LOG(INFO) << "The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
               << " the node id:" << node_info_.node_id_ << "is send scale_in_done to scheduler successful!";
}

void AbstractNode::BroadcastEvent(const uint32_t &event) {
  auto message_meta = std::make_shared<MessageMeta>();
  MS_EXCEPTION_IF_NULL(message_meta);
  message_meta->set_cmd(NodeCommand::SEND_EVENT);

  EventMessage event_message;
  event_message.set_event(event);
  event_message.set_node_id(node_info_.node_id_);

  if (!SendMessageSync(client_to_scheduler_, message_meta, Protos::PROTOBUF, event_message.SerializeAsString().data(),
                       event_message.ByteSizeLong())) {
    MS_LOG(ERROR) << "The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                  << " the node id:" << node_info_.node_id_ << " send event timeout!";
    return;
  }

  MS_LOG(INFO) << "The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
               << " the node id:" << node_info_.node_id_ << "is send event to scheduler!";
}

void AbstractNode::RegisterEventCallback(const core::ClusterEvent &event, const EventCallback &event_cb) {
  event_to_callback_.try_emplace(event, event_cb);
}

void AbstractNode::RegisterCustomEventCallback(const uint32_t &event, const EventCallback &event_cb) {
  custom_event_to_callback_.try_emplace(event, event_cb);
}

bool AbstractNode::Send(const NodeRole &node_role, const uint32_t &rank_id, const DataPtr &data, size_t len,
                        int command, const uint32_t &timeout) {
  if (current_cluster_state_ == ClusterState::NODE_TIMEOUT) {
    MS_LOG(DEBUG) << "The node is timeout, can not send message.";
    return false;
  }
  MS_EXCEPTION_IF_NULL(data);
  if (!CommUtil::ValidateRankId(node_role, rank_id, worker_num_, server_num_)) {
    MS_LOG(EXCEPTION) << "The node role or rank_id is illegal, the worker num:" << worker_num_
                      << ", the server num:" << server_num_ << ", the rank id:" << rank_id;
  }

  auto message_meta = std::make_shared<MessageMeta>();
  MS_EXCEPTION_IF_NULL(message_meta);
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
  if (current_cluster_state_ == ClusterState::NODE_TIMEOUT) {
    MS_LOG(DEBUG) << "The node is timeout, can not send message.";
    return false;
  }

  uint64_t request_id = AddMessageTrack(data.size());

  if (rank_ids.size() != data.size() || rank_ids.size() != lens.size()) {
    MS_LOG(EXCEPTION) << "The number of rank ids, data and lens are not equal!";
  }
  for (size_t it = 0; it < rank_ids.size(); ++it) {
    if (!CommUtil::ValidateRankId(node_role, rank_ids.at(it), worker_num_, server_num_)) {
      MS_LOG(EXCEPTION) << "The node role or rank_id is illegal, the worker num:" << worker_num_
                        << ", the server num:" << server_num_ << ", the rank id:" << rank_ids.at(it);
    }

    auto message_meta = std::make_shared<MessageMeta>();
    MS_EXCEPTION_IF_NULL(message_meta);
    message_meta->set_cmd(NodeCommand::SEND_DATA);
    message_meta->set_request_id(request_id);
    message_meta->set_rank_id(node_info_.rank_id_);
    message_meta->set_role(node_info_.node_role_);
    message_meta->set_user_cmd(command);

    auto send = data.at(it);
    auto len = lens.at(it);
    auto client = GetOrCreateTcpClient(rank_ids.at(it));
    MS_EXCEPTION_IF_NULL(client);
    if (!client->SendMessage(message_meta, Protos::RAW, send.get(), len)) {
      MS_LOG(WARNING) << "Client send message failed.";
    }
  }
  MS_LOG(DEBUG) << "The node role is:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                << ", the node id is:" << node_info_.node_id_ << " send the request id is:" << request_id;
  return Wait(request_id, timeout);
}

bool AbstractNode::Send(const NodeRole &node_role, const uint32_t &rank_id, const DataPtr &message, size_t len,
                        int command, VectorPtr *output, const uint32_t &timeout) {
  if (current_cluster_state_ == ClusterState::NODE_TIMEOUT) {
    MS_LOG(DEBUG) << "The node is timeout, can not send message.";
    return false;
  }
  MS_EXCEPTION_IF_NULL(message);
  MS_EXCEPTION_IF_NULL(output);
  if (!CommUtil::ValidateRankId(node_role, rank_id, worker_num_, server_num_)) {
    MS_LOG(EXCEPTION) << "The node role or rank_id is illegal, the worker num:" << worker_num_
                      << ", the server num:" << server_num_ << ", the rank id:" << rank_id;
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
  MS_EXCEPTION_IF_NULL(message_meta);
  message_meta->set_cmd(NodeCommand::SEND_DATA);
  message_meta->set_request_id(request_id);
  message_meta->set_rank_id(node_info_.rank_id_);
  message_meta->set_role(node_info_.node_role_);
  message_meta->set_user_cmd(command);

  auto client = GetOrCreateTcpClient(rank_id);
  MS_EXCEPTION_IF_NULL(client);
  if (!client->SendMessage(message_meta, Protos::RAW, message.get(), len)) {
    MS_LOG(WARNING) << "Client send message failed.";
  }
  MS_LOG(DEBUG) << "The node role is:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                << ", the node id is:" << node_info_.node_id_ << " send the request id is:" << request_id;
  return Wait(request_id, timeout);
}

bool AbstractNode::Send(const NodeRole &node_role, const std::vector<uint32_t> &rank_ids,
                        const std::vector<DataPtr> &data, const std::vector<size_t> &data_lens, int command,
                        std::vector<VectorPtr> *output, const uint32_t &timeout) {
  if (current_cluster_state_ == ClusterState::NODE_TIMEOUT) {
    MS_LOG(DEBUG) << "The node is timeout, can not send message.";
    return false;
  }
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
    if (!CommUtil::ValidateRankId(node_role, rank_ids.at(it), worker_num_, server_num_)) {
      MS_LOG(EXCEPTION) << "The node role or rank_id is illegal, the worker num:" << worker_num_
                        << ", the server num:" << server_num_ << ", the rank id:" << rank_ids.at(it);
    }

    auto message_meta = std::make_shared<MessageMeta>();
    MS_EXCEPTION_IF_NULL(message_meta);
    message_meta->set_cmd(NodeCommand::SEND_DATA);
    message_meta->set_request_id(request_id);
    message_meta->set_rank_id(node_info_.rank_id_);
    message_meta->set_role(node_info_.node_role_);
    message_meta->set_user_cmd(command);

    auto send = data.at(it);
    auto len = data_lens.at(it);

    auto client = GetOrCreateTcpClient(rank_ids.at(it));
    MS_EXCEPTION_IF_NULL(client);
    if (!client->SendMessage(message_meta, Protos::RAW, send.get(), len)) {
      MS_LOG(WARNING) << "Client send message failed.";
    }
  }
  MS_LOG(DEBUG) << "The node role is:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                << ", the node id is:" << node_info_.node_id_ << " send the request id is:" << request_id;
  return Wait(request_id, timeout);
}

uint64_t AbstractNode::CollectiveSendAsync(const NodeRole &node_role, const uint32_t &rank_id, const void *data,
                                           size_t size) {
  MS_EXCEPTION_IF_NULL(data);
  if (!CommUtil::ValidateRankId(node_role, rank_id, worker_num_, server_num_)) {
    MS_LOG(EXCEPTION) << "The node role or rank_id is illegal, the worker num:" << worker_num_
                      << ", the server num:" << server_num_ << ", the rank id:" << rank_id;
  }

  std::shared_ptr<MessageMeta> message_meta = std::make_shared<MessageMeta>();
  MS_EXCEPTION_IF_NULL(message_meta);
  message_meta->set_cmd(NodeCommand::COLLECTIVE_SEND_DATA);
  message_meta->set_rank_id(node_info_.rank_id_);
  message_meta->set_role(node_info_.node_role_);

  auto client = GetOrCreateTcpClient(rank_id);
  MS_EXCEPTION_IF_NULL(client);
  return SendMessageAsync(client, message_meta, Protos::RAW, data, size);
}

std::pair<uint32_t, uint64_t> AbstractNode::CollectiveReceiveAsync(const NodeRole &node_role, const uint32_t &rank_id,
                                                                   VectorPtr *output) {
  MS_EXCEPTION_IF_NULL(output);
  if (!CommUtil::ValidateRankId(node_role, rank_id, worker_num_, server_num_)) {
    MS_LOG(EXCEPTION) << "The node role or rank_id is illegal, the worker num:" << worker_num_
                      << ", the server num:" << server_num_ << ", the rank id:" << rank_id;
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

bool AbstractNode::InitFollowerScaler() {
  follower_scaler_ = std::make_unique<FollowerScaler>(this);
  MS_EXCEPTION_IF_NULL(follower_scaler_);
  follower_scaler_->RegisterScaleEventCallbacks();
  return true;
}

void AbstractNode::RegisterFollowerScalerBarrierBeforeScaleOut(const std::string &module,
                                                               const BarrierBeforeScaleOut &barrier) {
  MS_EXCEPTION_IF_NULL(follower_scaler_);
  follower_scaler_->RegisterBarrierBeforeScaleOut(module, barrier);
}

void AbstractNode::RegisterFollowerScalerBarrierBeforeScaleIn(const std::string &module,
                                                              const BarrierBeforeScaleIn &barrier) {
  MS_EXCEPTION_IF_NULL(follower_scaler_);
  follower_scaler_->RegisterBarrierBeforeScaleIn(module, barrier);
}

void AbstractNode::RegisterFollowerScalerHandlerAfterScaleOut(const std::string &module,
                                                              const HandlerAfterScaleOut &handler) {
  MS_EXCEPTION_IF_NULL(follower_scaler_);
  follower_scaler_->RegisterHandlerAfterScaleOut(module, handler);
}

void AbstractNode::RegisterFollowerScalerHandlerAfterScaleIn(const std::string &module,
                                                             const HandlerAfterScaleIn &handler) {
  MS_EXCEPTION_IF_NULL(follower_scaler_);
  follower_scaler_->RegisterHandlerAfterScaleIn(module, handler);
}

int32_t AbstractNode::worker_num() const { return worker_num_; }

int32_t AbstractNode::server_num() const { return server_num_; }

void AbstractNode::set_worker_num(const int32_t &worker_num) { worker_num_ = worker_num; }

void AbstractNode::set_server_num(const int32_t &server_num) { server_num_ = server_num; }

std::string AbstractNode::scheduler_ip() const { return scheduler_ip_; }

void AbstractNode::set_scheduler_ip(const std::string &scheduler_ip) { scheduler_ip_ = scheduler_ip; }

uint16_t AbstractNode::scheduler_port() const { return scheduler_port_; }

void AbstractNode::set_scheduler_port(const uint16_t &scheduler_port) { scheduler_port_ = scheduler_port; }

ClusterState AbstractNode::cluster_state() const { return current_cluster_state_; }

void AbstractNode::set_handler(const RequestHandler &handler) { request_handler_ = handler; }

void AbstractNode::Response(const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<MessageMeta> &meta,
                            const void *data, size_t size) {
  MS_EXCEPTION_IF_NULL(conn);
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  MS_EXCEPTION_IF_NULL(server_);
  meta->set_role(node_info_.node_role_);
  meta->set_rank_id(node_info_.rank_id_);
  MS_LOG(DEBUG) << "The node role is:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                << ", the node id is:" << node_info_.node_id_ << " send the request id is:" << meta->request_id();
  if (!server_->SendMessage(conn, meta, Protos::RAW, data, size)) {
    MS_LOG(WARNING) << "Server response message failed.";
  }
}

std::shared_ptr<CommunicatorBase> AbstractNode::GetOrCreateHttpComm(
  const std::string &ip, uint16_t port, const std::shared_ptr<TaskExecutor> &task_executor) {
  MS_EXCEPTION_IF_NULL(task_executor);
  std::lock_guard<std::mutex> lock(communicator_mutex_);
  if (!communicators_.count(kHttpCommunicator)) {
    MS_LOG(INFO) << "Create Http communicator.";
    auto http_comm = std::make_shared<HttpCommunicator>(ip, port, task_executor);
    MS_EXCEPTION_IF_NULL(http_comm);
    communicators_[kHttpCommunicator] = http_comm;
  }
  return communicators_[kHttpCommunicator];
}

std::shared_ptr<CommunicatorBase> AbstractNode::GetOrCreateTcpComm(const std::string &scheduler_ip,
                                                                   std::int16_t scheduler_port, uint32_t worker_num,
                                                                   uint32_t server_num,
                                                                   const std::shared_ptr<TaskExecutor> &task_executor) {
  MS_EXCEPTION_IF_NULL(task_executor);
  std::lock_guard<std::mutex> lock(communicator_mutex_);
  if (!communicators_.count(kTcpCommunicator)) {
    MS_LOG(INFO) << "Create Tcp communicator.";
    auto tcp_comm = std::make_shared<TcpCommunicator>(task_executor, this);
    PSContext::instance()->cluster_config().scheduler_host = scheduler_ip;
    PSContext::instance()->cluster_config().scheduler_port = static_cast<uint16_t>(scheduler_port);
    PSContext::instance()->cluster_config().initial_worker_num = worker_num;
    PSContext::instance()->cluster_config().initial_server_num = server_num;
    MS_EXCEPTION_IF_NULL(tcp_comm);
    PSContext::instance()->cluster_config().scheduler_host = scheduler_ip;
    PSContext::instance()->cluster_config().scheduler_port = static_cast<uint16_t>(scheduler_port);
    PSContext::instance()->cluster_config().initial_worker_num = worker_num;
    PSContext::instance()->cluster_config().initial_server_num = server_num;
    MS_LOG(INFO) << "Initialize cluster metadata for server. Worker number:" << worker_num
                 << ", Server number:" << server_num << ", Scheduler ip:" << scheduler_ip
                 << ", Scheduler port:" << scheduler_port;
    communicators_[kTcpCommunicator] = tcp_comm;
  }
  return communicators_[kTcpCommunicator];
}

void AbstractNode::StartHeartbeatTimer(const std::shared_ptr<TcpClient> &client) {
  MS_EXCEPTION_IF_NULL(client);
  MS_LOG(INFO) << "The node role: " << CommUtil::NodeRoleToString(node_info_.node_role_)
               << ", the node id:" << node_info_.node_id_ << ", the node rank id:" << node_info_.rank_id_
               << " begin send heartbeat to the scheduler!";
  heart_beat_thread_ = std::make_unique<std::thread>([&]() {
    while (!is_finish_.load()) {
      if (!Heartbeat(client)) {
        MS_LOG(WARNING) << "The node role is:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                        << ", the node id is:" << node_info_.node_id_ << " Send heartbeat timeout!";
        if (CheckSchedulerTimeout()) {
          MS_LOG(WARNING) << "The node role is:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                          << ", the node id is:" << node_info_.node_id_ << " exited due to scheduler timeout!";
          is_finish_ = true;
          wait_finish_cond_.notify_all();
          if (!is_already_stopped_) {
            OnEventCallback(ClusterEvent::SCHEDULER_TIMEOUT);
          }
        }
      } else {
        UpdateSchedulerTime();
      }

      std::this_thread::sleep_for(std::chrono::seconds(PSContext::instance()->cluster_config().heartbeat_interval));
    }
  });
  MS_EXCEPTION_IF_NULL(heart_beat_thread_);
  heart_beat_thread_->detach();
}

bool AbstractNode::Heartbeat(const std::shared_ptr<TcpClient> &client) {
  MS_EXCEPTION_IF_NULL(client);
  auto meta = std::make_shared<MessageMeta>();
  MS_EXCEPTION_IF_NULL(meta);
  meta->set_cmd(NodeCommand::HEARTBEAT);

  HeartbeatMessage heartbeat_message;
  heartbeat_message.set_node_id(node_info_.node_id_);

  if (!SendMessageSync(client, meta, Protos::PROTOBUF, heartbeat_message.SerializeAsString().data(),
                       heartbeat_message.ByteSizeLong(), kCommTimeoutInSeconds)) {
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
  int64_t old_time = scheduler_time_.tv_sec + PSContext::instance()->cluster_config().scheduler_timeout;
  if (old_time < current_time.tv_sec) {
    return true;
  }
  return false;
}

void AbstractNode::ProcessHeartbeatResp(const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size) {
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  HeartbeatRespMessage heartbeat_resp_message;
  CHECK_RETURN_TYPE(heartbeat_resp_message.ParseFromArray(data, SizeToInt(size)));

  current_cluster_state_ = heartbeat_resp_message.cluster_state();
  MS_LOG(DEBUG) << "The current cluster state from heartbeat:"
                << CommUtil::ClusterStateToString(current_cluster_state_);

  all_nodes_info_.clear();
  for (const auto &it : heartbeat_resp_message.servers_meta()) {
    NodeInfo info;
    info.ip_ = it.ip();
    info.node_id_ = it.node_id();
    info.port_ = static_cast<uint16_t>(it.port());
    info.node_role_ = it.role();
    info.rank_id_ = it.rank_id();
    info.is_alive = it.is_alive();

    all_nodes_info_[info.node_id_] = info;
    MS_LOG(DEBUG) << "The node id:" << info.node_id_ << ", the rank id:" << info.rank_id_
                  << ", the node role:" << CommUtil::NodeRoleToString(info.node_role_) << " is alive:" << info.is_alive;
  }

  bool is_worker_or_server0 = heartbeat_resp_message.is_worker_or_server0();

  if (current_cluster_state_ == ClusterState::NODE_TIMEOUT) {
    if (node_recovery_ == nullptr || is_worker_or_server0) {
      MS_LOG(INFO) << "The recovery is disable.";
      is_ready_ = true;
      wait_start_cond_.notify_all();
      OnEventCallback(ClusterEvent::NODE_TIMEOUT);
    } else {
      MS_LOG(INFO) << "The node is support recovery, users can pull up this node to restore the cluster.";
    }
  }
}

void AbstractNode::FetchServers(const std::shared_ptr<TcpClient> &client) {
  MS_EXCEPTION_IF_NULL(client);
  auto meta = std::make_shared<MessageMeta>();
  MS_EXCEPTION_IF_NULL(meta);
  meta->set_cmd(NodeCommand::FETCH_METADATA);

  FetchServersMessage fetch_servers;
  fetch_servers.set_node_id(node_info_.node_id_);
  if (!SendMessageSync(client, meta, Protos::PROTOBUF, fetch_servers.SerializeAsString().data(),
                       fetch_servers.ByteSizeLong())) {
    MS_LOG(EXCEPTION) << "Fetch servers address timeout!";
  }
}

void AbstractNode::ProcessFetchServersResp(const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size) {
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  FetchServersRespMessage fetch_servers_resp_message;
  CHECK_RETURN_TYPE(fetch_servers_resp_message.ParseFromArray(data, SizeToInt(size)));

  nodes_address_.clear();
  for (const auto &it : fetch_servers_resp_message.servers_meta()) {
    nodes_address_[std::make_pair(NodeRole::SERVER, it.rank_id())] = std::make_pair(it.ip(), it.port());
    MS_LOG(INFO) << "The server ip is:" << it.ip() << ", the port is:" << it.port();
  }
}

void AbstractNode::ProcessSendMetadata(const std::shared_ptr<TcpConnection> &conn,
                                       const std::shared_ptr<MessageMeta> &meta, const Protos &, const void *data,
                                       size_t size) {
  MS_EXCEPTION_IF_NULL(conn);
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  if (is_current_node_scale_in_) {
    MS_LOG(WARNING) << "Trigger cluster scale in done event.";
    node_info_.rank_id_ = UINT32_MAX;
    OnEventCallback(ClusterEvent::CLUSTER_SCALE_IN_DONE);
    return;
  }
  SendMetadataMessage send_meta_message;
  CHECK_RETURN_TYPE(send_meta_message.ParseFromArray(data, SizeToInt(size)));
  worker_num_ = send_meta_message.worker_num();
  server_num_ = send_meta_message.server_num();
  if (send_meta_message.rank_id() < 0) {
    MS_LOG(EXCEPTION) << "The rank id is wrong.";
  }
  node_info_.rank_id_ = send_meta_message.rank_id();
  current_cluster_state_ = send_meta_message.cluster_state();
  MS_LOG(INFO) << "The send metadata worker num:" << worker_num_ << ", server num:" << server_num_
               << ", cluster state is:" << CommUtil::ClusterStateToString(current_cluster_state_)
               << ", the rank id:" << node_info_.rank_id_;

  client_mutex_.lock();
  nodes_address_.clear();
  for (const auto &it : send_meta_message.servers_meta()) {
    nodes_address_[std::make_pair(NodeRole::SERVER, it.rank_id())] = std::make_pair(it.ip(), it.port());
    MS_LOG(INFO) << "The server ip is:" << it.ip() << ", the port is:" << it.port() << ", the rank id:" << it.rank_id();
  }
  client_mutex_.unlock();
  if (!server_->SendMessage(conn, meta, Protos::RAW, data, size)) {
    MS_LOG(WARNING) << "Sever response message failed.";
  }
  is_ready_ = true;
  wait_start_cond_.notify_all();

  if (current_cluster_state_ == ClusterState::CLUSTER_SCALE_OUT) {
    MS_LOG(WARNING) << "Trigger cluster scale out done event.";
    OnEventCallback(ClusterEvent::CLUSTER_SCALE_OUT_DONE);
  }

  if (current_cluster_state_ == ClusterState::CLUSTER_SCALE_IN) {
    MS_LOG(WARNING) << "Trigger cluster scale in done event.";
    OnEventCallback(ClusterEvent::CLUSTER_SCALE_IN_DONE);
  }

  std::lock_guard<std::mutex> lock(client_mutex_);
  connected_nodes_.clear();
}

void AbstractNode::ProcessFinish(const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<MessageMeta> &meta,
                                 const Protos &, const void *data, size_t size) {
  MS_EXCEPTION_IF_NULL(conn);
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  if (!server_->SendMessage(conn, meta, Protos::RAW, data, size)) {
    MS_LOG(WARNING) << "Server response message failed.";
  }
  is_finish_ = true;
  wait_finish_cond_.notify_all();
}

void AbstractNode::ProcessScaleOutDone(const std::shared_ptr<TcpConnection> &conn,
                                       const std::shared_ptr<MessageMeta> &meta, const Protos &, const void *data,
                                       size_t size) {
  MS_EXCEPTION_IF_NULL(conn);
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  if (!server_->SendMessage(conn, meta, Protos::RAW, data, size)) {
    MS_LOG(WARNING) << "Server response message failed.";
  }
  is_ready_ = true;
  current_cluster_state_ = ClusterState::CLUSTER_READY;
}

void AbstractNode::ProcessScaleInDone(const std::shared_ptr<TcpConnection> &conn,
                                      const std::shared_ptr<MessageMeta> &meta, const Protos &, const void *data,
                                      size_t size) {
  MS_EXCEPTION_IF_NULL(conn);
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  if (!server_->SendMessage(conn, meta, Protos::RAW, data, size)) {
    MS_LOG(WARNING) << "Server response message failed.";
  }
  is_ready_ = true;
  current_cluster_state_ = ClusterState::CLUSTER_READY;
}

void AbstractNode::ProcessEvent(const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<MessageMeta> &meta,
                                const Protos &, const void *data, size_t size) {
  MS_EXCEPTION_IF_NULL(conn);
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  EventRespMessage event_resp_message;
  CHECK_RETURN_TYPE(event_resp_message.ParseFromArray(data, SizeToInt(size)));
  uint32_t event = event_resp_message.event();
  if (!server_->SendMessage(conn, meta, Protos::RAW, data, size)) {
    MS_LOG(WARNING) << "Server response message failed.";
  }
  OnCustomEventCallback(event);
}

void AbstractNode::ProcessScaleOut(const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<MessageMeta> &meta,
                                   const Protos &, const void *data, size_t size) {
  MS_EXCEPTION_IF_NULL(conn);
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);

  ScaleOutMessage scale_out_message;
  CHECK_RETURN_TYPE(scale_out_message.ParseFromArray(data, SizeToInt(size)));
  int32_t worker_num = scale_out_message.worker_num();
  int32_t server_num = scale_out_message.server_num();
  MS_LOG(WARNING) << "The scale out worker num:" << worker_num << ", the server num:" << server_num;

  if (!server_->SendMessage(conn, meta, Protos::RAW, data, size)) {
    MS_LOG(WARNING) << "Server response message failed.";
  }
  OnEventCallback(ClusterEvent::READY_FOR_SCALE_OUT);
  current_cluster_state_ = ClusterState::CLUSTER_SCALE_OUT;
  is_ready_ = false;
}

void AbstractNode::ProcessScaleIn(const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<MessageMeta> &meta,
                                  const Protos &, const void *data, size_t size) {
  MS_EXCEPTION_IF_NULL(conn);
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);

  ScaleInMessage scale_in_message;
  CHECK_RETURN_TYPE(scale_in_message.ParseFromArray(data, SizeToInt(size)));
  int32_t worker_num = scale_in_message.worker_num();
  int32_t server_num = scale_in_message.server_num();
  MS_LOG(WARNING) << "The scale in worker num:" << worker_num << ", the server num:" << server_num;

  is_current_node_scale_in_ = scale_in_message.is_node_scale_in();
  if (is_current_node_scale_in_) {
    MS_LOG(WARNING) << "The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                    << " the node id:" << node_info_.node_id_ << " is a scale in node!";
  } else {
    MS_LOG(WARNING) << "The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                    << " the node id:" << node_info_.node_id_ << " is not a scale in node!";
  }

  if (!server_->SendMessage(conn, meta, Protos::RAW, data, size)) {
    MS_LOG(WARNING) << "Server response message failed.";
  }
  OnEventCallback(ClusterEvent::READY_FOR_SCALE_IN);
  current_cluster_state_ = ClusterState::CLUSTER_SCALE_IN;
  is_ready_ = false;
}

bool AbstractNode::Disconnect(const std::shared_ptr<TcpClient> &client, const uint32_t &timeout) {
  MS_EXCEPTION_IF_NULL(client);
  auto meta = std::make_shared<MessageMeta>();
  MS_EXCEPTION_IF_NULL(meta);
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
  if (config_ == nullptr) {
    MS_LOG(WARNING) << "The config is empty.";
    return false;
  }
  client_to_scheduler_ = std::make_shared<TcpClient>(scheduler_ip_, scheduler_port_, config_.get());
  MS_EXCEPTION_IF_NULL(client_to_scheduler_);
  client_to_scheduler_->SetMessageCallback(
    [&](const std::shared_ptr<MessageMeta> &meta, const Protos &, const void *data, size_t size) {
      try {
        MS_EXCEPTION_IF_NULL(meta);
        MS_EXCEPTION_IF_NULL(data);
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
    std::this_thread::sleep_for(std::chrono::milliseconds(PSContext::instance()->cluster_config().connect_interval));
    if (is_ready_.load() == false) {
      client_to_scheduler_->Init();
    }
  });
  bool wait_res = client_to_scheduler_->WaitConnected();
  if (!wait_res) {
    is_ready_ = true;
  }
  return wait_res;
}

const std::shared_ptr<TcpClient> &AbstractNode::GetOrCreateTcpClient(const uint32_t &rank_id) {
  std::lock_guard<std::mutex> lock(client_mutex_);
  if (connected_nodes_.find(rank_id) != connected_nodes_.end()) {
    return connected_nodes_[rank_id];
  } else {
    if (nodes_address_.find(std::make_pair(NodeRole::SERVER, rank_id)) == nodes_address_.end()) {
      MS_LOG(EXCEPTION) << "Worker receive nodes info from scheduler failed!";
    }
    if (config_ == nullptr) {
      MS_LOG(EXCEPTION) << "The config is empty.";
    }
    std::string ip = nodes_address_[std::make_pair(NodeRole::SERVER, rank_id)].first;
    uint16_t port = nodes_address_[std::make_pair(NodeRole::SERVER, rank_id)].second;
    auto client = std::make_shared<TcpClient>(ip, port, config_.get());
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
  MS_EXCEPTION_IF_NULL(client);
  uint64_t request_id = AddMessageTrack(1);
  const_cast<CommMessage &>(message).mutable_pb_meta()->set_request_id(request_id);
  client->SendMessage(message);
  MS_LOG(DEBUG) << "The node role is:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                << ", the node id is:" << node_info_.node_id_ << " send the request id is:" << request_id;
  return Wait(request_id, timeout);
}

uint64_t AbstractNode::SendMessageAsync(const std::shared_ptr<TcpClient> &client,
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

bool AbstractNode::SendMessageSync(const std::shared_ptr<TcpClient> &client, const std::shared_ptr<MessageMeta> &meta,
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

void AbstractNode::ProcessCollectiveSendData(const std::shared_ptr<TcpConnection> &conn,
                                             const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size) {
  MS_EXCEPTION_IF_NULL(conn);
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  if (!server_->SendMessage(conn, meta, Protos::RAW, data, size)) {
    MS_LOG(WARNING) << "Server response message failed.";
  }
}

void AbstractNode::ProcessSendData(const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<MessageMeta> &meta,
                                   const Protos &, const void *data, size_t size) {
  MS_EXCEPTION_IF_NULL(conn);
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  std::shared_ptr<unsigned char[]> res(new unsigned char[size]);
  if (size > 0) {
    size_t dest_size = size;
    size_t src_size = size;
    if (memcpy_s(res.get(), dest_size, data, src_size) != EOK) {
      MS_LOG(EXCEPTION) << "The memcpy_s error";
    }
  }
  MS_LOG(DEBUG) << "The node role is:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                << ", the node id is:" << node_info_.node_id_ << " send the request id is:" << meta->request_id()
                << " the current time is:"
                << std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now())
                     .time_since_epoch()
                     .count();
  request_handler_(conn, meta, res, size);
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
    receive_callbacks_mutex_.unlock();
    MS_LOG(EXCEPTION) << "The memcpy_s error, errorno(" << ret << ")";
  }
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
  handlers_[NodeCommand::HEARTBEAT] = &AbstractNode::ProcessHeartbeatResp;
  handlers_[NodeCommand::REGISTER] = &AbstractNode::ProcessRegisterResp;
  handlers_[NodeCommand::FETCH_METADATA] = &AbstractNode::ProcessFetchServersResp;
  handlers_[NodeCommand::FINISH] = nullptr;
  handlers_[NodeCommand::SCALE_OUT_DONE] = nullptr;
  handlers_[NodeCommand::SCALE_IN_DONE] = nullptr;
  handlers_[NodeCommand::SEND_EVENT] = nullptr;
}

void AbstractNode::InitServerHandler() {
  server_handler_[NodeCommand::SEND_METADATA] = &AbstractNode::ProcessSendMetadata;
  server_handler_[NodeCommand::FINISH] = &AbstractNode::ProcessFinish;
  server_handler_[NodeCommand::SEND_DATA] = nullptr;
  server_handler_[NodeCommand::COLLECTIVE_SEND_DATA] = nullptr;
  server_handler_[NodeCommand::SCALE_OUT] = &AbstractNode::ProcessScaleOut;
  server_handler_[NodeCommand::SCALE_IN] = &AbstractNode::ProcessScaleIn;
  server_handler_[NodeCommand::SCALE_OUT_DONE] = &AbstractNode::ProcessScaleOutDone;
  server_handler_[NodeCommand::SCALE_IN_DONE] = &AbstractNode::ProcessScaleInDone;
  server_handler_[NodeCommand::SEND_EVENT] = &AbstractNode::ProcessEvent;
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
  worker_num_ = SizeToInt(PSContext::instance()->cluster_config().initial_worker_num);
  server_num_ = SizeToInt(PSContext::instance()->cluster_config().initial_server_num);
  scheduler_ip_ = PSContext::instance()->cluster_config().scheduler_host;
  scheduler_port_ = PSContext::instance()->cluster_config().scheduler_port;
  MS_LOG(INFO) << "The worker num:" << worker_num_ << ", the server num:" << server_num_
               << ", the scheduler ip:" << scheduler_ip_ << ", the scheduler port:" << scheduler_port_;
}

bool AbstractNode::Recover() {
  MS_EXCEPTION_IF_NULL(config_);
  if (config_->Exists(kKeyRecovery)) {
    MS_LOG(INFO) << "The node is support recovery.";
    node_recovery_ = std::make_unique<NodeRecovery>(this);
    MS_EXCEPTION_IF_NULL(node_recovery_);
    node_recovery_->Initialize(config_->Get(kKeyRecovery, ""));
    return node_recovery_->Recover();
  }
  return false;
}

void AbstractNode::OnEventCallback(const ClusterEvent &event) {
  if (!event_to_callback_.count(event)) {
    MS_LOG(ERROR) << "[Event]:The event callback of " << event << " is not set.";
  } else {
    MS_LOG(INFO) << "[Event]:Trigger the event:" << event;
    if (event_to_callback_[event]) {
      event_to_callback_[event]();
    }
  }
}

void AbstractNode::OnCustomEventCallback(const uint32_t &event) {
  if (!custom_event_to_callback_.count(event)) {
    MS_LOG(WARNING) << "[Custom event]:The event callback of " << event << " is not set.";
  } else {
    MS_LOG(INFO) << "[Custom event]:Trigger the event:" << event;
    if (custom_event_to_callback_[event]) {
      custom_event_to_callback_[event]();
    }
  }
}

bool AbstractNode::IsWorkerOrServer0(const std::unordered_map<std::string, NodeInfo> &info) {
  for (const auto &it : info) {
    if (it.second.is_alive == true && it.second.node_role_ == NodeRole::WORKER) {
      return true;
    }

    if (it.second.is_alive == true && it.second.rank_id_ == 0 && it.second.node_role_ == NodeRole::SERVER) {
      return true;
    }
  }
  return false;
}

void AbstractNode::CreateTcpServer() {
  MS_EXCEPTION_IF_NULL(config_);
  std::string interface;
  std::string server_ip;
  CommUtil::GetAvailableInterfaceAndIP(&interface, &server_ip);
  server_ = std::make_shared<TcpServer>(server_ip, 0, config_.get());
  MS_EXCEPTION_IF_NULL(server_);
  server_->SetMessageCallback([&](const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<MessageMeta> &meta,
                                  const Protos &protos, const void *data, size_t size) {
    MS_EXCEPTION_IF_NULL(meta);
    MS_EXCEPTION_IF_NULL(conn);
    MS_EXCEPTION_IF_NULL(data);
    if (server_handler_.count(meta->cmd()) == 0) {
      MS_LOG(EXCEPTION) << "The cmd:" << meta->cmd() << " is not supported!";
    }

    if (meta->cmd() == NodeCommand::COLLECTIVE_SEND_DATA) {
      ProcessCollectiveSendData(conn, meta, data, size);
      RunReceiveCallback(meta, protos, data, size);
    } else if (meta->cmd() == NodeCommand::SEND_DATA) {
      ProcessSendData(conn, meta, protos, data, size);
    } else {
      const auto &handler_ptr = server_handler_[meta->cmd()];
      (this->*handler_ptr)(conn, meta, protos, data, size);
    }
  });
  server_->Init();
  server_thread_ = std::make_unique<std::thread>([this]() {
    MS_LOG(INFO) << "The server node start a tcp server!";
    this->server_->Start();
  });
  MS_EXCEPTION_IF_NULL(server_thread_);
  server_thread_->detach();
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
