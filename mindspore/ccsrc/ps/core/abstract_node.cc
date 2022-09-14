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

#include "include/common/debug/common.h"
#include "ps/core/communicator/http_communicator.h"
#include "ps/core/communicator/tcp_communicator.h"
#include "ps/core/node_recovery.h"

namespace mindspore {
namespace ps {
namespace core {
AbstractNode::~AbstractNode() {
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
}

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
  register_message.set_is_recover(is_recover.load());

  MS_LOG(INFO) << "The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
               << " the node id:" << node_info_.node_id_ << " begin to register to the scheduler!";

  if (!SendMessageAsync(client, message_meta, Protos::PROTOBUF, register_message.SerializeAsString().data(),
                        register_message.ByteSizeLong())) {
    MS_LOG(ERROR) << "The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                  << " the node id:" << node_info_.node_id_ << " register timeout!";
  } else {
    MS_LOG(INFO) << "The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                 << " the node id:" << node_info_.node_id_ << " send register success!";
  }
}

void AbstractNode::SendFailMessageToScheduler(const std::string &node_role, const std::string &event_info) {
  auto message_meta = std::make_shared<MessageMeta>();
  MS_EXCEPTION_IF_NULL(message_meta);
  message_meta->set_cmd(NodeCommand::FAILURE_EVENT_INFO);

  std::string now_time = ps::core::CommUtil::GetNowTime().time_str_mill;
  FailureEventMessage failure_event_message;
  failure_event_message.set_node_role(node_role);
  failure_event_message.set_ip(node_info_.ip_);
  failure_event_message.set_port(node_info_.port_);
  failure_event_message.set_time(now_time);
  failure_event_message.set_event(event_info);

  MS_LOG(INFO) << "The node role:" << node_role << "The node id:" << node_info_.node_id_
               << "begin to send failure message to scheduler!";

  if (!SendMessageAsync(client_to_scheduler_, message_meta, Protos::PROTOBUF,
                        failure_event_message.SerializeAsString().data(), failure_event_message.ByteSizeLong())) {
    MS_LOG(ERROR) << "The node role:" << node_role << " the node id:" << node_info_.node_id_
                  << " send failure message timeout!";
  } else {
    MS_LOG(INFO) << "The node role:" << node_role << " the node id:" << node_info_.node_id_ << " send failure message "
                 << event_info << "success!";
  }
}

void AbstractNode::ProcessRegisterResp(const std::shared_ptr<MessageMeta> &meta, const void *data, size_t size) {
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  RegisterRespMessage register_resp_message;
  CHECK_RETURN_TYPE(register_resp_message.ParseFromArray(data, SizeToInt(size)));
  MS_LOG(INFO) << "The node id get from scheduler is:" << register_resp_message.node_id()
               << ", rank_id is:" << register_resp_message.rank_id();

  if (register_resp_message.node_id() != node_info_.node_id_) {
    MS_LOG(ERROR) << "The node id received:" << register_resp_message.node_id()
                  << " is not match the current node id:" << node_info_.node_id_;
    return;
  }
  node_info_.rank_id_ = register_resp_message.rank_id();
  if (node_info_.rank_id_ == UINT32_MAX) {
    MS_LOG(ERROR) << "The rank id received:" << register_resp_message.rank_id();
    return;
  }

  // Receive the Register message, indicating that the scheduler is alive, so update the time point at which the
  // scheduler is alive
  UpdateSchedulerTime();

  MS_LOG(INFO) << "The node id is:" << node_info_.node_id_ << " registered scheduler success!";
}

bool AbstractNode::Broadcast(const NodeRole &node_role, const std::string &message, int command,
                             const uint32_t &timeout) {
  if (node_role != NodeRole::SERVER) {
    MS_LOG(EXCEPTION) << "Currently only supports broadcast to server nodes";
  }

  uint32_t broadcast_size = 0;
  (void)std::for_each(nodes_address_.begin(), nodes_address_.end(), [&broadcast_size, &node_role](const auto &addr) {
    if (addr.first.first == node_role) {
      ++broadcast_size;
    }
  });
  uint64_t request_id = AddMessageTrack(broadcast_size);

  for (auto it = nodes_address_.begin(); it != nodes_address_.end(); ++it) {
    if (it->first.first != node_role) {
      continue;
    }
    auto message_meta = std::make_shared<MessageMeta>();
    MS_EXCEPTION_IF_NULL(message_meta);
    message_meta->set_cmd(NodeCommand::SEND_DATA);
    message_meta->set_request_id(request_id);
    message_meta->set_rank_id(node_info_.rank_id_);
    message_meta->set_role(node_info_.node_role_);
    message_meta->set_user_cmd(command);

    auto client = GetOrCreateTcpClient((*it).first.second);
    if (!client->SendMessage(message_meta, Protos::RAW, message.data(), message.size())) {
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

  EventRespMessage event_resp_message;
  event_resp_message.set_event(event);

  for (auto it = nodes_address_.begin(); it != nodes_address_.end(); ++it) {
    const uint32_t rank_id = (*it).first.second;
    const NodeRole role = (*it).first.first;
    auto client = GetOrCreateTcpClient(rank_id, role);
    if (!SendMessageSync(client, message_meta, Protos::PROTOBUF, event_resp_message.SerializeAsString().data(),
                         event_resp_message.ByteSizeLong())) {
      MS_LOG(WARNING) << "send event to node role:" << CommUtil::NodeRoleToString(role) << ", rank id:" << rank_id
                      << " timeout!";
    }
  }
  MS_LOG(INFO) << "The node role:" << CommUtil::NodeRoleToString(node_info_.node_role_)
               << " the node id:" << node_info_.node_id_ << " send event to server/worker!";
}

void AbstractNode::RegisterEventCallback(const core::ClusterEvent &event, const EventCallback &event_cb) {
  event_to_callback_.try_emplace(event, event_cb);
}

void AbstractNode::RegisterCustomEventCallback(const uint32_t &event, const EventCallback &event_cb) {
  custom_event_to_callback_.try_emplace(event, event_cb);
}

bool AbstractNode::Send(const NodeRole &node_role, const uint32_t &rank_id, const void *message, size_t len,
                        int command, VectorPtr *output, const uint32_t &timeout) {
  MS_EXCEPTION_IF_NULL(message);
  if (!CommUtil::ValidateRankId(node_role, rank_id, worker_num_, server_num_)) {
    MS_LOG(ERROR) << "The node role or rank_id is illegal, the worker num:" << worker_num_
                  << ", the server num:" << server_num_ << ", the rank id:" << rank_id;
    return false;
  }

  uint64_t request_id = AddMessageTrack(1);
  if (output != nullptr) {
    set_message_callback(request_id, [this, request_id, rank_id, output]() {
      receive_messages_mutex_.lock();
      auto res = receive_messages_[request_id];
      *output = res[rank_id];
      receive_messages_.erase(request_id);
      receive_messages_mutex_.unlock();
    });
  }

  auto message_meta = std::make_shared<MessageMeta>();
  MS_EXCEPTION_IF_NULL(message_meta);
  message_meta->set_cmd(NodeCommand::SEND_DATA);
  message_meta->set_request_id(request_id);
  message_meta->set_rank_id(node_info_.rank_id_);
  message_meta->set_role(node_info_.node_role_);
  message_meta->set_user_cmd(command);

  auto client = GetOrCreateTcpClient(rank_id, node_role);
  MS_EXCEPTION_IF_NULL(client);
  if (!client->SendMessage(message_meta, Protos::RAW, message, len)) {
    MS_LOG(WARNING) << "Client send message failed.";
  }
  MS_LOG(DEBUG) << "The node role is:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                << ", the node id is:" << node_info_.node_id_ << " send the request id is:" << request_id;
  return Wait(request_id, timeout);
}

bool AbstractNode::Send(const NodeRole &node_role, const uint32_t &rank_id, const std::string &msg, int command,
                        VectorPtr *output, const uint32_t &timeout) {
  return Send(node_role, rank_id, msg.data(), msg.length(), command, output, timeout);
}

bool AbstractNode::Send(const NodeRole &node_role, const std::vector<uint32_t> &rank_ids,
                        const std::vector<std::string> &msgs, int command, std::vector<VectorPtr> *output,
                        const uint32_t &timeout) {
  uint64_t request_id = AddMessageTrack(msgs.size());

  if (rank_ids.size() != msgs.size()) {
    MS_LOG(EXCEPTION) << "The number of rank ids and messages are not equal!";
  }

  if (output != nullptr) {
    set_message_callback(request_id, [this, request_id, &rank_ids, output]() {
      receive_messages_mutex_.lock();
      auto &res = receive_messages_[request_id];
      for (auto &rank_id : rank_ids) {
        auto &response = res[rank_id];
        output->push_back(response);
      }
      receive_messages_.erase(request_id);
      receive_messages_mutex_.unlock();
    });
  }
  size_t size = rank_ids.size();
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

    auto &msg = msgs.at(it);

    auto client = GetOrCreateTcpClient(rank_ids.at(it), node_role);
    MS_EXCEPTION_IF_NULL(client);
    if (!client->SendMessage(message_meta, Protos::RAW, msg.data(), msg.size())) {
      MS_LOG(WARNING) << "Client send message failed.";
    }
  }
  MS_LOG(DEBUG) << "The node role is:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                << ", the node id is:" << node_info_.node_id_ << " send the request id is:" << request_id;
  return Wait(request_id, timeout);
}

bool AbstractNode::SendToScheduler(const void *message, size_t len, NodeCommand node_cmd, VectorPtr *output,
                                   const uint32_t &timeout) {
  MS_EXCEPTION_IF_NULL(message);

  uint32_t expected_reponse_num = 1;
  uint64_t request_id = AddMessageTrack(expected_reponse_num);
  auto message_meta = std::make_shared<MessageMeta>();
  MS_EXCEPTION_IF_NULL(message_meta);
  message_meta->set_cmd(node_cmd);
  message_meta->set_request_id(request_id);

  MS_EXCEPTION_IF_NULL(client_to_scheduler_);
  if (!client_to_scheduler_->SendMessage(message_meta, Protos::RAW, message, len)) {
    MS_LOG(WARNING) << "Failed to send message" << node_cmd << "to scheduler.";
  }

  bool ret = Wait(request_id, timeout);
  if (!ret) {
    MS_LOG(ERROR) << "Sending message " << node_cmd << " to scheduler timeout.";
    return ret;
  }

  // Assign the response value from scheduler.
  if (output != nullptr) {
    if (received_scheduler_messages_.count(request_id) == 0) {
      MS_LOG(ERROR) << "The response message of command " << node_cmd << ", request_id " << request_id
                    << " is not received yet.";
      return false;
    }
    *output = received_scheduler_messages_[request_id];
    (void)received_scheduler_messages_.erase(request_id);
  }
  return ret;
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

static std::string CollectiveMetaToString(const CollectiveMessageMeta &meta) {
  std::ostringstream os;
  os << "{iteration:" << meta.iteration() << ", data:" << meta.weight_name() << ", send rank:" << meta.send_rank_id()
     << ", recv rank:" << meta.recv_rank_id() << ", phase:" << meta.phase() << ", chunk index:" << meta.chunk_index()
     << ", for index:" << meta.for_index() << "}";
  return os.str();
}

uint64_t AbstractNode::FlCollectiveSendAsync(const CollectiveMessageMeta &collective_meta, const void *data,
                                             size_t size) {
  MS_EXCEPTION_IF_NULL(data);
  auto recv_rank_id = collective_meta.recv_rank_id();
  if (!CommUtil::ValidateRankId(SERVER, recv_rank_id, worker_num_, server_num_)) {
    MS_LOG(ERROR) << "The node role or rank_id is illegal, the worker num:" << worker_num_
                  << ", the server num:" << server_num_ << ", the rank id:" << recv_rank_id;
    return 0;
  }
  std::shared_ptr<MessageMeta> message_meta = std::make_shared<MessageMeta>();
  MS_EXCEPTION_IF_NULL(message_meta);
  message_meta->set_cmd(NodeCommand::COLLECTIVE_SEND_DATA);
  message_meta->set_rank_id(node_info_.rank_id_);
  message_meta->set_role(node_info_.node_role_);
  *(message_meta->mutable_collective_meta()) = collective_meta;
  message_meta->mutable_collective_meta()->set_enable_flag(true);
  message_meta->mutable_collective_meta()->set_send_rank_id(node_info_.rank_id_);

  MS_LOG(DEBUG) << "Send data to rank id:" << recv_rank_id
                << ", send meta:" << CollectiveMetaToString(message_meta->collective_meta());
  auto client = GetOrCreateTcpClient(recv_rank_id, SERVER);
  MS_EXCEPTION_IF_NULL(client);
  return SendCollectiveMeta(client, message_meta, Protos::RAW, data, size);
}

bool AbstractNode::FlCollectiveWaitInner(const CollectiveMessageMeta &expect_meta, VectorPtr *output,
                                         const uint32_t &timeout) {
  if (output == nullptr) {
    return false;
  }
  auto send_rank_id = expect_meta.send_rank_id();
  if (!CommUtil::ValidateRankId(SERVER, send_rank_id, worker_num_, server_num_)) {
    MS_LOG(ERROR) << "The node role or rank_id is illegal, the worker num:" << worker_num_
                  << ", the server num:" << server_num_ << ", the rank id:" << send_rank_id;
    return false;
  }
  auto check_meta = [](const CollectiveMessageMeta &left, const CollectiveMessageMeta &right) {
    return left.iteration() == right.iteration() && left.weight_name() == right.weight_name() &&
           left.recv_rank_id() == right.recv_rank_id() && left.send_rank_id() == right.send_rank_id() &&
           left.phase() == right.phase() && left.chunk_index() == right.chunk_index() &&
           left.for_index() == right.for_index();
  };
  auto iteration_num = expect_meta.iteration();
  std::unique_lock<std::mutex> lock(fl_receive_mutex_);
  auto &recv_data_list = fl_received_data_[send_rank_id];
  for (uint32_t i = 0; i < timeout; i++) {
    if (recv_data_list.empty()) {
      fl_receive_cond_.wait_for(lock, std::chrono::seconds(1), [&recv_data_list]() { return !recv_data_list.empty(); });
      if (recv_data_list.empty()) {               // timeout
        if (HasIterationFailed(iteration_num)) {  // if result of iteration reported by other server is failed
          MS_LOG(WARNING) << "Detect iteration " << iteration_num << " has failed";
          return false;
        }
        continue;
      }
    }
    while (!recv_data_list.empty()) {
      auto first = recv_data_list.begin();
      auto recv_meta = std::move(first->first);
      auto recv_data = std::move(first->second);
      recv_data_list.erase(first);
      MS_LOG(DEBUG) << "Handle receive data from rank id:" << send_rank_id
                    << ", recv meta:" << CollectiveMetaToString(recv_meta);
      if (recv_meta.iteration() != expect_meta.iteration()) {
        MS_LOG(WARNING) << "Skip recv data, iteration of recv meta " << recv_meta.iteration()
                        << " != iteration of expected meta " << expect_meta.iteration();
        continue;
      }
      // error data in the same iteration
      if (!check_meta(recv_meta, expect_meta)) {
        MS_LOG(WARNING) << "Recv meta not match expected meta, recv mata: " << CollectiveMetaToString(recv_meta)
                        << ", expected meta: " << CollectiveMetaToString(expect_meta);
        return false;
      }
      *output = recv_data;
      return true;  // success to recv data
    }
  }
  return false;
}

bool AbstractNode::FlCollectiveWait(const CollectiveMessageMeta &expect_meta, size_t expect_size, VectorPtr *output,
                                    const uint32_t &timeout) {
  if (output == nullptr) {
    MS_LOG(ERROR) << "FlCollectiveWait failed, parameter output invalid";
    return false;
  }
  auto data_recved = FlCollectiveWaitInner(expect_meta, output, timeout);
  if (!data_recved) {
    MS_LOG(ERROR) << "FlCollectiveWait failed, expect meta: " << CollectiveMetaToString(expect_meta);
    return false;
  }
  if (*output == nullptr) {
    MS_LOG(ERROR) << "FlCollectiveWait failed, recv buffer invalid";
    return false;
  }
  if (expect_size != (*output)->size()) {
    MS_LOG(ERROR) << "Expected data size " << expect_size << " != recv data size " << (*output)->size()
                  << CollectiveMetaToString(expect_meta);
    return false;
  }
  return true;
}

void AbstractNode::OnRecvCollectiveData(const MessageMeta &message_meta, const VectorPtr &data) {
  std::unique_lock<std::mutex> lock(fl_receive_mutex_);
  auto &recv_meta = message_meta.collective_meta();
  auto send_rank_id = recv_meta.send_rank_id();
  MS_LOG(DEBUG) << "Receive data from rank id:" << send_rank_id << ", recv meta:" << CollectiveMetaToString(recv_meta);
  fl_received_data_[send_rank_id].emplace_back(std::make_pair(recv_meta, data));
  fl_receive_cond_.notify_all();
}

void AbstractNode::SetIterationResult(size_t last_iteration, bool is_iteration_valid) {
  iteration_failed_ = !is_iteration_valid;
  failed_iteration_num_ = last_iteration;
}

bool AbstractNode::HasIterationFailed(uint32_t iteration_num) const {
  return iteration_num == failed_iteration_num_ && iteration_failed_;
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

PersistentState AbstractNode::persistent_state() const { return persistent_state_; }
void AbstractNode::set_persistent_state(PersistentState persistent_state) { persistent_state_ = persistent_state; }

uint32_t AbstractNode::worker_num() const { return worker_num_; }

uint32_t AbstractNode::server_num() const { return server_num_; }

void AbstractNode::set_worker_num(const uint32_t &worker_num) { worker_num_ = worker_num; }

void AbstractNode::set_server_num(const uint32_t &server_num) { server_num_ = server_num; }

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
    uint32_t connect_interval = PSContext::instance()->cluster_config().connect_interval;
    uint32_t heartbeat_interval = PSContext::instance()->cluster_config().heartbeat_interval * 1000;
    uint32_t reconnect_interval = 0;
    if (heartbeat_interval > connect_interval) {
      MS_LOG(WARNING) << "heartbeat_interval [" << heartbeat_interval << "] is larger than connect_interval ["
                      << connect_interval << "], reset connect_interval to " << heartbeat_interval;
    }
    while (!is_finish_.load()) {
      if (!Heartbeat(client)) {
        MS_LOG(WARNING) << "The node role is:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                        << ", the node id is:" << node_info_.node_id_ << " Send heartbeat failed!";
        if (CheckSchedulerTimeout()) {
          MS_LOG(WARNING) << "Scheduler is Timeout, please recovery.";
        }
      } else {
        UpdateSchedulerTime();
      }

      if (!is_already_finished_ && (client->connection_status() == -1)) {
        if (reconnect_interval > connect_interval) {
          MS_LOG(WARNING) << "Connection to Scheduler is disconnected, try to reconnect.";
          reconnect_interval = 0;
          ConnectToScheduler();
        } else {
          reconnect_interval += heartbeat_interval;
        }
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(heartbeat_interval));
    }
  });
  MS_EXCEPTION_IF_NULL(heart_beat_thread_);
}

bool AbstractNode::Heartbeat(const std::shared_ptr<TcpClient> &client) {
  MS_EXCEPTION_IF_NULL(client);
  if (client->connection_status() != 1) {
    return false;
  }
  auto meta = std::make_shared<MessageMeta>();
  MS_EXCEPTION_IF_NULL(meta);
  meta->set_cmd(NodeCommand::HEARTBEAT);

  HeartbeatMessage heartbeat_message;
  heartbeat_message.set_node_id(node_info_.node_id_);
  heartbeat_message.set_persistent_state(PersistentState::NOT_ENABLE_PERSIST);

  // The worker role does not support disaster recovery currently.
  if (EnableRecovery() && role() == NodeRole::SERVER) {
    heartbeat_message.set_persistent_state(persistent_state_);
  }

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

  if (heartbeat_resp_message.cluster_state() != current_cluster_state_ &&
      current_cluster_state_ != ClusterState::CLUSTER_SCALE_IN &&
      current_cluster_state_ != ClusterState::CLUSTER_SCALE_OUT) {
    UpdateClusterState(heartbeat_resp_message.cluster_state());
  }
  MS_LOG(DEBUG) << "The current cluster state from heartbeat:"
                << CommUtil::ClusterStateToString(current_cluster_state_);

  std::string timeoutNodeId;

  all_nodes_info_.clear();
  for (const auto &it : heartbeat_resp_message.servers_meta()) {
    NodeInfo info;
    info.ip_ = it.ip();
    info.node_id_ = it.node_id();
    info.port_ = it.port();
    info.node_role_ = it.role();
    info.rank_id_ = it.rank_id();
    info.is_alive = it.is_alive();

    if (!info.is_alive) {
      timeoutNodeId += (info.node_id_ + " ");
    }

    all_nodes_info_[info.node_id_] = info;
    MS_LOG(DEBUG) << "The node id:" << info.node_id_ << ", the rank id:" << info.rank_id_
                  << ", the node role:" << CommUtil::NodeRoleToString(info.node_role_) << " is alive:" << info.is_alive;
  }
  bool is_worker = heartbeat_resp_message.is_worker();
  bool is_ps_mode = PSContext::instance()->server_mode() == ps::kServerModePS;
  bool not_enable_recover_node_timeout = (is_worker && is_ps_mode);

  if (current_cluster_state_ == ClusterState::NODE_TIMEOUT) {
    if (node_recovery_ == nullptr || not_enable_recover_node_timeout) {
      MS_LOG(INFO) << "The recovery is disabled. Trigger NODE_TIMEOUT event.";
      // Avoid other methods blocking endlessly when NODE_TIMEOUT event is triggered.
      is_ready_ = true;
      wait_start_cond_.notify_all();
      is_finish_ = true;
      wait_finish_cond_.notify_all();
      OnEventCallback(ClusterEvent::NODE_TIMEOUT);
    } else {
      MS_LOG(INFO) << "The nodes:" << timeoutNodeId
                   << "is support recovery, users can pull up this node to restore the cluster.";
    }
  }

  if (!EnableRecovery()) {
    return;
  }

  PersistentCommand persistent_cmd = heartbeat_resp_message.persistent_cmd();
  // The worker role does not support disaster recovery for the time being.
  if (role() == NodeRole::SERVER && persistent_cmd == PersistentCommand::BEGIN_PERSIST &&
      persistent_state_ != PersistentState::PERSISTING) {
    OnEventCallback(ClusterEvent::ON_BEGIN_PERSIST);
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
    nodes_address_[std::make_pair(it.role(), it.rank_id())] = std::make_pair(it.ip(), it.port());
    MS_LOG(INFO) << "The server ip is:" << it.ip() << ", the port is:" << it.port();
  }
}

void AbstractNode::ProcessReceiveSchedulerResp(const std::shared_ptr<MessageMeta> &meta, const void *data,
                                               size_t size) {
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  std::lock_guard<std::mutex> lock(receive_messages_mutex_);

  const uint64_t request_id = meta->request_id();
  VectorPtr received_data = std::make_shared<std::vector<unsigned char>>(size, 0);
  if (size > 0) {
    size_t dest_size = size;
    size_t src_size = size;
    auto ret = memcpy_s(received_data.get()->data(), dest_size, data, src_size);
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "The memcpy_s error, errorno(" << ret << ")";
    }
  }
  received_scheduler_messages_[request_id] = received_data;
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
  send_meta_message.ParseFromArray(data, SizeToInt(size));
  worker_num_ = send_meta_message.worker_num();
  server_num_ = send_meta_message.server_num();
  if (send_meta_message.rank_id() < 0) {
    MS_LOG(EXCEPTION) << "The rank id is wrong.";
  }
  node_info_.rank_id_ = send_meta_message.rank_id();
  UpdateClusterState(send_meta_message.cluster_state());
  MS_LOG(INFO) << "The send metadata worker num:" << worker_num_ << ", server num:" << server_num_
               << ", cluster state is:" << CommUtil::ClusterStateToString(current_cluster_state_)
               << ", the rank id:" << node_info_.rank_id_;

  client_mutex_.lock();
  nodes_address_.clear();
  for (const auto &it : send_meta_message.servers_meta()) {
    nodes_address_[std::make_pair(it.role(), it.rank_id())] = std::make_pair(it.ip(), it.port());
    MS_LOG(INFO) << "The node role:" << CommUtil::NodeRoleToString(it.role()) << ", node id:" << it.node_id()
                 << ", rank id:" << it.rank_id() << ", ip:" << it.ip() << ", port:" << it.port();
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

  if (cancelSafeModeFn_ && current_cluster_state_ == ClusterState::CLUSTER_SCALE_OUT_ROLLBACK) {
    MS_LOG(WARNING) << "Trigger cluster scale out rollback done event.";
    OnEventCallback(ClusterEvent::CLUSTER_SCALE_OUT_ROLLBACK_DONE);
    cancelSafeModeFn_();
  }

  std::lock_guard<std::mutex> lock(client_mutex_);
  connected_nodes_.clear();

  OnEventCallback(ClusterEvent::ON_SEND_META_DATA);
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
  MS_LOG(INFO) << "This node receive a scale out done from scheduler.";
  if (!server_->SendMessage(conn, meta, Protos::RAW, data, size)) {
    MS_LOG(WARNING) << "Server response message failed.";
  }
  is_ready_ = true;
  UpdateClusterState(ClusterState::CLUSTER_READY);
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
  UpdateClusterState(ClusterState::CLUSTER_READY);
}

void AbstractNode::ProcessEvent(const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<MessageMeta> &meta,
                                const Protos &, const void *data, size_t size) {
  MS_EXCEPTION_IF_NULL(conn);
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  EventRespMessage event_resp_message;
  event_resp_message.ParseFromArray(data, SizeToInt(size));
  uint32_t event = event_resp_message.event();
  if (!server_->SendMessage(conn, meta, Protos::RAW, data, size)) {
    MS_LOG(WARNING) << "Server response message failed.";
  }
  MS_LOG(INFO) << "This node receive a event:" << event;
  if (event == static_cast<uint32_t>(ps::UserDefineEvent::kNodeTimeout)) {
    OnEventCallback(ClusterEvent::NODE_TIMEOUT);
  } else {
    OnCustomEventCallback(event);
  }
}

void AbstractNode::ProcessScaleOutRollback(const std::shared_ptr<TcpConnection> &conn,
                                           const std::shared_ptr<MessageMeta> &meta, const Protos &, const void *data,
                                           size_t size) {
  MS_EXCEPTION_IF_NULL(conn);
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);

  if (!server_->SendMessage(conn, meta, Protos::RAW, data, size)) {
    MS_LOG(WARNING) << "Server response message failed.";
  }

  UpdateClusterState(ClusterState::CLUSTER_SCALE_OUT_ROLLBACK);

  MS_LOG(INFO) << "[Scale out rollback]: begin to set scale out rollback.";
  Register(client_to_scheduler_);
  std::lock_guard<std::mutex> lock(client_mutex_);
  connected_nodes_.clear();

  MS_LOG(INFO) << "The node begin to start scale out rollback.";
}

void AbstractNode::ProcessScaleOut(const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<MessageMeta> &meta,
                                   const Protos &, const void *data, size_t size) {
  MS_EXCEPTION_IF_NULL(conn);
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);

  ScaleOutMessage scale_out_message;
  scale_out_message.ParseFromArray(data, SizeToInt(size));
  int32_t worker_num = scale_out_message.worker_num();
  int32_t server_num = scale_out_message.server_num();
  MS_LOG(WARNING) << "The scale out worker num:" << worker_num << ", the server num:" << server_num;

  if (!server_->SendMessage(conn, meta, Protos::RAW, data, size)) {
    MS_LOG(WARNING) << "Server response message failed.";
  }
  OnEventCallback(ClusterEvent::READY_FOR_SCALE_OUT);
  UpdateClusterState(ClusterState::CLUSTER_SCALE_OUT);
  is_ready_ = false;
}

void AbstractNode::ProcessScaleIn(const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<MessageMeta> &meta,
                                  const Protos &, const void *data, size_t size) {
  MS_EXCEPTION_IF_NULL(conn);
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);

  ScaleInMessage scale_in_message;
  scale_in_message.ParseFromArray(data, SizeToInt(size));
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
  UpdateClusterState(ClusterState::CLUSTER_SCALE_IN);
  is_ready_ = false;
}

void AbstractNode::ProcessSchedulerRecovery(const std::shared_ptr<TcpConnection> &conn,
                                            const std::shared_ptr<MessageMeta> &meta, const Protos &, const void *data,
                                            size_t size) {
  MS_EXCEPTION_IF_NULL(conn);
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  SendMetadataMessage scheduler_recovery_message;
  (void)scheduler_recovery_message.ParseFromArray(data, SizeToInt(size));
  worker_num_ = scheduler_recovery_message.worker_num();
  server_num_ = scheduler_recovery_message.server_num();
  uint32_t rank_id = scheduler_recovery_message.rank_id();

  MS_LOG(INFO) << "[Scheduler Recovery]: The scheduler recovery worker num:" << worker_num_
               << ", the server num:" << server_num_ << ", the rank id: " << rank_id;

  if (!server_->SendMessage(conn, meta, Protos::RAW, data, size)) {
    MS_LOG(WARNING) << "[Scheduler Recovery]: Server response message failed.";
  }
  MS_LOG(INFO) << "[Scheduler Recovery]: Server response message success!.";

  ConnectToScheduler();
  bool connected = client_to_scheduler_->WaitConnected();
  if (!connected) {
    MS_LOG(WARNING) << "[Scheduler Recovery]: Server node connect to scheduler timedout!";
  }

  Register(client_to_scheduler_);
  std::lock_guard<std::mutex> lock(client_mutex_);
  connected_nodes_.clear();
  MS_LOG(INFO) << "[Scheduler Recovery]: This node connect to scheduler successful!";

  if (cancelSafeModeFn_ && (current_cluster_state_ == ClusterState::CLUSTER_SCALE_IN ||
                            current_cluster_state_ == ClusterState::CLUSTER_SCALE_OUT)) {
    MS_LOG(INFO) << "[Scheduler Recovery]: Cancel Safe mode for " << kClusterState.at(current_cluster_state_);
    cancelSafeModeFn_();
  }

  UpdateClusterState(ClusterState::CLUSTER_SCHEDULER_RECOVERY);
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
  // If the cluster state is NODE_TIMEOUT, this node is already disconnected.
  if (current_cluster_state_ == ClusterState::NODE_TIMEOUT) {
    return true;
  }
  std::unique_lock<std::mutex> lock(wait_finish_mutex_);
  auto condition_func = [this] {
    if (is_finish_.load()) {
      MS_LOG(INFO) << "The node id:" << node_info_.node_id_ << " is success finish!";
    }
    return is_finish_.load();
  };

  bool res;
  if (timeout == UINT32_MAX) {
    // Caller should use this method to help block the thread.
    wait_finish_cond_.wait(lock, condition_func);
    res = true;
  } else {
    res = wait_finish_cond_.wait_for(lock, std::chrono::seconds(timeout), condition_func);
  }

  return res;
}

void AbstractNode::InitClientToServer() {
  // create tcp client to myself in case of event dispatch failed when Send msg to server 0 failed
  client_to_server_ = std::make_shared<TcpClient>(node_info_.ip_, node_info_.port_, node_info_.node_role_);
  MS_EXCEPTION_IF_NULL(client_to_server_);
  client_to_server_->Init();
  MS_LOG(INFO) << "The node start a tcp client to this node!";
}

bool AbstractNode::InitClientToScheduler() {
  if (config_ == nullptr) {
    MS_LOG(WARNING) << "The config is empty.";
    return false;
  }
  client_to_scheduler_ = std::make_shared<TcpClient>(scheduler_ip_, scheduler_port_, NodeRole::SCHEDULER);
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
  ConnectToScheduler();
  StartHeartbeatTimer(client_to_scheduler_);
  MS_LOG(INFO) << "Start heartbeat timer!";

  bool wait_res = client_to_scheduler_->WaitConnected();
  if (!wait_res) {
    is_ready_ = true;
  }
  return wait_res;
}
void AbstractNode::ConnectToScheduler() {
  client_to_scheduler_->Init();
  if (TcpClient::is_started()) {
    return;
  }

  if (client_to_scheduler_thread_ != nullptr && client_to_scheduler_thread_->joinable()) {
    client_to_scheduler_thread_->join();
  }
  client_to_scheduler_thread_ = std::make_unique<std::thread>([this]() {
    MS_LOG(INFO) << "The node start a tcp client!";
    client_to_scheduler_->Start();
  });
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

void AbstractNode::ProcessSendData(const std::shared_ptr<TcpConnection> &conn, const std::shared_ptr<MessageMeta> &meta,
                                   const Protos &, const void *data, size_t size) {
  MS_EXCEPTION_IF_NULL(conn);
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  MS_LOG(DEBUG) << "The node role is:" << CommUtil::NodeRoleToString(node_info_.node_role_)
                << ", the node id is:" << node_info_.node_id_ << " send the request id is:" << meta->request_id()
                << " the current time is:"
                << std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now())
                     .time_since_epoch()
                     .count();
  request_handler_(conn, meta, data, size);
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
  int ret = memcpy_s(received_data->data(), dest_size, data, src_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "The memcpy_s error, errorno(" << ret << ")";
  }
  if (meta->collective_meta().enable_flag()) {
    OnRecvCollectiveData(*meta, received_data);
    return;
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
  handlers_[NodeCommand::HEARTBEAT] = &AbstractNode::ProcessHeartbeatResp;
  handlers_[NodeCommand::REGISTER] = &AbstractNode::ProcessRegisterResp;
  handlers_[NodeCommand::FETCH_METADATA] = &AbstractNode::ProcessFetchServersResp;
  handlers_[NodeCommand::FINISH] = nullptr;
  handlers_[NodeCommand::SCALE_OUT_DONE] = nullptr;
  handlers_[NodeCommand::SCALE_IN_DONE] = nullptr;
  handlers_[NodeCommand::SEND_EVENT] = nullptr;
  RegisterActorRouteTableRspHandler();
  RegisterInitCollectCommResphandler();
  RegisterRecoveryRespHandler();
}

void AbstractNode::RegisterActorRouteTableRspHandler() {
  handlers_[NodeCommand::REGISTER_ACTOR_ROUTE] = &AbstractNode::ProcessReceiveSchedulerResp;
  handlers_[NodeCommand::DELETE_ACTOR_ROUTE] = &AbstractNode::ProcessReceiveSchedulerResp;
  handlers_[NodeCommand::LOOKUP_ACTOR_ROUTE] = &AbstractNode::ProcessReceiveSchedulerResp;
}

void AbstractNode::InitServerHandler() {
  server_handler_[NodeCommand::SEND_METADATA] = &AbstractNode::ProcessSendMetadata;
  server_handler_[NodeCommand::FINISH] = &AbstractNode::ProcessFinish;
  server_handler_[NodeCommand::SEND_DATA] = &AbstractNode::ProcessSendData;
  server_handler_[NodeCommand::COLLECTIVE_SEND_DATA] = &AbstractNode::ProcessCollectiveSendData;
  server_handler_[NodeCommand::SCALE_OUT] = &AbstractNode::ProcessScaleOut;
  server_handler_[NodeCommand::SCALE_IN] = &AbstractNode::ProcessScaleIn;
  server_handler_[NodeCommand::SCALE_OUT_DONE] = &AbstractNode::ProcessScaleOutDone;
  server_handler_[NodeCommand::SCALE_IN_DONE] = &AbstractNode::ProcessScaleInDone;
  server_handler_[NodeCommand::SEND_EVENT] = &AbstractNode::ProcessEvent;
  server_handler_[NodeCommand::SCHEDULER_RECOVERY] = &AbstractNode::ProcessSchedulerRecovery;
  server_handler_[NodeCommand::PREPARE_BUILDING_NETWORK] = &AbstractNode::ProcessPrepareBuildingNetwork;
  server_handler_[NodeCommand::SCALE_OUT_ROLLBACK] = &AbstractNode::ProcessScaleOutRollback;
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

bool AbstractNode::Recover() {
  MS_EXCEPTION_IF_NULL(config_);
  if (config_->Exists(kKeyRecovery)) {
    MS_LOG(INFO) << "The node is support recovery.";
    node_recovery_ = std::make_unique<NodeRecovery>(this);
    MS_EXCEPTION_IF_NULL(node_recovery_);
    if (node_recovery_->Initialize(config_->Get(kKeyRecovery, ""))) {
      MS_LOG(INFO) << "Initializing node recovery success.";
      return node_recovery_->Recover();
    }
  }
  return false;
}

void AbstractNode::OnEventCallback(const ClusterEvent &event) {
  if (!event_to_callback_.count(event)) {
    MS_LOG(INFO) << "[Event]:The event callback of " << event << " is not set.";
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

void AbstractNode::UpdateClusterState(const ClusterState &state) {
  std::lock_guard<std::mutex> lock(cluster_state_mutex_);
  std::string state_str = CommUtil::ClusterStateToString(state);
  if (state_str.empty()) {
    return;
  }

  if (state == current_cluster_state_) {
    return;
  }
  MS_LOG(INFO) << "[state]: Cluster state change from:" << CommUtil::ClusterStateToString(current_cluster_state_)
               << " to " << state_str;
  current_cluster_state_ = state;
}

void AbstractNode::PersistMetaData() {
  if (node_recovery_ == nullptr) {
    MS_LOG(WARNING) << "node recovery is null, so don't persist meta data";
    return;
  }
  if (config_->Exists(kKeyRecovery)) {
    ClusterConfig &clusterConfig = PSContext::instance()->cluster_config();
    clusterConfig.scheduler_host = this->scheduler_ip();
    clusterConfig.scheduler_port = this->scheduler_port();
    clusterConfig.initial_worker_num = worker_num_;
    clusterConfig.initial_server_num = server_num_;

    node_recovery_->Persist(clusterConfig);
  }
}

void AbstractNode::ProcessPrepareBuildingNetwork(const std::shared_ptr<TcpConnection> &conn,
                                                 const std::shared_ptr<MessageMeta> &meta, const Protos &,
                                                 const void *data, size_t size) {
  MS_EXCEPTION_IF_NULL(conn);
  MS_EXCEPTION_IF_NULL(meta);
  MS_EXCEPTION_IF_NULL(data);
  if (!server_->SendMessage(conn, meta, Protos::RAW, data, size)) {
    MS_LOG(ERROR) << "sever response message failed, prepare for building network failed.";
  } else {
    MS_LOG(INFO) << "prepare for building network success.";
  }
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
