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

#include <memory>
#include "ps/core/abstract_ps_node.h"

namespace mindspore {
namespace ps {
namespace core {
void AbstractPSNode::StartHeartbeatTimer() {
  MS_LOG(INFO) << "The node role: " << CommUtil::NodeRoleToString(node_info_.node_role_)
               << ", the node id:" << node_info_.node_id_ << ", the node rank id:" << node_info_.rank_id_
               << " begin send heartbeat to the scheduler!";
  heart_beat_thread_ = std::make_unique<std::thread>([&]() {
    while (!is_finish_.load() && !stop_heartbeat_.load()) {
      if (!DoHeartbeat()) {
        MS_LOG(WARNING)
          << "Heartbeat timeout, the tcp connection to scheduler is lost, please check the status of scheduler.";

        if (CheckSchedulerTimeout()) {
          MS_LOG(WARNING) << "Scheduler is Timeout, please recovery.";
        }

        HandleHeartbeatTimeout();
      } else {
        UpdateSchedulerTime();
      }
      std::this_thread::sleep_for(std::chrono::seconds(PSContext::instance()->cluster_config().heartbeat_interval));
    }
    MS_LOG(INFO) << "Heartbeat thread stopped normally.";
    heartbeat_stopped_ = true;
  });
  MS_EXCEPTION_IF_NULL(heart_beat_thread_);
  heart_beat_thread_->detach();
}

bool AbstractPSNode::DoHeartbeat() {
  MS_EXCEPTION_IF_NULL(client_to_scheduler_);
  auto meta = std::make_shared<MessageMeta>();
  MS_EXCEPTION_IF_NULL(meta);
  meta->set_cmd(NodeCommand::HEARTBEAT);

  HeartbeatMessage heartbeat_message;
  heartbeat_message.set_node_id(node_info_.node_id_);
  heartbeat_message.set_persistent_state(PersistentState::NOT_ENABLE_PERSIST);
  heartbeat_message.set_has_address(true);
  heartbeat_message.set_ip(node_info_.ip_);
  heartbeat_message.set_port(node_info_.port_);

  // The worker role does not support disaster recovery currently.
  if (EnableRecovery() && role() == NodeRole::SERVER) {
    heartbeat_message.set_persistent_state(persistent_state_);
  }

  // Send the heartbeat.
  if (!SendMessageSync(client_to_scheduler_, meta, Protos::PROTOBUF, heartbeat_message.SerializeAsString().data(),
                       heartbeat_message.ByteSizeLong(), kCommTimeoutInSeconds)) {
    MS_LOG(WARNING) << "The node id:" << node_info_.node_id_ << " Send heartbeat timeout!";
    return false;
  }
  return true;
}

bool AbstractPSNode::InitClientToScheduler() {
  if (config_ == nullptr) {
    MS_LOG(WARNING) << "The config is empty.";
    return false;
  }

  // Create the TCP client to scheduler.
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
  client_to_scheduler_->Init();
  client_to_scheduler_->set_connected_callback([&]() {
    is_connected_to_scheduler_ = true;
    MS_LOG(WARNING) << "The connection to scheduler has be established.";
  });

  client_to_scheduler_->set_disconnected_callback([&]() {
    is_connected_to_scheduler_ = false;
    std::this_thread::sleep_for(std::chrono::milliseconds(PSContext::instance()->cluster_config().connect_interval));
    if (is_ready_.load() == false) {
      client_to_scheduler_->Init();
    }
  });

  client_to_scheduler_thread_ = std::make_unique<std::thread>([this]() {
    MS_LOG(INFO) << "The node start a tcp client!";
    client_to_scheduler_->Start();
  });
  client_to_scheduler_thread_->detach();

  // Timeout for waiting for the tcp connection to the scheduler, 10 seconds in recovery mode, or 900 seconds for first
  // build connection to scheduler.
  const uint32_t timeout_for_reinit_in_recovery = 10;
  uint32_t timeout = heartbeat_stopped_ ? timeout_for_reinit_in_recovery
                                        : PSContext::instance()->cluster_config().cluster_available_timeout;
  bool wait_res = client_to_scheduler_->WaitConnected(timeout);
  if (!wait_res) {
    is_ready_ = true;
  }
  return wait_res;
}

bool AbstractPSNode::HandleHeartbeatTimeout() {
  std::lock_guard<std::mutex> lock(reinit_mutex_);
  auto stop_heartbeat_thread = std::make_unique<std::thread>([this]() {
    // Stop doing heartbeat.
    if (!stop_heartbeat_.load()) {
      stop_heartbeat_ = true;
      while (!heartbeat_stopped_.load()) {
        if (is_finish_.load()) {
          return;
        }
        MS_LOG(INFO) << "Waiting for heartbeat to stop...";

        // Time interval for waiting the heartbeat to stop.
        uint32_t interval = 1000;
        std::this_thread::sleep_for(std::chrono::milliseconds(interval));
      }
    }

    // Reconnect to the scheduler.
    MS_LOG(INFO) << "The heartbeat thread has been stopped successfully.";

    bool success = false;
    while (!success) {
      if (is_finish_.load()) {
        return;
      }
      MS_LOG(WARNING) << "Trying to reconnect to the scheduler...";
      success = InitClientToScheduler();
      if (success) {
        MS_LOG(INFO) << "Connection to the scheduler has been establised successfully.";
        // Restart the heartbeat.
        stop_heartbeat_ = false;
        heartbeat_stopped_ = false;
        StartHeartbeatTimer();
        break;
      } else {
        MS_LOG(WARNING) << "Failed to establish connection to the scheduler.";
      }
    }
  });
  stop_heartbeat_thread->detach();
  return true;
}

void AbstractPSNode::RegisterInitCollectCommResphandler() {
  handlers_[NodeCommand::SEND_HOST_NAME] = &AbstractPSNode::ProcessReceiveSchedulerResp;
  handlers_[NodeCommand::QUERY_HOST_NAMES] = &AbstractPSNode::ProcessReceiveSchedulerResp;
  handlers_[NodeCommand::SEND_UNIQUE_ID] = &AbstractPSNode::ProcessReceiveSchedulerResp;
  handlers_[NodeCommand::QUERY_UNIQUE_ID] = &AbstractPSNode::ProcessReceiveSchedulerResp;
}

void AbstractPSNode::RegisterRecoveryRespHandler() {
  handlers_[NodeCommand::SEND_FINISH_TRANSFORM] = &AbstractPSNode::ProcessReceiveSchedulerResp;
  handlers_[NodeCommand::QUERY_FINISH_TRANSFORM] = &AbstractPSNode::ProcessReceiveSchedulerResp;
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
