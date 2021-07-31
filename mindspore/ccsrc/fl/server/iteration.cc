/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "fl/server/iteration.h"
#include <memory>
#include <string>
#include <vector>
#include <numeric>
#include "fl/server/model_store.h"
#include "fl/server/server.h"

namespace mindspore {
namespace fl {
namespace server {
class Server;
void Iteration::RegisterMessageCallback(const std::shared_ptr<ps::core::TcpCommunicator> &communicator) {
  MS_EXCEPTION_IF_NULL(communicator);
  communicator_ = communicator;
  communicator_->RegisterMsgCallBack("syncIteration",
                                     std::bind(&Iteration::HandleSyncIterationRequest, this, std::placeholders::_1));
  communicator_->RegisterMsgCallBack(
    "notifyLeaderToNextIter",
    std::bind(&Iteration::HandleNotifyLeaderMoveToNextIterRequest, this, std::placeholders::_1));
  communicator_->RegisterMsgCallBack(
    "prepareForNextIter", std::bind(&Iteration::HandlePrepareForNextIterRequest, this, std::placeholders::_1));
  communicator_->RegisterMsgCallBack("proceedToNextIter",
                                     std::bind(&Iteration::HandleMoveToNextIterRequest, this, std::placeholders::_1));
  communicator_->RegisterMsgCallBack("endLastIter",
                                     std::bind(&Iteration::HandleEndLastIterRequest, this, std::placeholders::_1));
}

void Iteration::RegisterEventCallback(const std::shared_ptr<ps::core::ServerNode> &server_node) {
  MS_EXCEPTION_IF_NULL(server_node);
  server_node_ = server_node;
  server_node->RegisterCustomEventCallback(static_cast<uint32_t>(ps::CustomEvent::kIterationRunning),
                                           std::bind(&Iteration::HandleIterationRunningEvent, this));
  server_node->RegisterCustomEventCallback(static_cast<uint32_t>(ps::CustomEvent::kIterationCompleted),
                                           std::bind(&Iteration::HandleIterationCompletedEvent, this));
}

void Iteration::AddRound(const std::shared_ptr<Round> &round) {
  MS_EXCEPTION_IF_NULL(round);
  rounds_.push_back(round);
}

void Iteration::InitRounds(const std::vector<std::shared_ptr<ps::core::CommunicatorBase>> &communicators,
                           const TimeOutCb &timeout_cb, const FinishIterCb &finish_iteration_cb) {
  if (communicators.empty()) {
    MS_LOG(EXCEPTION) << "Communicators for rounds is empty.";
    return;
  }

  (void)std::for_each(communicators.begin(), communicators.end(),
                      [&](const std::shared_ptr<ps::core::CommunicatorBase> &communicator) {
                        for (auto &round : rounds_) {
                          MS_EXCEPTION_IF_NULL(round);
                          round->Initialize(communicator, timeout_cb, finish_iteration_cb);
                        }
                      });

  // The time window for one iteration, which will be used in some round kernels.
  size_t iteration_time_window = std::accumulate(rounds_.begin(), rounds_.end(), IntToSize(0),
                                                 [](size_t total, const std::shared_ptr<Round> &round) {
                                                   MS_EXCEPTION_IF_NULL(round);
                                                   return round->check_timeout() ? total + round->time_window() : total;
                                                 });
  LocalMetaStore::GetInstance().put_value(kCtxTotalTimeoutDuration, iteration_time_window);
  MS_LOG(INFO) << "Time window for one iteration is " << iteration_time_window;
  return;
}

void Iteration::MoveToNextIteration(bool is_last_iter_valid, const std::string &reason) {
  MS_LOG(INFO) << "Notify cluster starts to proceed to next iteration. Iteration is " << iteration_num_
               << " validation is " << is_last_iter_valid << ". Reason: " << reason;
  if (IsMoveToNextIterRequestReentrant(iteration_num_)) {
    return;
  }

  MS_ERROR_IF_NULL_WO_RET_VAL(server_node_);
  if (server_node_->rank_id() == kLeaderServerRank) {
    if (!BroadcastPrepareForNextIterRequest(is_last_iter_valid, reason)) {
      MS_LOG(ERROR) << "Broadcast prepare for next iteration request failed.";
      return;
    }
    if (!BroadcastMoveToNextIterRequest(is_last_iter_valid, reason)) {
      MS_LOG(ERROR) << "Broadcast proceed to next iteration request failed.";
      return;
    }
    if (!BroadcastEndLastIterRequest(iteration_num_)) {
      MS_LOG(ERROR) << "Broadcast end last iteration request failed.";
      return;
    }
  } else {
    // If this server is the follower server, notify leader server to control the cluster to proceed to next iteration.
    if (!NotifyLeaderMoveToNextIteration(is_last_iter_valid, reason)) {
      MS_LOG(ERROR) << "Server " << server_node_->rank_id() << " notifying the leader server failed.";
      return;
    }
  }
}

void Iteration::SetIterationRunning() {
  MS_LOG(INFO) << "Iteration " << iteration_num_ << " start running.";
  MS_ERROR_IF_NULL_WO_RET_VAL(server_node_);
  if (server_node_->rank_id() == kLeaderServerRank) {
    // This event helps worker/server to be consistent in iteration state.
    server_node_->BroadcastEvent(static_cast<uint32_t>(ps::CustomEvent::kIterationRunning));
  }
  iteration_state_ = IterationState::kRunning;
}

void Iteration::SetIterationCompleted() {
  MS_LOG(INFO) << "Iteration " << iteration_num_ << " completes.";
  MS_ERROR_IF_NULL_WO_RET_VAL(server_node_);
  if (server_node_->rank_id() == kLeaderServerRank) {
    // This event helps worker/server to be consistent in iteration state.
    server_node_->BroadcastEvent(static_cast<uint32_t>(ps::CustomEvent::kIterationCompleted));
  }
  iteration_state_ = IterationState::kCompleted;
}

void Iteration::ScalingBarrier() {
  MS_LOG(INFO) << "Starting Iteration scaling barrier.";
  while (iteration_state_.load() != IterationState::kCompleted) {
    std::this_thread::yield();
  }
  MS_LOG(INFO) << "Ending Iteration scaling barrier.";
}

bool Iteration::ReInitForScaling(uint32_t server_num, uint32_t server_rank) {
  for (auto &round : rounds_) {
    if (!round->ReInitForScaling(server_num)) {
      MS_LOG(WARNING) << "Reinitializing round " << round->name() << " for scaling failed.";
      return false;
    }
  }
  if (server_rank != kLeaderServerRank) {
    if (!SyncIteration(server_rank)) {
      MS_LOG(ERROR) << "Synchronizing iteration failed.";
      return false;
    }
  }
  return true;
}

const std::vector<std::shared_ptr<Round>> &Iteration::rounds() const { return rounds_; }

bool Iteration::is_last_iteration_valid() const { return is_last_iteration_valid_; }

bool Iteration::SyncIteration(uint32_t rank) {
  MS_ERROR_IF_NULL_W_RET_VAL(communicator_, false);
  SyncIterationRequest sync_iter_req;
  sync_iter_req.set_rank(rank);

  std::shared_ptr<std::vector<unsigned char>> sync_iter_rsp_msg = nullptr;
  if (!communicator_->SendPbRequest(sync_iter_req, kLeaderServerRank, ps::core::TcpUserCommand::kSyncIteration,
                                    &sync_iter_rsp_msg)) {
    MS_LOG(ERROR) << "Sending synchronizing iteration message to leader server failed.";
    return false;
  }

  MS_ERROR_IF_NULL_W_RET_VAL(sync_iter_rsp_msg, false);
  SyncIterationResponse sync_iter_rsp;
  (void)sync_iter_rsp.ParseFromArray(sync_iter_rsp_msg->data(), SizeToInt(sync_iter_rsp_msg->size()));
  iteration_num_ = sync_iter_rsp.iteration();
  MS_LOG(INFO) << "After synchronizing, server " << rank << " current iteration number is "
               << sync_iter_rsp.iteration();
  return true;
}

void Iteration::HandleSyncIterationRequest(const std::shared_ptr<ps::core::MessageHandler> &message) {
  MS_ERROR_IF_NULL_WO_RET_VAL(message);
  MS_ERROR_IF_NULL_WO_RET_VAL(communicator_);

  SyncIterationRequest sync_iter_req;
  (void)sync_iter_req.ParseFromArray(message->data(), SizeToInt(message->len()));
  uint32_t rank = sync_iter_req.rank();
  MS_LOG(INFO) << "Synchronizing iteration request from rank " << rank;

  SyncIterationResponse sync_iter_rsp;
  sync_iter_rsp.set_iteration(iteration_num_);
  std::string sync_iter_rsp_msg = sync_iter_rsp.SerializeAsString();
  if (!communicator_->SendResponse(sync_iter_rsp_msg.data(), sync_iter_rsp_msg.size(), message)) {
    MS_LOG(ERROR) << "Sending response failed.";
    return;
  }
}

bool Iteration::IsMoveToNextIterRequestReentrant(uint64_t iteration_num) {
  std::unique_lock<std::mutex> lock(pinned_mtx_);
  if (pinned_iter_num_ == iteration_num) {
    MS_LOG(WARNING) << "MoveToNextIteration is not reentrant. Ignore this call.";
    return true;
  }
  pinned_iter_num_ = iteration_num;
  return false;
}

bool Iteration::NotifyLeaderMoveToNextIteration(bool is_last_iter_valid, const std::string &reason) {
  MS_ERROR_IF_NULL_W_RET_VAL(communicator_, false);
  MS_LOG(INFO) << "Notify leader server to control the cluster to proceed to next iteration.";
  NotifyLeaderMoveToNextIterRequest notify_leader_to_next_iter_req;
  notify_leader_to_next_iter_req.set_rank(server_node_->rank_id());
  notify_leader_to_next_iter_req.set_is_last_iter_valid(is_last_iter_valid);
  notify_leader_to_next_iter_req.set_iter_num(iteration_num_);
  notify_leader_to_next_iter_req.set_reason(reason);
  if (!communicator_->SendPbRequest(notify_leader_to_next_iter_req, kLeaderServerRank,
                                    ps::core::TcpUserCommand::kNotifyLeaderToNextIter)) {
    MS_LOG(WARNING) << "Sending notify leader server to proceed next iteration request to leader server 0 failed.";
    return false;
  }
  return true;
}

void Iteration::HandleNotifyLeaderMoveToNextIterRequest(const std::shared_ptr<ps::core::MessageHandler> &message) {
  MS_ERROR_IF_NULL_WO_RET_VAL(message);
  MS_ERROR_IF_NULL_WO_RET_VAL(communicator_);
  NotifyLeaderMoveToNextIterResponse notify_leader_to_next_iter_rsp;
  notify_leader_to_next_iter_rsp.set_result("success");
  if (!communicator_->SendResponse(notify_leader_to_next_iter_rsp.SerializeAsString().data(),
                                   notify_leader_to_next_iter_rsp.SerializeAsString().size(), message)) {
    MS_LOG(ERROR) << "Sending response failed.";
    return;
  }

  NotifyLeaderMoveToNextIterRequest notify_leader_to_next_iter_req;
  (void)notify_leader_to_next_iter_req.ParseFromArray(message->data(), SizeToInt(message->len()));
  const auto &rank = notify_leader_to_next_iter_req.rank();
  const auto &is_last_iter_valid = notify_leader_to_next_iter_req.is_last_iter_valid();
  const auto &iter_num = notify_leader_to_next_iter_req.iter_num();
  const auto &reason = notify_leader_to_next_iter_req.reason();
  MS_LOG(INFO) << "Leader server receives NotifyLeaderMoveToNextIterRequest from rank " << rank
               << ". Iteration number: " << iter_num << ". Reason: " << reason;

  if (IsMoveToNextIterRequestReentrant(iter_num)) {
    return;
  }

  if (!BroadcastPrepareForNextIterRequest(is_last_iter_valid, reason)) {
    MS_LOG(ERROR) << "Broadcast prepare for next iteration request failed.";
    return;
  }
  if (!BroadcastMoveToNextIterRequest(is_last_iter_valid, reason)) {
    MS_LOG(ERROR) << "Broadcast proceed to next iteration request failed.";
    return;
  }
  if (!BroadcastEndLastIterRequest(iteration_num_)) {
    MS_LOG(ERROR) << "Broadcast end last iteration request failed.";
    return;
  }
}

bool Iteration::BroadcastPrepareForNextIterRequest(bool is_last_iter_valid, const std::string &reason) {
  MS_ERROR_IF_NULL_W_RET_VAL(communicator_, false);
  PrepareForNextIter();
  MS_LOG(INFO) << "Notify all follower servers to prepare for next iteration.";
  PrepareForNextIterRequest prepare_next_iter_req;
  prepare_next_iter_req.set_is_last_iter_valid(is_last_iter_valid);
  prepare_next_iter_req.set_reason(reason);

  std::vector<uint32_t> offline_servers = {};
  for (uint32_t i = 1; i < IntToUint(server_node_->server_num()); i++) {
    if (!communicator_->SendPbRequest(prepare_next_iter_req, i, ps::core::TcpUserCommand::kPrepareForNextIter)) {
      MS_LOG(WARNING) << "Sending prepare for next iteration request to server " << i << " failed. Retry later.";
      offline_servers.push_back(i);
      continue;
    }
  }

  // Retry sending to offline servers to notify them to prepare.
  (void)std::for_each(offline_servers.begin(), offline_servers.end(), [&](uint32_t rank) {
    // Should avoid endless loop if the server communicator is stopped.
    while (communicator_->running() &&
           !communicator_->SendPbRequest(prepare_next_iter_req, rank, ps::core::TcpUserCommand::kPrepareForNextIter)) {
      MS_LOG(WARNING) << "Retry sending prepare for next iteration request to server " << rank
                      << " failed. The server has not recovered yet.";
      std::this_thread::sleep_for(std::chrono::milliseconds(kRetryDurationForPrepareForNextIter));
    }
    MS_LOG(INFO) << "Offline server " << rank << " preparing for next iteration success.";
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(kServerSleepTimeForNetworking));
  return true;
}

void Iteration::HandlePrepareForNextIterRequest(const std::shared_ptr<ps::core::MessageHandler> &message) {
  MS_ERROR_IF_NULL_WO_RET_VAL(message);
  MS_ERROR_IF_NULL_WO_RET_VAL(communicator_);
  PrepareForNextIterRequest prepare_next_iter_req;
  (void)prepare_next_iter_req.ParseFromArray(message->data(), SizeToInt(message->len()));
  const auto &reason = prepare_next_iter_req.reason();
  MS_LOG(INFO) << "Prepare next iteration for this rank " << server_node_->rank_id() << ", reason: " << reason;
  PrepareForNextIter();

  PrepareForNextIterResponse prepare_next_iter_rsp;
  prepare_next_iter_rsp.set_result("success");
  if (!communicator_->SendResponse(prepare_next_iter_rsp.SerializeAsString().data(),
                                   prepare_next_iter_rsp.SerializeAsString().size(), message)) {
    MS_LOG(ERROR) << "Sending response failed.";
    return;
  }
}

void Iteration::PrepareForNextIter() {
  MS_LOG(INFO) << "Prepare for next iteration. Switch the server to safemode.";
  Server::GetInstance().SwitchToSafeMode();
}

bool Iteration::BroadcastMoveToNextIterRequest(bool is_last_iter_valid, const std::string &reason) {
  MS_ERROR_IF_NULL_W_RET_VAL(communicator_, false);
  MS_LOG(INFO) << "Notify all follower servers to proceed to next iteration. Set last iteration number "
               << iteration_num_;
  MoveToNextIterRequest proceed_to_next_iter_req;
  proceed_to_next_iter_req.set_is_last_iter_valid(is_last_iter_valid);
  proceed_to_next_iter_req.set_last_iter_num(iteration_num_);
  proceed_to_next_iter_req.set_reason(reason);
  for (uint32_t i = 1; i < IntToUint(server_node_->server_num()); i++) {
    if (!communicator_->SendPbRequest(proceed_to_next_iter_req, i, ps::core::TcpUserCommand::kProceedToNextIter)) {
      MS_LOG(WARNING) << "Sending proceed to next iteration request to server " << i << " failed.";
      continue;
    }
  }

  Next(is_last_iter_valid, reason);
  return true;
}

void Iteration::HandleMoveToNextIterRequest(const std::shared_ptr<ps::core::MessageHandler> &message) {
  MS_ERROR_IF_NULL_WO_RET_VAL(message);
  MS_ERROR_IF_NULL_WO_RET_VAL(communicator_);
  MoveToNextIterResponse proceed_to_next_iter_rsp;
  proceed_to_next_iter_rsp.set_result("success");
  if (!communicator_->SendResponse(proceed_to_next_iter_rsp.SerializeAsString().data(),
                                   proceed_to_next_iter_rsp.SerializeAsString().size(), message)) {
    MS_LOG(ERROR) << "Sending response failed.";
    return;
  }

  MoveToNextIterRequest proceed_to_next_iter_req;
  (void)proceed_to_next_iter_req.ParseFromArray(message->data(), SizeToInt(message->len()));
  const auto &is_last_iter_valid = proceed_to_next_iter_req.is_last_iter_valid();
  const auto &last_iter_num = proceed_to_next_iter_req.last_iter_num();
  const auto &reason = proceed_to_next_iter_req.reason();

  MS_LOG(INFO) << "Receive proceeding to next iteration request. This server current iteration is " << iteration_num_
               << ". The iteration number from leader server is " << last_iter_num
               << ". Last iteration is valid or not: " << is_last_iter_valid << ". Reason: " << reason;
  // Synchronize the iteration number with leader server.
  iteration_num_ = last_iter_num;
  Next(is_last_iter_valid, reason);
}

void Iteration::Next(bool is_iteration_valid, const std::string &reason) {
  MS_LOG(INFO) << "Prepare for next iteration.";
  is_last_iteration_valid_ = is_iteration_valid;
  if (is_iteration_valid) {
    // Store the model which is successfully aggregated for this iteration.
    const auto &model = Executor::GetInstance().GetModel();
    ModelStore::GetInstance().StoreModelByIterNum(iteration_num_, model);
    MS_LOG(INFO) << "Iteration " << iteration_num_ << " is successfully finished.";
  } else {
    // Store last iteration's model because this iteration is considered as invalid.
    const auto &model = ModelStore::GetInstance().GetModelByIterNum(iteration_num_ - 1);
    ModelStore::GetInstance().StoreModelByIterNum(iteration_num_, model);
    MS_LOG(WARNING) << "Iteration " << iteration_num_ << " is invalid. Reason: " << reason;
  }

  for (auto &round : rounds_) {
    MS_ERROR_IF_NULL_WO_RET_VAL(round);
    round->Reset();
  }
}

bool Iteration::BroadcastEndLastIterRequest(uint64_t last_iter_num) {
  MS_ERROR_IF_NULL_W_RET_VAL(communicator_, false);
  MS_LOG(INFO) << "Notify all follower servers to end last iteration.";
  EndLastIterRequest end_last_iter_req;
  end_last_iter_req.set_last_iter_num(last_iter_num);
  for (uint32_t i = 1; i < IntToUint(server_node_->server_num()); i++) {
    if (!communicator_->SendPbRequest(end_last_iter_req, i, ps::core::TcpUserCommand::kEndLastIter)) {
      MS_LOG(WARNING) << "Sending ending last iteration request to server " << i << " failed.";
      continue;
    }
  }

  EndLastIter();
  return true;
}

void Iteration::HandleEndLastIterRequest(const std::shared_ptr<ps::core::MessageHandler> &message) {
  MS_ERROR_IF_NULL_WO_RET_VAL(message);
  MS_ERROR_IF_NULL_WO_RET_VAL(communicator_);
  EndLastIterRequest end_last_iter_req;
  (void)end_last_iter_req.ParseFromArray(message->data(), SizeToInt(message->len()));
  const auto &last_iter_num = end_last_iter_req.last_iter_num();
  // If the iteration number is not matched, return error.
  if (last_iter_num != iteration_num_) {
    std::string reason = "The iteration of this server " + std::to_string(server_node_->rank_id()) + " is " +
                         std::to_string(iteration_num_) + ", iteration to be ended is " + std::to_string(last_iter_num);
    EndLastIterResponse end_last_iter_rsp;
    end_last_iter_rsp.set_result(reason);
    if (!communicator_->SendResponse(end_last_iter_rsp.SerializeAsString().data(),
                                     end_last_iter_rsp.SerializeAsString().size(), message)) {
      MS_LOG(ERROR) << "Sending response failed.";
      return;
    }
    return;
  }

  EndLastIter();

  EndLastIterResponse end_last_iter_rsp;
  end_last_iter_rsp.set_result("success");
  if (!communicator_->SendResponse(end_last_iter_rsp.SerializeAsString().data(),
                                   end_last_iter_rsp.SerializeAsString().size(), message)) {
    MS_LOG(ERROR) << "Sending response failed.";
    return;
  }
}

void Iteration::EndLastIter() {
  MS_LOG(INFO) << "End the last iteration " << iteration_num_;
  iteration_num_++;
  // After the job is done, reset the iteration to the initial number and reset ModelStore.
  if (iteration_num_ > ps::PSContext::instance()->fl_iteration_num()) {
    MS_LOG(INFO) << "Iteration loop " << iteration_loop_count_
                 << " is completed. Iteration number: " << ps::PSContext::instance()->fl_iteration_num();
    iteration_num_ = 1;
    iteration_loop_count_++;
    ModelStore::GetInstance().Reset();
  }

  std::unique_lock<std::mutex> lock(pinned_mtx_);
  pinned_iter_num_ = 0;
  lock.unlock();
  LocalMetaStore::GetInstance().set_curr_iter_num(iteration_num_);
  Server::GetInstance().CancelSafeMode();
  SetIterationCompleted();
  MS_LOG(INFO) << "Move to next iteration:" << iteration_num_ << "\n";
}
}  // namespace server
}  // namespace fl
}  // namespace mindspore
