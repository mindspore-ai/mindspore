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
#include <vector>
#include <string>
#include <numeric>
#include <unordered_map>
#include "fl/server/model_store.h"
#include "fl/server/server.h"

namespace mindspore {
namespace fl {
namespace server {
class Server;

Iteration::~Iteration() {
  move_to_next_thread_running_ = false;
  next_iteration_cv_.notify_all();
  if (move_to_next_thread_.joinable()) {
    move_to_next_thread_.join();
  }
}

void Iteration::RegisterMessageCallback(const std::shared_ptr<ps::core::TcpCommunicator> &communicator) {
  MS_EXCEPTION_IF_NULL(communicator);
  communicator_ = communicator;
  MS_EXCEPTION_IF_NULL(communicator_);
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
  server_node->RegisterCustomEventCallback(static_cast<uint32_t>(ps::UserDefineEvent::kIterationRunning),
                                           std::bind(&Iteration::ProcessIterationRunningEvent, this));
  server_node->RegisterCustomEventCallback(static_cast<uint32_t>(ps::UserDefineEvent::kIterationCompleted),
                                           std::bind(&Iteration::ProcessIterationEndEvent, this));
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
  // The time window for one iteration, which will be used in some round kernels.
  size_t iteration_time_window = 0;
  for (auto &round : rounds_) {
    MS_EXCEPTION_IF_NULL(round);
    round->Initialize(timeout_cb, finish_iteration_cb);
    for (auto &communicator : communicators) {
      round->RegisterMsgCallBack(communicator);
    }
    if (round->check_timeout()) {
      iteration_time_window += round->time_window();
    }
  }
  LocalMetaStore::GetInstance().put_value(kCtxTotalTimeoutDuration, iteration_time_window);
  MS_LOG(INFO) << "Time window for one iteration is " << iteration_time_window;

  // Initialize the thread which will handle the signal from round kernels.
  move_to_next_thread_ = std::thread([this]() {
    while (move_to_next_thread_running_.load()) {
      std::unique_lock<std::mutex> lock(next_iteration_mutex_);
      next_iteration_cv_.wait(lock);
      if (!move_to_next_thread_running_.load()) {
        break;
      }
      lock.unlock();
      MoveToNextIteration(is_last_iteration_valid_, move_to_next_reason_);
    }
  });
  return;
}

void Iteration::ClearRounds() { rounds_.clear(); }

void Iteration::NotifyNext(bool is_last_iter_valid, const std::string &reason) {
  std::unique_lock<std::mutex> lock(next_iteration_mutex_);
  is_last_iteration_valid_ = is_last_iter_valid;
  move_to_next_reason_ = reason;
  next_iteration_cv_.notify_one();
}

void Iteration::MoveToNextIteration(bool is_last_iter_valid, const std::string &reason) {
  iteration_num_ = LocalMetaStore::GetInstance().curr_iter_num();
  MS_LOG(INFO) << "Notify cluster starts to proceed to next iteration. Iteration is " << iteration_num_
               << " validation is " << is_last_iter_valid << ". Reason: " << reason;
  if (IsMoveToNextIterRequestReentrant(iteration_num_)) {
    return;
  }

  MS_ERROR_IF_NULL_WO_RET_VAL(server_node_);
  if (server_node_->rank_id() == kLeaderServerRank) {
    if (!BroadcastPrepareForNextIterRequest(iteration_num_, is_last_iter_valid, reason)) {
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
    server_node_->BroadcastEvent(static_cast<uint32_t>(ps::UserDefineEvent::kIterationRunning));
    if (server_recovery_ != nullptr) {
      // Save data to the persistent storage in case the recovery happens at the beginning.
      if (!server_recovery_->Save(iteration_num_)) {
        MS_LOG(WARNING) << "Save recovery data failed.";
      }
    }
  }

  std::unique_lock<std::mutex> lock(iteration_state_mtx_);
  iteration_state_ = IterationState::kRunning;
  start_timestamp_ = LongToUlong(CURRENT_TIME_MILLI.count());
  MS_LOG(INFO) << "Iteratoin " << iteration_num_ << " start global timer.";
  global_iter_timer_->Start(std::chrono::milliseconds(global_iteration_time_window_));
}

void Iteration::SetIterationEnd() {
  MS_LOG(INFO) << "Iteration " << iteration_num_ << " ends.";
  MS_ERROR_IF_NULL_WO_RET_VAL(server_node_);
  if (server_node_->rank_id() == kLeaderServerRank) {
    // This event helps worker/server to be consistent in iteration state.
    server_node_->BroadcastEvent(static_cast<uint32_t>(ps::UserDefineEvent::kIterationCompleted));
  }

  std::unique_lock<std::mutex> lock(iteration_state_mtx_);
  iteration_state_ = IterationState::kCompleted;
  complete_timestamp_ = LongToUlong(CURRENT_TIME_MILLI.count());
}

void Iteration::ScalingBarrier() {
  MS_LOG(INFO) << "Starting Iteration scaling barrier.";
  std::unique_lock<std::mutex> lock(iteration_state_mtx_);
  if (iteration_state_.load() != IterationState::kCompleted) {
    iteration_state_cv_.wait(lock);
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

bool Iteration::ReInitForUpdatingHyperParams(const std::vector<RoundConfig> &updated_rounds_config) {
  for (const auto &updated_round : updated_rounds_config) {
    for (const auto &round : rounds_) {
      if (updated_round.name == round->name()) {
        MS_LOG(INFO) << "Reinitialize for round " << round->name();
        if (!round->ReInitForUpdatingHyperParams(updated_round.threshold_count, updated_round.time_window)) {
          MS_LOG(ERROR) << "Reinitializing for round " << round->name() << " failed.";
          return false;
        }
      }
    }
  }
  return true;
}

const std::vector<std::shared_ptr<Round>> &Iteration::rounds() const { return rounds_; }

bool Iteration::is_last_iteration_valid() const { return is_last_iteration_valid_; }

void Iteration::set_metrics(const std::shared_ptr<IterationMetrics> &metrics) { metrics_ = metrics; }

void Iteration::set_loss(float loss) { loss_ = loss; }

void Iteration::set_accuracy(float accuracy) { accuracy_ = accuracy; }

InstanceState Iteration::instance_state() const { return instance_state_.load(); }

bool Iteration::EnableServerInstance(std::string *result) {
  MS_ERROR_IF_NULL_W_RET_VAL(result, false);
  // Before enabling server instance, we should judge whether this request should be handled.
  std::unique_lock<std::mutex> lock(instance_mtx_);
  if (is_instance_being_updated_) {
    *result = "The instance is being updated. Please retry enabling server later.";
    MS_LOG(WARNING) << *result;
    return false;
  }
  if (instance_state_.load() == InstanceState::kFinish) {
    *result = "The instance is completed. Please do not enabling server now.";
    MS_LOG(WARNING) << *result;
    return false;
  }

  // Start enabling server instance.
  is_instance_being_updated_ = true;

  instance_state_ = InstanceState::kRunning;
  *result = "Enabling FL-Server succeeded.";
  MS_LOG(INFO) << *result;

  // End enabling server instance.
  is_instance_being_updated_ = false;
  return true;
}

bool Iteration::DisableServerInstance(std::string *result) {
  MS_ERROR_IF_NULL_W_RET_VAL(result, false);
  // Before disabling server instance, we should judge whether this request should be handled.
  std::unique_lock<std::mutex> lock(instance_mtx_);
  if (is_instance_being_updated_) {
    *result = "The instance is being updated. Please retry disabling server later.";
    MS_LOG(WARNING) << *result;
    return false;
  }
  if (instance_state_.load() == InstanceState::kFinish) {
    *result = "The instance is completed. Please do not disabling server now.";
    MS_LOG(WARNING) << *result;
    return false;
  }
  if (instance_state_.load() == InstanceState::kDisable) {
    *result = "Disabling FL-Server succeeded.";
    MS_LOG(INFO) << *result;
    return true;
  }

  // Start disabling server instance.
  is_instance_being_updated_ = true;

  // If instance is running, we should drop current iteration and move to the next.
  instance_state_ = InstanceState::kDisable;
  if (!ForciblyMoveToNextIteration()) {
    *result = "Disabling instance failed. Can't drop current iteration and move to the next.";
    MS_LOG(ERROR) << result;
    return false;
  }
  *result = "Disabling FL-Server succeeded.";
  MS_LOG(INFO) << *result;

  // End disabling server instance.
  is_instance_being_updated_ = false;
  return true;
}

void Iteration::StartNewInstance() {
  iteration_num_ = 1;
  LocalMetaStore::GetInstance().set_curr_iter_num(iteration_num_);
  is_instance_being_updated_ = false;
  ModelStore::GetInstance().Reset();

  // Update the hyper-parameters on server and reinitialize rounds.
  UpdateHyperParams(new_instance_json_);
  if (!ReInitRounds()) {
    MS_LOG(ERROR) << "Reinitializing rounds failed.";
  }

  instance_state_ = InstanceState::kRunning;
  MS_LOG(INFO) << "Process iteration new instance successful.";
}

bool Iteration::NewInstance(const nlohmann::json &new_instance_json, std::string *result) {
  MS_ERROR_IF_NULL_W_RET_VAL(result, false);
  // Before new instance, we should judge whether this request should be handled.
  std::unique_lock<std::mutex> lock(instance_mtx_);
  if (is_instance_being_updated_) {
    *result = "The instance is being updated. Please retry new instance later.";
    MS_LOG(WARNING) << *result;
    return false;
  }

  if (iteration_num_ == 1) {
    MS_LOG(INFO) << "This is just the first iteration.";
    return true;
  }

  // Start new server instance.
  is_instance_being_updated_ = true;

  new_instance_json_ = new_instance_json;
  *result = "New FL-Server instance succeeded.";

  if (instance_state_.load() == InstanceState::kFinish || instance_state_.load() == InstanceState::kDisable) {
    StartNewInstance();
  } else {
    MS_LOG(INFO) << "Process new instance success, cluster will start new job after this iteration end.";
  }
  return true;
}

void Iteration::WaitAllRoundsFinish() const {
  while (running_round_num_.load() != 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(kThreadSleepTime));
  }
}

void Iteration::set_recovery_handler(const std::shared_ptr<ServerRecovery> &server_recovery) {
  MS_EXCEPTION_IF_NULL(server_recovery);
  server_recovery_ = server_recovery;
}

bool Iteration::SyncAfterRecovery(uint64_t) {
  NotifyNext(false, "Move to next iteration after recovery.");
  return true;
}

bool Iteration::SyncIteration(uint32_t rank) {
  MS_ERROR_IF_NULL_W_RET_VAL(communicator_, false);
  SyncIterationRequest sync_iter_req;
  sync_iter_req.set_rank(rank);

  std::shared_ptr<std::vector<unsigned char>> sync_iter_rsp_msg = nullptr;
  if (!communicator_->SendPbRequest(sync_iter_req, kLeaderServerRank, ps::core::TcpUserCommand::kSyncIteration,
                                    &sync_iter_rsp_msg)) {
    MS_LOG(ERROR) << "Sending sync iter message to leader server failed.";
    return false;
  }

  MS_ERROR_IF_NULL_W_RET_VAL(sync_iter_rsp_msg, false);
  SyncIterationResponse sync_iter_rsp;
  (void)sync_iter_rsp.ParseFromArray(sync_iter_rsp_msg->data(), SizeToInt(sync_iter_rsp_msg->size()));
  iteration_num_ = sync_iter_rsp.iteration();
  MS_LOG(INFO) << "After synchronizing, server " << rank << " current iteration number is " << iteration_num_;
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
    MS_LOG(DEBUG) << "MoveToNextIteration is not reentrant. Ignore this call.";
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
  while (communicator_->running() && !communicator_->SendPbRequest(notify_leader_to_next_iter_req, kLeaderServerRank,
                                                                   ps::core::TcpUserCommand::kNotifyLeaderToNextIter)) {
    MS_LOG(WARNING) << "Sending notify leader server to proceed next iteration request to leader server 0 failed.";
    std::this_thread::sleep_for(std::chrono::milliseconds(kRetryDurationForPrepareForNextIter));
  }
  MS_LOG(INFO) << "Notify leader server to control the cluster to proceed to next iteration success";
  return true;
}

void Iteration::HandleNotifyLeaderMoveToNextIterRequest(const std::shared_ptr<ps::core::MessageHandler> &message) {
  MS_ERROR_IF_NULL_WO_RET_VAL(message);
  MS_ERROR_IF_NULL_WO_RET_VAL(communicator_);
  NotifyLeaderMoveToNextIterResponse notify_leader_to_next_iter_rsp;
  notify_leader_to_next_iter_rsp.set_result("success");
  if (!communicator_->SendResponse(notify_leader_to_next_iter_rsp.SerializeAsString().data(),
                                   notify_leader_to_next_iter_rsp.SerializeAsString().size(), message)) {
    MS_LOG(WARNING) << "Sending response failed.";
    return;
  }

  std::unique_lock<std::mutex> lock(iter_move_mtx_);
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

  if (!BroadcastPrepareForNextIterRequest(iter_num, is_last_iter_valid, reason)) {
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

bool Iteration::BroadcastPrepareForNextIterRequest(size_t last_iteration, bool is_last_iter_valid,
                                                   const std::string &reason) {
  MS_ERROR_IF_NULL_W_RET_VAL(communicator_, false);
  PrepareForNextIter(last_iteration, is_last_iter_valid);
  MS_LOG(INFO) << "Notify all follower servers to prepare for next iteration.";
  PrepareForNextIterRequest prepare_next_iter_req;
  prepare_next_iter_req.set_is_last_iter_valid(is_last_iter_valid);
  prepare_next_iter_req.set_reason(reason);
  prepare_next_iter_req.set_last_iteration(last_iteration);

  std::vector<uint32_t> offline_servers = {};
  for (uint32_t i = 1; i < server_node_->server_num(); i++) {
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
  auto is_last_iter_valid = prepare_next_iter_req.is_last_iter_valid();
  auto last_iteration = prepare_next_iter_req.last_iteration();
  MS_LOG(INFO) << "Prepare next iteration for this rank " << server_node_->rank_id()
               << ", last iteration: " << last_iteration << ", last iteration valid: " << is_last_iter_valid
               << ", reason: " << reason;
  PrepareForNextIter(last_iteration, is_last_iter_valid);

  PrepareForNextIterResponse prepare_next_iter_rsp;
  prepare_next_iter_rsp.set_result("success");
  if (!communicator_->SendResponse(prepare_next_iter_rsp.SerializeAsString().data(),
                                   prepare_next_iter_rsp.SerializeAsString().size(), message)) {
    MS_LOG(ERROR) << "Sending response failed.";
    return;
  }
}

void Iteration::PrepareForNextIter(size_t last_iteration, bool is_last_iter_valid) {
  MS_LOG(INFO) << "Prepare for next iteration. Switch the server to safemode.";
  Server::GetInstance().SwitchToSafeMode();
  if (server_node_) {
    server_node_->SetIterationResult(last_iteration, is_last_iter_valid);
  }
  MS_LOG(INFO) << "Start waiting for rounds to finish.";
  WaitAllRoundsFinish();
  MS_LOG(INFO) << "End waiting for rounds to finish.";
}

bool Iteration::BroadcastMoveToNextIterRequest(bool is_last_iter_valid, const std::string &reason) {
  MS_ERROR_IF_NULL_W_RET_VAL(communicator_, false);
  MS_LOG(INFO) << "Notify all follower servers to proceed to next iteration. Set last iteration number "
               << iteration_num_;
  MoveToNextIterRequest proceed_to_next_iter_req;
  proceed_to_next_iter_req.set_is_last_iter_valid(is_last_iter_valid);
  proceed_to_next_iter_req.set_last_iter_num(iteration_num_);
  proceed_to_next_iter_req.set_reason(reason);
  for (uint32_t i = 1; i < server_node_->server_num(); i++) {
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

  MoveToNextIterResponse proceed_to_next_iter_rsp;
  proceed_to_next_iter_rsp.set_result("success");
  if (!communicator_->SendResponse(proceed_to_next_iter_rsp.SerializeAsString().data(),
                                   proceed_to_next_iter_rsp.SerializeAsString().size(), message)) {
    MS_LOG(ERROR) << "Sending response failed.";
    return;
  }
}

void Iteration::Next(bool is_iteration_valid, const std::string &reason) {
  MS_LOG(INFO) << "Prepare for next iteration.";
  is_last_iteration_valid_ = is_iteration_valid;
  if (is_iteration_valid) {
    // Store the model which is successfully aggregated for this iteration.
    const auto &model = Executor::GetInstance().GetModel();
    std::unordered_map<std::string, size_t> feature_map;
    for (auto weight : model) {
      std::string weight_fullname = weight.first;
      if (weight.second == nullptr) {
        continue;
      }
      size_t weight_size = weight.second->size;
      feature_map[weight_fullname] = weight_size;
    }

    if (LocalMetaStore::GetInstance().verifyAggregationFeatureMap(feature_map)) {
      ModelStore::GetInstance().StoreModelByIterNum(iteration_num_, model);
      iteration_result_ = IterationResult::kSuccess;
      MS_LOG(INFO) << "Iteration " << iteration_num_ << " is successfully finished.";
    } else {
      MS_LOG(WARNING) << "Verify feature maps failed, iteration " << iteration_num_ << " will not be stored.";
    }
  } else {
    // Store last iteration's model because this iteration is considered as invalid.
    const auto &iter_to_model = ModelStore::GetInstance().iteration_to_model();
    size_t latest_iter_num = iter_to_model.rbegin()->first;
    const auto &model = ModelStore::GetInstance().GetModelByIterNum(latest_iter_num);
    ModelStore::GetInstance().StoreModelByIterNum(iteration_num_, model);
    iteration_result_ = IterationResult::kFail;
    MS_LOG(WARNING) << "Iteration " << iteration_num_ << " is invalid. Reason: " << reason;
  }

  for (auto &round : rounds_) {
    MS_ERROR_IF_NULL_WO_RET_VAL(round);
    round->Reset();
  }
  MS_LOG(INFO) << "Iteratoin " << iteration_num_ << " stop global timer.";
  global_iter_timer_->Stop();

  for (const auto &round : rounds_) {
    MS_ERROR_IF_NULL_WO_RET_VAL(round);
    round->KernelSummarize();
  }

  for (const auto &round : rounds_) {
    if (round->name() == "startFLJob") {
      round_client_num_map_[kStartFLJobTotalClientNum] += round->kernel_total_client_num();
      round_client_num_map_[kStartFLJobAcceptClientNum] += round->kernel_accept_client_num();
      round_client_num_map_[kStartFLJobRejectClientNum] += round->kernel_reject_client_num();
    } else if (round->name() == "updateModel") {
      round_client_num_map_[kUpdateModelTotalClientNum] += round->kernel_total_client_num();
      round_client_num_map_[kUpdateModelAcceptClientNum] += round->kernel_accept_client_num();
      round_client_num_map_[kUpdateModelRejectClientNum] += round->kernel_reject_client_num();
      set_loss(loss_ + round->kernel_upload_loss());
    } else if (round->name() == "getModel") {
      round_client_num_map_[kGetModelTotalClientNum] += round->kernel_total_client_num();
      round_client_num_map_[kGetModelAcceptClientNum] += round->kernel_accept_client_num();
      round_client_num_map_[kGetModelRejectClientNum] += round->kernel_reject_client_num();
    }
  }
}

bool Iteration::BroadcastEndLastIterRequest(uint64_t last_iter_num) {
  MS_ERROR_IF_NULL_W_RET_VAL(communicator_, false);
  MS_LOG(INFO) << "Notify all follower servers to end last iteration.";
  EndLastIterRequest end_last_iter_req;
  end_last_iter_req.set_last_iter_num(last_iter_num);
  for (uint32_t i = 1; i < server_node_->server_num(); i++) {
    std::shared_ptr<std::vector<unsigned char>> client_info_rsp_msg = nullptr;
    if (!communicator_->SendPbRequest(end_last_iter_req, i, ps::core::TcpUserCommand::kEndLastIter,
                                      &client_info_rsp_msg)) {
      MS_LOG(WARNING) << "Sending ending last iteration request to server " << i << " failed.";
      continue;
    }
    UpdateRoundClientNumMap(client_info_rsp_msg);
    UpdateRoundClientUploadLoss(client_info_rsp_msg);
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

  EndLastIterResponse end_last_iter_rsp;
  end_last_iter_rsp.set_result("success");

  for (const auto &round : rounds_) {
    if (round == nullptr) {
      continue;
    }
    if (round->name() == "startFLJob") {
      end_last_iter_rsp.set_startfljob_total_client_num(round->kernel_total_client_num());
      end_last_iter_rsp.set_startfljob_accept_client_num(round->kernel_accept_client_num());
      end_last_iter_rsp.set_startfljob_reject_client_num(round->kernel_reject_client_num());
    } else if (round->name() == "updateModel") {
      end_last_iter_rsp.set_updatemodel_total_client_num(round->kernel_total_client_num());
      end_last_iter_rsp.set_updatemodel_accept_client_num(round->kernel_accept_client_num());
      end_last_iter_rsp.set_updatemodel_reject_client_num(round->kernel_reject_client_num());
      end_last_iter_rsp.set_upload_loss(round->kernel_upload_loss());
    } else if (round->name() == "getModel") {
      end_last_iter_rsp.set_getmodel_total_client_num(round->kernel_total_client_num());
      end_last_iter_rsp.set_getmodel_accept_client_num(round->kernel_accept_client_num());
      end_last_iter_rsp.set_getmodel_reject_client_num(round->kernel_reject_client_num());
    }
  }

  EndLastIter();
  if (!communicator_->SendResponse(end_last_iter_rsp.SerializeAsString().data(),
                                   end_last_iter_rsp.SerializeAsString().size(), message)) {
    MS_LOG(ERROR) << "Sending response failed.";
    return;
  }
}

void Iteration::EndLastIter() {
  MS_LOG(INFO) << "End the last iteration " << iteration_num_;
  if (iteration_num_ == ps::PSContext::instance()->fl_iteration_num()) {
    MS_LOG(INFO) << "Iteration loop " << iteration_loop_count_
                 << " is completed. Iteration number: " << ps::PSContext::instance()->fl_iteration_num();
    iteration_loop_count_++;
    instance_state_ = InstanceState::kFinish;
  }

  std::unique_lock<std::mutex> lock(pinned_mtx_);
  pinned_iter_num_ = 0;
  lock.unlock();

  SetIterationEnd();
  if (!SummarizeIteration()) {
    MS_LOG(WARNING) << "Summarizing iteration data failed.";
  }

  if (is_instance_being_updated_) {
    StartNewInstance();
  } else {
    iteration_num_++;
  }

  LocalMetaStore::GetInstance().set_curr_iter_num(iteration_num_);

  MS_ERROR_IF_NULL_WO_RET_VAL(server_node_);
  if (server_node_->rank_id() == kLeaderServerRank) {
    // Save current iteration number for recovery.
    MS_ERROR_IF_NULL_WO_RET_VAL(server_recovery_);
    if (!server_recovery_->Save(iteration_num_)) {
      MS_LOG(WARNING) << "Can't save current iteration number into persistent storage.";
    }
  }
  for (const auto &round : rounds_) {
    MS_ERROR_IF_NULL_WO_RET_VAL(round);
    round->InitkernelClientVisitedNum();
    round->InitkernelClientUploadLoss();
  }
  round_client_num_map_.clear();
  set_loss(0.0f);
  Server::GetInstance().CancelSafeMode();
  iteration_state_cv_.notify_all();
  MS_LOG(INFO) << "Move to next iteration:" << iteration_num_ << "\n";
}

bool Iteration::ForciblyMoveToNextIteration() {
  NotifyNext(false, "Forcibly move to next iteration.");
  return true;
}

bool Iteration::SummarizeIteration() {
  // If the metrics_ is not initialized or the server is not the leader server, do not summarize.
  if (server_node_->rank_id() != kLeaderServerRank || metrics_ == nullptr) {
    MS_LOG(INFO) << "This server will not summarize for iteration.";
    return true;
  }

  metrics_->set_fl_name(ps::PSContext::instance()->fl_name());
  metrics_->set_fl_iteration_num(ps::PSContext::instance()->fl_iteration_num());
  metrics_->set_cur_iteration_num(iteration_num_);
  metrics_->set_instance_state(instance_state_.load());
  uint64_t update_model_threshold =
    ps::PSContext::instance()->start_fl_job_threshold() * ps::PSContext::instance()->update_model_ratio();
  if (update_model_threshold > 0) {
    metrics_->set_loss(loss_ / update_model_threshold);
  }
  metrics_->set_accuracy(accuracy_);
  metrics_->set_round_client_num_map(round_client_num_map_);
  metrics_->set_iteration_result(iteration_result_.load());

  if (complete_timestamp_ < start_timestamp_) {
    MS_LOG(ERROR) << "The complete_timestamp_: " << complete_timestamp_ << ", start_timestamp_: " << start_timestamp_
                  << ". One of them is invalid.";
    metrics_->set_iteration_time_cost(UINT64_MAX);
  } else {
    metrics_->set_iteration_time_cost(complete_timestamp_ - start_timestamp_);
  }

  if (!metrics_->Summarize()) {
    MS_LOG(ERROR) << "Summarizing metrics failed.";
    return false;
  }
  return true;
}

bool Iteration::UpdateHyperParams(const nlohmann::json &json) {
  for (const auto &item : json.items()) {
    std::string key = item.key();
    if (key == "start_fl_job_threshold") {
      ps::PSContext::instance()->set_start_fl_job_threshold(item.value().get<uint64_t>());
      continue;
    }
    if (key == "start_fl_job_time_window") {
      ps::PSContext::instance()->set_start_fl_job_time_window(item.value().get<uint64_t>());
      continue;
    }
    if (key == "update_model_ratio") {
      ps::PSContext::instance()->set_update_model_ratio(item.value().get<float>());
      continue;
    }
    if (key == "update_model_time_window") {
      ps::PSContext::instance()->set_update_model_time_window(item.value().get<uint64_t>());
      continue;
    }
    if (key == "fl_iteration_num") {
      ps::PSContext::instance()->set_fl_iteration_num(item.value().get<uint64_t>());
      continue;
    }
    if (key == "client_epoch_num") {
      ps::PSContext::instance()->set_client_epoch_num(item.value().get<uint64_t>());
      continue;
    }
    if (key == "client_batch_size") {
      ps::PSContext::instance()->set_client_batch_size(item.value().get<uint64_t>());
      continue;
    }
    if (key == "client_learning_rate") {
      ps::PSContext::instance()->set_client_learning_rate(item.value().get<float>());
      continue;
    }
    if (key == "global_iteration_time_window") {
      ps::PSContext::instance()->set_global_iteration_time_window(item.value().get<uint64_t>());
      continue;
    }
  }

  MS_LOG(INFO) << "start_fl_job_threshold: " << ps::PSContext::instance()->start_fl_job_threshold();
  MS_LOG(INFO) << "start_fl_job_time_window: " << ps::PSContext::instance()->start_fl_job_time_window();
  MS_LOG(INFO) << "update_model_ratio: " << ps::PSContext::instance()->update_model_ratio();
  MS_LOG(INFO) << "update_model_time_window: " << ps::PSContext::instance()->update_model_time_window();
  MS_LOG(INFO) << "fl_iteration_num: " << ps::PSContext::instance()->fl_iteration_num();
  MS_LOG(INFO) << "client_epoch_num: " << ps::PSContext::instance()->client_epoch_num();
  MS_LOG(INFO) << "client_batch_size: " << ps::PSContext::instance()->client_batch_size();
  MS_LOG(INFO) << "client_learning_rate: " << ps::PSContext::instance()->client_learning_rate();
  MS_LOG(INFO) << "global_iteration_time_window: " << ps::PSContext::instance()->global_iteration_time_window();
  return true;
}

bool Iteration::ReInitRounds() {
  size_t start_fl_job_threshold = ps::PSContext::instance()->start_fl_job_threshold();
  float update_model_ratio = ps::PSContext::instance()->update_model_ratio();
  size_t update_model_threshold = static_cast<size_t>(std::ceil(start_fl_job_threshold * update_model_ratio));
  uint64_t start_fl_job_time_window = ps::PSContext::instance()->start_fl_job_time_window();
  uint64_t update_model_time_window = ps::PSContext::instance()->update_model_time_window();
  std::vector<RoundConfig> new_round_config = {
    {"startFLJob", true, start_fl_job_time_window, true, start_fl_job_threshold},
    {"updateModel", true, update_model_time_window, true, update_model_threshold}};
  if (!ReInitForUpdatingHyperParams(new_round_config)) {
    MS_LOG(ERROR) << "Reinitializing for updating hyper-parameters failed.";
    return false;
  }

  size_t executor_threshold = 0;
  const std::string &server_mode = ps::PSContext::instance()->server_mode();
  uint32_t worker_num = ps::PSContext::instance()->initial_worker_num();
  if (server_mode == ps::kServerModeFL || server_mode == ps::kServerModeHybrid) {
    executor_threshold = update_model_threshold;
  } else if (server_mode == ps::kServerModePS) {
    executor_threshold = worker_num;
  } else {
    MS_LOG(ERROR) << "Server mode " << server_mode << " is not supported.";
    return false;
  }
  if (!Executor::GetInstance().ReInitForUpdatingHyperParams(executor_threshold)) {
    MS_LOG(ERROR) << "Reinitializing executor failed.";
    return false;
  }
  return true;
}

void Iteration::InitGlobalIterTimer(const TimeOutCb &timeout_cb) {
  global_iteration_time_window_ = ps::PSContext::instance()->global_iteration_time_window();
  global_iter_timer_ = std::make_shared<IterationTimer>();

  MS_LOG(INFO) << "Global iteration time window is: " << global_iteration_time_window_;
  // Set the timeout callback for the timer.
  global_iter_timer_->SetTimeOutCallBack([this, timeout_cb](bool, const std::string &) -> void {
    std::string reason = "Global Iteration " + std::to_string(iteration_num_) +
                         " timeout! This iteration is invalid. Proceed to next iteration.";
    timeout_cb(false, reason);
  });
}

void Iteration::UpdateRoundClientNumMap(const std::shared_ptr<std::vector<unsigned char>> &client_info_rsp_msg) {
  MS_ERROR_IF_NULL_WO_RET_VAL(client_info_rsp_msg);
  EndLastIterResponse end_last_iter_rsp;
  (void)end_last_iter_rsp.ParseFromArray(client_info_rsp_msg->data(), SizeToInt(client_info_rsp_msg->size()));

  round_client_num_map_[kStartFLJobTotalClientNum] += end_last_iter_rsp.startfljob_total_client_num();
  round_client_num_map_[kStartFLJobAcceptClientNum] += end_last_iter_rsp.startfljob_accept_client_num();
  round_client_num_map_[kStartFLJobRejectClientNum] += end_last_iter_rsp.startfljob_reject_client_num();

  round_client_num_map_[kUpdateModelTotalClientNum] += end_last_iter_rsp.updatemodel_total_client_num();
  round_client_num_map_[kUpdateModelAcceptClientNum] += end_last_iter_rsp.updatemodel_accept_client_num();
  round_client_num_map_[kUpdateModelRejectClientNum] += end_last_iter_rsp.updatemodel_reject_client_num();

  round_client_num_map_[kGetModelTotalClientNum] += end_last_iter_rsp.getmodel_total_client_num();
  round_client_num_map_[kGetModelAcceptClientNum] += end_last_iter_rsp.getmodel_accept_client_num();
  round_client_num_map_[kGetModelRejectClientNum] += end_last_iter_rsp.getmodel_reject_client_num();
}

void Iteration::UpdateRoundClientUploadLoss(const std::shared_ptr<std::vector<unsigned char>> &client_info_rsp_msg) {
  MS_ERROR_IF_NULL_WO_RET_VAL(client_info_rsp_msg);
  EndLastIterResponse end_last_iter_rsp;
  (void)end_last_iter_rsp.ParseFromArray(client_info_rsp_msg->data(), SizeToInt(client_info_rsp_msg->size()));

  set_loss(loss_ + end_last_iter_rsp.upload_loss());
}
}  // namespace server
}  // namespace fl
}  // namespace mindspore
