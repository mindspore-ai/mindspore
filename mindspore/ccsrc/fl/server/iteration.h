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

#ifndef MINDSPORE_CCSRC_FL_SERVER_ITERATION_H_
#define MINDSPORE_CCSRC_FL_SERVER_ITERATION_H_

#include <memory>
#include <vector>
#include <string>
#include "ps/core/communicator/communicator_base.h"
#include "fl/server/common.h"
#include "fl/server/round.h"
#include "fl/server/local_meta_store.h"

namespace mindspore {
namespace fl {
namespace server {
enum class IterationState {
  // This iteration is still in process.
  kRunning,
  // This iteration is completed and the next iteration is not started yet.
  kCompleted
};

// The time duration between retrying when sending prepare for next iteration request failed.
constexpr uint32_t kRetryDurationForPrepareForNextIter = 500;

// In server's logic, Iteration is the minimum execution unit. For each execution, it consists of multiple kinds of
// Rounds, only after all the rounds are finished, this iteration is considered as completed.
class Iteration {
 public:
  static Iteration &GetInstance() {
    static Iteration instance;
    return instance;
  }

  // Register callbacks for other servers to synchronize iteration information from leader server.
  void RegisterMessageCallback(const std::shared_ptr<ps::core::TcpCommunicator> &communicator);

  // Register event callbacks for iteration state synchronization.
  void RegisterEventCallback(const std::shared_ptr<ps::core::ServerNode> &server_node);

  // Add a round for the iteration. This method will be called multiple times for each round.
  void AddRound(const std::shared_ptr<Round> &round);

  // Initialize all the rounds in the iteration.
  void InitRounds(const std::vector<std::shared_ptr<ps::core::CommunicatorBase>> &communicators,
                  const TimeOutCb &timeout_cb, const FinishIterCb &finish_iteration_cb);

  // This method will control servers to proceed to next iteration.
  // There's communication between leader and follower servers in this method.
  // The server moves to next iteration only after the last round finishes or the time expires.
  void MoveToNextIteration(bool is_last_iter_valid, const std::string &reason);

  // Set current iteration state to running and trigger events about kIterationRunning.
  void SetIterationRunning();

  // Set current iteration state to completed and trigger the event about kIterationCompleted.
  void SetIterationCompleted();

  // The barrier function for elastic scaling. The scaling out/in operation should be done only after this iteration is
  // completed.
  void ScalingBarrier();

  // Reinitialize rounds after scaling operations are done.
  // The server number after scaling is required in some rounds.
  bool ReInitForScaling(uint32_t server_num, uint32_t server_rank);

  const std::vector<std::shared_ptr<Round>> &rounds() const;

  bool is_last_iteration_valid() const;

 private:
  Iteration()
      : server_node_(nullptr),
        communicator_(nullptr),
        iteration_state_(IterationState::kCompleted),
        iteration_loop_count_(0),
        iteration_num_(1),
        is_last_iteration_valid_(true),
        pinned_iter_num_(0) {
    LocalMetaStore::GetInstance().set_curr_iter_num(iteration_num_);
  }
  ~Iteration() = default;
  Iteration(const Iteration &) = delete;
  Iteration &operator=(const Iteration &) = delete;

  // The server does not need to handle the iteration events for now.
  void HandleIterationRunningEvent() {}
  void HandleIterationCompletedEvent() {}

  // Synchronize iteration form the leader server(Rank 0).
  bool SyncIteration(uint32_t rank);
  void HandleSyncIterationRequest(const std::shared_ptr<ps::core::MessageHandler> &message);

  // The request for moving to next iteration is not reentrant.
  bool IsMoveToNextIterRequestReentrant(uint64_t iteration_num);

  // The methods for moving to next iteration for all the servers.
  // Step 1: follower servers notify leader server that they need to move to next iteration.
  bool NotifyLeaderMoveToNextIteration(bool is_last_iter_valid, const std::string &reason);
  void HandleNotifyLeaderMoveToNextIterRequest(const std::shared_ptr<ps::core::MessageHandler> &message);

  // Step 2: leader server broadcast to all follower servers to prepare for next iteration and switch to safemode.
  bool BroadcastPrepareForNextIterRequest(bool is_last_iter_valid, const std::string &reason);
  void HandlePrepareForNextIterRequest(const std::shared_ptr<ps::core::MessageHandler> &message);
  // The server prepare for the next iteration. This method will switch the server to safemode.
  void PrepareForNextIter();

  // Step 3: leader server broadcast to all follower servers to move to next iteration.
  bool BroadcastMoveToNextIterRequest(bool is_last_iter_valid, const std::string &reason);
  void HandleMoveToNextIterRequest(const std::shared_ptr<ps::core::MessageHandler> &message);
  // Move to next iteration. Store last iterations model and reset all the rounds.
  void Next(bool is_iteration_valid, const std::string &reason);

  // Step 4: leader server broadcasts to all follower servers to end last iteration and cancel the safemode.
  bool BroadcastEndLastIterRequest(uint64_t iteration_num);
  void HandleEndLastIterRequest(const std::shared_ptr<ps::core::MessageHandler> &message);
  // The server end the last iteration. This method will increase the iteration number and cancel the safemode.
  void EndLastIter();

  std::shared_ptr<ps::core::ServerNode> server_node_;
  std::shared_ptr<ps::core::TcpCommunicator> communicator_;

  // All the rounds in the server.
  std::vector<std::shared_ptr<Round>> rounds_;

  // The iteration is either running or completed at any time.
  std::atomic<IterationState> iteration_state_;

  // The count of iteration loops which are completed.
  size_t iteration_loop_count_;

  // Server's current iteration number.
  size_t iteration_num_;

  // Last iteration is successfully finished.
  bool is_last_iteration_valid_;

  // To avoid Next method is called multiple times in one iteration, we should mark the iteration number.
  uint64_t pinned_iter_num_;
  std::mutex pinned_mtx_;
};
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_ITERATION_H_
