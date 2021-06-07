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

#ifndef MINDSPORE_CCSRC_PS_SERVER_ITERATION_H_
#define MINDSPORE_CCSRC_PS_SERVER_ITERATION_H_

#include <memory>
#include <vector>
#include "ps/core/communicator/communicator_base.h"
#include "ps/server/common.h"
#include "ps/server/round.h"
#include "ps/server/local_meta_store.h"

namespace mindspore {
namespace ps {
namespace server {
enum class IterationState {
  // This iteration is still in process.
  kRunning,
  // This iteration is completed and the next iteration is not started yet.
  kCompleted
};

// In server's logic, Iteration is the minimum execution unit. For each execution, it consists of multiple kinds of
// Rounds, only after all the rounds are finished, this iteration is considered as completed.
class Iteration {
 public:
  static Iteration &GetInstance() {
    static Iteration instance;
    return instance;
  }

  // Register callbacks for other servers to synchronize iteration information from leader server.
  void RegisterMessageCallback(const std::shared_ptr<core::TcpCommunicator> &communicator);

  // Register event callbacks for iteration state synchronization.
  void RegisterEventCallback(const std::shared_ptr<core::ServerNode> &server_node);

  // Add a round for the iteration. This method will be called multiple times for each round.
  void AddRound(const std::shared_ptr<Round> &round);

  // Initialize all the rounds in the iteration.
  void InitRounds(const std::vector<std::shared_ptr<core::CommunicatorBase>> &communicators,
                  const TimeOutCb &timeout_cb, const FinishIterCb &finish_iteration_cb);

  // The server proceeds to the next iteration only after the last round finishes or the timer expires.
  // If the timer expires, we consider this iteration as invalid.
  void ProceedToNextIter(bool is_iteration_valid);

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

  const std::vector<std::shared_ptr<Round>> &rounds();

  bool is_last_iteration_valid() const;

 private:
  Iteration()
      : server_node_(nullptr),
        communicator_(nullptr),
        iteration_state_(IterationState::kCompleted),
        iteration_num_(1),
        is_last_iteration_valid_(true) {
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
  void HandleSyncIterationRequest(const std::shared_ptr<core::MessageHandler> &message);

  std::shared_ptr<core::ServerNode> server_node_;
  std::shared_ptr<core::TcpCommunicator> communicator_;

  // All the rounds in the server.
  std::vector<std::shared_ptr<Round>> rounds_;

  // The iteration is either running or completed at any time.
  std::atomic<IterationState> iteration_state_;

  // Server's current iteration number.
  size_t iteration_num_;

  // Last iteration is successfully finished.
  bool is_last_iteration_valid_;
};
}  // namespace server
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_SERVER_ITERATION_H_
