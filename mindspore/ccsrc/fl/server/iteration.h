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
#include "fl/server/iteration_metrics.h"

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

class IterationMetrics;
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

  // Release all the round objects in Iteration instance. Used for reinitializing round and round kernels.
  void ClearRounds();

  // Notify move_to_next_thread_ to move to next iteration.
  void NotifyNext(bool is_last_iter_valid, const std::string &reason);

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

  // After hyper-parameters are updated, some rounds and kernels should be reinitialized.
  bool ReInitForUpdatingHyperParams(const std::vector<RoundConfig> &updated_rounds_config);

  const std::vector<std::shared_ptr<Round>> &rounds() const;

  bool is_last_iteration_valid() const;

  // Set the instance metrics which will be called for each iteration.
  void set_metrics(const std::shared_ptr<IterationMetrics> &metrics);
  void set_loss(float loss);
  void set_accuracy(float accuracy);

  // Return state of current training job instance.
  InstanceState instance_state() const;

  // Return whether current instance is being updated.
  bool IsInstanceBeingUpdated() const;

  // EnableFLS/disableFLS the current training instance.
  bool EnableServerInstance(std::string *result);
  bool DisableServerInstance(std::string *result);

  // Finish current instance and start a new one. FLPlan could be changed in this method.
  bool NewInstance(const nlohmann::json &new_instance_json, std::string *result);

  // Query information of current instance.
  bool QueryInstance(std::string *result);

  // Need to wait all the rounds to finish before proceed to next iteration.
  void WaitAllRoundsFinish() const;

  // The round kernels whose Launch method has not returned yet.
  std::atomic_uint32_t running_round_num_;

 private:
  Iteration()
      : running_round_num_(0),
        server_node_(nullptr),
        communicator_(nullptr),
        iteration_state_(IterationState::kCompleted),
        start_timestamp_(0),
        complete_timestamp_(0),
        iteration_loop_count_(0),
        iteration_num_(1),
        is_last_iteration_valid_(true),
        move_to_next_reason_(""),
        move_to_next_thread_running_(true),
        pinned_iter_num_(0),
        metrics_(nullptr),
        instance_state_(InstanceState::kRunning),
        is_instance_being_updated_(false),
        loss_(0.0),
        accuracy_(0.0),
        joined_client_num_(0),
        rejected_client_num_(0),
        time_cost_(0) {
    LocalMetaStore::GetInstance().set_curr_iter_num(iteration_num_);
  }
  ~Iteration();
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

  // Drop current iteration and move to the next immediately.
  bool ForciblyMoveToNextIteration();

  // Summarize metrics for the completed iteration, including iteration time cost, accuracy, loss, etc.
  bool SummarizeIteration();

  // Update server's hyper-parameters according to the given serialized json(hyper_params_data).
  bool UpdateHyperParams(const nlohmann::json &json);

  // Reinitialize rounds and round kernels.
  bool ReInitRounds();

  std::shared_ptr<ps::core::ServerNode> server_node_;
  std::shared_ptr<ps::core::TcpCommunicator> communicator_;

  // All the rounds in the server.
  std::vector<std::shared_ptr<Round>> rounds_;

  // The iteration is either running or completed at any time.
  std::mutex iteration_state_mtx_;
  std::condition_variable iteration_state_cv_;
  std::atomic<IterationState> iteration_state_;
  uint64_t start_timestamp_;
  uint64_t complete_timestamp_;

  // The count of iteration loops which are completed.
  size_t iteration_loop_count_;

  // Server's current iteration number.
  size_t iteration_num_;

  // Whether last iteration is successfully finished and the reason.
  bool is_last_iteration_valid_;
  std::string move_to_next_reason_;

  // It will be notified by rounds that the instance moves to the next iteration.
  std::thread move_to_next_thread_;
  std::atomic_bool move_to_next_thread_running_;
  std::mutex next_iteration_mutex_;
  std::condition_variable next_iteration_cv_;

  // To avoid Next method is called multiple times in one iteration, we should mark the iteration number.
  uint64_t pinned_iter_num_;
  std::mutex pinned_mtx_;

  std::shared_ptr<IterationMetrics> metrics_;

  // The state for current instance.
  std::atomic<InstanceState> instance_state_;

  // Every instance is not reentrant.
  // This flag represents whether the instance is being updated.
  std::mutex instance_mtx_;
  bool is_instance_being_updated_;

  // The training loss after this federated learning iteration, passed by worker.
  float loss_;

  // The evaluation result after this federated learning iteration, passed by worker.
  float accuracy_;

  // The number of clients which join the federated aggregation.
  size_t joined_client_num_;

  // The number of clients which are not involved in federated aggregation.
  size_t rejected_client_num_;

  // The time cost in millisecond for this completed iteration.
  uint64_t time_cost_;
};
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_ITERATION_H_
