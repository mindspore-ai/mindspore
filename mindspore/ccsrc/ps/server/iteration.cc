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

#include "ps/server/iteration.h"
#include <memory>
#include <vector>
#include <numeric>
#include "ps/server/model_store.h"

namespace mindspore {
namespace ps {
namespace server {
void Iteration::AddRound(const std::shared_ptr<Round> &round) {
  MS_EXCEPTION_IF_NULL(round);
  rounds_.push_back(round);
}

void Iteration::InitRounds(const std::vector<std::shared_ptr<core::CommunicatorBase>> &communicators,
                           const TimeOutCb &timeout_cb, const FinishIterCb &finish_iteration_cb) {
  if (communicators.empty()) {
    MS_LOG(EXCEPTION) << "Communicators for rounds is empty.";
    return;
  }

  std::for_each(communicators.begin(), communicators.end(),
                [&](const std::shared_ptr<core::CommunicatorBase> &communicator) {
                  for (auto &round : rounds_) {
                    if (round == nullptr) {
                      continue;
                    }
                    round->Initialize(communicator, timeout_cb, finish_iteration_cb);
                  }
                });

  // The time window for one iteration, which will be used in some round kernels.
  size_t iteration_time_window =
    std::accumulate(rounds_.begin(), rounds_.end(), 0, [](size_t total, const std::shared_ptr<Round> &round) {
      return round->check_timeout() ? total + round->time_window() : total;
    });
  LocalMetaStore::GetInstance().put_value(kCtxTotalTimeoutDuration, iteration_time_window);
  MS_LOG(INFO) << "Time window for one iteration is " << iteration_time_window;
  return;
}

void Iteration::ProceedToNextIter(bool is_iteration_valid) {
  iteration_num_ = LocalMetaStore::GetInstance().curr_iter_num();
  if (is_iteration_valid) {
    // Store the model which is successfully aggregated for this iteration.
    const auto &model = Executor::GetInstance().GetModel();
    ModelStore::GetInstance().StoreModelByIterNum(iteration_num_, model);
    MS_LOG(INFO) << "Iteration " << iteration_num_ << " is successfully finished.";
  } else {
    // Store last iteration's model because this iteration is considered as invalid.
    const auto &model = ModelStore::GetInstance().GetModelByIterNum(iteration_num_ - 1);
    ModelStore::GetInstance().StoreModelByIterNum(iteration_num_, model);
    MS_LOG(WARNING) << "Iteration " << iteration_num_ << " is invalid.";
  }

  for (auto &round : rounds_) {
    round->Reset();
  }

  iteration_num_++;
  // After the job is done, reset the iteration to the initial number and reset ModelStore.
  if (iteration_num_ > PSContext::instance()->fl_iteration_num()) {
    MS_LOG(INFO) << PSContext::instance()->fl_iteration_num() << " iterations are completed.";
    iteration_num_ = 1;
    ModelStore::GetInstance().Reset();
  }

  is_last_iteration_valid_ = is_iteration_valid;
  iteration_state_ = IterationState::kEnd;
  LocalMetaStore::GetInstance().set_curr_iter_num(iteration_num_);
  MS_LOG(INFO) << "Proceed to next iteration:" << iteration_num_ << "\n";
}

void Iteration::SetIterationRunning() {
  MS_LOG(INFO) << "Iteration " << iteration_num_ << " start running.";
  iteration_state_ = IterationState::kRunning;
}

void Iteration::ScalingBarrier() {
  MS_LOG(INFO) << "Starting Iteration scaling barrier.";
  while (iteration_state_.load() != IterationState::kEnd) {
    std::this_thread::yield();
  }
  MS_LOG(INFO) << "Ending Iteration scaling barrier.";
}

bool Iteration::ReInitForScaling() {
  for (auto &round : rounds_) {
    if (!round->ReInitForScaling()) {
      MS_LOG(ERROR) << "Reinitializing round " << round->name() << " for scaling failed.";
      return false;
    }
  }
  return true;
}

const std::vector<std::shared_ptr<Round>> &Iteration::rounds() { return rounds_; }

bool Iteration::is_last_iteration_valid() const { return is_last_iteration_valid_; }
}  // namespace server
}  // namespace ps
}  // namespace mindspore
