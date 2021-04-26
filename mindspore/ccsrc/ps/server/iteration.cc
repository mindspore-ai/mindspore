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
Iteration::Iteration() : iteration_num_(1) { LocalMetaStore::GetInstance().set_curr_iter_num(iteration_num_); }

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
    std::accumulate(rounds_.begin(), rounds_.end(), 0,
                    [](size_t total, const std::shared_ptr<Round> &round) { return total + round->time_window(); });
  LocalMetaStore::GetInstance().put_value(kCtxTotalTimeoutDuration, iteration_time_window);
  return;
}

void Iteration::ProceedToNextIter() {
  iteration_num_ = LocalMetaStore::GetInstance().curr_iter_num();
  // Store the model for each iteration.
  const auto &model = Executor::GetInstance().GetModel();
  ModelStore::GetInstance().StoreModelByIterNum(iteration_num_, model);

  for (auto &round : rounds_) {
    round->Reset();
  }

  iteration_num_++;
  LocalMetaStore::GetInstance().set_curr_iter_num(iteration_num_);
  MS_LOG(INFO) << "Proceed to next iteration:" << iteration_num_ << "\n";
}

const std::vector<std::shared_ptr<Round>> &Iteration::rounds() { return rounds_; }
}  // namespace server
}  // namespace ps
}  // namespace mindspore
