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

#ifndef MINDSPORE_CCSRC_FL_SERVER_ROUND_H_
#define MINDSPORE_CCSRC_FL_SERVER_ROUND_H_

#include <memory>
#include <string>
#include "ps/core/communicator/communicator_base.h"
#include "fl/server/common.h"
#include "fl/server/iteration_timer.h"
#include "fl/server/distributed_count_service.h"
#include "fl/server/kernel/round/round_kernel.h"

namespace mindspore {
namespace fl {
namespace server {
// Round helps server to handle network round messages and launch round kernels. One iteration in server consists of
// multiple rounds like startFLJob, updateModel, Push, Pull, etc. Some round kernels may be stateful because of counting
// and timing. So Round helps register counter and timer so that the round kernels only need to focus on the logic.
class Round {
 public:
  explicit Round(const std::string &name, bool check_timeout = true, size_t time_window = 3000,
                 bool check_count = false, size_t threshold_count = 8, bool server_num_as_threshold = false);
  ~Round() = default;

  void Initialize(const std::shared_ptr<ps::core::CommunicatorBase> &communicator, const TimeOutCb &timeout_cb,
                  const FinishIterCb &finish_iteration_cb);

  // Reinitialize count service and round kernel of this round after scaling operations are done.
  bool ReInitForScaling(uint32_t server_num);

  // After hyper-parameters are updated, some rounds and kernels should be reinitialized.
  bool ReInitForUpdatingHyperParams(size_t updated_threshold_count, size_t updated_time_window);

  // Bind a round kernel to this Round. This method should be called after Initialize.
  void BindRoundKernel(const std::shared_ptr<kernel::RoundKernel> &kernel);

  // This method is the callback which will be set to the communicator and called after the corresponding round message
  // is sent to the server.
  void LaunchRoundKernel(const std::shared_ptr<ps::core::MessageHandler> &message);

  // Round needs to be reset after each iteration is finished or its timer expires.
  void Reset();

  const std::string &name() const;
  size_t threshold_count() const;
  bool check_timeout() const;
  size_t time_window() const;

 private:
  // The callbacks which will be set to DistributedCounterService.
  void OnFirstCountEvent(const std::shared_ptr<ps::core::MessageHandler> &message);
  void OnLastCountEvent(const std::shared_ptr<ps::core::MessageHandler> &message);

  // Judge whether the training service is available.
  bool IsServerAvailable(std::string *reason);

  std::string name_;

  // Whether this round needs to use timer. Most rounds in federated learning with mobile devices scenario need to set
  // check_timeout_ to true.
  bool check_timeout_;

  // The time window duration for this round when check_timeout_ is set to true.
  size_t time_window_;

  // If check_count_ is true, it means the round has to do counting for every round message and the first/last count
  // event will be triggered.
  bool check_count_;

  // The threshold count for this round when check_count_ is set to true. The logic of this round has to check whether
  // the round message count has reached threshold_count_.
  size_t threshold_count_;

  // Whether this round uses the server number as its threshold count.
  bool server_num_as_threshold_;

  std::shared_ptr<ps::core::CommunicatorBase> communicator_;

  // The round kernel for this Round.
  std::shared_ptr<kernel::RoundKernel> kernel_;

  // Some rounds may need timer to eliminate the long tail effect.
  std::shared_ptr<IterationTimer> iter_timer_;

  // The callbacks which will be set to the round kernel.
  StopTimerCb stop_timer_cb_;
  FinishIterCb finish_iteration_cb_;
  FinalizeCb finalize_cb_;
};
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_ROUND_H_
