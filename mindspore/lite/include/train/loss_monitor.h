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
#ifndef MINDSPORE_LITE_INCLUDE_TRAIN_LOSS_MONITOR_H_
#define MINDSPORE_LITE_INCLUDE_TRAIN_LOSS_MONITOR_H_
#include <vector>
#include <string>
#include <utility>
#include <climits>
#include <unordered_map>
#include "include/train/train_loop_callback.h"

using GraphPoint = std::pair<int, float>;

namespace mindspore {
namespace lite {

class LossMonitor : public session::TrainLoopCallBack {
 public:
  /// \brief constructor
  ///
  /// \param[in] print_every_n_steps prints loss into stdout every n_steps.
  //             print_every_n_steps=0 means never print
  //             print_every_n_steps=INT_MAX will print every epoch
  /// \param[in] dataset Pointer to MindData Dataset object
  /// \param[in] cbs A vector of TrainLoopCallBack objects
  ///
  /// \return 0 on success or -1 in case of error
  explicit LossMonitor(int print_every_n_steps = INT_MAX) : print_every_n_(print_every_n_steps) {}
  virtual ~LossMonitor() = default;
  void Begin(const session::TrainLoopCallBackData &cb_data) override;
  void EpochBegin(const session::TrainLoopCallBackData &cb_data) override;
  int EpochEnd(const session::TrainLoopCallBackData &cb_data) override;
  void StepEnd(const session::TrainLoopCallBackData &cb_data) override;
  const std::vector<GraphPoint> &GetLossPoints() const { return losses_; }

 private:
  std::vector<GraphPoint> losses_;
  int print_every_n_;
};

}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_INCLUDE_TRAIN_LOSS_MONITOR_H_
