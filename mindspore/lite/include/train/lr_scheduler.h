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
#ifndef MINDSPORE_LITE_INCLUDE_TRAIN_LR_SCHEDULER_H_
#define MINDSPORE_LITE_INCLUDE_TRAIN_LR_SCHEDULER_H_
#include <vector>
#include <string>
#include <utility>
#include <functional>
#include <unordered_map>
#include "include/train/train_loop_callback.h"

namespace mindspore {
namespace lite {

constexpr int DONT_UPDATE_LR = 0;
constexpr int UPDATE_LR = 1;

using LR_Lambda = std::function<int(float *lr, int epoch, void *cb_data)>;

/// \brief Multiply the LR by a factor of gamma every epoch
int MultiplicativeLRLambda(float *lr, int epoch, void *multiplication);

/// \brief Multiply the LR by a factor of gamma every step_size
int StepLRLambda(float *lr, int epoch, void *step_size);
struct StepLRLambda {
  StepLRLambda(int step, float g) : step_size(step), gamma(g) {}

  int step_size;  // period of LR decay
  float gamma;    // LR decay factor
};

class LRScheduler : public session::TrainLoopCallBack {
 public:
  explicit LRScheduler(LR_Lambda lambda_func, void *lr_cb_data = nullptr, int step_ = 1);
  virtual ~LRScheduler() = default;
  int EpochEnd(const session::TrainLoopCallBackData &cb_data) override;

 private:
  LR_Lambda lambda_func_;
  void *lr_data_ = nullptr;
  int step_ = 1;
};

}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_INCLUDE_TRAIN_LR_SCHEDULER_H_
