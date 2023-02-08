/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_INCLUDE_API_CALLBACK_LR_SCHEDULER_H
#define MINDSPORE_INCLUDE_API_CALLBACK_LR_SCHEDULER_H

#include <cstddef>
#include <string>
#include <vector>
#include <memory>
#include "include/api/callback/callback.h"

namespace mindspore {

constexpr int DONT_UPDATE_LR = 0;
constexpr int UPDATE_LR = 1;

using LR_Lambda = std::function<int(float *lr, int epoch, void *cb_data)>;

/// \brief Multiply the LR by a factor of gamma every epoch
MS_API int MultiplicativeLRLambda(float *lr, int epoch, void *multiplication);

/// \brief Multiply the LR by a factor of gamma every step_size
MS_API int StepLRLambda(float *lr, int epoch, void *step_size);
struct MS_API StepLRLambda {
  StepLRLambda(int step, float g) : step_size(step), gamma(g) {}

  int step_size;  // period of LR decay
  float gamma;    // LR decay factor
};

class MS_API LRScheduler : public TrainCallBack {
 public:
  explicit LRScheduler(LR_Lambda lambda_func, void *lr_cb_data = nullptr, int step = 1);
  virtual ~LRScheduler();
};

}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_CALLBACK_LR_SCHEDULER_H
