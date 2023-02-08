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
#ifndef MINDSPORE_INCLUDE_API_CALLBACK_LOSS_MONITOR_H
#define MINDSPORE_INCLUDE_API_CALLBACK_LOSS_MONITOR_H

#include <cstddef>
#include <vector>
#include <utility>
#include "include/api/callback/callback.h"

namespace mindspore {

class MS_API LossMonitor : public TrainCallBack {
 public:
  explicit LossMonitor(int print_every_n_steps = INT_MAX);
  virtual ~LossMonitor();
  const std::vector<GraphPoint> &GetLossPoints();
};
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_CALLBACK_LOSS_MONITOR_H
