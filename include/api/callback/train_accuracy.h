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
#ifndef MINDSPORE_INCLUDE_API_CALLBACK_TRAIN_ACCURACY_H
#define MINDSPORE_INCLUDE_API_CALLBACK_TRAIN_ACCURACY_H

#include <cstddef>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include "include/api/callback/callback.h"
#include "include/api/metrics/accuracy.h"

namespace mindspore {

class MS_API TrainAccuracy : public TrainCallBack {
 public:
  explicit TrainAccuracy(int print_every_n = INT_MAX, int accuracy_metrics = METRICS_CLASSIFICATION,
                         const std::vector<int> &input_indexes = {1}, const std::vector<int> &output_indexes = {0});
  virtual ~TrainAccuracy();
  const std::vector<GraphPoint> &GetAccuracyPoints();
};
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_CALLBACK_TRAIN_ACCURACY_H
