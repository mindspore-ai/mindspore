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
#ifndef MINDSPORE_INCLUDE_API_METRICS_ACCURACY_H
#define MINDSPORE_INCLUDE_API_METRICS_ACCURACY_H
#include <vector>
#include "include/api/metrics/metrics.h"

namespace mindspore {

constexpr int METRICS_CLASSIFICATION = 0;
constexpr int METRICS_MULTILABEL = 1;

class MS_API AccuracyMetrics : public Metrics {
 public:
  explicit AccuracyMetrics(int accuracy_metrics = METRICS_CLASSIFICATION, const std::vector<int> &input_indexes = {1},
                           const std::vector<int> &output_indexes = {0});
  virtual ~AccuracyMetrics();
  void Clear() override;
  float Eval() override;
};

}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_METRICS_ACCURACY_H
