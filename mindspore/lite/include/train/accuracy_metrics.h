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
#ifndef MINDSPORE_LITE_INCLUDE_TRAIN_ACCURACY_METRICS_H_
#define MINDSPORE_LITE_INCLUDE_TRAIN_ACCURACY_METRICS_H_
#include <vector>
#include "include/train/metrics.h"

using mindspore::session::Metrics;

namespace mindspore {
namespace lite {

constexpr int METRICS_CLASSIFICATION = 0;
constexpr int METRICS_MULTILABEL = 1;

class AccuracyMetrics : public Metrics {
 public:
  explicit AccuracyMetrics(int accuracy_metrics = METRICS_CLASSIFICATION, const std::vector<int> &input_indexes = {1},
                           const std::vector<int> &output_indexes = {0});
  virtual ~AccuracyMetrics() = default;
  void Clear() override { total_accuracy_ = total_steps_ = 0.0; }
  float Eval() override;
  void Update(std::vector<tensor::MSTensor *> inputs, std::vector<tensor::MSTensor *> outputs) override;

 protected:
  int accuracy_metrics_ = METRICS_CLASSIFICATION;
  std::vector<int> input_indexes_ = {1};
  std::vector<int> output_indexes_ = {0};
  float total_accuracy_ = 0.0;
  float total_steps_ = 0.0;
};

}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_INCLUDE_TRAIN_ACCURACY_METRICS_H_
