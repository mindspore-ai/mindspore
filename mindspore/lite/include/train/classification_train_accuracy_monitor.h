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
#ifndef MINDSPORE_LITE_INCLUDE_TRAIN_CLASSIFICATION_TRAIN_ACCURACY_MONITOR_H_
#define MINDSPORE_LITE_INCLUDE_TRAIN_CLASSIFICATION_TRAIN_ACCURACY_MONITOR_H_
#include <vector>
#include <string>
#include <utility>
#include <climits>
#include <unordered_map>
#include "include/train/train_loop.h"
#include "include/train/accuracy_metrics.h"

using GraphPoint = std::pair<int, float>;

namespace mindspore {
namespace lite {

class ClassificationTrainAccuracyMonitor : public session::TrainLoopCallBack {
 public:
  explicit ClassificationTrainAccuracyMonitor(int print_every_n = INT_MAX,
                                              int accuracy_metrics = METRICS_CLASSIFICATION,
                                              const std::vector<int> &input_indexes = {1},
                                              const std::vector<int> &output_indexes = {0});
  virtual ~ClassificationTrainAccuracyMonitor() = default;

  void Begin(const session::TrainLoopCallBackData &cb_data) override;
  void EpochBegin(const session::TrainLoopCallBackData &cb_data) override;
  int EpochEnd(const session::TrainLoopCallBackData &cb_data) override;
  void StepEnd(const session::TrainLoopCallBackData &cb_data) override;
  const std::vector<GraphPoint> &GetAccuracyPoints() const { return accuracies_; }

 private:
  std::vector<GraphPoint> accuracies_;
  int accuracy_metrics_ = METRICS_CLASSIFICATION;
  std::vector<int> input_indexes_ = {1};
  std::vector<int> output_indexes_ = {0};
  int print_every_n_ = 0;
};

}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_INCLUDE_TRAIN_CLASSIFICATION_TRAIN_ACCURACY_MONITOR_H_
