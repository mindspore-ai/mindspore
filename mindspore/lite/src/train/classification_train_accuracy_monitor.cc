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

#include "include/train/classification_train_accuracy_monitor.h"
#include <sys/stat.h>
#include <vector>
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "include/train/train_session.h"
#include "src/common/utils.h"
#include "src/train/train_utils.h"

using mindspore::WARNING;

namespace mindspore {
namespace lite {

ClassificationTrainAccuracyMonitor::ClassificationTrainAccuracyMonitor(int print_every_n, int accuracy_metrics,
                                                                       const std::vector<int> &input_indexes,
                                                                       const std::vector<int> &output_indexes) {
  if (input_indexes.size() == output_indexes.size()) {
    input_indexes_ = input_indexes;
    output_indexes_ = output_indexes;
  } else {
    MS_LOG(WARNING) << "input to output mapping vectors sizes do not match";
  }
  if (accuracy_metrics != METRICS_CLASSIFICATION) {
    MS_LOG(WARNING) << "Only classification metrics is supported";
  } else {
    accuracy_metrics_ = accuracy_metrics;
  }
  print_every_n_ = print_every_n;
}

void ClassificationTrainAccuracyMonitor::Begin(const session::TrainLoopCallBackData &cb_data) {
  if (cb_data.epoch_ == 0) accuracies_.clear();
}

void ClassificationTrainAccuracyMonitor::EpochBegin(const session::TrainLoopCallBackData &cb_data) {
  if (accuracies_.size() != cb_data.epoch_) {
    MS_LOG(WARNING) << "Accuracies array does not match epoch number";
  } else {
    accuracies_.push_back(std::make_pair(cb_data.epoch_, 0.0));
  }
}

int ClassificationTrainAccuracyMonitor::EpochEnd(const session::TrainLoopCallBackData &cb_data) {
  if (cb_data.step_ > 0) accuracies_.at(cb_data.epoch_).second /= static_cast<float>(cb_data.step_ + 1);
  if ((cb_data.epoch_ + 1) % print_every_n_ == 0) {
    std::cout << "Epoch (" << cb_data.epoch_ + 1 << "):\tTraining Accuracy is " << accuracies_.at(cb_data.epoch_).second
              << std::endl;
  }
  return mindspore::session::RET_CONTINUE;
}

void ClassificationTrainAccuracyMonitor::StepEnd(const session::TrainLoopCallBackData &cb_data) {
  auto inputs = cb_data.session_->GetInputs();
  auto outputs = cb_data.session_->GetPredictions();

  float accuracy = 0.0;
  for (unsigned int i = 0; i < input_indexes_.size(); i++) {
    if ((inputs.size() <= static_cast<unsigned int>(input_indexes_[i])) ||
        (outputs.size() <= static_cast<unsigned int>(output_indexes_[i]))) {
      MS_LOG(WARNING) << "indices " << input_indexes_[i] << "/" << output_indexes_[i]
                      << " is outside of input/output range";
      return;
    }
    if (inputs.at(input_indexes_[i])->data_type() == kNumberTypeInt32) {
      accuracy += CalculateSparseClassification(inputs.at(input_indexes_[i]), outputs.at(output_indexes_[i]));
    } else {
      accuracy += CalculateOneHotClassification(inputs.at(input_indexes_[i]), outputs.at(output_indexes_[i]));
    }
  }
  accuracies_.at(cb_data.epoch_).second += accuracy;
}

}  // namespace lite
}  // namespace mindspore
