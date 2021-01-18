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
#include <algorithm>
#include <utility>
#include <vector>
#include <iostream>
#include <fstream>
#include <memory>
#include "include/errorcode.h"
#include "include/train_session.h"
#include "src/common/utils.h"
#include "src/tensor.h"
#include "src/train/loss_kernel.h"
#include "src/train/optimizer_kernel.h"
#include "src/sub_graph_kernel.h"
#include "src/train/train_populate_parameter.h"
#include "src/runtime/runtime_api.h"
#include "src/executor.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/fp32_grad/convolution.h"

namespace mindspore {
namespace lite {

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
  if (cb_data.step_ > 0) accuracies_.at(cb_data.epoch_).second /= static_cast<float>(cb_data.step_);
  if ((cb_data.epoch_ + 1) % print_every_n_ == 0) {
    std::cout << cb_data.epoch_ + 1 << ":\tTraining Accuracy is " << accuracies_.at(cb_data.epoch_).second << std::endl;
  }
  return mindspore::session::RET_CONTINUE;
}

void ClassificationTrainAccuracyMonitor::StepEnd(const session::TrainLoopCallBackData &cb_data) {
  auto inputs = cb_data.session_->GetInputs();
  auto outputs = cb_data.session_->GetPredictions();
  auto labels = reinterpret_cast<float *>(inputs.at(1)->MutableData());
  for (auto it = outputs.begin(); it != outputs.end(); ++it) {
    if (it->second->ElementsNum() == inputs.at(1)->ElementsNum()) {
      int batch_size = inputs.at(1)->shape().at(0);
      int num_of_classes = inputs.at(1)->shape().at(1);
      auto predictions = reinterpret_cast<float *>(it->second->MutableData());
      float accuracy = 0.0;
      for (int b = 0; b < batch_size; b++) {
        int label = 0;
        int max_idx = 0;
        float max_label_score = labels[num_of_classes * b];
        float max_score = predictions[num_of_classes * b];
        for (int c = 1; c < num_of_classes; c++) {
          if (predictions[num_of_classes * b + c] > max_score) {
            max_score = predictions[num_of_classes * b + c];
            max_idx = c;
          }
          if (labels[num_of_classes * b + c] > max_label_score) {
            max_label_score = labels[num_of_classes * b + c];
            label = c;
          }
        }
        if (label == max_idx) accuracy += 1.0;
      }
      accuracy /= static_cast<float>(batch_size);
      accuracies_.at(cb_data.epoch_).second = accuracy;
      return;
    }
  }

  MS_LOG(WARNING) << "Model does not have a loss output tensor of size 1";
}

}  // namespace lite
}  // namespace mindspore
