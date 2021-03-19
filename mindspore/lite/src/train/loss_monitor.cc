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

#include "include/train/loss_monitor.h"
#include <sys/stat.h>
#include <algorithm>
#include <utility>
#include <vector>
#include <iostream>
#include "include/train/train_session.h"
#include "src/common/utils.h"
#include "src/tensor.h"

namespace mindspore {
namespace lite {

void LossMonitor::Begin(const session::TrainLoopCallBackData &cb_data) {
  if (cb_data.epoch_ == 0) losses_.clear();
}

void LossMonitor::EpochBegin(const session::TrainLoopCallBackData &cb_data) {
  if (losses_.size() != cb_data.epoch_) {
    MS_LOG(WARNING) << "losses array does not match epoch number";
  } else {
    losses_.push_back(std::make_pair(cb_data.epoch_, 0.0));
  }
}

int LossMonitor::EpochEnd(const session::TrainLoopCallBackData &cb_data) {
  if (cb_data.step_ > 0) losses_.at(cb_data.epoch_).second /= static_cast<float>(cb_data.step_ + 1);
  if (print_every_n_ > 0) {
    std::cout << "Epoch (" << cb_data.epoch_ + 1 << "):\tLoss is " << losses_.at(cb_data.epoch_).second << std::endl;
  }
  return mindspore::session::RET_CONTINUE;
}

void LossMonitor::StepEnd(const session::TrainLoopCallBackData &cb_data) {
  auto outputs = cb_data.session_->GetOutputs();
  for (auto it = outputs.begin(); it != outputs.end(); ++it) {
    if (it->second->ElementsNum() == 1) {
      auto loss = reinterpret_cast<float *>(it->second->MutableData());
      losses_.at(cb_data.epoch_).second += loss[0];
      if ((cb_data.step_ + 1) % print_every_n_ == 0)
        std::cout << cb_data.epoch_ + 1 << "." << cb_data.step_ + 1 << ":\tLoss is " << loss[0] << std::endl;
      return;
    }
  }
  MS_LOG(WARNING) << "Model does not have a loss output tensor of size 1";
}

}  // namespace lite
}  // namespace mindspore
