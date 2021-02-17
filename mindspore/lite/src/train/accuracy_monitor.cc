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

#include "include/train/accuracy_monitor.h"
#include <sys/stat.h>
#include <algorithm>
#include <utility>
#include <vector>
#include <iostream>
#include <fstream>
#include <memory>
#include "include/errorcode.h"
#include "include/train/train_loop.h"
#include "src/common/utils.h"
#include "src/tensor.h"

namespace mindspore {
namespace lite {

void AccuracyMonitor::Begin(const session::TrainLoopCallBackData &cb_data) {
  if (cb_data.epoch_ == 0) accuracies_.clear();
}

int AccuracyMonitor::EpochEnd(const session::TrainLoopCallBackData &cb_data) {
  if ((cb_data.epoch_ + 1) % check_every_n_ == 0) cb_data.loop_->Eval(ds_, {}, nullptr, max_steps_);

  accuracies_.push_back(std::make_pair(cb_data.epoch_, 0.0));
  return mindspore::session::RET_CONTINUE;
}

}  // namespace lite
}  // namespace mindspore
