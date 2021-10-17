/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include <cstddef>
#include <string>
#include <vector>
#include <memory>
#include "include/train/loss_monitor.h"
#include "include/api/callback/loss_monitor.h"
#include "src/cxx_api/callback/callback_impl.h"
#include "src/common/log_adapter.h"

namespace mindspore {
LossMonitor::LossMonitor(int print_every_n_steps) {
  callback_impl_ = new (std::nothrow) CallbackImpl(new (std::nothrow) lite::LossMonitor(print_every_n_steps));
  if (callback_impl_ == nullptr) {
    MS_LOG(ERROR) << "Callback implement new failed";
  }
}

LossMonitor::~LossMonitor() {
  if (callback_impl_ == nullptr) {
    MS_LOG(ERROR) << "Callback implement is null.";
    return;
  }
  auto internal_call_back = callback_impl_->GetInternalCallback();
  if (internal_call_back != nullptr) {
    delete internal_call_back;
  }
  delete callback_impl_;
  callback_impl_ = nullptr;
}

const std::vector<GraphPoint> &LossMonitor::GetLossPoints() {
  static std::vector<GraphPoint> empty_vector;
  if (callback_impl_ == nullptr) {
    MS_LOG(ERROR) << "Callback implement is null.";
    return empty_vector;
  }

  session::TrainLoopCallBack *internal_call_back = callback_impl_->GetInternalCallback();
  if (internal_call_back == nullptr) {
    MS_LOG(ERROR) << "Internal callback is null.";
    return empty_vector;
  }

  return (reinterpret_cast<lite::LossMonitor *>(internal_call_back))->GetLossPoints();
}
}  // namespace mindspore
