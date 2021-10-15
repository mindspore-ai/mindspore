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
#include "include/train/lr_scheduler.h"
#include "include/api/callback/lr_scheduler.h"
#include "src/cxx_api/callback/callback_impl.h"
#include "src/common/log_adapter.h"

namespace mindspore {
int StepLRLambda(float *lr, int epoch, void *lr_cb_data) {
  if ((lr == nullptr) || (lr_cb_data == nullptr)) {
    MS_LOG(ERROR) << "nullptr passed as input to MultiplicativeLRLambda";
    return DONT_UPDATE_LR;
  }
  struct StepLRLambda *step_lr_data = (static_cast<struct StepLRLambda *>(lr_cb_data));
  if (step_lr_data->step_size <= 0) {
    MS_LOG(ERROR) << "lr data step size need bigger than 0.";
    return DONT_UPDATE_LR;
  }
  if (((epoch + 1) % step_lr_data->step_size) == 0) {
    *lr = *lr * step_lr_data->gamma;
    return UPDATE_LR;
  }
  return DONT_UPDATE_LR;
}

LRScheduler::LRScheduler(LR_Lambda lambda_func, void *lr_cb_data, int step) {
  callback_impl_ = new (std::nothrow) CallbackImpl(new (std::nothrow) lite::LRScheduler(lambda_func, lr_cb_data, step));
  if (callback_impl_ == nullptr) {
    MS_LOG(ERROR) << "callback implement new failed";
  }
}

LRScheduler::~LRScheduler() {
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
}  // namespace mindspore
