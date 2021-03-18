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

#include "include/train/lr_scheduler.h"
#include <sys/stat.h>
#include <algorithm>
#include <utility>
#include <vector>
#include <iostream>
#include <fstream>
#include <memory>
#include "include/errorcode.h"
#include "include/train/train_session.h"
#include "src/common/utils.h"
#include "src/tensor.h"

namespace mindspore {
namespace lite {

int MultiplicativeLRLambda(float *lr, int epoch, void *lr_cb_data) {
  if ((lr == nullptr) || (lr_cb_data == nullptr)) {
    MS_LOG(ERROR) << "nullptr passed as input to MultiplicativeLRLambda";
    return DONT_UPDATE_LR;
  }
  float mult = *(static_cast<float *>(lr_cb_data));
  *lr = *lr * mult;
  return UPDATE_LR;
}

int StepLRLambda(float *lr, int epoch, void *lr_cb_data) {
  if ((lr == nullptr) || (lr_cb_data == nullptr)) {
    MS_LOG(ERROR) << "nullptr passed as input to MultiplicativeLRLambda";
    return DONT_UPDATE_LR;
  }
  struct StepLRLambda *step_lr_data = (static_cast<struct StepLRLambda *>(lr_cb_data));
  if (((epoch + 1) % step_lr_data->step_size) == 0) {
    *lr = *lr * step_lr_data->gamma;
    return UPDATE_LR;
  }
  return DONT_UPDATE_LR;
}

LRScheduler::LRScheduler(LR_Lambda lambda_func, void *lr_cb_data, int step)
    : lambda_func_(lambda_func), lr_data_(lr_cb_data), step_(step) {}

int LRScheduler::EpochEnd(const session::TrainLoopCallBackData &cb_data) {
  if (((cb_data.epoch_ + 1) % step_) == 0) {
    float lr = cb_data.session_->GetLearningRate();
    int update = lambda_func_(&lr, cb_data.epoch_, lr_data_);
    if (update == UPDATE_LR) {
      int ret = cb_data.session_->SetLearningRate(lr);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Error setting Leraning rate in train session";
        return mindspore::session::RET_EXIT;
      }
    }
  }
  return mindspore::session::RET_CONTINUE;
}

}  // namespace lite
}  // namespace mindspore
