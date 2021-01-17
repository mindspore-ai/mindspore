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

#include "src/train/train_loop.h"
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

using session::RET_CONTINUE;
using session::RET_EXIT;
using session::RET_STOP_TRAINING;

TrainLoop::~TrainLoop() {
  if (train_session_ != nullptr) delete train_session_;
}

int TrainLoop::Train(int epochs, std::vector<session::TrainLoopCallBack *> cbs) {
  train_session_->Train();
  session::TrainLoopCallBackData cb_data(true, epoch_, train_session_, this);

  for (auto cb : cbs) cb->Begin(cb_data);

  int steps_in_epoch = 1;  // should be data_size/batch_size
  for (int i = 0; i < epochs; i++) {
    cb_data.epoch_ = epoch_++;
    for (auto cb : cbs) cb->EpochBegin(cb_data);

    for (int s = 0; s < steps_in_epoch; s++) {
      cb_data.step_ = s;
      for (auto cb : cbs) cb->StepBegin(cb_data);
      train_session_->RunGraph(before_cb_, after_cb_);
      for (auto cb : cbs) cb->StepEnd(cb_data);
    }

    int break_loop = false;
    for (auto cb : cbs) {
      int ret = cb->EpochEnd(cb_data);
      if (ret != RET_CONTINUE) {
        if (ret == RET_EXIT) {
          MS_LOG(ERROR) << "Error in TrainLoop callback";
          return RET_ERROR;
        }
        if (ret == RET_STOP_TRAINING) {
          break_loop = true;
        }
      }
    }
    if (break_loop) {
      break;
    }
  }

  for (auto cb : cbs) cb->End(cb_data);
  return RET_OK;
}

}  // namespace lite

session::TrainLoop *session::TrainLoop::CreateTrainLoop(const std::string &model_filename, lite::Context *context,
                                                        int batch_size) {
  auto train_session = session::TrainSession::CreateSession(model_filename, context);
  auto loop = new (std::nothrow) lite::TrainLoop(train_session);

  return loop;
}

}  // namespace mindspore
