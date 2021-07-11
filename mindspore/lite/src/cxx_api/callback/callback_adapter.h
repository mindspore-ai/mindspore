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

#ifndef MINDSPORE_LITE_SRC_CXX_API_CALLBACK_CALLBACK_ADAPTER_H_
#define MINDSPORE_LITE_SRC_CXX_API_CALLBACK_CALLBACK_ADAPTER_H_

#include <functional>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <unordered_map>
#include "include/api/model.h"
#include "include/api/context.h"
#include "include/api/cell.h"
#include "include/lite_session.h"
#include "include/train/train_loop_callback.h"

namespace mindspore {

class TrainLoopCallBackAdapter : public session::TrainLoopCallBack {
 public:
  explicit TrainLoopCallBackAdapter(Model *model, TrainCallBack *call_back) : model_(model), call_back_(call_back) {}
  TrainLoopCallBackAdapter() = delete;

  void Begin(const session::TrainLoopCallBackData &i_cb_data) override {
    call_back_->Begin(TrainCallBackData(i_cb_data.train_mode_, i_cb_data.epoch_, i_cb_data.step_, model_));
  };

  void End(const session::TrainLoopCallBackData &i_cb_data) override {
    call_back_->End(TrainCallBackData(i_cb_data.train_mode_, i_cb_data.epoch_, i_cb_data.step_, model_));
  };

  void EpochBegin(const session::TrainLoopCallBackData &i_cb_data) override {
    call_back_->EpochBegin(TrainCallBackData(i_cb_data.train_mode_, i_cb_data.epoch_, i_cb_data.step_, model_));
  };

  int EpochEnd(const session::TrainLoopCallBackData &i_cb_data) override {
    return call_back_->EpochEnd(TrainCallBackData(i_cb_data.train_mode_, i_cb_data.epoch_, i_cb_data.step_, model_));
  };

  void StepBegin(const session::TrainLoopCallBackData &i_cb_data) override {
    call_back_->StepBegin(TrainCallBackData(i_cb_data.train_mode_, i_cb_data.epoch_, i_cb_data.step_, model_));
  };

  void StepEnd(const session::TrainLoopCallBackData &i_cb_data) override {
    call_back_->StepEnd(TrainCallBackData(i_cb_data.train_mode_, i_cb_data.epoch_, i_cb_data.step_, model_));
  };

 private:
  Model *model_;
  TrainCallBack *call_back_;
};
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_CXX_API_CALLBACK_CALLBACK_ADAPTER_H_
