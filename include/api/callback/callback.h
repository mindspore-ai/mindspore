/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_INCLUDE_API_CALLBACK_CALLBACK_H
#define MINDSPORE_INCLUDE_API_CALLBACK_CALLBACK_H

#include <cstddef>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include "include/api/data_type.h"
#include "include/api/dual_abi_helper.h"
#include "include/api/types.h"

namespace mindspore {
class Model;
class ModelImpl;
class CallbackImpl;

using GraphPoint = std::pair<int, float>;

struct MS_API TrainCallBackData {
  TrainCallBackData(bool train_mode, int epoch, int step, Model *model)
      : train_mode_(train_mode), epoch_(epoch), step_(step), model_(model) {}

  bool train_mode_;       /**< training mode of LiteSession object */
  unsigned int epoch_;    /**< the current training epoch (starts at 0) */
  unsigned int step_ = 0; /**< the current step within the epoch */
  Model *model_;          /**< pointer to the Model object */
};

enum CallbackRetValue : uint32_t { kContinue = 0, kStopTraining = 1, kExit = 2, kUnknownRetValue = 0xFFFFFFFF };

class MS_API TrainCallBack {
 public:
  virtual ~TrainCallBack() = default;

  /// \brief This method is called once before the network executing
  ///
  /// \param[in] cb_data info about current execution
  virtual void Begin(const TrainCallBackData &cb_data) {}

  /// \brief This method is called once following the network execution
  ///
  /// \param[in] cb_data info about current execution
  virtual void End(const TrainCallBackData &cb_data) {}

  /// \brief This method is called at the beginning of each epoch
  ///
  /// \param[in] cb_data info about current execution
  virtual void EpochBegin(const TrainCallBackData &cb_data) {}

  /// \brief This method is called after the run of each epoch
  ///
  /// \param[in] cb_data info about current execution
  ///
  /// \return indication if to continue in the train loop:
  ///         RET_CONTINUE -- continue training
  ///         RET_STOP_TRAINING -- stop training (e.g., due to achieved accuracy)
  ///         RET_EXIT -- Exit training (due to error of some sort)
  virtual CallbackRetValue EpochEnd(const TrainCallBackData &cb_data) { return kContinue; }

  /// \brief This method is called at the beginning of each step
  ///
  /// \param[in] cb_data info about current execution
  virtual void StepBegin(const TrainCallBackData &cb_data) {}

  /// \brief This method is called after each step is ran
  ///
  /// \param[in] cb_data info about current execution
  virtual void StepEnd(const TrainCallBackData &cb_data) {}

 protected:
  friend class Model;
  friend class ModelImpl;
  CallbackImpl *callback_impl_ = nullptr;
};

}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_CALLBACK_CALLBACK_H
