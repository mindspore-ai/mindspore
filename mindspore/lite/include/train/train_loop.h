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
#ifndef MINDSPORE_LITE_INCLUDE_TRAIN_TRAIN_LOOP_H_
#define MINDSPORE_LITE_INCLUDE_TRAIN_TRAIN_LOOP_H_
#include <vector>
#include <string>
#include <tuple>
#include <unordered_map>
#include "include/train/train_loop_callback.h"
#include "include/train_session.h"

namespace mindspore {
namespace session {

class TrainLoop {
 public:
  /// \brief Static method to create a TrainLoop object
  ///
  /// \param[in] filename Filename to read flatbuffer from
  /// \param[in] context Defines the context of the session to be created
  ///
  /// \return Pointer of MindSpore Lite TrainLoop
  static TrainLoop *CreateTrainLoop(const std::string &model_filename, lite::Context *context, int batch_size = -1);

  /// \brief Class destructor
  virtual ~TrainLoop() = default;

  /// \brief Resets the epoch counter
  ///
  /// \return 0 on success or -1 in case of error
  virtual int Reset() = 0;  // resets the epoch counter to 0.

  /// \brief Accessor to the TrainSession
  ///
  /// \return pointer of the train_session
  virtual session::TrainSession *train_session() = 0;

  /// \brief Accessor to the Session KernelCallbacks
  ///
  /// \param[in] before Define a call_back_function to be called before running each node.
  /// \param[in] after Define a call_back_function called after running each node.
  ///
  /// \return 0 on success or -1 in case of error
  virtual int SetKernelCallBack(const KernelCallBack &before, const KernelCallBack &after) = 0;

  /// \brief Performs the training Loop
  ///
  /// \param[in] epoch The number of epochs to run
  /// \param[in] cbs A vector of TrainLoopCallBack objects
  ///
  /// \return 0 on success or -1 in case of error
  virtual int Train(int epochs, std::vector<TrainLoopCallBack *> cbs) = 0;
};
}  // namespace session
}  // namespace mindspore
#endif  // MINDSPORE_LITE_INCLUDE_TRAIN_TRAIN_LOOP_H_
