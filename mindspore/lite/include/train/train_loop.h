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
#include <climits>
#include <unordered_map>
#include "include/train/train_loop_callback.h"
#include "include/train/metrics.h"
#include "include/train/train_session.h"

namespace mindspore {
class MSTensor;

namespace dataset {
class Dataset;
using MSTensorVec = std::vector<mindspore::MSTensor>;
}  // namespace dataset

using LoadDataFunc = std::function<int(std::vector<tensor::MSTensor *> inputs, dataset::MSTensorVec *dataset_vec)>;

namespace session {

class TrainLoop {
 public:
  /// \brief Static method to create a TrainLoop object
  ///
  /// \param[in] train_session Train session object as return from CreateSession\CreateTransferSession API
  ///
  /// \return Pointer of MindSpore Lite TrainLoop
  static TrainLoop *CreateTrainLoop(session::TrainSession *train_session);

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

  /// \brief Initialize object with metrics
  ///
  /// \param[in] verctor of metrics
  ///
  /// \return 0 on success or -1 in case of error
  virtual int Init(std::vector<mindspore::session::Metrics *> metrics) = 0;

  /// \brief Accessor to TrainLoop metric objects
  ///
  /// \return vector of metrics
  virtual std::vector<mindspore::session::Metrics *> GetMetrics() = 0;

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
  /// \param[in] dataset Pointer to MindData Dataset object
  /// \param[in] cbs A vector of TrainLoopCallBack objects
  /// \param[in] load_func a function that load (and can manipulate) data from Minddata Dataset array into model
  ///
  /// \return 0 on success or -1 in case of error
  virtual int Train(int epochs, mindspore::dataset::Dataset *dataset, std::vector<TrainLoopCallBack *> cbs,
                    LoadDataFunc load_func = nullptr) = 0;

  /// \brief Performs loop over all data in Eval Mode
  ///
  /// \param[in] dataset Pointer to MindData Dataset object
  /// \param[in] cbs A vector of TrainLoopCallBack objects
  /// \param[in] load_func a function that load (and can manipulate) data from Minddata Dataset array into model
  /// \param[in] max_steps (with default = INT_MAX the method iterates all dataset)
  ///
  /// \return 0 on success or -1 in case of error
  virtual int Eval(mindspore::dataset::Dataset *dataset, std::vector<TrainLoopCallBack *> cbs,
                   LoadDataFunc load_func = nullptr, int max_steps = INT_MAX) = 0;
};
}  // namespace session
}  // namespace mindspore
#endif  // MINDSPORE_LITE_INCLUDE_TRAIN_TRAIN_LOOP_H_
