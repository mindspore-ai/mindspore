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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_DS_CALLBACK_H
#define MINDSPORE_CCSRC_MINDDATA_DATASET_DS_CALLBACK_H

#include <memory>
#include <utility>
#include <vector>

#include "minddata/dataset/callback/callback_param.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

class DSCallback {
 public:
  /// \brief constructor of DSCallback, this is the base class for all front end specific callbacks
  /// \param step_size number of steps to call DSNStepBegin()
  explicit DSCallback(int32_t step_size = 1) : step_size_(step_size) {}

  /// \brief Destructor
  virtual ~DSCallback() = default;

  /// \brief actual callback function for begin, needs to be overridden in the derived class
  /// \param cb_param, callback parameter passed in from DatasetOp when calling the callback
  /// \return Status
  virtual Status DSBegin(const CallbackParam &cb_param) = 0;

  /// \brief actual callback function for epoch_begin, needs to be overridden in the derived class
  /// \param cb_param, callback parameter passed in from DatasetOp when calling the callback
  /// \return Status
  virtual Status DSEpochBegin(const CallbackParam &cb_param) = 0;

  /// \brief actual callback function for step_begin, needs to be overridden in the derived class
  /// \param cb_param, callback parameter passed in from DatasetOp when calling the callback
  /// \return Status
  virtual Status DSNStepBegin(const CallbackParam &cb_param) = 0;

  /// \brief actual callback function for end, needs to be overridden in the derived class
  /// \param cb_param, callback parameter passed in from DatasetOp when calling the callback
  /// \return Status
  virtual Status DSEnd(const CallbackParam &cb_param) = 0;

  /// \brief actual callback function epoch_end begin, needs to be overridden in the derived class
  /// \param cb_param, callback parameter passed in from DatasetOp when calling the callback
  /// \return Status
  virtual Status DSEpochEnd(const CallbackParam &cb_param) = 0;

  /// \brief actual callback function for step_end, needs to be overridden in the derived class
  /// \param cb_param, callback parameter passed in from DatasetOp when calling the callback
  /// \return Status
  virtual Status DSNStepEnd(const CallbackParam &cb_param) = 0;

  /// \brief predicate function, whether begin callback is needed
  /// \return bool
  virtual bool IsBeginNeeded() = 0;

  /// \brief predicate function, whether epoch_begin callback is needed
  /// \return bool
  virtual bool IsEpochBeginNeeded() = 0;

  /// \brief predicate function, whether step_begin callback is needed
  /// \return bool
  virtual bool IsNStepBeginNeeded() = 0;

  /// \brief predicate function, whether end callback is needed
  /// \return bool
  virtual bool IsEndNeeded() = 0;

  /// \brief predicate function, whether epoch_end callback is needed
  /// \return bool
  virtual bool IsEpochEndNeeded() = 0;

  /// \brief predicate function, whether step_end callback is needed
  /// \return bool
  virtual bool IsNStepEndNeeded() = 0;

  /// \brief getter
  /// \return step_size
  int32_t step_size() const { return step_size_; }

 protected:
  int32_t step_size_;  // step begin/end will be called every step_size_
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_DS_CALLBACK_H
