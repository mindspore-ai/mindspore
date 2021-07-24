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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_PY_DS_CALLBACK_H
#define MINDSPORE_CCSRC_MINDDATA_DATASET_PY_DS_CALLBACK_H

#include <memory>
#include <utility>
#include <vector>

#include "minddata/dataset/callback/ds_callback.h"
#include "minddata/dataset/util/status.h"
#include "pybind11/pybind11.h"

namespace mindspore {
namespace dataset {

namespace py = pybind11;

class PyDSCallback : public DSCallback {
 public:
  /// \brief constructor for PyDSCallback. This callback is for python front end
  explicit PyDSCallback(int32_t step_size = 1)
      : DSCallback(step_size),
        begin_needed_(false),
        epoch_begin_needed_(false),
        step_begin_needed_(false),
        end_needed_(false),
        epoch_end_needed_(false),
        step_end_needed_(false) {}

  ~PyDSCallback() = default;

  void SetBegin(const py::function &f);
  void SetEnd(const py::function &f);
  void SetEpochBegin(const py::function &f);
  void SetEpochEnd(const py::function &f);
  void SetStepBegin(const py::function &f);
  void SetStepEnd(const py::function &f);

  /// \brief actual callback function for begin, needs to be overridden in the derived class
  /// \param cb_param, callback parameter passed in from DatasetOp when calling the callback
  /// \return Status
  Status DSBegin(const CallbackParam &cb_param) override;

  /// \brief actual callback function for epoch_begin, needs to be overridden in the derived class
  /// \param cb_param, callback parameter passed in from DatasetOp when calling the callback
  /// \return Status
  Status DSEpochBegin(const CallbackParam &cb_param) override;

  /// \brief actual callback function for step_begin, needs to be overridden in the derived class
  /// \param cb_param, callback parameter passed in from DatasetOp when calling the callback
  /// \return Status
  Status DSNStepBegin(const CallbackParam &cb_param) override;

  /// \brief actual callback function for end, needs to be overridden in the derived class
  /// \param cb_param, callback parameter passed in from DatasetOp when calling the callback
  /// \return Status
  Status DSEnd(const CallbackParam &cb_param) override;

  /// \brief actual callback function epoch_end begin, needs to be overridden in the derived class
  /// \param cb_param, callback parameter passed in from DatasetOp when calling the callback
  /// \return Status
  Status DSEpochEnd(const CallbackParam &cb_param) override;

  /// \brief actual callback function for step_end, needs to be overridden in the derived class
  /// \param cb_param, callback parameter passed in from DatasetOp when calling the callback
  /// \return Status
  Status DSNStepEnd(const CallbackParam &cb_param) override;

  /// \brief predicate function, whether begin callback is needed
  /// \return bool
  bool IsBeginNeeded() override;

  /// \brief predicate function, whether epoch_begin callback is needed
  /// \return bool
  bool IsEpochBeginNeeded() override;

  /// \brief predicate function, whether step_begin callback is needed
  /// \return bool
  bool IsNStepBeginNeeded() override;

  /// \brief predicate function, whether end callback is needed
  /// \return bool
  bool IsEndNeeded() override;

  /// \brief predicate function, whether epoch_end callback is needed
  /// \return bool
  bool IsEpochEndNeeded() override;

  /// \brief predicate function, whether step_end callback is needed
  /// \return bool
  bool IsNStepEndNeeded() override;

  /// \brief helper function to acquire GIL then execute a pyfunc
  /// \param f the python function
  /// \param cb_param
  /// \return Status
  static Status ExecutePyfunc(py::function f, const CallbackParam &cb_param);

 private:
  py::function begin_func_;
  py::function epoch_begin_func_;
  py::function step_begin_func_;
  py::function end_func_;
  py::function epoch_end_func_;
  py::function step_end_func_;

  bool begin_needed_;
  bool epoch_begin_needed_;
  bool step_begin_needed_;
  bool end_needed_;
  bool epoch_end_needed_;
  bool step_end_needed_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_PY_DS_CALLBACK_H
