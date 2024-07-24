/*
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_GIL_SCOPED_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_GIL_SCOPED_H_

#include <memory>

#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace mindspore {
namespace dataset {

class GilAcquireWithCheck {
 public:
  GilAcquireWithCheck() {
    if (Py_IsInitialized() != 0 && PyGILState_Check() == 0) {
      MS_LOG(INFO) << "Begin acquire gil.";
      acquire_ = std::make_unique<py::gil_scoped_acquire>();  // acquire the gil
      MS_LOG(INFO) << "End acquire gil.";
    } else {
      MS_LOG(INFO) << "Py_IsInitialized is 1, PyGILState_Check is 1, no need to acquire gil.";
      acquire_ = nullptr;
    }
    if (PyGILState_Check() == 0) {
      MS_LOG(EXCEPTION) << "PyGILState_Check(): 0, except 1. Acquire gil failed in current thread.";
    }
  }

  ~GilAcquireWithCheck() {
    if (PyGILState_Check() == 1 && acquire_ != nullptr) {
      MS_LOG(INFO) << "Begin release gil.";
      acquire_ = nullptr;
      MS_LOG(INFO) << "End release gil.";
    } else {
      MS_LOG(INFO) << "No need to release gil.";
    }
  }

 private:
  std::unique_ptr<py::gil_scoped_acquire> acquire_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_GIL_SCOPED_H_
