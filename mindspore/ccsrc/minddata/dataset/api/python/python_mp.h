/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_API_PYTHON_MP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_API_PYTHON_MP_H_

#include <map>
#include <string>
#include <memory>
#include <functional>
#include <utility>
#include <vector>

#ifdef ENABLE_PYTHON
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
namespace py = pybind11;
#endif

namespace mindspore {
namespace dataset {
class PythonMultiprocessingRuntime {
 public:
  virtual void Launch(int32_t id) = 0;
  virtual void Terminate() = 0;
  virtual bool IsMPEnabled() = 0;
  virtual void AddNewWorkers(int32_t num_new_workers) = 0;
  virtual void RemoveWorkers(int32_t num_removed_workers) = 0;
  virtual std::vector<int32_t> GetPIDs() = 0;
  virtual ~PythonMultiprocessingRuntime() {}
};

#ifdef ENABLE_PYTHON
class PyPythonMultiprocessingRuntime : public PythonMultiprocessingRuntime {
 public:
  // inherit constructors
  using PythonMultiprocessingRuntime::PythonMultiprocessingRuntime;
  //  Trampoline (need one for each virtual function)
  //  PYBIND11_OVERLOAD_PURE(void,                         /* Return type */
  //                        PythonMultiprocessingRuntime,  /* Parent class */
  //                        Launch                         /* Name of function in C++ (must match Python name) */

  void Launch(int32_t id) override { PYBIND11_OVERLOAD_PURE(void, PythonMultiprocessingRuntime, Launch, id); }
  void Terminate() override { PYBIND11_OVERLOAD_PURE(void, PythonMultiprocessingRuntime, Terminate); }
  bool IsMPEnabled() override { PYBIND11_OVERLOAD_PURE(bool, PythonMultiprocessingRuntime, IsMPEnabled); }
  void AddNewWorkers(int32_t num_workers) override {
    PYBIND11_OVERLOAD_PURE(void, PythonMultiprocessingRuntime, AddNewWorkers, num_workers);
  }
  void RemoveWorkers(int32_t num_workers) override {
    PYBIND11_OVERLOAD_PURE(void, PythonMultiprocessingRuntime, RemoveWorkers, num_workers);
  }
  std::vector<int32_t> GetPIDs() override {
    PYBIND11_OVERLOAD_PURE(std::vector<int32_t>, PythonMultiprocessingRuntime, GetPIDs);
  }
};
#endif
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_API_PYTHON_MP_H_
