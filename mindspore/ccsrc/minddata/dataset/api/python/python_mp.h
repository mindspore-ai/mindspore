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

#include <functional>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

#ifdef ENABLE_PYTHON
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
namespace py = pybind11;
#endif
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class PythonMultiprocessingRuntime {
 public:
  virtual void launch(int32_t id) = 0;
  virtual void terminate() = 0;
  virtual bool is_mp_enabled() = 0;
  virtual void add_new_workers(int32_t num_new_workers) = 0;
  virtual void remove_workers(int32_t num_removed_workers) = 0;
  virtual std::vector<int32_t> get_pids() = 0;
  virtual bool is_running() = 0;
  virtual ~PythonMultiprocessingRuntime() {}
  virtual void set_thread_to_worker(int32_t worker_id) = 0;
  virtual Status get_thread_to_worker(int32_t *const worker_id) const = 0;
  virtual void reset() = 0;
};

#ifdef ENABLE_PYTHON
class PyPythonMultiprocessingRuntime : public PythonMultiprocessingRuntime {
 public:
  // inherit constructors
  using PythonMultiprocessingRuntime::PythonMultiprocessingRuntime;
  //  Trampoline (need one for each virtual function)
  //  PYBIND11_OVERLOAD_PURE(void,                         /* Return type */
  //                        PythonMultiprocessingRuntime,  /* Parent class */
  //                        launch                         /* Name of function in C++ (must match Python name) */

  void launch(int32_t id) override { PYBIND11_OVERLOAD_PURE(void, PythonMultiprocessingRuntime, launch, id); }
  void terminate() override { PYBIND11_OVERLOAD_PURE(void, PythonMultiprocessingRuntime, terminate); }
  bool is_mp_enabled() override { PYBIND11_OVERLOAD_PURE(bool, PythonMultiprocessingRuntime, is_mp_enabled); }
  void add_new_workers(int32_t num_workers) override {
    PYBIND11_OVERLOAD_PURE(void, PythonMultiprocessingRuntime, add_new_workers, num_workers);
  }
  void remove_workers(int32_t num_workers) override {
    PYBIND11_OVERLOAD_PURE(void, PythonMultiprocessingRuntime, remove_workers, num_workers);
  }
  std::vector<int32_t> get_pids() override {
    PYBIND11_OVERLOAD_PURE(std::vector<int32_t>, PythonMultiprocessingRuntime, get_pids);
  }

  void set_thread_to_worker(int32_t worker_id) override {
    std::lock_guard<std::mutex> guard(lock_);
    threads_to_workers_[std::this_thread::get_id()] = worker_id;
  }

  Status get_thread_to_worker(int32_t *const worker_id) const override {
    auto itr = threads_to_workers_.find(std::this_thread::get_id());
    CHECK_FAIL_RETURN_UNEXPECTED(itr != threads_to_workers_.end(), "[Internal] This thread is not a worker!");
    *worker_id = itr->second;
    return Status::OK();
  }

  void reset() override { threads_to_workers_.clear(); }

  bool is_running() override { PYBIND11_OVERLOAD_PURE(bool, PythonMultiprocessingRuntime, is_running); }

 private:
  std::map<std::thread::id, int32_t> threads_to_workers_{};
  std::mutex lock_;  // used when writing into threads_to_workers_
};
#endif
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_API_PYTHON_MP_H_
