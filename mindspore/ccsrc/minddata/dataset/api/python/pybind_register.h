/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_API_PYTHON_PYBIND_REGISTER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_API_PYTHON_PYBIND_REGISTER_H_

#include <functional>
#include <map>
#include <stdexcept>
#include <string>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "utils/status.h"
#include "minddata/dataset/util/md_log_adapter.h"

namespace py = pybind11;

namespace mindspore::dataset {
inline void THROW_IF_ERROR(Status status) {
  if (status.IsError()) {
    std::string error_msg = MDLogAdapter::Apply(&status).ToString();
    {
      py::gil_scoped_acquire gil_acquire;
      // Decode the error message to UTF-8 and replace non UTF-8 characters with backslash.
      py::handle utf8_str_handle = PyUnicode_DecodeUTF8(error_msg.data(), error_msg.size(), "backslashreplace");
      if (!utf8_str_handle) {
        throw py::error_already_set();
      }
      // The handle must be stolen to avoid memory leak.
      py::str utf8_str = py::reinterpret_steal<py::str>(utf8_str_handle);
      error_msg = utf8_str.cast<std::string>();
    }
    throw std::runtime_error(error_msg);
  }
  return;
}

using PybindDefineFunc = std::function<void(py::module *)>;

class PybindDefinedFunctionRegister {
 public:
  static void Register(const std::string &name, const uint8_t &priority, const PybindDefineFunc &fn) {
    return GetSingleton().RegisterFn(name, priority, fn);
  }

  PybindDefinedFunctionRegister(const PybindDefinedFunctionRegister &) = delete;

  PybindDefinedFunctionRegister &operator=(const PybindDefinedFunctionRegister &) = delete;

  static std::map<uint8_t, std::map<std::string, PybindDefineFunc>> &AllFunctions() {
    return GetSingleton().module_fns_;
  }
  std::map<uint8_t, std::map<std::string, PybindDefineFunc>> module_fns_;

 protected:
  PybindDefinedFunctionRegister() = default;

  virtual ~PybindDefinedFunctionRegister() = default;

  static PybindDefinedFunctionRegister &GetSingleton();

  void RegisterFn(const std::string &name, const uint8_t &priority, const PybindDefineFunc &fn) {
    module_fns_[priority][name] = fn;
  }
};

class PybindDefineRegisterer {
 public:
  PybindDefineRegisterer(const std::string &name, const uint8_t &priority, const PybindDefineFunc &fn) {
    PybindDefinedFunctionRegister::Register(name, priority, fn);
  }
  ~PybindDefineRegisterer() = default;
};

#ifdef ENABLE_PYTHON
#define PYBIND_REGISTER(name, priority, define) PybindDefineRegisterer g_pybind_define_f_##name(#name, priority, define)
#endif
}  // namespace mindspore::dataset
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_API_PYTHON_PYBIND_REGISTER_H_
