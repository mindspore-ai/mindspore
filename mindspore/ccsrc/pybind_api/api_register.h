/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef PYBIND_API_API_REGISTER_H_
#define PYBIND_API_API_REGISTER_H_

#include <map>
#include <string>
#include <memory>
#include <functional>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

namespace mindspore {

using PybindDefineFunc = std::function<void(py::module*)>;

class PybindDefineRegister {
 public:
  static void Register(const std::string& name, const PybindDefineFunc& fn) {
    return GetSingleton().RegisterFn(name, fn);
  }

  PybindDefineRegister(const PybindDefineRegister&) = delete;

  PybindDefineRegister& operator=(const PybindDefineRegister&) = delete;

  static std::map<std::string, PybindDefineFunc>& AllFuncs() { return GetSingleton().fns_; }

  std::map<std::string, PybindDefineFunc> fns_;

 protected:
  PybindDefineRegister() = default;

  virtual ~PybindDefineRegister() = default;

  static PybindDefineRegister& GetSingleton();

  void RegisterFn(const std::string& name, const PybindDefineFunc& fn) { fns_[name] = fn; }
};

class PybindDefineRegisterer {
 public:
  PybindDefineRegisterer(const std::string& name, const PybindDefineFunc& fn) {
    PybindDefineRegister::Register(name, fn);
  }
  ~PybindDefineRegisterer() = default;
};

#define REGISTER_PYBIND_DEFINE(name, define) PybindDefineRegisterer g_pybind_define_f_##name(#name, define)

}  // namespace mindspore

#endif  // PYBIND_API_API_REGISTER_H_
