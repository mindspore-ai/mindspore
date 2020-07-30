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
#include "frontend/optimizer/py_pass_manager.h"

#include <functional>
#include <algorithm>
#include <utility>
#include <initializer_list>

#include "ir/manager.h"
#include "frontend/optimizer/pass_group.h"

namespace mindspore {
namespace opt {
namespace python_pass {
PyPassManagerPtr PyPassManager::global_instance = nullptr;
std::unordered_map<Phase, PassGroupPtr> PyPassManager::phase_to_group_;

PassGroupPtr PyPassManager::GetPassGroup(Phase phase) {
  auto pm = phase_to_group_.find(phase);
  if (pm == phase_to_group_.end()) {
    return nullptr;
  }
  return pm->second;
}

PyPassManagerPtr PyPassManager::GetInstance() {
  if (global_instance == nullptr) {
    global_instance = std::shared_ptr<PyPassManager>(new (std::nothrow) PyPassManager());
  }
  return global_instance;
}

PyPassManager::PyPassManager() {
  phase_to_group_[Phase::RESOLVE] = std::make_shared<PassGroup>();
  phase_to_group_[Phase::OPT] = std::make_shared<PassGroup>();
}

void PyPassManager::Registe(const std::string &pass_name, const PatternPtr &pattern, const PatternPtr &target,
                            Phase phase, bool run_only_once, bool multigraph) {
  auto cur_pm = GetPassGroup(phase);
  MS_EXCEPTION_IF_NULL(cur_pm);
  PythonPassPtr new_pass = std::make_shared<PythonPass>(pass_name, pattern, target, run_only_once, multigraph);
  cur_pm->AddPass(new_pass);
}

void PyPassManager::Unregiste(const std::string &pass_name, Phase phase) {
  auto cur_pm = GetPassGroup(phase);
  MS_EXCEPTION_IF_NULL(cur_pm);
  if (!cur_pm->DeletePass(pass_name)) {
    MS_LOG(WARNING) << "No such pass : " + pass_name + "\n";
  }
}

void PyPassManager::ClearRes() {
  MS_LOG(INFO) << "Clear PyPassManager resources!";
  global_instance = nullptr;
  phase_to_group_.clear();
}

REGISTER_PYBIND_DEFINE(
  PyPassManager_, ([](const py::module *m) {
    (void)py::enum_<Phase>(*m, "phase", py::arithmetic()).value("resolve", Phase::RESOLVE).value("opt", Phase::OPT);
    (void)py::class_<PyPassManager, std::shared_ptr<PyPassManager>>(*m, "PyPassManager_")
      .def(py::init([]() { return PyPassManager::GetInstance(); }))
      .def("registe", &PyPassManager::Registe, "Registe python pass")
      .def("unregiste", &PyPassManager::Unregiste, "Delete Python Pass");
  }));
}  // namespace python_pass
}  // namespace opt
}  // namespace mindspore
