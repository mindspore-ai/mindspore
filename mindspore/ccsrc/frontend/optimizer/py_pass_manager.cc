/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "ir/manager.h"
#include "frontend/optimizer/pass_group.h"

namespace mindspore {
namespace opt {
namespace python_pass {
PyPassManagerPtr PyPassManager::global_instance = nullptr;
mindspore::HashMap<Phase, PassGroupPtr> PyPassManager::phase_to_group_;

PassGroupPtr PyPassManager::GetPassGroup(Phase phase) const {
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
  phase_to_group_[Phase::PREAD] = std::make_shared<PassGroup>("Pre_AD_PassGroup");
  phase_to_group_[Phase::OPT] = std::make_shared<PassGroup>("After_OPT_PassGroup");
  res_ = std::make_shared<MatchResult>();
}

void PyPassManager::Register(const std::string &pass_name, const PatternPtr &pattern, const PatternPtr &target,
                             bool requires_grad, bool run_only_once) {
  PassGroupPtr cur_pg;
  if (requires_grad) {
    cur_pg = GetPassGroup(Phase::PREAD);
  } else {
    cur_pg = GetPassGroup(Phase::OPT);
  }
  MS_EXCEPTION_IF_NULL(cur_pg);
  cur_pg->SetRunOnlyOnce(run_only_once);
  MS_EXCEPTION_IF_NULL(pattern);
  MS_EXCEPTION_IF_NULL(target);
  MS_EXCEPTION_IF_NULL(cur_pg);
  PythonPassPtr new_pass = std::make_shared<PythonPass>(pass_name, pattern, target, run_only_once);
  cur_pg->AddPass(new_pass);
}

void PyPassManager::Unregister(const std::string &pass_name) {
  auto opt_pm = GetPassGroup(Phase::OPT);
  if (!opt_pm->DeletePass(pass_name)) {
    MS_LOG(WARNING) << "Opt has no such pass : " + pass_name + "\n";
  }
  auto pre_ad_pm = GetPassGroup(Phase::PREAD);
  if (!pre_ad_pm->DeletePass(pass_name)) {
    MS_LOG(WARNING) << "Pre_AD has no such pass : " + pass_name + "\n";
  }
}

void PyPassManager::GenNewParameter(const PatternPtr &parameter) {
  MS_EXCEPTION_IF_NULL(parameter);
  // NOTE: Add NewParameter at early stage will cause CSE problems
  auto cur_pg = GetPassGroup(Phase::OPT);
  MS_EXCEPTION_IF_NULL(cur_pg);
  cur_pg->SetRunOnlyOnce(true);
  auto new_para_pattern = parameter->cast_ptr<NewParameter>();
  MS_EXCEPTION_IF_NULL(new_para_pattern);
  auto pass_name = new_para_pattern->para_name();
  new_para_pattern->set_last(true);
  auto new_pass = std::make_shared<PythonPass>(pass_name, nullptr, parameter, true);
  cur_pg->AddPass(new_pass);
}

void PyPassManager::ClearRes() {
  MS_LOG(INFO) << "Clear PyPassManager resources!";
  global_instance = nullptr;
  phase_to_group_.clear();
}

void RegPyPassManager(const py::module *m) {
  (void)py::enum_<Phase>(*m, "phase", py::arithmetic()).value("pre_ad", Phase::PREAD).value("opt", Phase::OPT);
  (void)py::class_<PyPassManager, std::shared_ptr<PyPassManager>>(*m, "PyPassManager_")
    .def(py::init([]() { return PyPassManager::GetInstance(); }))
    .def("register", &PyPassManager::Register, "Register python pass")
    .def("unregister", &PyPassManager::Unregister, "Unregister Python Pass")
    .def("gen_new_parameter", &PyPassManager::GenNewParameter, "Generate new parameter")
    .def("set_renorm", &PyPassManager::SetRenorm, "Set whether or not to do renorm after modified graph")
    .def("set_reopt", &PyPassManager::SetReOpt, "Set whether or not to do optimization after modified graph");
}
}  // namespace python_pass
}  // namespace opt
}  // namespace mindspore
