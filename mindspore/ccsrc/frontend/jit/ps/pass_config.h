/**
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

#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_PS_PASS_CONFIG_H_
#define MINDSPORE_CCSRC_FRONTEND_JIT_PS_PASS_CONFIG_H_

#include <vector>
#include <functional>
#include <utility>
#include <string>
#include <set>
#include "pybind11/pybind11.h"
#include "frontend/jit/ps/resource.h"
#include "include/frontend/optimizer/optimizer.h"
#include "frontend/jit/ps/pass.h"

namespace mindspore {
namespace opt {

bool FilterPass(const std::string &pass_key);
void UpdateRunningPasses(const std::string &pass_key);

class PassConfigure {
 public:
  void Clear() {
    opt_func_map_.clear();
    substitution_map_.clear();
    passes_.clear();
  }
  ~PassConfigure() = default;
  void SetOptimizeConfig(const py::list &optimize_cfg);
  using PassFunc = std::function<bool(pipeline::ResourcePtr)>;
  using PassItem = std::pair<std::string, PassFunc>;
  void RegisterOptimizeFunc(const std::string &name, const OptimizeGraphFunc &func) { opt_func_map_[name] = func; }

  OptimizeGraphFunc GetOptimizeFunc(const std::string &name) {
    auto it = opt_func_map_.find(name);
    if (it != opt_func_map_.end()) {
      return it->second;
    }
    return OptimizeGraphFunc();
  }

  void RegisterPassFunc(const std::string &name, const PassFunc &func) { pass_func_map_[name] = func; }

  PassFunc GetPassFunc(const std::string &name) {
    auto it = pass_func_map_.find(name);
    if (it != pass_func_map_.end()) {
      return it->second;
    }
    return PassFunc();
  }

  void RegisterSubstitution(const SubstitutionPtr &substitution) {
    substitution_map_[substitution->name_] = substitution;
  }

  SubstitutionPtr GetSubstitution(const std::string &name) {
    auto it = substitution_map_.find(name);
    if (it != substitution_map_.end()) {
      return it->second;
    }
    return nullptr;
  }

  void SetPasses(const std::vector<PassItem> &passes) { passes_ = passes; }

  std::vector<PassItem> GetPasses() { return passes_; }

  static PassConfigure &Instance();

  std::string GetOptimizeConfig();
  py::list GetRunningPasses();
  void SetConfigPasses(const py::list &cfg_passes);

  void UpdateRunningPasses(const std::string &pass_key) {
    if (cfg_passes_.empty()) {
      running_passes_.insert(pass_key);
    }
  }

  bool FilterPass(const std::string &pass_key) {
    if (!cfg_passes_.empty() && cfg_passes_.count(pass_key) == 0) {
      return true;
    }
    return false;
  }

 private:
  PassConfigure() = default;
  mindspore::HashMap<std::string, OptimizeGraphFunc> opt_func_map_;
  mindspore::HashMap<std::string, SubstitutionPtr> substitution_map_;
  mindspore::HashMap<std::string, PassFunc> pass_func_map_;
  std::vector<PassItem> passes_;
  std::set<std::string> cfg_passes_;
  std::set<std::string> running_passes_;
};

class RegisterOptimizeFunc {
 public:
  RegisterOptimizeFunc(const std::string &name, const opt::OptimizeGraphFunc &func) {
    PassConfigure::Instance().RegisterOptimizeFunc(name, func);
  }

  RegisterOptimizeFunc() = delete;
  ~RegisterOptimizeFunc() = default;
};

class RegisterPassFunc {
 public:
  RegisterPassFunc(const std::string &name, const PassConfigure::PassFunc &func) {
    PassConfigure::Instance().RegisterPassFunc(name, func);
  }
  RegisterPassFunc() = delete;
  ~RegisterPassFunc() = default;
};
void SavePassesConfig(const std::string &func_graph);
void LoadPassesConfig(const std::string &func_graph);
}  // namespace opt

#define REGISTER_OPT_PASS_FUNC(name)                                                                   \
  namespace {                                                                                          \
  static auto helper_opt_pass_##name = opt::RegisterOptimizeFunc(#name, opt::OptimizeGraphFunc(name)); \
  }

#define REGISTER_OPT_PASS_CLASS(name)                                            \
  namespace {                                                                    \
  static auto helper_opt_pass_##name = opt::RegisterOptimizeFunc(#name, name()); \
  }

#define REGISTER_PASS_FUNC_IMPL(name)                                                                     \
  namespace {                                                                                             \
  static auto helper_pass_func_##name = opt::RegisterPassFunc(#name, opt::PassConfigure::PassFunc(name)); \
  }
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_JIT_PS_PASS_CONFIG_H_
