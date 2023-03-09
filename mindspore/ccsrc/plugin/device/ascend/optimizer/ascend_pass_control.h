/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_ASCEND_PASS_CONTROL_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_ASCEND_PASS_CONTROL_H_

#include <string>
#include <map>
#include "include/backend/optimizer/optimizer.h"
#include "plugin/device/ascend/hal/device/lic_manager.h"

namespace mindspore {
namespace opt {
class PassSwitchManager {
 public:
  static PassSwitchManager &GetInstance();
  void RegistPass(const std::string &pass_name) { (void)env_pass_switch_.emplace(pass_name, true); }
  void RegistLicPass(const std::string &pass_name, enum OptPassEnum pass) {
    (void)pass_enum_map_.emplace(pass_name, pass);
  }
  bool GetPassSwitch(const std::string &pass_name) const;

 private:
  PassSwitchManager();
  ~PassSwitchManager() = default;

  enum OptPassEnum GetPassEnum(const std::string &pass_name) const;
  void SetSwitchFromEnv();

  std::map<std::string, enum OptPassEnum> pass_enum_map_ = {};
  std::map<std::string, bool> env_pass_switch_ = {};
  bool env_switch_ = true;
};

class PassWithSwitch : public Pass {
 public:
  explicit PassWithSwitch(const std::string &name = "pass") : Pass(name) {
    PassSwitchManager::GetInstance().RegistPass(name);
  }

  virtual ~PassWithSwitch() = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 protected:
  virtual bool RunPass(const FuncGraphPtr &func_graph) = 0;
};

class PatternProcessPassWithSwitch : public PatternProcessPass {
 public:
  explicit PatternProcessPassWithSwitch(const std::string &name = "", bool multigraph = true)
      : PatternProcessPass(name, multigraph) {
    PassSwitchManager::GetInstance().RegistPass(name);
  }

  ~PatternProcessPassWithSwitch() override = default;
  AnfNodePtr Run(const FuncGraphPtr &func_graph, const AnfNodePtr &node) override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_ASCEND_PASS_CONTROL_H_
