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
#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_COMMON_PASS_MANAGER_EXTENDS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_COMMON_PASS_MANAGER_EXTENDS_H_

#include <memory>
#include <string>
#include <vector>
#include "include/backend/optimizer/pass_manager.h"

namespace mindspore {
namespace opt {
class LitePassManager : public PassManager {
 public:
  explicit LitePassManager(const std::string &name = "pm", bool run_only_once = true)
      : PassManager(name, run_only_once) {}
  virtual ~LitePassManager() = default;
  void AddPass(const PassPtr &pass) override;
  bool Run(const FuncGraphPtr &func_graph) const override;
  bool Run(const FuncGraphPtr &func_graph, const std::vector<PassPtr> &passes) const override;

 protected:
  bool RunPass(const FuncGraphPtr &func_graph, size_t pass_id, const PassPtr &pass) const override;
  std::string GetPassFullname(size_t pass_id, const PassPtr &pass) const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_COMMON_PASS_MANAGER_EXTENDS_H_
