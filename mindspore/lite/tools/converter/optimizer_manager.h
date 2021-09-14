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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_OPTIMIZER_MANAGER_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_OPTIMIZER_MANAGER_H

#include <map>
#include <string>
#include <vector>
#include "backend/optimizer/common/pass.h"
#include "include/errorcode.h"
#include "include/registry/pass_registry.h"
#include "ir/func_graph.h"

namespace mindspore {
namespace lite {
class PassStorage {
 public:
  static int StorePass(const std::string &pass_name, const opt::PassPtr &pass) {
    if (registry::PassRegistry::GetPassFromStoreRoom(pass_name) != nullptr) {
      return RET_ERROR;
    }
    pass_stroge_[pass_name] = pass;
    return RET_OK;
  }
  static opt::PassPtr GetPassFromStorage(const std::string &pass_name) { return pass_stroge_[pass_name]; }

 private:
  static std::map<std::string, opt::PassPtr> pass_stroge_;
};

bool RunOptimizerPass(const FuncGraphPtr &func_graph, const std::vector<std::string> &pass_names);
bool RunExternalPass(const FuncGraphPtr &func_graph, registry::PassPosition position);
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_OPTIMIZER_MANAGER_H
