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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_OPTIMIZER_MANAGER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_OPTIMIZER_MANAGER_H_

#include <map>
#include <set>
#include <string>
#include <vector>
#include "include/backend/optimizer/pass.h"
#include "include/errorcode.h"
#include "include/registry/pass_registry.h"
#include "ir/func_graph.h"

namespace mindspore {
namespace lite {
class PassStorage {
 public:
  static void StorePass(const std::string &pass_name, const opt::PassPtr &pass, bool access_for_outer) {
    pass_storage_[pass_name] = pass;
    if (!access_for_outer) {
      (void)inaccessible_for_outer_.insert(pass_name);
    }
  }
  static void ClearPass() {
    pass_storage_.clear();
    inaccessible_for_outer_.clear();
  }
  static opt::PassPtr GetPassFromStorage(const std::string &pass_name) { return pass_storage_[pass_name]; }
  static bool IsAccessibleForOuter(const std::string &pass_name) {
    return inaccessible_for_outer_.find(pass_name) == inaccessible_for_outer_.end();
  }

 private:
  static std::map<std::string, opt::PassPtr> pass_storage_;
  static std::set<std::string> inaccessible_for_outer_;
};

bool RunOptimizerPass(const FuncGraphPtr &func_graph, const std::vector<std::string> &pass_names);
bool RunExternalPass(const FuncGraphPtr &func_graph, registry::PassPosition position);
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_OPTIMIZER_MANAGER_H_
