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
#include "frontend/optimizer/pass_group.h"
#include "frontend/optimizer/py_pass_manager.h"

namespace mindspore {
namespace opt {
namespace python_pass {
void PassGroup::AddPass(const PythonPassPtr &pass) {
  if (pass != nullptr) {
    passes_.push_back(pass);
  }
}

bool PassGroup::DeletePass(const std::string &pass_name) {
  for (auto iter = passes_.begin(); iter != passes_.end(); iter++) {
    if ((*iter)->name() == pass_name) {
      *iter = nullptr;
      passes_.erase(iter);
      return true;
    }
  }
  return false;
}

bool PassGroup::Run(const FuncGraphPtr &func_graph, const std::vector<PythonPassPtr> &passes,
                    const MatchResultPtr &res) const {
  if (func_graph == nullptr) {
    return false;
  }
  bool changed = false;
  for (const auto &pass : passes) {
    if (pass != nullptr) {
      if (pass->Run(func_graph, res)) {
        changed = true;
      }
    }
  }
  return changed;
}

bool PassGroup::Run(const FuncGraphPtr &func_graph) const {
  bool changed = false;
  // run all passes
  bool change = true;
  auto res = PyPassManager::GetInstance()->GetMatchResult();
  while (change) {
    change = Run(func_graph, passes_, res);
    changed = change || changed;
    if (run_only_once_) {
      break;
    }
  }
  return changed;
}
}  // namespace python_pass
}  // namespace opt
}  // namespace mindspore
