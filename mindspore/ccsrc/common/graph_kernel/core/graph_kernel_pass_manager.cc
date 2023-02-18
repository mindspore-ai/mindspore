/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include "common/graph_kernel/core/graph_kernel_pass_manager.h"

#include <iomanip>
#include <algorithm>
#include <memory>
#include <ostream>

#include "utils/log_adapter.h"

namespace mindspore::graphkernel {
void GraphKernelPassManager::Add(const opt::PassPtr &pass, unsigned int pass_level, bool supported_device) {
  MS_EXCEPTION_IF_NULL(pass);
  auto pass_id = passes_.size();
  auto pass_name = pass->name();
  auto pass_in_list = [this, pass_id, &pass_name](const std::vector<std::string> &pass_list) {
    // the config format can be "stage_id.pass_id" or "stage_name.pass_name"
    return std::find(pass_list.begin(), pass_list.end(),
                     std::to_string(this->stage_) + "." + std::to_string(pass_id)) != pass_list.end() ||
           std::find(pass_list.begin(), pass_list.end(), this->name_ + "." + pass_name) != pass_list.end();
  };
  bool enable = supported_device && flags_.opt_level >= pass_level;
  if (enable) {
    // if it meets the condition to enable, check whether it's in the disabled list.
    enable = !pass_in_list(flags_.disable_pass);
  } else {
    // if it doesn't meet the condition to enable, check whether it's in the enabled list.
    enable = pass_in_list(flags_.enable_pass);
  }
  passes_.push_back(pass);
  enabled_.push_back(enable);
}

std::string GraphKernelPassManager::GetPassFullname(size_t pass_id, const opt::PassPtr &pass) const {
  return "stage" + std::to_string(stage_) + "_" + name() + "_" + std::to_string(pass_id) + "_" + pass->name();
}

bool GraphKernelPassManager::Run(const FuncGraphPtr &func_graph) const {
  bool changed = false;
  for (size_t i = 0; i < passes_.size(); i++) {
    if (enabled_[i]) {
      changed = RunPass(func_graph, i, passes_[i]) || changed;
      // dump ir to a graph_kernel subdir, and set a global id in front of the filename
      std::ostringstream oss;
      static int g_id = 0;
      constexpr int id_length = 4;
      oss << "graph_kernel/" << std::setfill('0') << std::setw(id_length) << g_id++ << "_"
          << GetPassFullname(i, passes_[i]);
      DumpPassIR(func_graph, oss.str());
    } else {
      MS_LOG(INFO) << "pass " << GetPassFullname(i, passes_[i]) << " is disabled.";
    }
  }
  return changed;
}
}  // namespace mindspore::graphkernel
