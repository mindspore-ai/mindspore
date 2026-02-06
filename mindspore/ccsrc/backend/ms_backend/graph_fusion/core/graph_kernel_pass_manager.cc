/**
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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
#include "backend/ms_backend/graph_fusion/core/graph_kernel_pass_manager.h"

#include <chrono>
#include <iomanip>
#include <limits>
#include <ratio>
#include <utility>

#include "utils/log_adapter.h"
#include "backend/ms_backend/graph_fusion/core/graph_kernel_utils.h"

namespace mindspore::graphkernel {
namespace {
constexpr size_t fusion_stage = std::numeric_limits<size_t>::max();
}
void GraphKernelPassManager::Add(const opt::PassPtr &pass, unsigned int pass_level, bool supported_device) {
  MS_EXCEPTION_IF_NULL(pass);
  auto pass_id = passes_.size();
  auto pass_name = pass->name();

  auto pass_in_list = [this, pass_id,
                       &pass_name](const std::vector<std::string> &pass_list) -> std::pair<bool, size_t> {
    if (pass_list.empty()) {
      return std::make_pair(false, 0);
    }

    // the config format can be "stage_id.pass_id" or "stage_name.pass_name"
    auto number_name = std::to_string(this->stage_) + "." + std::to_string(pass_id);
    auto stage_pass_name = this->name_ + "." + pass_name;
    for (size_t i = 0; i < pass_list.size(); ++i) {
      if (pass_list[i] == number_name || pass_list[i] == stage_pass_name ||
          (this->stage_ == fusion_stage && pass_list[i] == pass_name)) {
        return std::make_pair(true, i);
      }
    }
    return std::make_pair(false, pass_list.size());
  };

  bool enable = supported_device && GraphKernelFlags::GetInstance().opt_level >= pass_level;
  if (enable) {
    // if it meets the condition to enable, check whether it's in the disabled list.
    auto [match, idx] = pass_in_list(GraphKernelFlags::GetInstance().disable_pass);
    enable = !match;
    GraphKernelPassChecker::GetInstance().SetDisablePassActive(idx, true);
  } else {
    // if it doesn't meet the condition to enable, check whether it's in the enabled list.
    auto [match, idx] = pass_in_list(GraphKernelFlags::GetInstance().enable_pass);
    enable = match;
    GraphKernelPassChecker::GetInstance().SetEnablePassActive(idx, true);
  }

  passes_.push_back(pass);
  fusion_passes_switch_[pass] = enable;
}

std::string GraphKernelPassManager::GetPassFullname(size_t pass_id, const opt::PassPtr &pass) const {
  std::string full_name = "";
  if (stage_ != fusion_stage) {
    full_name += "stage" + std::to_string(stage_) + "_";
  }
  full_name += name() + "_" + std::to_string(pass_id) + "_" + pass->name();

  return full_name;
}

bool GraphKernelPassManager::Run(const FuncGraphPtr &func_graph) const {
  bool changed = false;
  for (size_t i = 0; i < passes_.size(); i++) {
    auto pass_name = GetPassFullname(i, passes_[i]);
    auto kv = fusion_passes_switch_.find(passes_[i]);
    if (kv != fusion_passes_switch_.end() && kv->second) {
      GK_PROF_START(pass_name);
      MS_LOG(INFO) << "graph kernel pass " << pass_name << " is enabled.";
      auto pass_changed = RunPass(func_graph, i, passes_[i]);
      GK_PROF_END_WITH_VAR(pass_name);
      if (pass_changed) {
        MS_LOG(INFO) << "graph kernel pass " << pass_name << " changed.";
      }
      changed = pass_changed || changed;
      // dump ir to a graph_kernel subdir, and set a global id in front of the filename
      std::ostringstream oss;
      static int g_id = 0;
      constexpr int id_length = 4;
      oss << "graph_kernel/" << std::setfill('0') << std::setw(id_length) << g_id++ << "_" << pass_name;
      DumpPassIR(func_graph, oss.str());
    } else {
      MS_LOG(INFO) << "graph kernel pass " << pass_name << " is disabled.";
    }
  }
  return changed;
}

bool GraphKernelPassManager::RunPass(const FuncGraphPtr &func_graph, size_t pass_id, const opt::PassPtr &pass) const {
  auto start_time = std::chrono::steady_clock::now();
  bool changed = pass->Run(func_graph);
  auto stop_time = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::micro> cost = stop_time - start_time;
  MS_LOG(INFO) << "Run graph kernel pass " + GetPassFullname(pass_id, pass) + " in " << cost.count() << " us";
  return changed;
}
}  // namespace mindspore::graphkernel
