/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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
#include "include/backend/common/pass_manager/pass_manager.h"
#include <deque>
#include <string>
#include "utils/ms_context.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel_flags.h"

namespace mindspore {
namespace opt {
PassManager::PassManager(const std::string &name, bool run_only_once)
    : name_(name), passes_{}, run_only_once_(run_only_once) {}

void PassManager::AddPass(const PassPtr &pass) {
  if (pass != nullptr) {
    passes_.push_back(pass);
  }
}

void PassManager::AddFusionPass(const PassPtr &pass, bool condition) {
  MS_EXCEPTION_IF_NULL(pass);
  auto pass_name = pass->name();
  auto pass_in_list = [this, &pass_name](const std::vector<std::string> &pass_list) -> std::pair<bool, size_t> {
    if (pass_list.empty()) {
      return std::make_pair(false, 0);
    }

    // the config format should be "pass_name or stage_name.pass_name"
    auto stage_pass_name = this->name_ + "." + pass_name;
    for (size_t i = 0; i < pass_list.size(); ++i) {
      if (pass_list[i] == stage_pass_name || pass_list[i] == pass_name) {
        return std::make_pair(true, i);
      }
    }
    return std::make_pair(false, pass_list.size());
  };

  if (condition == true) {
    // if it meets the condition to enable, check whether it's in the disabled list.
    auto [match, idx] = pass_in_list(graphkernel::GraphKernelFlags::GetInstance().disable_pass);
    condition = !match;
    graphkernel::GraphKernelPassChecker::GetInstance().SetDisablePassActive(idx, true);
  } else {
    // if it doesn't meet the condition to enable, check whether it's in the enabled list.
    auto [match, idx] = pass_in_list(graphkernel::GraphKernelFlags::GetInstance().enable_pass);
    condition = match;
    graphkernel::GraphKernelPassChecker::GetInstance().SetEnablePassActive(idx, true);
  }

  passes_.push_back(pass);
  fusion_passes_switch_[pass] = condition;
}

bool PassManager::RunPass(const FuncGraphPtr &func_graph, size_t pass_id, const PassPtr &pass) const {
  auto start_time = std::chrono::steady_clock::now();
  bool changed = pass->Run(func_graph);
  constexpr auto kMicroSendUnit = 1000000;
  auto end_time = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::ratio<1, kMicroSendUnit>> cost = end_time - start_time;
  MS_LOG(INFO) << "Run pass " + GetPassFullname(pass_id, pass) + " in " << cost.count() << " us";
  return changed;
}

std::string PassManager::GetPassFullname(size_t pass_id, const PassPtr &pass) const {
  auto header_name = std::string("hwopt_");
  auto kv = fusion_passes_switch_.find(pass);
  if (kv != fusion_passes_switch_.end()) {
    header_name += "fusion_";
  }
  return header_name + name() + "_" + std::to_string(pass_id) + "_" + pass->name();
}

void PassManager::DumpPassIR(const FuncGraphPtr &func_graph, const std::string &pass_fullname) const {
#ifdef ENABLE_DUMP_IR
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  static const auto enable_dump = !GetDumpConfig().disable_backend_dump;
  if (context_ptr->CanDump(kAdvanced) && enable_dump) {
    std::ostringstream oss;
    oss << "verbose_ir_files"
        << "/";
    oss << (pass_fullname + ".ir");
    DumpIR(oss.str(), func_graph, true);
  }
#endif
}

bool PassManager::Run(const FuncGraphPtr &func_graph, const std::vector<PassPtr> &passes) const {
  if (func_graph == nullptr) {
    return false;
  }
  bool changed = false;
  size_t num = 0;
  for (const auto &pass : passes) {
    if (pass != nullptr) {
      auto pass_name = GetPassFullname(num, pass);
      bool enable = true;
      auto kv = fusion_passes_switch_.find(pass);
      if (kv != fusion_passes_switch_.end()) {
        if (kv->second) {
          MS_LOG(INFO) << "graph kernel pass " << pass_name << " is enabled.";
        } else {
          MS_LOG(INFO) << "graph kernel pass " << pass_name << " is disabled.";
          enable = false;
        }
      }
      if (enable) {
        changed = RunPass(func_graph, num, pass) || changed;
#ifdef ENABLE_DUMP_IR
        DumpPassIR(func_graph, pass_name);
#endif
      }
      num++;
    }
  }
  return changed;
}

bool PassManager::Run(const FuncGraphPtr &func_graph) const {
  bool changed = false;
  // run all passes
  bool change = true;
  while (change) {
    change = Run(func_graph, passes_);
    changed = change || changed;
    if (run_only_once_) {
      break;
    }
  }
  return changed;
}
}  // namespace opt
}  // namespace mindspore
