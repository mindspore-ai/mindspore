/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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
#include "include/backend/optimizer/pass_manager.h"
#include <deque>
#include <string>
#include "ir/anf.h"
#include "utils/ms_context.h"
#include "include/common/debug/anf_ir_dump.h"
#include "backend/common/optimizer/cache_manager.h"

namespace mindspore {
namespace opt {
PassManager::PassManager(const std::string &name, bool run_only_once)
    : name_(name), passes_{}, run_only_once_(run_only_once), cache_manager_(std::make_shared<CacheManager>()) {}

void PassManager::AddPass(const PassPtr &pass) {
  if (pass != nullptr) {
    passes_.push_back(pass);
  }
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
  return std::string("hwopt_") + name() + "_" + std::to_string(pass_id) + "_" + pass->name();
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
      pass->SetCacheManager(cache_manager_);
      changed = RunPass(func_graph, num, pass) || changed;
#ifdef ENABLE_DUMP_IR
      DumpPassIR(func_graph, GetPassFullname(num, pass));
#endif
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
