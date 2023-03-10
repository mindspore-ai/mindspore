/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "tools/optimizer/common/pass_manager_extends.h"
#ifndef _MSC_VER
#include <sys/time.h>
#endif
#include <deque>
#include <string>
#include <algorithm>
#include "ir/anf.h"
#include "backend/common/optimizer/cache_manager.h"

namespace mindspore {
namespace opt {
constexpr size_t kMaxRepassTimes = 12;
constexpr uint64_t kUSecondInSecond = 1000000;

PassManager::PassManager(const std::string &name, bool run_only_once)
    : name_(name), passes_{}, run_only_once_(run_only_once), cache_manager_(std::make_shared<CacheManager>()) {}

void PassManager::AddPass(const PassPtr &pass) {
  if (pass != nullptr) {
    passes_.push_back(pass);
  }
}

bool PassManager::RunPass(const FuncGraphPtr &func_graph, size_t pass_id, const PassPtr &pass) const {
  MS_LOG(ERROR) << "stub func";
  return false;
}

std::string PassManager::GetPassFullname(size_t pass_id, const PassPtr &pass) const {
  return std::string("hwopt_") + name() + "_" + std::to_string(pass_id) + "_" + pass->name();
}

void PassManager::DumpPassIR(const FuncGraphPtr &func_graph, const std::string &pass_fullname) const {
  MS_LOG(ERROR) << "stub func";
}

bool PassManager::Run(const FuncGraphPtr &func_graph, const std::vector<PassPtr> &passes) const {
  MS_LOG(ERROR) << "stub func";
  return false;
}

bool PassManager::Run(const FuncGraphPtr &func_graph) const {
  MS_LOG(ERROR) << "stub func";
  return false;
}

void LitePassManager::AddPass(const PassPtr &pass) {
  if (pass != nullptr) {
    passes_.push_back(pass);
  }
}

bool LitePassManager::RunPass(const FuncGraphPtr &func_graph, size_t pass_id, const PassPtr &pass) const {
  bool changed = false;
#if defined(_WIN32) || defined(_WIN64)
  auto start_time = std::chrono::steady_clock::now();
#else
  struct timeval start_time {};
  struct timeval end_time {};
  (void)gettimeofday(&start_time, nullptr);
#endif
  if (pass->Run(func_graph)) {
    MS_LOG(DEBUG) << "Run pass and find change";
    changed = true;
  }
#if defined(_WIN32) || defined(_WIN64)
  auto end_time = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::ratio<1, kUSecondInSecond>> cost = end_time - start_time;
  MS_LOG(INFO) << "Run pass " << GetPassFullname(pass_id, pass) << " in " << cost.count() << " us.";
#else
  (void)gettimeofday(&end_time, nullptr);
  uint64_t cost = kUSecondInSecond * static_cast<uint64_t>(end_time.tv_sec - start_time.tv_sec);
  cost += static_cast<uint64_t>(end_time.tv_usec - start_time.tv_usec);
  MS_LOG(INFO) << "Run pass " << GetPassFullname(pass_id, pass) << " in " << cost << " us.";
#endif
  return changed;
}

std::string LitePassManager::GetPassFullname(size_t pass_id, const PassPtr &pass) const {
  return "hwopt_" + name() + "_" + std::to_string(pass_id) + "_" + pass->name();
}

bool LitePassManager::Run(const FuncGraphPtr &func_graph, const std::vector<PassPtr> &passes) const {
  if (func_graph == nullptr) {
    return false;
  }
  bool changed = false;
  size_t num = 0;
  for (const auto &pass : passes) {
    if (pass != nullptr) {
      changed = RunPass(func_graph, num, pass) || changed;
    } else {
      MS_LOG(INFO) << "pass is null";
    }
    num++;
  }
  return changed;
}

bool LitePassManager::Run(const FuncGraphPtr &func_graph) const {
  if (func_graph == nullptr) {
    return false;
  }
  bool changed = false;
  size_t count = 0;
  // run all passes
  bool change = true;
  while (change) {
    change = Run(func_graph, passes_);
    changed = change || changed;
    if (run_only_once_ || count > kMaxRepassTimes) {
      break;
    }
    count++;
    MS_LOG(INFO) << "Run pass counts:" << count;
  }
  return changed;
}
}  // namespace opt
}  // namespace mindspore
