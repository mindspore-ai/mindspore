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
#include "backend/optimizer/common/pass_manager.h"

#include <sys/time.h>
#include <deque>
#include <string>
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ir/manager.h"
#include "utils/ms_context.h"
#include "debug/anf_ir_dump.h"

namespace mindspore {
namespace opt {
const std::vector<PassPtr> &PassManager::Passes() const { return passes_; }

void PassManager::AddPass(const PassPtr &pass) {
  if (pass != nullptr) {
    passes_.push_back(pass);
  }
}

bool PassManager::Run(const FuncGraphPtr &func_graph, const std::vector<PassPtr> &passes) const {
  if (func_graph == nullptr) {
    return false;
  }
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG);
  bool changed = false;
  size_t num = 0;
  for (const auto &pass : passes) {
    if (pass != nullptr) {
#if defined(_WIN32) || defined(_WIN64)
      auto start_time = std::chrono::steady_clock::now();
#else
      struct timeval start_time {};
      struct timeval end_time {};
      (void)gettimeofday(&start_time, nullptr);
#endif
      if (pass->Run(func_graph)) {
        changed = true;
      }
#if defined(_WIN32) || defined(_WIN64)
      auto end_time = std::chrono::steady_clock::now();
      std::chrono::duration<double, std::ratio<1, 1000000>> cost = end_time - start_time;
      MS_LOG(INFO) << "Run pass hwopt_" + name() + "_" << num << "_" + pass->name() + " in " << cost.count() << " us";
#else
      (void)gettimeofday(&end_time, nullptr);
      // time unit: us
      uint64_t cost = 1000000 * static_cast<uint64_t>(end_time.tv_sec - start_time.tv_sec);
      cost += static_cast<uint64_t>(end_time.tv_usec - start_time.tv_usec);
      MS_LOG(INFO) << "Run pass hwopt_" + name() + "_" << num << "_" + pass->name() + " in " << cost << " us";
#endif
      static const auto enable_dump = (common::GetEnv("ENV_NO_DUMP_BE_PASS_IR") != "1");
      if (save_graphs && enable_dump) {
        std::ostringstream oss;
        oss << "verbose_ir_files"
            << "/";
        oss << "hwopt_" + name() + "_" + std::to_string(num) + "_" + pass->name() + ".ir";
        DumpIR(oss.str(), func_graph, true);
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
