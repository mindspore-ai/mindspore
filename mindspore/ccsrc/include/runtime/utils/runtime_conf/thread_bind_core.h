/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_RUNTIME_UTILS_RUNTIME_CONF_THREAD_BIND_CORE_H_
#define MINDSPORE_CCSRC_INCLUDE_RUNTIME_UTILS_RUNTIME_CONF_THREAD_BIND_CORE_H_

#include <string>
#include <memory>
#include <map>
#include <vector>
#include <set>
#include <mutex>
#include <iostream>

#include "runtime/utils/visible.h"

namespace mindspore {
namespace runtime {
using ModuleBindCorePolicy = std::map<std::string, std::vector<int>>;
enum kBindCoreModule : int { kMAIN = 0, kRUNTIME, kPYNATIVE, kMINDDATA, kBATCHLAUNCH };

class RUNTIME_UTILS_EXPORT ThreadBindCore {
 public:
  static ThreadBindCore &GetInstance() {
    static ThreadBindCore instance;
    return instance;
  }
  void enable_thread_bind_core(const ModuleBindCorePolicy &module_bind_core_strategy);
  bool parse_thread_bind_core_policy(const kBindCoreModule &module_name);
  std::vector<int> get_thread_bind_core_list(const kBindCoreModule &module_name);
  void bind_thread_core(const std::vector<int> &cpu_list);

  /// @brief Bind thread id or process id to cores.
  /// @param cpu_list List of bound cores.
  /// @param thread_or_process_id The thread id or process id.
  /// @param is_thread Whether the binding is a thread or not, false is bound to a thread, true is bound to a process,
  ///     default is false
  void bind_thread_core(const std::vector<int> &cpu_list, int64_t thread_or_process_id, bool is_thread = false);
  bool unbind_thread_core(const std::string &thread_name);
  bool is_enable_thread_bind_core_{false};

 private:
  ModuleBindCorePolicy process_bind_core_policy_;
  std::vector<int> available_cpu_list_;
  std::map<kBindCoreModule, std::vector<int>> thread_bind_core_policy_;
  std::map<kBindCoreModule, bool> thread_bind_core_status_;
  std::mutex mtx_;
  ThreadBindCore() = default;
  ~ThreadBindCore() = default;
};
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_RUNTIME_UTILS_RUNTIME_CONF_THREAD_BIND_CORE_H_
