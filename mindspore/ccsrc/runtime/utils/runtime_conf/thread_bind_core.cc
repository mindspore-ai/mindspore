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

#include "include/runtime/utils/runtime_conf/thread_bind_core.h"

#ifdef __linux__
#define BIND_CORE
#include <sched.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#endif
#include <utility>
#include <algorithm>
#include <sstream>
#include <nlohmann/json.hpp>

#include "utils/ms_utils.h"
#include "utils/ms_context.h"
#include "utils/file_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/distributed_meta.h"
#include "include/runtime/utils/runtime_conf/runtime_conf.h"
#include "include/runtime/utils/runtime_conf/runtime_env.h"

namespace mindspore {
namespace runtime {
namespace {
constexpr char kMainThread[] = "main";
constexpr char kPynativeThread[] = "pynative";
constexpr char kRunTimeThread[] = "runtime";
constexpr char kDataThread[] = "minddata";
}  // namespace

void ThreadBindCore::enable_thread_bind_core(const ModuleBindCorePolicy &bind_core_policy,
                                             const std::string &actor_thread_fix_bind) {
  if (is_enable_thread_bind_core_) {
    MS_LOG(WARNING)
      << "Thread bind core has already been enabled and will be implemented based on the first binding policy.";
    return;
  }

  if (!actor_thread_fix_bind.empty()) {
    actor_thread_fix_bind_ = actor_thread_fix_bind;
  } else {
    const auto &env_actor_thread_fix_bind = runtime::GetRuntimeConfigValue(runtime::kRuntimeActorThreadFixBind);
    actor_thread_fix_bind_ = env_actor_thread_fix_bind;
  }

  process_bind_core_policy_ = bind_core_policy;
  is_enable_thread_bind_core_ = true;
}

bool ThreadBindCore::parse_thread_bind_core_policy(const kBindCoreModule &module_name) {
  auto it = thread_bind_core_policy_.find(module_name);
  if (it != thread_bind_core_policy_.end()) {
    return true;
  }

  auto runtime_conf_instance = RuntimeConf::GetInstance();
  MS_EXCEPTION_IF_NULL(runtime_conf_instance);
  bool enable_batch_launch_kernel = runtime_conf_instance->IsKernelLaunchGroupConfigured();
  uint32_t group_launch_thread_num = 0;
  if (enable_batch_launch_kernel) {
    group_launch_thread_num = runtime_conf_instance->group_launch_thread_num();
  }
  // Record remaining core excluding mian, runtime, pynative module.
  std::vector<int> remaining_core_list;

  thread_bind_core_policy_[kBindCoreModule::kMAIN] = process_bind_core_policy_[kMainThread];
  thread_bind_core_policy_[kBindCoreModule::kRUNTIME] = process_bind_core_policy_[kRunTimeThread];
  thread_bind_core_policy_[kBindCoreModule::kPYNATIVE] = process_bind_core_policy_[kPynativeThread];

  remaining_core_list = process_bind_core_policy_[kDataThread];

  // Allocate core resource for minddata, runtime batch launch.
  if (SizeToUint(remaining_core_list.size()) < group_launch_thread_num) {
    MS_LOG(WARNING) << "The current process does not have enough thread resources for CPU affinity binding, thread "
                       "bind core function is not enabled.";
    thread_bind_core_policy_.clear();
    return false;
  }
  thread_bind_core_policy_[kBindCoreModule::kMINDDATA] =
    std::vector<int>(remaining_core_list.begin(), remaining_core_list.end() - group_launch_thread_num);
  thread_bind_core_policy_[kBindCoreModule::kBATCHLAUNCH] =
    std::vector<int>(remaining_core_list.end() - group_launch_thread_num, remaining_core_list.end());

  return true;
}

std::vector<int> ThreadBindCore::get_thread_bind_core_list(const kBindCoreModule &module_name) {
  if (!is_enable_thread_bind_core_) {
    MS_LOG(EXCEPTION) << "Cannot get thread bind core list for this module: " << module_name
                      << ", if 'set_cpu_affinity' is turned off.";
  }
  std::lock_guard<std::mutex> locker(mtx_);

  auto status_it = thread_bind_core_status_.find(module_name);
  if (status_it != thread_bind_core_status_.end() && status_it->second) {
    MS_LOG(INFO) << "This module: " << module_name
                 << " has already been assigned a bind core list: " << thread_bind_core_policy_[module_name];
    return thread_bind_core_policy_[module_name];
  }

  bool res = parse_thread_bind_core_policy(module_name);
  if (!res) {
    return {};
  }

  auto it = thread_bind_core_policy_.find(module_name);
  if (it == thread_bind_core_policy_.end()) {
    MS_LOG(INFO) << "This module: " << module_name << " has no bind core policy to be assigned.";
    return {};
  }
  MS_LOG(INFO) << "This module: " << module_name << " is assigned a bind core list: " << it->second;
  thread_bind_core_status_[module_name] = true;
  return it->second;
}

void ThreadBindCore::bind_thread_core(const std::vector<int> &cpu_list) {
#if defined(BIND_CORE)
  if (!is_enable_thread_bind_core_) {
    MS_LOG(EXCEPTION) << "Cannot bind core to this thread if 'set_cpu_affinity' is turned off.";
  }

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);

  for (const auto &cpu_id : cpu_list) {
    CPU_SET(static_cast<size_t>(cpu_id), &cpuset);
  }

  int result = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
  if (result != 0) {
    MS_LOG(ERROR) << "Failed to bind thread to core list.";
    return;
  }
  MS_LOG(INFO) << "Enable bind core to core list: " << cpu_list;
#endif  // BIND_CORE
}

void ThreadBindCore::bind_thread_core(const std::vector<int> &cpu_list, int64_t thread_or_process_id, bool is_thread) {
#if defined(BIND_CORE)
  if (!is_enable_thread_bind_core_) {
    MS_LOG(EXCEPTION) << "Cannot bind core to this thread if 'set_cpu_affinity' is turned off.";
  }

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);

  for (const auto &cpu_id : cpu_list) {
    CPU_SET(static_cast<size_t>(cpu_id), &cpuset);
  }

  if (is_thread) {
    MS_LOG(INFO) << "Start binding thread [" << thread_or_process_id << "] to cores";
    int result = pthread_setaffinity_np(static_cast<pthread_t>(thread_or_process_id), sizeof(cpu_set_t), &cpuset);
    if (result != 0) {
      MS_LOG(ERROR) << "Failed to bind thread to core list.";
      return;
    }
    MS_LOG(INFO) << "Enable bind thread [" << thread_or_process_id << "] to core list: " << cpu_list;
  } else {
    MS_LOG(INFO) << "Start binding process [" << thread_or_process_id << "] to cores";
    int result = sched_setaffinity(static_cast<pid_t>(thread_or_process_id), sizeof(cpu_set_t), &cpuset);
    if (result != 0) {
      MS_LOG(ERROR) << "Failed to bind process to core list.";
      return;
    }
    MS_LOG(INFO) << "Enable bind process [" << thread_or_process_id << "] to core list: " << cpu_list;
  }
#endif  // BIND_CORE
}

bool ThreadBindCore::unbind_thread_core(const std::string &thread_name) { return true; }
}  // namespace runtime
}  // namespace mindspore
