/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_UTILS_H_
#define MINDSPORE_CCSRC_MINDDATA_UTILS_H_

#include <string>
#include <vector>

#include "minddata/dataset/util/status.h"
#include "minddata/dataset/util/log_adapter.h"
#include "include/runtime/utils/runtime_conf/thread_bind_core.h"

namespace mindspore {
namespace dataset {

inline void BindThreadCoreForMindDataOp(std::string name) {
  std::string prefix_word = "dataset::";
  if (name.rfind(prefix_word, 0) != 0) {
    name = prefix_word + name;
  }
  auto &bind_core_manager = runtime::ThreadBindCore::GetInstance();
  auto env_data_reserved = std::getenv("CONFIG_BIND_MINDDATA_LIST");
  if (!bind_core_manager.is_enable_thread_bind_core_ && env_data_reserved == nullptr) {
    MS_LOG(INFO) << "[" << name << "]: Core binding is not enabled.";
    return;
  } else {
    MS_LOG(INFO) << "[" << name << "]: Start core binding.";
  }

  if (bind_core_manager.is_enable_thread_bind_core_) {
    const auto &core_list = bind_core_manager.get_thread_bind_core_list(runtime::kBindCoreModule::kMINDDATA);
    if (core_list.empty()) {
      MS_LOG(INFO) << "[" << name
                   << "]: No core list assigned, make sure set_cpu_affinity method is configured correctly.";
      return;
    } else {
      bind_core_manager.bind_thread_core(core_list);
      MS_LOG(INFO) << "[" << name << "]: Current thread has been bound to core list.";
    }
  } else if (env_data_reserved != nullptr) {
#ifdef __linux__
    std::vector<int> cpu_list;
    std::stringstream ss(env_data_reserved);
    std::string item;

    while (std::getline(ss, item, ',')) {
      cpu_list.push_back(std::stoi(item));
    }

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    for (const auto &cpu_id : cpu_list) {
      CPU_SET(static_cast<size_t>(cpu_id), &cpuset);
    }

    int result = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (result != 0) {
      MS_LOG(ERROR) << "[" << name << "]: Current thread failed to core list:" << cpu_list;
      return;
    }

    MS_LOG(INFO) << "[" << name << "]: Current thread has been bound to core list:" << cpu_list;
#endif
  }
}

inline void BindThreadCoreForMindDataOp(std::string name, int64_t id, bool is_thread = false) {
  std::string prefix_word = "dataset::";
  if (name.rfind(prefix_word, 0) != 0) {
    name = prefix_word + name;
  }
  auto &bind_core_manager = runtime::ThreadBindCore::GetInstance();
  if (!bind_core_manager.is_enable_thread_bind_core_) {
    MS_LOG(INFO) << "[" << name << "]: Core binding is not enabled.";
    return;
  } else {
    MS_LOG(INFO) << "[" << name << "]: Start core binding.";
  }

  const auto &core_list = bind_core_manager.get_thread_bind_core_list(runtime::kBindCoreModule::kMINDDATA);
  if (core_list.empty()) {
    MS_LOG(INFO) << "[" << name
                 << "]: No core list assigned, make sure set_cpu_affinity method is configured correctly.";
    return;
  } else {
    bind_core_manager.bind_thread_core(core_list, id, is_thread);
    if (is_thread) {
      MS_LOG(INFO) << "[" << name << "]: Current thread [" << id << "] has been bound to core list.";
    } else {
      MS_LOG(INFO) << "[" << name << "]: Current process [" << id << "] has been bound to core list.";
    }
  }
}
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_UTILS_H_
