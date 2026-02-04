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

#ifndef MINDSPORE_CCSRC_INCLUDE_RUNTIME_UTILS_RUNTIME_CONF_RUNTIME_CONF_H_
#define MINDSPORE_CCSRC_INCLUDE_RUNTIME_UTILS_RUNTIME_CONF_RUNTIME_CONF_H_

#include <memory>
#include <string>
#include <map>
#include <vector>
#include <utility>
#include "utils/ms_context.h"
#include "runtime/utils/visible.h"
#include "include/runtime/utils/runtime_conf/thread_bind_core.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace runtime {

const char kMemoryConf[] = "MemoryConf";
const char kDispatchThreadsNumConf[] = "DispatchThreadsNumConf";
const char kOpThreadsNumConf[] = "OpThreadsNumConf";
const char kLaunchBlocking[] = "launch_blocking";
const char kThreadBindCore[] = "thread_bind_core";
const char kKernelLaunchGroupConf[] = "KernelLaunchGroupConf";
const char kSimulationLevelKey[] = "MS_SIMULATION_LEVEL";
constexpr auto kMsDevLaunchBlocking = "MS_DEV_LAUNCH_BLOCKING";

class RUNTIME_UTILS_EXPORT RuntimeConf {
 public:
  RuntimeConf();
  ~RuntimeConf();
  RuntimeConf(const RuntimeConf &) = delete;
  RuntimeConf &operator=(const RuntimeConf &) = delete;
  static std::shared_ptr<RuntimeConf> GetInstance();

  void set_launch_blocking() {
    conf_status_[kLaunchBlocking] = true;
    launch_blocking_ = true;
  }
  bool launch_blocking() {
    static bool launch_blocking = common::GetEnv(kMsDevLaunchBlocking) == "1";
    if (launch_blocking) {
      return true;
    }
    if (is_sim_level_three_) {
      MS_LOG(INFO) << "Run in simulation level 3, always run in blocking";
      return true;
    }
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    return launch_blocking_ || ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE);
  }
  bool IsSetLaunchBlocking() { return conf_status_.count(kLaunchBlocking); }

  void set_dispatch_threads_num(uint32_t threads_num) {
    dispatch_threads_num_ = threads_num;
    conf_status_[kDispatchThreadsNumConf] = true;
  }
  uint32_t dispatch_threads_num() { return dispatch_threads_num_; }
  bool IsDispatchThreadsNumConfigured() { return conf_status_.count(kDispatchThreadsNumConf); }

  void set_kernel_launch_group(uint32_t group_launch_thread_num, uint32_t kernel_group_num) {
    group_launch_thread_num_ = group_launch_thread_num;
    kernel_group_num_ = kernel_group_num;
    conf_status_[kKernelLaunchGroupConf] = true;
  }
  bool IsKernelLaunchGroupConfigured() { return conf_status_.count(kKernelLaunchGroupConf); }
  uint32_t group_launch_thread_num() { return group_launch_thread_num_; }
  uint32_t kernel_group_num() { return kernel_group_num_; }

  void set_op_threads_num(uint32_t threads_num) {
    op_threads_num_ = threads_num;
    conf_status_[kOpThreadsNumConf] = true;
  }
  uint32_t op_threads_num() { return op_threads_num_; }
  bool IsOpThreadsNumConfigured() { return conf_status_.count(kOpThreadsNumConf); }

  void set_memory(float mem_init_size, float mem_block_increase_size, float mem_max_size, int mem_optimize_level,
                  float mem_huge_page_reserve_size) {
    mem_init_size_ = mem_init_size;
    mem_block_increase_size_ = mem_block_increase_size;
    mem_max_size_ = mem_max_size;
    mem_optimize_level_ = mem_optimize_level;
    mem_huge_page_reserve_size_ = mem_huge_page_reserve_size;
    conf_status_[kMemoryConf] = true;
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    ms_context->set_param<int>(MS_CTX_MEMORY_OPTIMIZE_LEVEL, mem_optimize_level);
  }
  bool IsMemoryConfigured() { return conf_status_.count(kMemoryConf); }
  float mem_init_size() { return IsMemoryConfigured() ? mem_init_size_ : 0; }
  float mem_block_increase_size() {
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    return IsMemoryConfigured() ? mem_block_increase_size_ : ms_context->get_param<float>(MS_CTX_MEMPOOL_BLOCK_SIZE);
  }
  float mem_max_size() {
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    return IsMemoryConfigured() ? mem_max_size_ : ms_context->get_param<float>(MS_CTX_MAX_DEVICE_MEMORY);
  }
  int mem_optimize_level() {
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    return IsMemoryConfigured() ? mem_optimize_level_ : ms_context->get_param<int>(MS_CTX_MEMORY_OPTIMIZE_LEVEL);
  }

  float mem_huge_page_reserve_size() { return mem_huge_page_reserve_size_; }

  // Methods for Thread bind core
  bool IsThreadBindCoreConfigured() { return conf_status_.count(kThreadBindCore); }

  void EnableResLimit() { enable_res_limit_ = true; }
  bool IsEnableResLimit() { return enable_res_limit_; }

  void SetThreadBindCoreConfigured() { conf_status_[kThreadBindCore] = true; }

  void BindThreadCore(const ModuleBindCorePolicy &module_bind_core_strategy) {
    conf_status_[kThreadBindCore] = true;
    ThreadBindCore::GetInstance().enable_thread_bind_core(module_bind_core_strategy);
  }

  void SetEnableKernelLaunchCapture(bool enable_capture_graph, const std::vector<std::string> &op_capture_skip = {}) {
    enable_capture_graph_ = enable_capture_graph;
    op_capture_skip_ = op_capture_skip;
  }

  bool GetEnableKernelLaunchCapture() { return enable_capture_graph_; }

  const std::vector<std::string> &GetNotCaptureOpList() const { return op_capture_skip_; }

 private:
  static std::shared_ptr<RuntimeConf> inst_context_;

  bool launch_blocking_;
  bool is_sim_level_three_ = (common::GetEnv(kSimulationLevelKey) == "3");
  uint32_t dispatch_threads_num_;
  uint32_t op_threads_num_;
  uint32_t group_launch_thread_num_;
  uint32_t kernel_group_num_;

  float mem_init_size_;
  float mem_block_increase_size_;
  float mem_max_size_;
  int mem_optimize_level_;
  float mem_huge_page_reserve_size_;
  bool enable_capture_graph_;
  bool enable_res_limit_{false};
  std::vector<std::string> op_capture_skip_{};
  std::map<std::string, bool> conf_status_;
};

RUNTIME_UTILS_EXPORT void ComputeThreadNums(size_t *actor_thread_num, size_t *actor_and_kernel_thread_num);
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_RUNTIME_UTILS_RUNTIME_CONF_RUNTIME_CONF_H_
