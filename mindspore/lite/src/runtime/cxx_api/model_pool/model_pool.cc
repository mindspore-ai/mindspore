/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "src/runtime/cxx_api/model_pool/model_pool.h"
#include <unistd.h>
#include <future>
#include <algorithm>
#include "src/common/log_adapter.h"
#include "include/lite_types.h"
#include "src/runtime/inner_allocator.h"
#include "src/common/file_utils.h"
#include "src/runtime/pack_weight_manager.h"
#include "src/runtime/numa_adapter.h"
#include "src/common/common.h"
namespace mindspore {
namespace {
constexpr int kNumDeviceInfo = 2;
constexpr int kNumIndex = 2;
constexpr int kNumMaxTaskQueueSize = 1000;
constexpr int kNumPhysicalCoreThreshold = 16;
constexpr int kDefaultWorkerNumPerPhysicalCpu = 2;
constexpr int kDefaultThreadsNum = 8;
constexpr int kInvalidNumaId = -1;
constexpr int kNumDefaultInterOpParallel = 4;
constexpr int kCoreNumTimes = 5;

std::vector<int> ParseCpusetFile(int *percentage) {
  std::vector<int> cpu_core = {};
  std::ifstream infile("/sys/fs/cgroup/cpuset/cpuset.cpus", std::ios::in);
  std::string line;
  const char ch1 = ',';
  const char ch2 = '-';
  while (getline(infile, line, ch1)) {
    if (line[line.size() - 1] == '\n') {
      line = line.substr(0, line.size() - 1);
    }
    auto it = find(line.begin(), line.end(), ch2);
    if (it != line.end()) {
      auto begin_s = line.substr(0, it - line.begin());
      auto end_s = line.substr(it - line.begin() + 1, line.size() - (it - line.begin()));
      int begin = std::atoi(begin_s.c_str());
      int end = std::atoi(end_s.c_str());
      for (int i = begin; i <= end; i++) {
        cpu_core.push_back(i);
      }
    } else {
      cpu_core.push_back(std::atoi(line.c_str()));
    }
  }
  std::ifstream quota_file("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", std::ios::in);
  std::string quota_line;
  getline(quota_file, quota_line);
  if (quota_line[quota_line.size() - 1] == '\n') {
    quota_line = quota_line.substr(0, quota_line.size() - 1);
  }
  auto quota = std::atoi(quota_line.c_str());
  if (quota == -1) {
    *percentage = -1;
    return cpu_core;
  }
  std::ifstream period_file("/sys/fs/cgroup/cpu/cpu.cfs_period_us", std::ios::in);
  std::string period_line;
  getline(period_file, period_line);
  if (period_line[period_line.size() - 1] == '\n') {
    period_line = period_line.substr(0, period_line.size() - 1);
  }
  auto period = std::atoi(period_line.c_str());
  if (period == 0) {
    return {};
  }
  *percentage = quota / period;
  return cpu_core;
}

Status DistinguishPhysicalAndLogical(std::vector<int> *physical_list, std::vector<int> *logical_list) {
  // physical id <=> one physical cpu core id
  std::unordered_map<int, std::vector<int>> ids;
  std::ifstream infile("/proc/cpuinfo", std::ios::in);
  std::string line;
  std::vector<int> processor_ids = {};
  std::vector<int> physical_ids = {};
  std::vector<int> core_ids = {};
  while (getline(infile, line)) {
    auto line_size = line.size();
    if (line.find("processor") != std::string::npos) {
      auto it = line.find(": ") + kNumIndex;
      processor_ids.push_back(std::atoi(line.substr(it, line_size - 1).c_str()));
    }
    if (line.find("physical id") != std::string::npos) {
      auto it = line.find(": ") + kNumIndex;
      physical_ids.push_back(std::atoi(line.substr(it, line_size - 1).c_str()));
    }
    if (line.find("core id") != std::string::npos) {
      auto it = line.find(": ") + kNumIndex;
      core_ids.push_back(std::atoi(line.substr(it, line_size - 1).c_str()));
    }
  }
  if (core_ids.empty() && physical_ids.empty()) {
    MS_LOG(DEBUG) << "All cores are physical cores.";
    for (size_t i = 0; i < processor_ids.size(); i++) {
      physical_list->push_back(processor_ids[i]);
    }
    return kSuccess;
  }
  if (core_ids.size() == physical_ids.size() && physical_ids.size() == processor_ids.size()) {
    for (size_t i = 0; i < processor_ids.size(); i++) {
      if (ids.find(physical_ids[i]) == ids.end()) {
        std::vector<int> core_id_list = {core_ids[i]};
        ids.insert(std::make_pair(physical_ids[i], core_id_list));
        physical_list->push_back(processor_ids[i]);
        continue;
      }
      if (find(ids[physical_ids[i]].begin(), ids[physical_ids[i]].end(), core_ids[i]) == ids[physical_ids[i]].end()) {
        ids[physical_ids[i]].push_back(core_ids[i]);
        physical_list->push_back(processor_ids[i]);
        continue;
      } else {
        logical_list->push_back(processor_ids[i]);
      }
    }
  }
  return kSuccess;
}
}  // namespace

int ModelPool::GetDefaultThreadNum(int worker_num) {
  int default_thread_num = -1;
  if (can_use_core_num_ <= kNumPhysicalCoreThreshold) {
    default_thread_num = can_use_core_num_ >= kDefaultWorkerNumPerPhysicalCpu
                           ? can_use_core_num_ / kDefaultWorkerNumPerPhysicalCpu
                           : can_use_core_num_;
  } else {
    default_thread_num = kDefaultThreadsNum;
  }
  if (worker_num * default_thread_num > can_use_core_num_) {
    default_thread_num = can_use_core_num_ >= worker_num ? can_use_core_num_ / worker_num : can_use_core_num_;
  }
  return default_thread_num;
}

Status ModelPool::DistinguishPhysicalAndLogicalByNuma(const std::vector<int> &physical_core_list,
                                                      const std::vector<int> &logical_core_list) {
  std::vector<std::vector<int>> all_numa_core_list;
  auto numa_num = numa::NUMAAdapter::GetInstance()->NodesNum();
  for (int i = 0; i < numa_num; i++) {
    std::vector<int> numa_cpu_list = numa::NUMAAdapter::GetInstance()->GetCPUList(i);
    if (numa_cpu_list.empty()) {
      MS_LOG(ERROR) << i << "-th numa node does not exist";
      return kLiteError;
    }
    all_numa_core_list.push_back(numa_cpu_list);
  }
  MS_CHECK_TRUE_MSG(!all_numa_core_list.empty(), kLiteError, "numa core list is empty.");
  for (auto one_numa_list : all_numa_core_list) {
    MS_CHECK_TRUE_MSG(!one_numa_list.empty(), kLiteError, "one numa core list is empty.");
    std::vector<int> physical_cores;
    std::vector<int> logical_cores;
    for (auto core_id : one_numa_list) {
      if (find(physical_core_list.begin(), physical_core_list.end(), core_id) != physical_core_list.end()) {
        physical_cores.push_back(core_id);
      } else if (find(logical_core_list.begin(), logical_core_list.end(), core_id) != logical_core_list.end()) {
        logical_cores.push_back(core_id);
      } else {
        MS_LOG(ERROR) << "core id not belong physical/logical core id.";
        return kLiteError;
      }
    }
    numa_physical_cores_.push_back(physical_cores);
    numa_logical_cores_.push_back(logical_cores);
  }
  return kSuccess;
}

/*
 * bind numa:
 *      worker 1    worker 2    worker 3    worker 4
 *         |           |           |          |
 *       numa 0  ->  numa 1  ->  numa 0  ->  numa 1
 *
 *  core num: 16          worker num: 4          thread num: 4
 *  physical core id: 0,2,4,6,8,10,12,14    logic core id: 1,3,5,7,9,11,13,15
 *  numa 0: 0,1,2,3,4,5,6,7    numa 1: 8,9,10,11,12,13,14,15
 *
 * result of bind numa:
 *                physical                 logic
 *  numa 0:   worker1: 0,2,4,6       worker3: 1,3,5,7
 *  numa 1:   worker2: 8,10,12,14    worker4: 9,11,13,15
 *
 * */
Status ModelPool::SetNumaBindStrategy(std::vector<std::vector<int>> *all_worker_bind_list,
                                      std::vector<int> *numa_node_id, int thread_num) {
  if (MS_UNLIKELY(thread_num == 0)) {
    MS_LOG(ERROR) << "thread num is zero.";
    return kLiteError;
  }
  if (numa_physical_cores_.empty()) {
    MS_LOG(ERROR) << "numa physical cores is empty.";
    return kLiteError;
  }
  if (numa_physical_cores_.front().size() < static_cast<size_t>(thread_num)) {
    MS_LOG(ERROR) << "thread num more than physical core num. one numa physical core size: "
                  << numa_physical_cores_.front().size();
    return kLiteError;
  }
  std::vector<int> physical_index(numa_physical_cores_.size(), 0);  // numa node size
  std::vector<int> logical_index(numa_logical_cores_.size(), 0);
  size_t bind_numa_id = 0;
  for (size_t i = 0; i < workers_num_; i++) {
    if (bind_numa_id >= numa_physical_cores_.size()) {
      used_numa_node_num_ = bind_numa_id;
      bind_numa_id = 0;
    }
    std::vector<int> worker_bind_list;
    if (physical_index[bind_numa_id] + static_cast<size_t>(thread_num) <= numa_physical_cores_[bind_numa_id].size()) {
      worker_bind_list.insert(worker_bind_list.begin(),
                              numa_physical_cores_[bind_numa_id].begin() + physical_index[bind_numa_id],
                              numa_physical_cores_[bind_numa_id].begin() + physical_index[bind_numa_id] + thread_num);
      physical_index[bind_numa_id] += thread_num;
    } else if (logical_index[bind_numa_id] + static_cast<size_t>(thread_num) <=
               numa_logical_cores_[bind_numa_id].size()) {
      worker_bind_list.insert(worker_bind_list.begin(),
                              numa_logical_cores_[bind_numa_id].begin() + logical_index[bind_numa_id],
                              numa_logical_cores_[bind_numa_id].begin() + logical_index[bind_numa_id] + thread_num);
      logical_index[bind_numa_id] += thread_num;
    } else {
      MS_LOG(ERROR) << "In the core-bound scenario, the product of the number of threads and the number of workers "
                       "should not exceed the number of cores of the machine. Please check the parameter settings: \n"
                    << "workers num: " << workers_num_ << " | thread num: " << thread_num
                    << " | numa physical cores: " << numa_physical_cores_
                    << " | numa logical cores: " << numa_logical_cores_;
      return kLiteError;
    }
    all_worker_bind_list->push_back(worker_bind_list);
    numa_node_id->push_back(bind_numa_id);
    bind_numa_id++;
  }
  used_numa_node_num_ = used_numa_node_num_ != 0 ? used_numa_node_num_ : bind_numa_id;
  return kSuccess;
}

Status ModelPool::SetBindStrategy(std::vector<std::vector<int>> *all_model_bind_list, std::vector<int> *numa_node_id,
                                  int thread_num) {
  if (thread_num == 0) {
    MS_LOG(ERROR) << "thread num is zero.";
    return kLiteError;
  }
  std::vector<int> physical_core_list;
  std::vector<int> logical_core_list;
  auto status = DistinguishPhysicalAndLogical(&physical_core_list, &logical_core_list);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "distinguish physical and logical failed.";
    return kLiteError;
  }
  std::vector<int> all_core_list = {};
  if (!can_use_all_physical_core_) {
    int percentage;
    std::vector<int> can_use_core_list = ParseCpusetFile(&percentage);
    if (percentage != -1) {
      return kLiteFileError;
    }
    std::sort(physical_core_list.begin(), physical_core_list.end());
    std::sort(logical_core_list.begin(), logical_core_list.end());
    std::vector<int> can_use_physical_list = {};
    std::vector<int> can_use_logical_list = {};
    std::set_intersection(physical_core_list.begin(), physical_core_list.end(), can_use_core_list.begin(),
                          can_use_core_list.end(), std::back_inserter(can_use_physical_list));
    std::set_intersection(logical_core_list.begin(), logical_core_list.end(), can_use_core_list.begin(),
                          can_use_core_list.end(), std::back_inserter(can_use_logical_list));
    all_core_list.insert(all_core_list.end(), can_use_physical_list.begin(), can_use_physical_list.end());
    all_core_list.insert(all_core_list.end(), can_use_logical_list.begin(), can_use_logical_list.end());
  } else {
    all_core_list = physical_core_list;
    all_core_list.insert(all_core_list.end(), logical_core_list.begin(), logical_core_list.end());
  }
  size_t core_id = 0;
  for (size_t i = 0; i < workers_num_; i++) {
    std::vector<int> bind_id;
    for (int j = 0; j < thread_num; j++) {
      if (core_id >= all_core_list.size()) {
        core_id = 0;
      }
      bind_id.push_back(all_core_list[core_id]);
      core_id++;
    }
    all_model_bind_list->push_back(bind_id);
    numa_node_id->push_back(kInvalidNumaId);
  }
  return kSuccess;
}

Status ModelPool::SetDefaultOptimalModelNum(int thread_num) {
  if (thread_num <= 0) {
    MS_LOG(ERROR) << "the number of threads set in the context is less than 1.";
    return kLiteError;
  }
  if (!can_use_all_physical_core_) {
    workers_num_ = can_use_core_num_ > thread_num ? can_use_core_num_ / thread_num : 1;
  } else if (numa_available_) {
    // now only supports the same number of cores per numa node
    // do not use if there are extra cores
    auto worker_num = 0;
    for (auto one_numa_physical : numa_physical_cores_) {
      worker_num += (one_numa_physical.size() / thread_num);
    }
    for (auto one_numa_logical : numa_logical_cores_) {
      worker_num += (one_numa_logical.size() / thread_num);
    }
    workers_num_ = worker_num;
  } else {
    // each worker binds all available cores in order
    workers_num_ = lite::GetCoreNum() / thread_num;
  }
  return kSuccess;
}

std::shared_ptr<mindspore::Context> ModelPool::GetDefaultContext() {
  MS_LOG(DEBUG) << "use default config.";
  auto context = std::make_shared<Context>();
  if (context == nullptr) {
    MS_LOG(ERROR) << "create default context failed.";
    return nullptr;
  }
  auto thread_num = GetDefaultThreadNum();
  if (thread_num == 0) {
    MS_LOG(ERROR) << "computer thread num failed.";
    return nullptr;
  }
  context->SetThreadNum(thread_num);
  if (thread_num > kNumDefaultInterOpParallel) {
    context->SetInterOpParallelNum(kNumDefaultInterOpParallel);
  } else {
    context->SetInterOpParallelNum(1);
  }
  auto &device_list = context->MutableDeviceInfo();
  auto device_info = std::make_shared<CPUDeviceInfo>();
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "device_info is nullptr.";
    return nullptr;
  }
  device_info->SetEnableFP16(false);
  device_list.push_back(device_info);
  return context;
}

Status ModelPool::CheckAffinityCoreList(const std::shared_ptr<RunnerConfig> &runner_config) {
  auto context = runner_config->GetContext();
  auto all_bind_core_list = context->GetThreadAffinityCoreList();
  auto worker_num = runner_config->GetWorkersNum();
  if (worker_num != 0 && !all_bind_core_list.empty() &&
      static_cast<int>(all_bind_core_list.size()) != context->GetThreadNum() * worker_num) {
    MS_LOG(ERROR) << "user set core list size != " << context->GetThreadNum() * worker_num
                  << " If the user sets the Bind core list, the size must be equal to the number of threads "
                     "multiplied by the number of workers";
    return kLiteError;
  }
  if (worker_num != 0 && !all_bind_core_list.empty() &&
      static_cast<int>(all_bind_core_list.size()) == context->GetThreadNum() * worker_num) {
    // Use the core id set by the user. Currently, this function can only be enabled when the worker num is not 0.
    auto max_core_id = lite::GetCoreNum();
    for (size_t i = 0; i < all_bind_core_list.size(); i++) {
      if (all_bind_core_list[i] < 0 || all_bind_core_list[i] >= max_core_id) {
        MS_LOG(ERROR) << "Please set correct core id, core id should be less than " << max_core_id
                      << "and greater than 0";
        return kLiteError;
      }
    }
    is_user_core_list_ = true;
  }
  return kSuccess;
}

Status ModelPool::CheckThreadNum(const std::shared_ptr<RunnerConfig> &runner_config) {
  auto context = runner_config->GetContext();
  auto thread_num = context->GetThreadNum();
  if (thread_num < 0) {
    MS_LOG(ERROR) << "Invalid thread num " << thread_num;
    return kLiteError;
  }

  int core_num = static_cast<int>(std::max<size_t>(1, std::thread::hardware_concurrency()));
  int threshold_thread_num = kCoreNumTimes * core_num;
  if (thread_num > threshold_thread_num) {
    MS_LOG(WARNING) << "Thread num: " << thread_num << " is more than 5 times core num: " << threshold_thread_num
                    << ", change it to 5 times core num. Please check whether Thread num is reasonable.";
    context->SetThreadNum(threshold_thread_num);
  }

  if (thread_num == 0) {
    // Defaults are automatically adjusted based on computer performance
    auto default_thread_num = GetDefaultThreadNum(runner_config->GetWorkersNum());
    if (default_thread_num == 0) {
      MS_LOG(ERROR) << "computer thread num failed, worker num: " << runner_config->GetWorkersNum()
                    << " | can use core num: " << can_use_core_num_;
      return kLiteError;
    }
    context->SetThreadNum(default_thread_num);
  }
  if (context->GetThreadNum() > can_use_core_num_) {
    MS_LOG(WARNING) << "thread num[" << context->GetThreadNum() << "] more than core num[" << can_use_core_num_ << "]";
    if (context->GetThreadAffinityMode() != BindMode::Power_NoBind || !context->GetThreadAffinityCoreList().empty()) {
      MS_LOG(ERROR) << "thread num more than core num, can not bind cpu core.";
      return kLiteError;
    }
  }
  return kSuccess;
}

std::shared_ptr<Context> ModelPool::GetUserDefineContext(const std::shared_ptr<RunnerConfig> &runner_config) {
  auto context = runner_config->GetContext();
  MS_CHECK_TRUE_MSG(context != nullptr, nullptr, "user set config context nullptr.");
  auto status = CheckThreadNum(runner_config);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "user set thread num failed.";
    return nullptr;
  }
  status = CheckAffinityCoreList(runner_config);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "user set core list failed.";
    return nullptr;
  }
  auto device_list = context->MutableDeviceInfo();
  if (device_list.size() > kNumDeviceInfo) {
    MS_LOG(ERROR) << "model pool only support device CPU or GPU.";
    return nullptr;
  }
  for (size_t i = 0; i < device_list.size(); i++) {
    auto device = device_list[i];
    if (device->GetDeviceType() != kCPU && device->GetDeviceType() != kGPU && device->GetDeviceType() != kAscend) {
      MS_LOG(ERROR) << "model pool only support cpu or gpu or ascend type.";
      return nullptr;
    }
    if (device->GetDeviceType() == kGPU && device_list.size() == kNumDeviceInfo) {
      if (context->GetInterOpParallelNum() == 0) {
        context->SetInterOpParallelNum(1);  // do not use InterOpParallel
      }
      return context;
    } else if (device->GetDeviceType() == kCPU) {
      if (context->GetInterOpParallelNum() == 0) {
        int32_t inter_op_parallel =
          context->GetThreadNum() >= kNumDefaultInterOpParallel ? kNumDefaultInterOpParallel : 1;
        context->SetInterOpParallelNum(inter_op_parallel);
      }
    } else if (device->GetDeviceType() == kAscend) {
      if (context->GetInterOpParallelNum() == 0) {
        context->SetInterOpParallelNum(1);  // do not use InterOpParallel
      }
      return context;
    } else {
      MS_LOG(ERROR) << "context is invalid; If you want run in GPU, you must set gpu device first, and then set cpu "
                       "device";
      return nullptr;
    }
  }
  return context;
}

std::shared_ptr<Context> ModelPool::GetInitContext(const std::shared_ptr<RunnerConfig> &runner_config) {
  std::shared_ptr<mindspore::Context> context = nullptr;
  if (runner_config != nullptr && runner_config->GetContext() != nullptr) {
    use_numa_bind_mode_ = numa_available_ && runner_config->GetContext()->GetThreadAffinityMode() == lite::HIGHER_CPU;
    context = GetUserDefineContext(runner_config);
  } else {
    context = GetDefaultContext();
  }
  if (context == nullptr) {
    MS_LOG(ERROR) << "Init context failed. status=" << kLiteNullptr;
    return nullptr;
  }
  if (!bind_core_available_) {
    MS_LOG(WARNING) << "Cannot use all hardware resources, does not support core binding.";
    context->SetThreadAffinity(mindspore::Power_NoBind);
    std::vector<int> empty = {};
    context->SetThreadAffinity(empty);
  }
  return context;
}

Status ModelPool::SetModelBindMode(std::vector<std::vector<int>> *all_worker_bind_list, std::vector<int> *numa_node_id,
                                   std::shared_ptr<Context> model_context) {
  Status status;
  if (numa_available_) {
    status = SetNumaBindStrategy(all_worker_bind_list, numa_node_id, static_cast<int>(model_context->GetThreadNum()));
  } else {
    status = SetBindStrategy(all_worker_bind_list, numa_node_id, static_cast<int>(model_context->GetThreadNum()));
  }
  if (status != kSuccess) {
    MS_LOG(ERROR) << "Set  bind strategy failed.";
    return kLiteError;
  }
  return kSuccess;
}

Status ModelPool::SetWorkersNum(const std::shared_ptr<RunnerConfig> &runner_config,
                                const std::shared_ptr<Context> &context) {
  if ((runner_config != nullptr && runner_config->GetWorkersNum() == 0) || runner_config == nullptr) {
    // the user does not define the number of models, the default optimal number of models is used
    auto status = SetDefaultOptimalModelNum(context->GetThreadNum());
    if (status != kSuccess) {
      MS_LOG(ERROR) << "SetDefaultOptimalModelNum failed.";
      return kLiteError;
    }
  } else if (runner_config != nullptr && runner_config->GetWorkersNum() > 0 &&
             runner_config->GetWorkersNum() <= can_use_core_num_ * kCoreNumTimes) {
    // User defined number of models
    workers_num_ = runner_config->GetWorkersNum();
  } else {
    MS_LOG(ERROR) << "user set worker num: " << runner_config->GetWorkersNum() << "is invalid";
    return kLiteError;
  }
  if (workers_num_ == 0) {
    MS_LOG(ERROR) << "worker num is zero.";
    return kLiteError;
  }
  return kSuccess;
}

Status ModelPool::SetWorkersNumaId(std::vector<int> *numa_node_id) {
  if (!numa_available_) {
    MS_LOG(WARNING) << "numa is not available.";
    numa_node_id->resize(workers_num_, kInvalidNumaId);
  } else {
    size_t numa_id = 0;
    for (size_t i = 0; i < workers_num_; i++) {
      if (numa_id == numa_node_num_) {
        numa_id = 0;
      }
      numa_node_id->push_back(numa_id);
      numa_id++;
    }
    used_numa_node_num_ = workers_num_ < numa_node_num_ ? workers_num_ : numa_node_num_;
  }
  return kSuccess;
}

ModelPoolConfig ModelPool::CreateGpuModelPoolConfig(const std::shared_ptr<RunnerConfig> &runner_config,
                                                    const std::shared_ptr<Context> &init_context) {
  ModelPoolConfig model_pool_gpu_config;
  auto worker_config = std::make_shared<WorkerConfig>();
  if (worker_config == nullptr) {
    MS_LOG(ERROR) << "new worker config failed.";
    return {};
  }
  if (runner_config != nullptr) {
    worker_config->config_info = runner_config->GetConfigInfo();
    worker_config->config_path = runner_config->GetConfigPath();
  }
  worker_config->worker_id = 0;
  worker_config->context = init_context;
  worker_config->numa_id = -1;
  used_numa_node_num_ = 1;
  workers_num_ = 1;
  if (worker_config->config_info.find(lite::kWeight) == worker_config->config_info.end()) {
    std::map<std::string, std::string> config;
    config[lite::kWeightPath] = model_path_;
    worker_config->config_info[lite::kWeight] = config;
  }
  model_pool_gpu_config.push_back(worker_config);
  return model_pool_gpu_config;
}

ModelPoolConfig ModelPool::CreateCpuModelPoolConfig(const std::shared_ptr<RunnerConfig> &runner_config,
                                                    const std::shared_ptr<Context> &init_context,
                                                    const std::vector<std::vector<int>> &all_worker_bind_list,
                                                    const std::vector<int> &numa_node_id) {
  ModelPoolConfig model_pool_config;
  for (size_t i = 0; i < workers_num_; i++) {
    auto context = std::make_shared<Context>();
    if (context == nullptr) {
      MS_LOG(ERROR) << "New Context failed.";
      return {};
    }
    auto worker_config = std::make_shared<WorkerConfig>();
    if (worker_config == nullptr) {
      MS_LOG(ERROR) << "new worker config failed.";
      return {};
    }
    context->SetThreadNum(init_context->GetThreadNum());
    context->SetEnableParallel(init_context->GetEnableParallel());
    context->SetInterOpParallelNum(init_context->GetInterOpParallelNum());
    if (init_context->GetThreadAffinityMode() != lite::NO_BIND) {
      // bind by core id
      context->SetThreadAffinity(init_context->GetThreadAffinityMode());
      context->SetThreadAffinity(all_worker_bind_list[i]);
    } else {
      // not bind core , not use numa
      context->SetThreadAffinity(init_context->GetThreadAffinityMode());
    }
    worker_config->numa_id = numa_node_id[i];
    auto &new_device_list = context->MutableDeviceInfo();
    std::shared_ptr<CPUDeviceInfo> device_info = std::make_shared<CPUDeviceInfo>();
    if (device_info == nullptr) {
      MS_LOG(ERROR) << "device_info is nullptr.";
      return {};
    }
    std::shared_ptr<Allocator> allocator = nullptr;
    if (init_context->MutableDeviceInfo().front()->GetAllocator() == nullptr) {
      allocator = std::make_shared<DynamicMemAllocator>(worker_config->numa_id);
    } else {
      allocator = init_context->MutableDeviceInfo().front()->GetAllocator();
    }
    if (allocator == nullptr) {
      MS_LOG(ERROR) << "new allocator failed.";
      return {};
    }
    if (numa_allocator_.find(worker_config->numa_id) == numa_allocator_.end()) {
      numa_allocator_.insert(std::make_pair(worker_config->numa_id, allocator));
    }
    device_info->SetAllocator(allocator);
    auto init_cpu_info = init_context->MutableDeviceInfo().front()->Cast<CPUDeviceInfo>();
    auto enable_fp16 = init_cpu_info->GetEnableFP16();
    device_info->SetEnableFP16(enable_fp16);
    new_device_list.push_back(device_info);
    if (runner_config != nullptr) {
      worker_config->config_info = runner_config->GetConfigInfo();
      worker_config->config_path = runner_config->GetConfigPath();
    }
    worker_config->context = context;
    worker_config->worker_id = i;
    if (worker_config->config_info.find(lite::kWeight) == worker_config->config_info.end()) {
      std::map<std::string, std::string> config;
      config[lite::kWeightPath] = model_path_;
      worker_config->config_info[lite::kWeight] = config;
    }
    model_pool_config.push_back(worker_config);
  }
  return model_pool_config;
}

Status ModelPool::InitModelPoolBindList(const std::shared_ptr<Context> &init_context,
                                        std::vector<std::vector<int>> *bind_core_list,
                                        std::vector<int> *bind_numa_list) {
  if (!bind_core_available_ || init_context->GetThreadAffinityMode() == lite::NO_BIND) {
    auto status = SetWorkersNumaId(bind_numa_list);
    if (status != kSuccess) {
      MS_LOG(ERROR) << "set worker numa id failed in NO_BIND";
      return kLiteError;
    }
  } else if (bind_core_available_ && init_context->GetThreadAffinityMode() == lite::HIGHER_CPU) {
    // The user specified the id of the bundled core
    if (is_user_core_list_) {
      auto user_core_list = init_context->GetThreadAffinityCoreList();
      auto thread_num = init_context->GetThreadNum();
      for (size_t work_index = 0; work_index < workers_num_; work_index++) {
        std::vector<int> core_list;
        core_list.insert(core_list.end(), user_core_list.begin() + work_index * thread_num,
                         user_core_list.begin() + work_index * thread_num + thread_num);
        bind_core_list->push_back(core_list);
        bind_numa_list->push_back(kInvalidNumaId);
      }
      return kSuccess;
    }
    auto status = SetModelBindMode(bind_core_list, bind_numa_list, init_context);
    if (status != kSuccess) {
      MS_LOG(ERROR) << "SetModelBindMode failed.";
      return kLiteError;
    }
  } else {
    MS_LOG(ERROR) << "not support bind MID_CPU.";
    return kLiteError;
  }
  return kSuccess;
}

ModelPoolConfig ModelPool::CreateModelPoolConfig(const std::shared_ptr<RunnerConfig> &runner_config) {
  auto init_context = GetInitContext(runner_config);
  if (init_context == nullptr) {
    MS_LOG(ERROR) << "context is nullptr.";
    return {};
  }
  auto status = SetWorkersNum(runner_config, init_context);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "set worker num failed.";
    return {};
  }
  // init all bind list
  std::vector<std::vector<int>> bind_core_list;
  std::vector<int> bind_numa_list;
  status = InitModelPoolBindList(init_context, &bind_core_list, &bind_numa_list);
  // if numa is not applicable or numa is not available, numa id is -1
  if (status != kSuccess || bind_numa_list.empty()) {
    MS_LOG(ERROR) << "init model pool bind list failed.";
    return {};
  }
  auto device_num = init_context->MutableDeviceInfo().size();
  if (device_num == 0) {
    MS_LOG(ERROR) << "device_info is null.";
    return {};
  }
  if (device_num > 1) {
    return CreateGpuModelPoolConfig(runner_config, init_context);
  }
  // create all worker config
  ModelPoolConfig model_pool_config =
    CreateCpuModelPoolConfig(runner_config, init_context, bind_core_list, bind_numa_list);
  if (model_pool_config.empty()) {
    MS_LOG(ERROR) << "create cpu model pool config failed.";
    return {};
  }
  return model_pool_config;
}

std::vector<MSTensor> ModelPool::GetInputs() {
  std::shared_lock<std::shared_mutex> l(model_pool_mutex_);
  std::vector<MSTensor> inputs;
  if (inputs_info_.empty()) {
    MS_LOG(ERROR) << "model input is empty.";
    return {};
  }
  for (size_t i = 0; i < inputs_info_.size(); i++) {
    auto tensor =
      mindspore::MSTensor::CreateTensor(inputs_info_.at(i).name, inputs_info_.at(i).data_type, {}, nullptr, 0);
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "create tensor failed.";
      return {};
    }
    tensor->SetShape(inputs_info_.at(i).shape);
    tensor->SetFormat(inputs_info_.at(i).format);
    tensor->SetQuantParams(inputs_info_.at(i).quant_param);
    inputs.push_back(*tensor);
    delete tensor;
  }
  return inputs;
}

std::vector<MSTensor> ModelPool::GetOutputs() {
  std::shared_lock<std::shared_mutex> l(model_pool_mutex_);
  std::vector<MSTensor> outputs;
  if (outputs_info_.empty()) {
    MS_LOG(ERROR) << "model output is empty.";
    return {};
  }
  for (size_t i = 0; i < outputs_info_.size(); i++) {
    auto tensor =
      mindspore::MSTensor::CreateTensor(outputs_info_.at(i).name, outputs_info_.at(i).data_type, {}, nullptr, 0);
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "create tensor failed.";
      return {};
    }
    tensor->SetShape(outputs_info_.at(i).shape);
    tensor->SetFormat(outputs_info_.at(i).format);
    tensor->SetQuantParams(outputs_info_.at(i).quant_param);
    outputs.push_back(*tensor);
    delete tensor;
  }
  return outputs;
}

Status ModelPool::InitNumaParameter(const std::shared_ptr<RunnerConfig> &runner_config) {
  numa_available_ = numa::NUMAAdapter::GetInstance()->Available();
  if (!numa_available_) {
    MS_LOG(DEBUG) << "numa node is unavailable.";
    return kSuccess;
  }
  if (runner_config != nullptr && runner_config->GetWorkersNum() != 0 && runner_config->GetContext() != nullptr &&
      runner_config->GetContext()->GetThreadAffinityCoreList().size() != 0) {
    MS_LOG(DEBUG) << "If the user explicitly sets the core list, the numa binding is not performed by default.";
    numa_available_ = false;
    return kSuccess;
  }
  numa_node_num_ = numa::NUMAAdapter::GetInstance()->NodesNum();
  std::vector<int> physical_core_lite;
  std::vector<int> logical_core_list;
  auto status = DistinguishPhysicalAndLogical(&physical_core_lite, &logical_core_list);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "distinguish physical and logical failed.";
    return kLiteError;
  }
  status = DistinguishPhysicalAndLogicalByNuma(physical_core_lite, logical_core_list);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "distinguish physical and logical by numa failed.";
    return kLiteError;
  }
  return kSuccess;
}

Status ModelPool::CreateWorkers(char *graph_buf, size_t size, const ModelPoolConfig &model_pool_config) {
  std::shared_ptr<ModelWorker> model_worker = nullptr;
  if (model_pool_config.size() != workers_num_) {
    MS_LOG(ERROR) << "model pool config size is wrong.";
    return kLiteError;
  }
  bool create_worker_success = true;
  for (size_t i = 0; i < workers_num_; i++) {
    int numa_node_id = model_pool_config[i]->numa_id;
    auto ret = lite::PackWeightManager::GetInstance()->InitPackWeight(graph_buf, size, numa_node_id);
    MS_CHECK_FALSE_MSG(ret != kSuccess, kLiteError, "InitWeightManagerByBuf failed.");
    auto new_model_buf = lite::PackWeightManager::GetInstance()->GetNumaModelBuf(graph_buf, numa_node_id);
    model_bufs_.push_back(new_model_buf);
    MS_CHECK_TRUE_MSG(new_model_buf != nullptr, kLiteError, "get model buf is nullptr from PackWeightManager");
    model_worker = std::make_shared<ModelWorker>();
    if (model_worker == nullptr) {
      MS_LOG(ERROR) << "model worker is nullptr.";
      return kLiteNullptr;
    }
    int task_queue_id = numa_node_id != -1 ? numa_node_id : 0;
    predict_task_queue_->IncreaseWaitModelNum(1, task_queue_id);
    MS_LOG(INFO) << "create worker index: " << i << " | numa id: " << model_pool_config[i]->numa_id
                 << " | worker affinity mode: " << model_pool_config[i]->context->GetThreadAffinityMode()
                 << " | worker bind core list: " << model_pool_config[i]->context->GetThreadAffinityCoreList()
                 << " | worker thread num: " << model_pool_config[i]->context->GetThreadNum()
                 << " | inter op parallel num: " << model_pool_config[i]->context->GetInterOpParallelNum();
    if (!model_pool_config[i]->config_info.empty()) {
      for (auto &item : model_pool_config[i]->config_info) {
        auto section = item.first;
        MS_LOG(INFO) << "section: " << section;
        auto configs = item.second;
        for (auto &config : configs) {
          MS_LOG(INFO) << "\t key: " << config.first << " | value: " << config.second;
        }
      }
    }
    worker_thread_vec_.push_back(std::thread(&ModelWorker::CreateThreadWorker, model_worker, new_model_buf, size,
                                             model_pool_config[i], predict_task_queue_, &create_worker_success));
    if (all_model_workers_.find(task_queue_id) != all_model_workers_.end()) {
      all_model_workers_[task_queue_id].push_back(model_worker);
    } else {
      all_model_workers_[task_queue_id] = {model_worker};
    }
  }
  // wait for all workers to be created successfully
  MS_LOG(INFO) << "wait for all workers to be created successfully.";
  for (auto &item : all_model_workers_) {
    auto &workers = item.second;
    for (auto &worker : workers) {
      worker->WaitCreateWorkerDone();
      if (!create_worker_success) {
        MS_LOG(ERROR) << "worker init failed.";
        return kLiteError;
      }
    }
  }
  MS_LOG(INFO) << "All models are initialized.";
  // init model pool input and output
  if (model_worker != nullptr) {
    auto inputs = model_worker->GetInputs();
    for (auto &tensor : inputs) {
      TensorInfo tensor_info;
      tensor_info.name = tensor.Name();
      tensor_info.format = tensor.format();
      tensor_info.data_type = tensor.DataType();
      tensor_info.shape = tensor.Shape();
      tensor_info.quant_param = tensor.QuantParams();
      inputs_info_.push_back(tensor_info);
    }
    auto output = model_worker->GetOutputs();
    for (auto &tensor : output) {
      TensorInfo tensor_info;
      tensor_info.name = tensor.Name();
      tensor_info.format = tensor.format();
      tensor_info.data_type = tensor.DataType();
      tensor_info.shape = tensor.Shape();
      tensor_info.quant_param = tensor.QuantParams();
      outputs_info_.push_back(tensor_info);
    }
  }
  return kSuccess;
}

Status ModelPool::CanUseAllPhysicalResources(int *percentage) {
  auto can_use_cores = ParseCpusetFile(percentage);
  if (can_use_cores.empty()) {
    MS_LOG(ERROR) << "parse cpu files failed, | can use core list: " << can_use_cores
                  << " | percentage: " << *percentage;
    return kLiteError;
  }
  if (*percentage == -1) {
    can_use_core_num_ = can_use_cores.size();
    all_core_num_ = lite::GetCoreNum();
    can_use_all_physical_core_ = can_use_core_num_ == all_core_num_;
  } else {
    can_use_core_num_ = *percentage;
    can_use_all_physical_core_ = false;
  }
  return kSuccess;
}

Status ModelPool::Init(const std::string &model_path, const std::shared_ptr<RunnerConfig> &runner_config) {
  std::unique_lock<std::shared_mutex> l(model_pool_mutex_);
  model_path_ = model_path;
  int percentage;
  auto status = CanUseAllPhysicalResources(&percentage);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "parser sys file failed.";
    return kLiteError;
  }
  if (percentage != -1) {
    numa_available_ = false;
    bind_core_available_ = false;
  } else if (!can_use_all_physical_core_) {
    MS_LOG(INFO) << "the number of usable cores is less than the number of hardware cores of the machine.";
    numa_available_ = false;
  } else {
    status = InitNumaParameter(runner_config);
    if (status != kSuccess) {
      MS_LOG(ERROR) << "Init numa parameter failed.";
      return kLiteError;
    }
  }
  // create model pool config
  auto model_pool_config = CreateModelPoolConfig(runner_config);
  if (model_pool_config.empty()) {
    MS_LOG(ERROR) << "CreateModelPoolConfig failed, context is empty.";
    return kLiteError;
  }
  // create task queue for model pool
  predict_task_queue_ = std::make_shared<PredictTaskQueue>();
  if (predict_task_queue_ == nullptr) {
    MS_LOG(ERROR) << "create PredictTaskQueue failed, predict task queue is nullptr.";
    return kLiteNullptr;
  }
  if (numa_available_) {
    status = predict_task_queue_->InitTaskQueue(used_numa_node_num_, kNumMaxTaskQueueSize);
  } else {
    status = predict_task_queue_->InitTaskQueue(1, kNumMaxTaskQueueSize);
  }
  if (status != kSuccess) {
    MS_LOG(ERROR) << "predict task queue init failed, status=" << status;
    return kLiteError;
  }
  // read model by path and init packed weight by buffer
  size_t size = 0;
  graph_buf_ = lite::ReadFile(model_path.c_str(), &size);
  if (graph_buf_ == nullptr) {
    MS_LOG(ERROR) << "read file failed.";
    return kLiteNullptr;
  }
  status = CreateWorkers(graph_buf_, size, model_pool_config);
  if (status != kSuccess) {
    lite::PackWeightManager::GetInstance()->DeleteOriginModelBufInfo(graph_buf_);
    MS_LOG(ERROR) << "create worker failed.";
    return kLiteError;
  }
  // initialize the task pool
  tasks_ = new (std::nothrow) PredictTask[kNumMaxTaskQueueSize]();
  if (tasks_ == nullptr) {
    MS_LOG(ERROR) << "new task failed.";
    return kLiteNullptr;
  }
  for (size_t i = 0; i < kNumMaxTaskQueueSize; i++) {
    free_tasks_id_.push(i);
  }
  is_initialized_ = true;
  return kSuccess;
}

Status ModelPool::UpdateConfig(const std::string &section, const std::pair<std::string, std::string> &config) {
  for (auto &item : all_model_workers_) {
    auto &workers = item.second;
    for (auto &worker : workers) {
      auto status = worker->UpdateConfig(section, config);
      if (status != kSuccess) {
        MS_LOG(ERROR) << "model pool update config failed, status=" << status;
        return status;
      }
    }
  }
  return kSuccess;
}

std::shared_ptr<ModelWorker> ModelPool::GetMaxWaitWorkerNum(int *max_wait_worker_node_id, int *max_wait_worker_num) {
  std::unique_lock<std::mutex> l(predict_task_mutex_);
  *max_wait_worker_node_id = 0;
  *max_wait_worker_num = predict_task_queue_->GetWaitModelNum(0);
  for (size_t i = 1; i < used_numa_node_num_; i++) {
    int worker_num = predict_task_queue_->GetWaitModelNum(i);
    if (*max_wait_worker_num < worker_num) {
      *max_wait_worker_num = worker_num;
      *max_wait_worker_node_id = i;
    }
  }
  if (*max_wait_worker_num > 0) {
    auto &workers = all_model_workers_[*max_wait_worker_node_id];
    auto task_queue_id = *max_wait_worker_node_id;
    for (auto &worker : workers) {
      if (worker->IsAvailable()) {
        *max_wait_worker_num = predict_task_queue_->GetWaitModelNum(task_queue_id);
        *max_wait_worker_node_id = task_queue_id;
        predict_task_queue_->DecreaseWaitModelNum(1, *max_wait_worker_node_id);
        return worker;
      }
    }
  }
  return nullptr;
}

PredictTask *ModelPool::CreatePredictTask(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                                          const MSKernelCallBack &before, const MSKernelCallBack &after,
                                          size_t *task_id) {
  std::lock_guard<std::mutex> lock(task_id_mutex_);
  if (!free_tasks_id_.empty()) {
    auto item = free_tasks_id_.front();
    *task_id = item;
    free_tasks_id_.pop();
    PredictTask *task = &tasks_[*task_id];
    task->inputs = &inputs;
    task->outputs = outputs;
    task->before = before;
    task->after = after;
    return task;
  } else {
    return nullptr;
  }
}

void ModelPool::UpdateFreeTaskId(size_t id) {
  std::lock_guard<std::mutex> lock(task_id_mutex_);
  free_tasks_id_.push(id);
}

Status ModelPool::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                          const MSKernelCallBack &before, const MSKernelCallBack &after) {
  std::shared_lock<std::shared_mutex> l(model_pool_mutex_);
  int max_wait_worker_node_id = 0;
  int max_wait_worker_num = 0;
  if (inputs.size() == 0 || inputs.front().Shape().size() == 0) {
    MS_LOG(ERROR) << "inputs is invalid. input size: " << inputs.size();
    return kLiteError;
  }
  auto available_worker = GetMaxWaitWorkerNum(&max_wait_worker_node_id, &max_wait_worker_num);
  if (available_worker != nullptr) {
    // dispatch tasks directly to workers
    auto ret = available_worker->Predict(inputs, outputs, before, after);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "direct predict failed.";
      return kLiteError;
    }
    predict_task_queue_->IncreaseWaitModelNum(1, max_wait_worker_node_id);
    return kSuccess;
  } else {
    // do predict
    size_t task_id;
    auto task = CreatePredictTask(inputs, outputs, before, after, &task_id);
    if (task == nullptr) {
      MS_LOG(ERROR) << "The number of waiting tasks in the queue exceeds the limit, ret=" << kLiteServiceDeny;
      return kLiteServiceDeny;
    }
    predict_task_queue_->PushPredictTask(task, max_wait_worker_node_id);
    predict_task_queue_->WaitUntilPredictActive(task, max_wait_worker_node_id);
    UpdateFreeTaskId(task_id);
  }
  return kSuccess;
}

ModelPool::~ModelPool() {
  MS_LOG(INFO) << "free model pool.";
  std::unique_lock<std::shared_mutex> l(model_pool_mutex_);
  is_initialized_ = false;
  if (predict_task_queue_ != nullptr) {
    predict_task_queue_->SetPredictTaskDone();
  }
  MS_LOG(INFO) << "Wait for all threads to finish tasks.";
  for (auto &th : worker_thread_vec_) {
    if (th.joinable()) {
      th.join();
    }
  }
  MS_LOG(INFO) << "delete model pool task.";
  if (tasks_ != nullptr) {
    delete[] tasks_;
    tasks_ = nullptr;
  }
  // free weight sharing related memory
  MS_LOG(INFO) << "free pack weight model buf.";
  if (graph_buf_ != nullptr) {
    lite::PackWeightManager::GetInstance()->DeleteOriginModelBufInfo(graph_buf_);
    delete[] graph_buf_;
    graph_buf_ = nullptr;
  }
  lite::PackWeightManager::GetInstance()->FreePackWeight(model_bufs_);
  model_bufs_.clear();
  MS_LOG(INFO) << "free model pool done.";
}
}  // namespace mindspore
