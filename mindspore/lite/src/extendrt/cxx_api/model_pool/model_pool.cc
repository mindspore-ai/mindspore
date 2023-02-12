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

#include "src/extendrt/cxx_api/model_pool/model_pool.h"
#include <unistd.h>
#include <future>
#include <algorithm>
#include "mindspore/ccsrc/plugin/device/cpu/kernel/nnacl/op_base.h"
#include "src/extendrt/cxx_api/model_pool/resource_manager.h"
#include "src/common/log_adapter.h"
#include "include/lite_types.h"
#include "src/litert/inner_allocator.h"
#include "src/common/file_utils.h"
#include "src/litert/pack_weight_manager.h"
#include "src/extendrt/numa_adapter.h"
#include "src/common/common.h"
#include "thread/parallel_thread_pool_manager.h"
#include "src/common/config_file.h"
namespace mindspore {
namespace {
constexpr int kNumDeviceInfo = 2;
constexpr int kNumIndex = 2;
constexpr int kNumCoreDataLen = 3;
constexpr int kNumMaxTaskQueueSize = 1000;
constexpr int kNumPhysicalCoreThreshold = 16;
constexpr int kDefaultWorkerNumPerPhysicalCpu = 2;
constexpr int kDefaultThreadsNum = 8;
constexpr int kInvalidNumaId = -1;
constexpr int kNumDefaultInterOpParallel = 4;
constexpr int kNumCoreNumTimes = 5;
constexpr int kDefaultThreadNumTimes = 2;
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
  auto status = ResourceManager::GetInstance()->DistinguishPhysicalAndLogical(&physical_core_list, &logical_core_list);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "distinguish physical and logical failed.";
    return kLiteError;
  }
  std::vector<int> all_core_list = {};
  if (!can_use_all_physical_core_) {
    size_t percentage;
    std::vector<int> can_use_core_list = ResourceManager::GetInstance()->ParseCpuCoreList(&percentage);
    if (percentage != can_use_core_list.size()) {
      MS_LOG(ERROR) << "can not use all resource, percentage: " << percentage
                    << ", can use core list size: " << can_use_core_list.size();
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
  int threshold_thread_num = kNumCoreNumTimes * core_num;
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
    if (context->GetThreadAffinityMode() != lite::NO_BIND || !context->GetThreadAffinityCoreList().empty()) {
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
    if (device->GetDeviceType() == kGPU) {
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
      MS_LOG(ERROR) << "Please set the correct DeviceType";
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
    context->SetThreadAffinity(lite::NO_BIND);
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
             runner_config->GetWorkersNum() <= can_use_core_num_ * kNumCoreNumTimes) {
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

std::shared_ptr<Context> ModelPool::CopyContext(const std::shared_ptr<Context> &context) {
  auto new_context = std::make_shared<Context>();
  new_context->SetThreadNum(context->GetThreadNum());
  new_context->SetEnableParallel(context->GetEnableParallel());
  new_context->SetInterOpParallelNum(context->GetInterOpParallelNum());
  new_context->SetThreadAffinity(context->GetThreadAffinityMode());
  new_context->SetThreadAffinity(context->GetThreadAffinityCoreList());
  auto &device_list = context->MutableDeviceInfo();
  auto &new_device_list = new_context->MutableDeviceInfo();
  for (auto &device : device_list) {
    if (device->GetDeviceType() == DeviceType::kCPU) {
      auto cpu_info = device->Cast<CPUDeviceInfo>();
      auto new_cpu_info = std::make_shared<CPUDeviceInfo>();
      new_cpu_info->SetEnableFP16(cpu_info->GetEnableFP16());
      new_device_list.push_back(new_cpu_info);
    } else if (device->GetDeviceType() == DeviceType::kGPU) {
      auto gpu_info = device->Cast<GPUDeviceInfo>();
      auto new_gpu_info = std::make_shared<GPUDeviceInfo>();
      new_gpu_info->SetEnableFP16(gpu_info->GetEnableFP16());
      new_gpu_info->SetDeviceID(gpu_info->GetDeviceID());
      new_device_list.push_back(new_gpu_info);
    } else if (device->GetDeviceType() == DeviceType::kAscend) {
      auto asscend_info = device->Cast<AscendDeviceInfo>();
      auto new_asscend_info = std::make_shared<AscendDeviceInfo>();
      new_asscend_info->SetDeviceID(asscend_info->GetDeviceID());
      new_device_list.push_back(new_asscend_info);
    } else {
      MS_LOG(ERROR) << "device type is: " << device->GetDeviceType();
    }
  }
  return new_context;
}

ModelPoolConfig ModelPool::CreateCpuModelPoolConfig(const std::shared_ptr<RunnerConfig> &runner_config,
                                                    const std::shared_ptr<Context> &init_context,
                                                    const std::vector<std::vector<int>> &all_worker_bind_list,
                                                    const std::vector<int> &numa_node_id) {
  ModelPoolConfig model_pool_config;
  for (size_t i = 0; i < workers_num_; i++) {
    auto worker_config = std::make_shared<WorkerConfig>();
    if (worker_config == nullptr) {
      MS_LOG(ERROR) << "new worker config failed.";
      return {};
    }
    auto context = CopyContext(init_context);
    if (init_context->GetThreadAffinityMode() != lite::NO_BIND) {
      // bind by core id
      context->SetThreadAffinity(init_context->GetThreadAffinityMode());
      context->SetThreadAffinity(all_worker_bind_list[i]);
    }
    worker_config->numa_id = numa_node_id[i];
    if (init_context->MutableDeviceInfo().front()->GetDeviceType() == DeviceType::kCPU) {
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
      context->MutableDeviceInfo().front()->SetAllocator(allocator);
    }
    if (runner_config != nullptr) {
      worker_config->config_info = runner_config->GetConfigInfo();
      worker_config->config_path = runner_config->GetConfigPath();
    }
    worker_config->context = context;
    worker_config->worker_id = i;
    if (worker_config->config_info.find(lite::kWeightSection) != worker_config->config_info.end()) {
      MS_LOG(WARNING) << "It is not recommended to use the 'weigh' and 'weight_path' parameters. "
                         "Please use the 'model_info' and 'mindir_path' parameters auto data_path";
      auto ms_weight_iter = worker_config->config_info.find(lite::kWeightSection)->second;
      if (ms_weight_iter.find(lite::kWeightPathKey) == ms_weight_iter.end()) {
        MS_LOG(ERROR) << "The 'weight' parameter has been set. Please set the 'weight_path' parameter synchronously";
        return {};
      }
      std::map<std::string, std::string> config;
      config[lite::kConfigMindIRPathKey] = ms_weight_iter[lite::kWeightPathKey];
      worker_config->config_info[lite::kConfigModelFileSection] = config;
    }
    if (worker_config->config_info.find(lite::kConfigModelFileSection) == worker_config->config_info.end()) {
      std::map<std::string, std::string> config;
      config[lite::kConfigMindIRPathKey] = model_path_;
      worker_config->config_info[lite::kConfigModelFileSection] = config;
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
  auto status =
    ResourceManager::GetInstance()->DistinguishPhysicalAndLogicalByNuma(&numa_physical_cores_, &numa_logical_cores_);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "distinguish physical and logical by numa failed.";
    return kLiteError;
  }
  return kSuccess;
}

Status ModelPool::CheckSharingThreadPoolParam(const ModelPoolConfig &model_pool_config) {
  if (!enable_shared_thread_pool_) {
    return kSuccess;
  }
  if (model_pool_config.front()->context->GetInterOpParallelNum() <= 1) {
    enable_shared_thread_pool_ = false;
    MS_LOG(WARNING) << "Not enable parallelThreadPool, not enable sharing thread pool.";
    return kSuccess;
  }
  if (remaining_thread_num_ < 0) {
    MS_LOG(ERROR) << "remaining thread num is invalid, remaining_thread_num_: " << remaining_thread_num_;
    return kLiteParamInvalid;
  }
  if (remaining_thread_num_ > model_pool_config.front()->context->GetThreadNum()) {
    MS_LOG(ERROR) << "remaining thread num must less then thread num, remaining_thread_num is: "
                  << remaining_thread_num_ << ", thread num: " << model_pool_config.front()->context->GetThreadNum();
    return kLiteParamInvalid;
  }
  if (thread_num_limit_ == 0) {
    thread_num_limit_ = model_pool_config.front()->context->GetThreadNum() * kDefaultThreadNumTimes;
  }
  if (thread_num_limit_ <= model_pool_config.front()->context->GetThreadNum()) {
    MS_LOG(ERROR) << "thread_num_limit_ is:" << thread_num_limit_ << "  thread num."
                  << model_pool_config.front()->context->GetThreadNum()
                  << " thread_num_limit_ should more than thread num.";
    return kLiteParamInvalid;
  }
  if (thread_num_limit_ > model_pool_config.front()->context->GetThreadNum() * kNumCoreNumTimes) {
    MS_LOG(ERROR) << "thread num limit: " << thread_num_limit_
                  << " is more than 5 times thread num: " << model_pool_config.front()->context->GetThreadNum()
                  << ", change it to 5 times thread num. Please check whether Thread num is reasonable.";
    return kLiteParamInvalid;
  }
  return kSuccess;
}

Status ModelPool::CreateWorkers(const char *graph_buf, size_t size, const ModelPoolConfig &model_pool_config,
                                bool copy_model) {
  std::shared_ptr<ModelWorker> model_worker = nullptr;
  bool create_worker_success = true;
  runner_id_ = ResourceManager::GetInstance()->GenRunnerID();
  auto status = CheckSharingThreadPoolParam(model_pool_config);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "CheckSharingThreadPoolParam failed.";
    return kLiteError;
  }
  MS_LOG(INFO) << "runner_id_: " << runner_id_ << " | enable_shared_thread_pool_: " << enable_shared_thread_pool_
               << " | workers_num_: " << workers_num_ << " | remaining_thread_num_: " << remaining_thread_num_
               << " | thread_num_limit_: " << thread_num_limit_;
  for (size_t i = 0; i < workers_num_; i++) {
    int numa_node_id = model_pool_config[i]->numa_id;
    std::map<std::string, std::string> ids;
    ids[lite::kInnerModelIDKey] = std::to_string(i);
    ids[lite::kInnerRunnerIDKey] = runner_id_;
    ids[lite::kInnerNumaIDKey] = std::to_string(model_pool_config[i]->numa_id);
    if (enable_shared_thread_pool_) {
      ids[lite::kInnerWorkerNumKey] = std::to_string(workers_num_);
      ids[lite::kEnableSharedThreadPoolKey] = "true";
      ids[lite::kThreadNumRemainingPerWorkerKey] = std::to_string(remaining_thread_num_);
      ids[lite::kThreadNumLimitPerWorkerKey] = std::to_string(thread_num_limit_);
    }
    if (!copy_model || model_pool_config[i]->numa_id == 0) {
      ids[lite::kInnerSharingWeightCopyBufKey] = "false";
    }
    model_pool_config[i]->config_info[lite::kInnerModelParallelRunnerSection] = ids;
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
    if (i == 0) {
      model_worker->InitModelWorker(graph_buf, size, model_pool_config[i], predict_task_queue_, &create_worker_success);
      thread_ = std::thread(&ModelWorker::Run, model_worker);
      model_worker->WaitCreateWorkerDone();
    } else {
      InitWorkerManager::GetInstance()->InitModelWorker(model_worker, graph_buf, size, model_pool_config[i],
                                                        predict_task_queue_, &create_worker_success);
    }
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
    }
  }
  if (!create_worker_success) {
    MS_LOG(ERROR) << "worker init failed.";
    return kLiteError;
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

Status ModelPool::CanUseAllPhysicalResources() {
  size_t percentage;
  auto can_use_cores = ResourceManager::GetInstance()->ParseCpuCoreList(&percentage);
  if (can_use_cores.empty()) {
    MS_LOG(ERROR) << "parse cpu files failed, | can use core list: " << can_use_cores
                  << " | percentage: " << percentage;
    return kLiteError;
  }
  if (percentage == can_use_cores.size()) {
    MS_LOG(INFO) << "percentage: " << percentage << "can_use_cores size: " << can_use_cores.size();
    can_use_core_num_ = can_use_cores.size();
    all_core_num_ = lite::GetCoreNum();
    can_use_all_physical_core_ = can_use_core_num_ == all_core_num_;
    bind_core_available_ = can_use_core_num_ == all_core_num_;
  } else {
    can_use_core_num_ = percentage;
    can_use_all_physical_core_ = false;
    bind_core_available_ = false;
  }
  return kSuccess;
}

Status ModelPool::InitByBuf(const char *model_data, size_t size, const std::shared_ptr<RunnerConfig> &runner_config) {
  auto model_pool_config = Init(runner_config);
  if (model_pool_config.empty() || model_pool_config.size() != workers_num_) {
    MS_LOG(ERROR) << "InitModelPoolConfig failed.";
    return kLiteFileError;
  }
  auto status = CreateWorkers(model_data, size, model_pool_config, numa_available_ && (used_numa_node_num_ > 1));
  if (status != kSuccess) {
    MS_LOG(ERROR) << "create worker failed.";
    return kLiteError;
  }
  return kSuccess;
}

Status ModelPool::InitByPath(const std::string &model_path, const std::shared_ptr<RunnerConfig> &runner_config) {
  model_path_ = model_path;
  auto model_pool_config = Init(runner_config);
  if (model_pool_config.empty() || model_pool_config.size() != workers_num_) {
    MS_LOG(ERROR) << "InitModelPoolConfig failed.";
    return kLiteFileError;
  }
  size_t size = 0;
  bool numa_copy_buf = numa_available_ && (used_numa_node_num_ > 1);
  if (numa_copy_buf) {
    allocator_ = std::make_shared<DynamicMemAllocator>(0);
    if (allocator_ == nullptr) {
      MS_LOG(ERROR) << "new dynamic allocator failed.";
      return kLiteNullptr;
    }
  }
  graph_buf_ = lite::ReadFile(model_path.c_str(), &size, allocator_);
  if (graph_buf_ == nullptr) {
    MS_LOG(ERROR) << "read model failed, model path: " << model_path;
    return kLiteNullptr;
  }
  auto status = CreateWorkers(graph_buf_, size, model_pool_config, numa_copy_buf);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "create worker failed.";
    return kLiteError;
  }
  return kSuccess;
}

Status ModelPool::ParseParamByConfigInfo(std::map<std::string, std::map<std::string, std::string>> config_info) {
  auto shared_thread_pool = config_info.find(lite::kSharedThreadPoolSection);
  if (shared_thread_pool == config_info.end()) {
    MS_LOG(INFO) << "not set shared thread pool.";
    return kSuccess;
  }
  auto shared_thread_pool_param = shared_thread_pool->second;
  if (shared_thread_pool_param.find(lite::kEnableSharedThreadPoolKey) == shared_thread_pool_param.end()) {
    MS_LOG(INFO) << "not find key of enable_shared_thread_pool";
    return kLiteParamInvalid;
  }
  if (shared_thread_pool_param[lite::kEnableSharedThreadPoolKey] == "false") {
    MS_LOG(INFO) << "Not use shared thread pool";
    enable_shared_thread_pool_ = false;
    return kSuccess;
  }
  if (shared_thread_pool_param[lite::kEnableSharedThreadPoolKey] == "true") {
    if (shared_thread_pool_param.find(lite::kThreadNumLimitPerWorkerKey) != shared_thread_pool_param.end() &&
        !shared_thread_pool_param[lite::kThreadNumLimitPerWorkerKey].empty()) {
      thread_num_limit_ = std::atoi(shared_thread_pool_param[lite::kThreadNumLimitPerWorkerKey].c_str());
      if (thread_num_limit_ <= 0) {
        MS_LOG(WARNING) << "thread_num_limit is invalid, thread_num_limit: " << thread_num_limit_;
        return kLiteParamInvalid;
      }
    }
  }
  if (shared_thread_pool_param[lite::kEnableSharedThreadPoolKey] == "true" &&
      shared_thread_pool_param.find(lite::kThreadNumLimitPerWorkerKey) != shared_thread_pool_param.end()) {
    MS_LOG(INFO) << "use shared thread pool";
    enable_shared_thread_pool_ = true;
    if (shared_thread_pool_param.find(lite::kThreadNumRemainingPerWorkerKey) != shared_thread_pool_param.end()) {
      if (!shared_thread_pool_param[lite::kThreadNumRemainingPerWorkerKey].empty()) {
        remaining_thread_num_ = std::atoi(shared_thread_pool_param[lite::kThreadNumRemainingPerWorkerKey].c_str());
        if (remaining_thread_num_ < 0) {
          MS_LOG(WARNING) << "remaining_thread_num_ is invalid, remaining_thread_num_: " << remaining_thread_num_;
          return kLiteParamInvalid;
        }
      } else {
        MS_LOG(INFO) << "not set thread_num_remaining_per_worker param, default remaining thread num is 0.";
      }
    }
  }
  MS_LOG(INFO) << "use thread pool shared, remaining thread num: " << remaining_thread_num_
               << " | Limit thread num: " << thread_num_limit_;
  return kSuccess;
}

Status ModelPool::ParseSharedThreadPoolParam(const std::shared_ptr<RunnerConfig> &runner_config) {
  if (runner_config == nullptr) {
    MS_LOG(INFO) << "runner config is nullptr.";
    return kSuccess;
  }
  std::map<std::string, std::map<std::string, std::string>> config_file_info;
  if (!runner_config->GetConfigPath().empty()) {
    int ret = lite::GetAllSectionInfoFromConfigFile(runner_config->GetConfigPath(), &config_file_info);
    if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << "GetAllSectionInfoFromConfigFile failed.";
      return kLiteError;
    }
  }
  if (config_file_info.find(lite::kSharedThreadPoolSection) != config_file_info.end()) {
    MS_LOG(INFO) << "parse shared thread pool parm by config file.";
    if (ParseParamByConfigInfo(config_file_info) != kSuccess) {
      MS_LOG(WARNING) << "config file param is wrong.";
    }
  }
  return ParseParamByConfigInfo(runner_config->GetConfigInfo());
}

ModelPoolConfig ModelPool::Init(const std::shared_ptr<RunnerConfig> &runner_config) {
  auto status = ParseSharedThreadPoolParam(runner_config);
  if (status != kSuccess) {
    MS_LOG(WARNING) << "ParseSharedThreadPoolParam failed, Not use thread pool shared.";
    enable_shared_thread_pool_ = false;
  }
  ModelPoolConfig model_pool_config = {};
  status = CanUseAllPhysicalResources();
  if (status != kSuccess) {
    MS_LOG(ERROR) << "parser sys file failed.";
    return model_pool_config;
  }
  if (!can_use_all_physical_core_) {
    numa_available_ = false;
  } else {
    status = InitNumaParameter(runner_config);
    if (status != kSuccess) {
      MS_LOG(ERROR) << "Init numa parameter failed.";
      return model_pool_config;
    }
  }
  // create model pool config
  model_pool_config = CreateModelPoolConfig(runner_config);
  if (model_pool_config.empty()) {
    MS_LOG(ERROR) << "CreateModelPoolConfig failed, context is empty.";
    return model_pool_config;
  }
  if (!model_pool_config[0]->config_info.empty()) {
    for (auto &item : model_pool_config[0]->config_info) {
      auto section = item.first;
      MS_LOG(INFO) << "Model Parallel Runner config info:";
      MS_LOG(INFO) << "section: " << section;
      auto configs = item.second;
      for (auto &config : configs) {
        MS_LOG(INFO) << "\t key: " << config.first << " | value: " << config.second;
      }
    }
  }
  // create task queue for model pool
  predict_task_queue_ = std::make_shared<PredictTaskQueue>();
  if (predict_task_queue_ == nullptr) {
    MS_LOG(ERROR) << "create PredictTaskQueue failed, predict task queue is nullptr.";
    return model_pool_config;
  }
  if (numa_available_) {
    status = predict_task_queue_->InitTaskQueue(used_numa_node_num_, kNumMaxTaskQueueSize);
  } else {
    status = predict_task_queue_->InitTaskQueue(1, kNumMaxTaskQueueSize);
  }
  if (status != kSuccess) {
    MS_LOG(ERROR) << "predict task queue init failed, status=" << status;
    return model_pool_config;
  }
  // initialize the task pool
  tasks_ = new (std::nothrow) PredictTask[kNumMaxTaskQueueSize]();
  if (tasks_ == nullptr) {
    MS_LOG(ERROR) << "new task failed.";
    return model_pool_config;
  }
  for (size_t i = 0; i < kNumMaxTaskQueueSize; i++) {
    free_tasks_id_.push(i);
  }
  return model_pool_config;
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
Status ModelPool::WarmUpForAllWorker(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) {
  bool user_data = true;
  for (auto &item : all_model_workers_) {
    for (auto &worker : item.second) {
      if (user_data) {
        auto ret = worker->Predict(inputs, outputs);
        if (ret != kSuccess) {
          MS_LOG(ERROR) << "predict failed.";
          return kLiteError;
        }
        user_data = false;
        continue;
      }
      std::vector<MSTensor> outs;
      auto ret = worker->Predict(inputs, &outs);
      if (ret != kSuccess) {
        MS_LOG(WARNING) << "warm up failed.";
        return kSuccess;
      }
    }
  }
  return kSuccess;
}

Status ModelPool::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                          const MSKernelCallBack &before, const MSKernelCallBack &after) {
  if (MS_UNLIKELY((inputs.size() == 0 || inputs.front().Shape().size() == 0))) {
    MS_LOG(ERROR) << "inputs is invalid. input size: " << inputs.size();
    return kLiteError;
  }
  if (MS_UNLIKELY(!is_warm_up_)) {
    std::unique_lock<std::mutex> warm_up_l(warm_up_mutex);
    if (!is_warm_up_) {
      MS_LOG(INFO) << "do warm up.";
      if (WarmUpForAllWorker(inputs, outputs) != kSuccess) {
        return kLiteError;
      }
      is_warm_up_ = true;
      MS_LOG(INFO) << "do warm up done.";
      return kSuccess;
    }
  }
  int max_wait_worker_node_id = 0;
  int max_wait_worker_num = 0;
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
  if (predict_task_queue_ != nullptr) {
    predict_task_queue_->SetPredictTaskDone();
  }
  if (allocator_ != nullptr && graph_buf_ != nullptr) {
    allocator_->Free(graph_buf_);
    graph_buf_ = nullptr;
  }
  MS_LOG(INFO) << "delete model worker.";
  for (auto &item : all_model_workers_) {
    auto model_workers = item.second;
    for (auto &model_worker : model_workers) {
      while (!model_worker->ModelIsNull()) {
        MS_LOG(INFO) << "wait model of model worker destroy";
        std::this_thread::yield();
      }
    }
  }
  if (thread_.joinable()) {
    thread_.join();
  }
  MS_LOG(INFO) << "delete model pool task.";
  if (tasks_ != nullptr) {
    delete[] tasks_;
    tasks_ = nullptr;
  }
  MS_LOG(INFO) << "free model pool done.";
}
}  // namespace mindspore
