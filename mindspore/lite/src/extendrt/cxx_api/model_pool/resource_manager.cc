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
#include "src/extendrt/cxx_api/model_pool/resource_manager.h"
#include <unordered_map>
#include <memory>
#include <fstream>
#include <utility>
#include "include/api/status.h"
#include "src/common/log_adapter.h"
#include "src/common/utils.h"
#include "src/extendrt/numa_adapter.h"
#include "nnacl/op_base.h"

namespace {
constexpr int kNumIndex = 2;
}
namespace mindspore {
ResourceManager *ResourceManager::GetInstance() {
  static ResourceManager instance;
  return &instance;
}

std::string ResourceManager::GenRunnerID() {
  std::unique_lock<std::mutex> l(manager_mutex_);
  std::string runner_id = "runner_" + std::to_string(runner_id_);
  runner_id_++;
  MS_LOG(INFO) << "generate runner id: " << runner_id;
  return runner_id;
}

std::vector<int> ResourceManager::ParseCpuCoreList(size_t *can_use_core_num) {
  std::unique_lock<std::mutex> l(manager_mutex_);
  if (can_use_core_num_ != 0) {
    *can_use_core_num = can_use_core_num_;
    return cpu_cores_;
  }
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
        cpu_cores_.push_back(i);
      }
    } else {
      cpu_cores_.push_back(std::atoi(line.c_str()));
    }
  }
  infile.close();
  std::ifstream quota_file("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", std::ios::in);
  std::string quota_line;
  getline(quota_file, quota_line);
  if (quota_line[quota_line.size() - 1] == '\n') {
    quota_line = quota_line.substr(0, quota_line.size() - 1);
  }
  auto quota = std::atoi(quota_line.c_str());
  quota_file.close();
  if (quota == -1) {
    *can_use_core_num = cpu_cores_.size();
    can_use_core_num_ = *can_use_core_num;
    return cpu_cores_;
  }
  std::ifstream period_file("/sys/fs/cgroup/cpu/cpu.cfs_period_us", std::ios::in);
  std::string period_line;
  getline(period_file, period_line);
  if (period_line[period_line.size() - 1] == '\n') {
    period_line = period_line.substr(0, period_line.size() - 1);
  }
  period_file.close();
  auto period = std::atoi(period_line.c_str());
  if (period == 0) {
    MS_LOG(ERROR) << "read cpu.cfs_period_us file failed.";
    *can_use_core_num = 0;
    return {};
  }
  *can_use_core_num = quota / period;
  can_use_core_num_ = *can_use_core_num;
  return cpu_cores_;
}

Status ResourceManager::DistinguishPhysicalAndLogical(std::vector<int> *physical_cores,
                                                      std::vector<int> *logical_cores) {
  std::unique_lock<std::mutex> l(manager_mutex_);
  if (!physical_core_ids_.empty()) {
    if (physical_cores != nullptr && logical_cores != nullptr) {
      *physical_cores = physical_core_ids_;
      *logical_cores = logical_core_ids_;
    }
    return kSuccess;
  }
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
    MS_LOG(DEBUG) << "All cores are physical core..";
    for (size_t i = 0; i < processor_ids.size(); i++) {
      physical_core_ids_.push_back(processor_ids[i]);
    }
  } else if (core_ids.size() == physical_ids.size() && physical_ids.size() == processor_ids.size()) {
    for (size_t i = 0; i < processor_ids.size(); i++) {
      if (ids.find(physical_ids[i]) == ids.end()) {
        std::vector<int> core_id_list = {core_ids[i]};
        ids.insert(std::make_pair(physical_ids[i], core_id_list));
        physical_core_ids_.push_back(processor_ids[i]);
        continue;
      }
      if (find(ids[physical_ids[i]].begin(), ids[physical_ids[i]].end(), core_ids[i]) == ids[physical_ids[i]].end()) {
        ids[physical_ids[i]].push_back(core_ids[i]);
        physical_core_ids_.push_back(processor_ids[i]);
        continue;
      } else {
        logical_core_ids_.push_back(processor_ids[i]);
      }
    }
  }
  if (physical_cores != nullptr && logical_cores != nullptr) {
    *physical_cores = physical_core_ids_;
    *logical_cores = logical_core_ids_;
  }
  return kSuccess;
}

Status ResourceManager::DistinguishPhysicalAndLogicalByNuma(std::vector<std::vector<int>> *numa_physical_cores,
                                                            std::vector<std::vector<int>> *numa_logical_cores) {
  if (numa_logical_cores == nullptr || numa_physical_cores == nullptr) {
    MS_LOG(ERROR) << "numa_logical_cores/numa_logical_cores is nullptr";
    return kLiteError;
  }
  (void)DistinguishPhysicalAndLogical(nullptr, nullptr);
  std::unique_lock<std::mutex> l(manager_mutex_);
  if (!numa_physical_core_ids_.empty()) {
    *numa_physical_cores = numa_physical_core_ids_;
    *numa_logical_cores = numa_logical_core_ids_;
    return kSuccess;
  }
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
      if (find(physical_core_ids_.begin(), physical_core_ids_.end(), core_id) != physical_core_ids_.end()) {
        physical_cores.push_back(core_id);
      } else if (find(logical_core_ids_.begin(), logical_core_ids_.end(), core_id) != logical_core_ids_.end()) {
        logical_cores.push_back(core_id);
      } else {
        MS_LOG(ERROR) << "core id not belong physical/logical core id.";
        numa_physical_core_ids_.clear();
        numa_logical_core_ids_.clear();
        return kLiteError;
      }
    }
    numa_physical_core_ids_.push_back(physical_cores);
    numa_logical_core_ids_.push_back(logical_cores);
  }
  *numa_physical_cores = numa_physical_core_ids_;
  *numa_logical_cores = numa_logical_core_ids_;
  return kSuccess;
}

void InitWorkerThread::Destroy() {
  if (model_worker_ != nullptr) {
    predict_task_queue_->SetPredictTaskDone();
    predict_task_queue_ = nullptr;
  }
  std::unique_lock<std::mutex> l(mtx_init_);
  is_destroy_ = true;
  is_launch_ = true;
  init_cond_var_.notify_one();
}

void InitWorkerThread::Run() {
  while (!is_destroy_) {
    std::unique_lock<std::mutex> l(mtx_init_);
    while (!is_launch_) {
      init_cond_var_.wait(l);
    }
    if (is_destroy_) {
      is_idle_ = true;
      return;
    }
    if (model_worker_ == nullptr) {
      continue;
    }
    model_worker_->InitModelWorker(model_buf_, size_, worker_config_, predict_task_queue_, create_success_);
    model_worker_->Run();
    model_worker_ = nullptr;
    is_idle_ = true;
    is_launch_ = false;
  }
}

InitWorkerThread::~InitWorkerThread() {
  if (thread_.joinable()) {
    thread_.join();
  }
}

void InitWorkerThread::Launch(std::shared_ptr<ModelWorker> worker, const char *model_buf, size_t size,
                              const std::shared_ptr<WorkerConfig> &worker_config,
                              const std::shared_ptr<PredictTaskQueue> &predict_task_queue, bool *create_success) {
  std::unique_lock<std::mutex> l(mtx_init_);
  is_idle_ = false;
  model_worker_ = worker;
  model_buf_ = model_buf;
  size_ = size;
  worker_config_ = worker_config;
  predict_task_queue_ = predict_task_queue;
  create_success_ = create_success;
  is_launch_ = true;
  init_cond_var_.notify_one();
}

InitWorkerManager *InitWorkerManager::GetInstance() {
  static InitWorkerManager instance;
  return &instance;
}

void InitWorkerManager::InitModelWorker(std::shared_ptr<ModelWorker> worker, const char *model_buf, size_t size,
                                        const std::shared_ptr<WorkerConfig> &worker_config,
                                        const std::shared_ptr<PredictTaskQueue> &predict_task_queue,
                                        bool *create_success) {
  std::unique_lock<std::mutex> l(manager_mutex_);
  auto numa_id = worker_config->numa_id;
  if (all_init_worker_.find(numa_id) != all_init_worker_.end()) {
    for (auto &worker_thread : all_init_worker_[numa_id]) {
      if (worker_thread->IsIdle()) {
        MS_LOG(INFO) << "reuse init worker thread.";
        worker_thread->Launch(worker, model_buf, size, worker_config, predict_task_queue, create_success);
        return;
      }
    }
  }
  auto init_worker = std::make_shared<InitWorkerThread>();
  if (init_worker == nullptr) {
    MS_LOG(ERROR) << "create init worker thread failed.";
    *create_success = false;
    return;
  }
  init_worker->CreateInitThread();
  init_worker->Launch(worker, model_buf, size, worker_config, predict_task_queue, create_success);
  if (all_init_worker_.find(numa_id) == all_init_worker_.end()) {
    all_init_worker_[numa_id] = {init_worker};
  } else {
    all_init_worker_[numa_id].push_back(init_worker);
  }
}

InitWorkerManager::~InitWorkerManager() {
  MS_LOG(INFO) << "~InitWorkerManager() begin.";
  for (auto &numa_threads : all_init_worker_) {
    for (auto &thread : numa_threads.second) {
      thread->Destroy();
    }
  }
  MS_LOG(INFO) << "~InitWorkerManager() end.";
}
}  // namespace mindspore
