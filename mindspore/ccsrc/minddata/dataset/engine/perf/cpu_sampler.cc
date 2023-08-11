/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/engine/perf/cpu_sampler.h"

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
#include <sys/syscall.h>
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <fstream>
#include <string>
#include <utility>

#include "minddata/dataset/api/python/pybind_conversion.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/util/path.h"

namespace mindspore {
namespace dataset {
using json = nlohmann::json;
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
#define USING_LINUX
#endif

#if defined(USING_LINUX)
int32_t SystemInfo::num_cpu_ = get_nprocs_conf();
#else
int32_t SystemInfo::num_cpu_ = 0;
#endif

constexpr uint64_t kBInMB = 1024;  // Constant for kByte to MByte division conversion

Status SystemInfo::ParseCpuInfo(const std::string &str) {
  SystemStat system_cpu_stat;
  uint64_t nice = 0;
  uint64_t irq = 0;
  uint64_t softirq = 0;
  if (sscanf_s(str.c_str(), "%*s %lu %lu %lu %lu %lu %lu %lu", &system_cpu_stat.user_stat, &nice,
               &system_cpu_stat.sys_stat, &system_cpu_stat.idle_stat, &system_cpu_stat.io_stat, &irq,
               &softirq) == EOF) {
    return Status(StatusCode::kMDUnexpectedError, "Get System CPU failed.");
  }

  system_cpu_stat.total_stat = system_cpu_stat.user_stat + nice + system_cpu_stat.sys_stat + system_cpu_stat.idle_stat +
                               system_cpu_stat.io_stat + irq + softirq;
  SystemUtil system_cpu_util = {0, 0, 0, 0};
  // Calculate the utilization from the second sampling
  if (!first_sample_) {
    int one_hundred = 100;
    system_cpu_util.user_utilization =
      static_cast<uint8_t>(round((system_cpu_stat.user_stat - prev_sys_stat_.user_stat) * 1.0 /
                                 (system_cpu_stat.total_stat - prev_sys_stat_.total_stat) * one_hundred));
    system_cpu_util.sys_utilization =
      static_cast<uint8_t>(round((system_cpu_stat.sys_stat - prev_sys_stat_.sys_stat) * 1.0 /
                                 (system_cpu_stat.total_stat - prev_sys_stat_.total_stat) * one_hundred));
    system_cpu_util.io_utilization =
      static_cast<uint8_t>(round((system_cpu_stat.io_stat - prev_sys_stat_.io_stat) * 1.0 /
                                 (system_cpu_stat.total_stat - prev_sys_stat_.total_stat) * one_hundred));
    system_cpu_util.idle_utilization =
      static_cast<uint8_t>(round((system_cpu_stat.idle_stat - prev_sys_stat_.idle_stat) * 1.0 /
                                 (system_cpu_stat.total_stat - prev_sys_stat_.total_stat) * one_hundred));
  }
  // append the 0 util as well to maintain sys_cpu_util_.size == ts_.size
  (void)sys_cpu_util_.emplace_back(system_cpu_util);
  prev_sys_stat_ = system_cpu_stat;
  return Status::OK();
}

Status SystemInfo::ParseCtxt(const std::string &str) {
  uint64_t ctxt;
  if (sscanf_s(str.c_str(), "%*s %lu", &ctxt) == EOF) {
    return Status(StatusCode::kMDUnexpectedError, "Get context switch count failed.");
  }
  // first context switch count will be 0
  auto val = first_sample_ ? 0 : ctxt - prev_context_switch_count_;
  context_switch_count_.push_back(val);
  prev_context_switch_count_ = ctxt;
  return Status::OK();
}

Status SystemInfo::ParseRunningProcess(const std::string &str) {
  uint32_t running_process;
  if (sscanf_s(str.c_str(), "%*s %ud", &running_process) == EOF) {
    return Status(StatusCode::kMDUnexpectedError, "Get context switch count failed.");
  }
  running_process_.push_back(running_process);
  return Status::OK();
}

Status SystemInfo::SampleAndGetCurrPrevStat(SystemStat *current_stat, SystemStat *previous_stat) {
  RETURN_UNEXPECTED_IF_NULL(previous_stat);
  std::ifstream file("/proc/stat");
  if (!file.is_open()) {
    MS_LOG(INFO) << "Failed to open /proc/stat file.";
    return {StatusCode::kMDUnexpectedError, "Failed to open /proc/stat file."};
  }
  *previous_stat = prev_sys_stat_;
  bool first_line = true;
  std::string line;
  Status s;
  while (getline(file, line)) {
    if (first_line) {
      first_line = false;
      RETURN_IF_NOT_OK(ParseCpuInfo(line));
      s = ParseCpuInfo(line);
      if (s.IsError()) {
        file.close();
        return s;
      }
    }
    if (line.find("ctxt") != std::string::npos) {
      RETURN_IF_NOT_OK(ParseCtxt(line));
      s = ParseCtxt(line);
      if (s.IsError()) {
        file.close();
        return s;
      }
    }
    if (line.find("procs_running") != std::string::npos) {
      RETURN_IF_NOT_OK(ParseRunningProcess(line));
      s = ParseRunningProcess(line);
      if (s.IsError()) {
        file.close();
        return s;
      }
    }
  }
  // after the loop above, prev_sys_stat_ has the current value
  *current_stat = prev_sys_stat_;
  file.close();

  first_sample_ = false;
  RETURN_IF_NOT_OK(SampleSystemMemInfo());
  return Status::OK();
}

Status SystemInfo::SampleSystemMemInfo() {
  std::ifstream file("/proc/meminfo");
  if (!file.is_open()) {
    MS_LOG(INFO) << "Unable to open /proc/meminfo. Continue processing.";
    last_mem_sampling_failed_ = true;
    // Note: Return Status:OK() although failed to open /proc/meminfo file
    return Status::OK();
  }
  std::string line;
  uint64_t total = 0;
  uint64_t available = 0;
  uint64_t used = 0;
  uint64_t curr_val = 0;

  (void)getline(file, line);
  if (sscanf_s(line.c_str(), "%*[MemTotal:] %lu %*[kB]", &curr_val) == 1) {
    total = curr_val;
    (void)getline(file, line);
    (void)getline(file, line);
    if (sscanf_s(line.c_str(), "%*[MemAvailable:] %lu %*[kB]", &curr_val) == 1) {
      available = curr_val;
      used = total - available;
      last_mem_sampling_failed_ = false;
    } else {
      prev_system_memory_info_ = {0, 0, 0};
      last_mem_sampling_failed_ = true;
    }
  } else {
    prev_system_memory_info_ = {0, 0, 0};
    last_mem_sampling_failed_ = true;
  }
  // Note: Must close file before returning from this function.
  file.close();

  if (last_mem_sampling_failed_) {
    return Status::OK();
  }

  prev_system_memory_info_.total_mem = static_cast<float>(total) / kBInMB;
  prev_system_memory_info_.available_mem = static_cast<float>(available) / kBInMB;
  prev_system_memory_info_.used_mem = static_cast<float>(used) / kBInMB;

  system_memory_info_.push_back(SystemMemInfo{
    prev_system_memory_info_.total_mem, prev_system_memory_info_.available_mem, prev_system_memory_info_.used_mem});

  return Status::OK();
}

Status SystemInfo::GetUserCpuUtil(uint64_t start_index, uint64_t end_index, std::vector<uint8_t> *result) const {
  RETURN_UNEXPECTED_IF_NULL(result);
  MS_LOG(DEBUG) << "start_index: " << start_index << " end_index: " << end_index
                << " sys_cpu_util_.size: " << sys_cpu_util_.size();
  CHECK_FAIL_RETURN_UNEXPECTED(start_index <= end_index,
                               "Expected start_index <= end_index. Got start_index: " + std::to_string(start_index) +
                                 " end_index: " + std::to_string(end_index));
  CHECK_FAIL_RETURN_UNEXPECTED(
    end_index <= sys_cpu_util_.size(),
    "Expected end_index <= sys_cpu_util_.size(). Got end_index: " + std::to_string(end_index) +
      " sys_cpu_util_.size: " + std::to_string(sys_cpu_util_.size()));
  (void)std::transform(sys_cpu_util_.begin() + start_index, sys_cpu_util_.begin() + end_index,
                       std::back_inserter(*result), [&](const SystemUtil &info) { return info.user_utilization; });
  return Status::OK();
}

Status SystemInfo::GetSysCpuUtil(uint64_t start_index, uint64_t end_index, std::vector<uint8_t> *result) const {
  RETURN_UNEXPECTED_IF_NULL(result);
  MS_LOG(DEBUG) << "start_index: " << start_index << " end_index: " << end_index
                << "sys_cpu_util_.size: " << sys_cpu_util_.size();
  CHECK_FAIL_RETURN_UNEXPECTED(start_index <= end_index,
                               "Expected start_index <= end_index. Got start_index: " + std::to_string(start_index) +
                                 " end_index: " + std::to_string(end_index));
  CHECK_FAIL_RETURN_UNEXPECTED(
    end_index <= sys_cpu_util_.size(),
    "Expected end_index <= sys_cpu_util_.size(). Got end_index: " + std::to_string(end_index) +
      " sys_cpu_util_.size: " + std::to_string(sys_cpu_util_.size()));
  (void)std::transform(sys_cpu_util_.begin() + start_index, sys_cpu_util_.begin() + end_index,
                       std::back_inserter(*result), [&](const SystemUtil &info) { return info.sys_utilization; });
  return Status::OK();
}

std::vector<uint8_t> SystemInfo::GetIOCpuUtil() const {
  std::vector<uint8_t> io_util;
  (void)std::transform(sys_cpu_util_.begin(), sys_cpu_util_.end(), std::back_inserter(io_util),
                       [&](const SystemUtil &info) { return info.io_utilization; });
  return io_util;
}

std::vector<uint8_t> SystemInfo::GetIdleCpuUtil() const {
  std::vector<uint8_t> idle_util;
  (void)std::transform(sys_cpu_util_.begin(), sys_cpu_util_.end(), std::back_inserter(idle_util),
                       [&](const SystemUtil &info) { return info.idle_utilization; });
  return idle_util;
}

std::vector<uint16_t> TaskCpuInfo::GetSysCpuUtil() const {
  std::vector<uint16_t> sys_util;
  (void)std::transform(task_cpu_util_.begin(), task_cpu_util_.end(), std::back_inserter(sys_util),
                       [&](const TaskUtil &info) {
                         return static_cast<uint16_t>(info.sys_utilization * static_cast<float>(SystemInfo::num_cpu_));
                       });
  return sys_util;
}

std::vector<uint16_t> TaskCpuInfo::GetUserCpuUtil() const {
  std::vector<uint16_t> user_util;
  (void)std::transform(task_cpu_util_.begin(), task_cpu_util_.end(), std::back_inserter(user_util),
                       [&](const TaskUtil &info) {
                         return static_cast<uint16_t>(info.user_utilization * static_cast<float>(SystemInfo::num_cpu_));
                       });
  return user_util;
}

TaskUtil TaskCpuInfo::GetLatestCpuUtil() const {
  TaskUtil ret = {0, 0};
  if (!task_cpu_util_.empty() && !last_sampling_failed_) {
    ret = task_cpu_util_.back();
  }
  return ret;
}

Status ProcessInfo::SampleMemInfo() {
  std::ifstream file("/proc/" + std::to_string(pid_) + "/smaps");
  if (!file.is_open()) {
    MS_LOG(INFO) << "Unable to open /proc/" << pid_ << "/smaps file. Continue processing.";
    last_mem_sampling_failed_ = true;
    // Note: Return Status:OK() although failed to open /proc/<pid>/smaps file
    return Status::OK();
  }
  std::string line;
  uint64_t total_vss = 0;
  uint64_t total_rss = 0;
  uint64_t total_pss = 0;
  uint64_t curr_val = 0;
  while (getline(file, line)) {
    if (sscanf_s(line.c_str(), "%*[Size:] %lu %*[kB]", &curr_val) == 1) {
      total_vss += curr_val;
    } else if (sscanf_s(line.c_str(), "%*[Rss:] %lu %*[kB]", &curr_val) == 1) {
      total_rss += curr_val;
    } else if (sscanf_s(line.c_str(), "%*[Pss:] %lu %*[kB]", &curr_val) == 1) {
      total_pss += curr_val;
    }
  }
  file.close();
  last_mem_sampling_failed_ = false;

  prev_memory_info_.vss = static_cast<float>(total_vss) / kBInMB;
  prev_memory_info_.rss = static_cast<float>(total_rss) / kBInMB;
  prev_memory_info_.pss = static_cast<float>(total_pss) / kBInMB;

  // Sum the memory usage of all child processes and add to parent process
  if (IsParent()) {
    for (auto child : child_processes_) {
      MemoryInfo child_mem_info = child->GetLatestMemoryInfo();
      prev_memory_info_.vss += child_mem_info.vss;
      prev_memory_info_.rss += child_mem_info.rss;
      prev_memory_info_.pss += child_mem_info.pss;
    }
  }

  // Append latest data to vector if we want to track history for this process
  if (track_sampled_history_) {
    process_memory_info_.push_back(MemoryInfo{prev_memory_info_.vss, prev_memory_info_.rss, prev_memory_info_.pss});
  }

  return Status::OK();
}

Status ProcessInfo::Sample(uint64_t total_time_elapsed) {
  std::ifstream file("/proc/" + std::to_string(pid_) + "/stat");
  if (!file.is_open()) {
    MS_LOG(INFO) << "Unable to open /proc/" << pid_ << "/stat file. Continue processing.";
    last_sampling_failed_ = true;
    RETURN_IF_NOT_OK(SampleMemInfo());
    // Note: Return Status:OK() although failed to open /proc/<pid>/stat file
    return Status::OK();
  }
  std::string str;
  (void)getline(file, str);
  uint64_t utime = 0, stime = 0;
  if (sscanf_s(str.c_str(), "%*d %*s %*s %*lu %*lu %*lu %*lu %*lu %*lu %*lu %*lu %*lu %*lu %lu %lu", &utime, &stime) ==
      EOF) {
    file.close();
    last_sampling_failed_ = true;
    return Status(StatusCode::kMDUnexpectedError, "Get device CPU failed.");
  }
  file.close();
  last_sampling_failed_ = false;
  if (!first_sample_ && total_time_elapsed > 0) {
    float user_util = (utime - prev_task_stat_.user_stat) * 1.0 / (total_time_elapsed)*100.0;
    float sys_util = (stime - prev_task_stat_.sys_stat) * 1.0 / (total_time_elapsed)*100.0;
    (void)task_cpu_util_.emplace_back(TaskUtil{user_util, sys_util});
  }
  prev_task_stat_.user_stat = utime;
  prev_task_stat_.sys_stat = stime;
  first_sample_ = false;
  RETURN_IF_NOT_OK(SampleMemInfo());
  return Status::OK();
}

Status ThreadCpuInfo::Sample(uint64_t total_time_elapsed) {
  if (last_sampling_failed_) {
    // thread is probably terminated
    return Status::OK();
  }
  std::ifstream file("/proc/" + std::to_string(pid_) + "/task/" + std::to_string(tid_) + "/stat");
  if (!file.is_open()) {
    MS_LOG(INFO) << "Unable to open /proc/" << pid_ << "/task/" << tid_ << "/stat file. Continue processing.";
    last_sampling_failed_ = true;
    return Status::OK();
  }
  std::string str;
  (void)getline(file, str);
  uint64_t utime;
  uint64_t stime;
  if (sscanf_s(str.c_str(), "%*d %*s %*s %*lu %*lu %*lu %*lu %*lu %*lu %*lu %*lu %*lu %*lu %lu %lu", &utime, &stime) ==
      EOF) {
    file.close();
    last_sampling_failed_ = true;
    return Status(StatusCode::kMDUnexpectedError, "Get thread CPU failed.");
  }
  file.close();
  last_sampling_failed_ = false;
  if (!first_sample_) {
    float user_util = ((utime - prev_task_stat_.user_stat) * 1.0 / total_time_elapsed) * 100.0;
    float sys_util = ((stime - prev_task_stat_.sys_stat) * 1.0 / total_time_elapsed) * 100.0;
    (void)task_cpu_util_.emplace_back(TaskUtil{user_util, sys_util});
  }
  prev_task_stat_.user_stat = utime;
  prev_task_stat_.sys_stat = stime;
  first_sample_ = false;
  return Status::OK();
}

bool MDOperatorCpuInfo::TaskExists(pid_t id) const { return task_by_id_.find(id) != task_by_id_.end(); }

void MDOperatorCpuInfo::AddTask(const std::shared_ptr<TaskCpuInfo> &task_ptr) {
  auto id = task_ptr->GetId();
  if (!TaskExists(id)) {
    (void)task_by_id_.emplace(id, task_ptr);
  }
}

void MDOperatorCpuInfo::CalculateOperatorUtilization() {
  OpUtil op_util{0, 0};
  for (auto const &[task_id, task_ptr] : task_by_id_) {
    MS_LOG(DEBUG) << "Processing task_id: " << task_id;
    auto task_util = task_ptr->GetLatestCpuUtil();
    op_util.user_utilization += task_util.user_utilization;
    op_util.sys_utilization += task_util.sys_utilization;
  }
  (void)op_cpu_util_.emplace_back(op_util);
}

Status MDOperatorCpuInfo::GetUserCpuUtil(uint64_t start_index, uint64_t end_index,
                                         std::vector<uint16_t> *result) const {
  RETURN_UNEXPECTED_IF_NULL(result);
  MS_LOG(DEBUG) << "start_index: " << start_index << " end_index: " << end_index
                << " op_cpu_util_.size: " << op_cpu_util_.size();
  CHECK_FAIL_RETURN_UNEXPECTED(start_index <= end_index,
                               "Expected start_index <= end_index. Got start_index: " + std::to_string(start_index) +
                                 " end_index: " + std::to_string(end_index));
  CHECK_FAIL_RETURN_UNEXPECTED(
    end_index <= op_cpu_util_.size(),
    "Expected end_index <= op_cpu_util_.size(). Got end_index: " + std::to_string(end_index) +
      " op_cpu_util_.size: " + std::to_string(op_cpu_util_.size()));
  auto first_iter = op_cpu_util_.begin() + start_index;
  auto last_iter = op_cpu_util_.begin() + end_index;
  (void)std::transform(first_iter, last_iter, std::back_inserter(*result), [&](const OpUtil &info) {
    return static_cast<uint16_t>(info.user_utilization * static_cast<float>(SystemInfo::num_cpu_));
  });
  return Status::OK();
}

Status MDOperatorCpuInfo::GetSysCpuUtil(uint64_t start_index, uint64_t end_index, std::vector<uint16_t> *result) const {
  RETURN_UNEXPECTED_IF_NULL(result);
  MS_LOG(DEBUG) << "start_index: " << start_index << " end_index: " << end_index
                << " op_cpu_util_.size: " << op_cpu_util_.size();
  CHECK_FAIL_RETURN_UNEXPECTED(start_index <= end_index,
                               "Expected start_index <= end_index. Got start_index: " + std::to_string(start_index) +
                                 " end_index: " + std::to_string(end_index));
  CHECK_FAIL_RETURN_UNEXPECTED(
    end_index <= op_cpu_util_.size(),
    "Expected end_index <= op_cpu_util_.size(). Got end_index: " + std::to_string(end_index) +
      " op_cpu_util_.size: " + std::to_string(op_cpu_util_.size()));
  auto first_iter = op_cpu_util_.begin() + start_index;
  auto last_iter = op_cpu_util_.begin() + end_index;
  (void)std::transform(first_iter, last_iter, std::back_inserter(*result), [&](const OpUtil &info) {
    return static_cast<uint16_t>(info.sys_utilization * static_cast<float>(SystemInfo::num_cpu_));
  });
  return Status::OK();
}

Status CpuSampler::Sample() {
  if (active_ == false) {
    return Status::OK();
  }
  std::lock_guard<std::mutex> guard(lock_);
  // Function to Update TaskList
  // Loop through all tasks to find any new threads
  // Get all multi-processing Ops from Python only if fetched_all_process = False
  // Create new TaskCpuInfo as required and update OpInfo
  RETURN_IF_NOT_OK(UpdateTaskList());

  // Sample SystemInfo - Update current and move current to previous stat and calc Util
  SystemStat current_sys_stat;
  SystemStat previous_sys_stat;
  RETURN_IF_NOT_OK(sys_info_.SampleAndGetCurrPrevStat(&current_sys_stat, &previous_sys_stat));
  auto total_time_elapsed = current_sys_stat.total_stat - previous_sys_stat.total_stat;

  // Call Sample on all
  // Read /proc/ files and get stat, calculate util
  for (auto &task_ptr : tasks_) {
    (void)task_ptr->Sample(total_time_elapsed);
  }

  // Call after Sample is called on all child processes
  (void)main_process_info_->Sample(total_time_elapsed);

  // Calculate OperatorCpuInfo
  for (auto &[op_id, op_info] : op_info_by_id_) {
    MS_LOG(DEBUG) << "Calculate operator cpu utilization for OpId: " << op_id;
    op_info.CalculateOperatorUtilization();
  }

  // Get sampling time.
  (void)ts_.emplace_back(ProfilingTime::GetCurMilliSecond());

  return Status::OK();
}

Status CpuSampler::UpdateTaskList() {
  List<Task> allTasks = tree->AllTasks()->GetTask();
  for (auto &task : allTasks) {
    int32_t op_id = task.get_operator_id();
    // check if the op_info was initialized in Init
    auto iter = op_info_by_id_.find(op_id);
    if (iter != op_info_by_id_.end()) {
      int32_t tid = task.get_linux_id();
      if (!iter->second.TaskExists(tid)) {
        auto task_cpu_info_ptr = std::make_shared<ThreadCpuInfo>(main_pid_, tid);
        (void)tasks_.emplace_back(task_cpu_info_ptr);
        iter->second.AddTask(task_cpu_info_ptr);
      }
    }
  }
  for (const auto &op : *tree) {
    std::vector<int32_t> pids = op.GetMPWorkerPIDs();
    int32_t op_id = op.id();
    auto iter = op_info_by_id_.find(op_id);
    if (iter != op_info_by_id_.end()) {
      for (auto pid : pids) {
        if (!iter->second.TaskExists(pid)) {
          auto task_cpu_info_ptr = std::make_shared<ProcessInfo>(pid);
          (void)tasks_.emplace_back(task_cpu_info_ptr);
          main_process_info_->AddChildProcess(task_cpu_info_ptr);
          iter->second.AddTask(task_cpu_info_ptr);
        }
      }
    }
  }

  if (!fetched_all_python_multiprocesses_ && tree->IsPython()) {
    py::gil_scoped_acquire gil_acquire;
    py::module ds = py::module::import("mindspore.dataset.engine.datasets");
    py::tuple process_info = ds.attr("_get_operator_process")();
    auto sub_process = py::reinterpret_borrow<py::dict>(process_info[0]);
    fetched_all_python_multiprocesses_ = py::reinterpret_borrow<py::bool_>(process_info[1]);
    // parse dict value
    auto op_to_process = toIntMap(sub_process);
    for (auto const &[op_id, process_list] : op_to_process) {
      for (auto pid : process_list) {
        auto iter = op_info_by_id_.find(op_id);
        if (iter != op_info_by_id_.end()) {
          if (!iter->second.TaskExists(pid)) {
            auto task_cpu_info_ptr = std::make_shared<ProcessInfo>(pid);
            (void)tasks_.emplace_back(task_cpu_info_ptr);
            main_process_info_->AddChildProcess(task_cpu_info_ptr);
            iter->second.AddTask(task_cpu_info_ptr);
          }
        }
      }
    }
  }

  return Status::OK();
}

Status CpuSampler::Init() {
#if defined(USING_LINUX)
  main_pid_ = syscall(SYS_getpid);
#endif
  for (auto iter = tree->begin(); iter != tree->end(); (void)iter++) {
    auto op_id = iter->id();
    (void)op_info_by_id_.emplace(std::make_pair(op_id, MDOperatorCpuInfo(op_id)));
  }
  // thread id of main thread is same as the process ID
  main_thread_cpu_info_ = std::make_shared<ThreadCpuInfo>(main_pid_, main_pid_);
  (void)tasks_.emplace_back(main_thread_cpu_info_);
  main_process_info_ = std::make_shared<ProcessInfo>(main_pid_, true);
  return Status::OK();
}

void CpuSampler::Clear() {
  ts_.clear();
  tasks_.clear();
  main_thread_cpu_info_.reset();
  main_process_info_.reset();
  op_info_by_id_.clear();
  fetched_all_python_multiprocesses_ = false;
}

Status CpuSampler::ChangeFileMode(const std::string &dir_path, const std::string &rank_id) {
  Path path = GetFileName(dir_path, rank_id);
  std::string file_path = path.ToString();
  if (chmod(common::SafeCStr(file_path), S_IRUSR | S_IWUSR) == -1) {
    std::string err_str = "Change file mode failed," + file_path;
    return Status(StatusCode::kMDUnexpectedError, err_str);
  }
  return Status::OK();
}

Status CpuSampler::SaveToFile(const std::string &dir_path, const std::string &rank_id) {
  Path path = GetFileName(dir_path, rank_id);
  // Remove the file if it exists (from prior profiling usage)
  RETURN_IF_NOT_OK(path.Remove());
  std::string file_path = path.ToString();

  // construct json obj to write to file
  json output;
  output["cpu_processor_num"] = SystemInfo::num_cpu_;
  std::vector<uint8_t> system_user_util, system_sys_util;
  // end_index = ts_.size() essentially means to get all sampled points
  (void)sys_info_.GetUserCpuUtil(0, ts_.size(), &system_user_util);
  (void)sys_info_.GetSysCpuUtil(0, ts_.size(), &system_sys_util);
  output["device_info"] = {{"context_switch_count", sys_info_.GetContextSwitchCount()},
                           {"idle_utilization", sys_info_.GetIdleCpuUtil()},
                           {"io_utilization", sys_info_.GetIOCpuUtil()},
                           {"sys_utilization", system_sys_util},
                           {"user_utilization", system_user_util},
                           {"runnable_process", sys_info_.GetRunningProcess()}};
  // array of op_info json objects
  json op_infos;
  for (auto &[op_id, op_info] : op_info_by_id_) {
    MS_LOG(INFO) << "Processing op_id: " << op_id;
    std::vector<uint16_t> user_util, sys_util;
    (void)op_info.GetSysCpuUtil(0, ts_.size(), &sys_util);
    (void)op_info.GetUserCpuUtil(0, ts_.size(), &user_util);
    json op_info_json = {{"metrics", {{"user_utilization", user_util}, {"sys_utilization", sys_util}}},
                         {"op_id", op_id}};
    (void)op_infos.emplace_back(op_info_json);
  }
  output["op_info"] = op_infos;

  output["process_info"] = {{"user_utilization", main_process_info_->GetUserCpuUtil()},
                            {"sys_utilization", main_process_info_->GetSysCpuUtil()}};

  output["sampling_interval"] = GlobalContext::config_manager()->monitor_sampling_interval();
  output["time_stamp"] = ts_;

  std::vector<float> vss, rss, pss;
  (void)main_process_info_->GetMemoryInfo(ProcessMemoryMetric::kVSS, 0, ts_.size(), &vss);
  (void)main_process_info_->GetMemoryInfo(ProcessMemoryMetric::kRSS, 0, ts_.size(), &rss);
  (void)main_process_info_->GetMemoryInfo(ProcessMemoryMetric::kPSS, 0, ts_.size(), &pss);
  output["process_memory_info"] = {{"vss_mbytes", vss}, {"rss_mbytes", rss}, {"pss_mbytes", pss}};

  std::vector<float> mem_total, mem_avail, mem_used;
  (void)sys_info_.GetSystemMemInfo(SystemMemoryMetric::kMemoryTotal, 0, ts_.size(), &mem_total);
  (void)sys_info_.GetSystemMemInfo(SystemMemoryMetric::kMemoryAvailable, 0, ts_.size(), &mem_avail);
  (void)sys_info_.GetSystemMemInfo(SystemMemoryMetric::kMemoryUsed, 0, ts_.size(), &mem_used);
  output["system_memory_info"] = {{"total_sys_memory_mbytes", mem_total},
                                  {"available_sys_memory_mbytes", mem_avail},
                                  {"used_sys_memory_mbytes", mem_used}};

  // Discard the content of the file when opening.
  std::ofstream os(file_path, std::ios::trunc);
  os << output;
  os.close();

  return Status::OK();
}

Status CpuSampler::GetOpUserCpuUtil(int32_t op_id, uint64_t start_ts, uint64_t end_ts, std::vector<uint16_t> *result) {
  std::lock_guard<std::mutex> guard(lock_);
  // find first ts that is not less than start_ts
  auto lower = std::lower_bound(ts_.begin(), ts_.end(), start_ts);
  // find first ts that is greater than end_ts
  auto upper = std::upper_bound(ts_.begin(), ts_.end(), end_ts);
  // std::distance is O(1) since vector allows random access
  auto start_index = std::distance(ts_.begin(), lower);
  auto end_index = std::distance(ts_.begin(), upper);
  auto op_info = op_info_by_id_.find(op_id);
  CHECK_FAIL_RETURN_UNEXPECTED(op_info != op_info_by_id_.end(), "Op Id: " + std::to_string(op_id) + " not found.");
  return op_info->second.GetUserCpuUtil(start_index, end_index, result);
}

Status CpuSampler::GetOpSysCpuUtil(int32_t op_id, uint64_t start_ts, uint64_t end_ts, std::vector<uint16_t> *result) {
  std::lock_guard<std::mutex> guard(lock_);
  // find first ts that is not less than start_ts
  auto lower = std::lower_bound(ts_.begin(), ts_.end(), start_ts);
  // find first ts that is greater than end_ts
  auto upper = std::upper_bound(ts_.begin(), ts_.end(), end_ts);
  // std::distance is O(1) since vector allows random access
  auto start_index = std::distance(ts_.begin(), lower);
  auto end_index = std::distance(ts_.begin(), upper);
  auto op_info = op_info_by_id_.find(op_id);
  CHECK_FAIL_RETURN_UNEXPECTED(op_info != op_info_by_id_.end(), "Op Id: " + std::to_string(op_id) + " not found.");
  return op_info->second.GetSysCpuUtil(start_index, end_index, result);
}

Status CpuSampler::GetSystemUserCpuUtil(uint64_t start_ts, uint64_t end_ts, std::vector<uint8_t> *result) {
  std::lock_guard<std::mutex> guard(lock_);
  // find first ts that is not less than start_ts
  auto lower = std::lower_bound(ts_.begin(), ts_.end(), start_ts);
  // find first ts that is greater than end_ts
  auto upper = std::upper_bound(ts_.begin(), ts_.end(), end_ts);
  // std::distance is O(1) since vector allows random access
  auto start_index = std::distance(ts_.begin(), lower);
  auto end_index = std::distance(ts_.begin(), upper);
  return sys_info_.GetUserCpuUtil(start_index, end_index, result);
}

Status CpuSampler::GetSystemSysCpuUtil(uint64_t start_ts, uint64_t end_ts, std::vector<uint8_t> *result) {
  std::lock_guard<std::mutex> guard(lock_);
  // find first ts that is not less than start_ts
  auto lower = std::lower_bound(ts_.begin(), ts_.end(), start_ts);
  // find first ts that is greater than end_ts
  auto upper = std::upper_bound(ts_.begin(), ts_.end(), end_ts);
  // std::distance is O(1) since vector allows random access
  auto start_index = std::distance(ts_.begin(), lower);
  auto end_index = std::distance(ts_.begin(), upper);
  return sys_info_.GetSysCpuUtil(start_index, end_index, result);
}

Path CpuSampler::GetFileName(const std::string &dir_path, const std::string &rank_id) {
  return Path(dir_path) / Path("minddata_cpu_utilization_" + rank_id + ".json");
}

MemoryInfo ProcessInfo::GetLatestMemoryInfo() const {
  MemoryInfo ret = {0, 0, 0};
  if (!last_mem_sampling_failed_) {
    ret = prev_memory_info_;
  }
  return ret;
}

Status ProcessInfo::GetMemoryInfo(ProcessMemoryMetric metric, uint64_t start_index, uint64_t end_index,
                                  std::vector<float> *result) const {
  RETURN_UNEXPECTED_IF_NULL(result);
  MS_LOG(DEBUG) << "start_index: " << start_index << " end_index: " << end_index
                << "process_memory_info_.size: " << process_memory_info_.size();
  CHECK_FAIL_RETURN_UNEXPECTED(start_index <= end_index,
                               "Expected start_index <= end_index. Got start_index: " + std::to_string(start_index) +
                                 " end_index: " + std::to_string(end_index));
  CHECK_FAIL_RETURN_UNEXPECTED(
    end_index <= process_memory_info_.size(),
    "Expected end_index <= process_memory_info_.size(). Got end_index: " + std::to_string(end_index) +
      " process_memory_info_.size: " + std::to_string(process_memory_info_.size()));
  if (metric == ProcessMemoryMetric::kVSS) {
    (void)std::transform(process_memory_info_.begin() + start_index, process_memory_info_.begin() + end_index,
                         std::back_inserter(*result),
                         [&](const MemoryInfo &info) { return static_cast<float>(info.vss); });
  } else if (metric == ProcessMemoryMetric::kRSS) {
    (void)std::transform(process_memory_info_.begin() + start_index, process_memory_info_.begin() + end_index,
                         std::back_inserter(*result),
                         [&](const MemoryInfo &info) { return static_cast<float>(info.rss); });
  } else if (metric == ProcessMemoryMetric::kPSS) {
    (void)std::transform(process_memory_info_.begin() + start_index, process_memory_info_.begin() + end_index,
                         std::back_inserter(*result),
                         [&](const MemoryInfo &info) { return static_cast<float>(info.pss); });
  }
  return Status::OK();
}

Status CpuSampler::GetProcessMemoryInfo(ProcessMemoryMetric metric, uint64_t start_index, uint64_t end_index,
                                        std::vector<float> *result) {
  return (main_process_info_->GetMemoryInfo(metric, start_index, end_index, result));
}

void ProcessInfo::AddChildProcess(const std::shared_ptr<ProcessInfo> &child_ptr) {
  (void)child_processes_.emplace_back(child_ptr);
}

bool ProcessInfo::IsParent() { return !(child_processes_.empty()); }

Status SystemInfo::GetSystemMemInfo(SystemMemoryMetric metric, uint64_t start_index, uint64_t end_index,
                                    std::vector<float> *result) const {
  RETURN_UNEXPECTED_IF_NULL(result);
  MS_LOG(DEBUG) << "start_index: " << start_index << " end_index: " << end_index
                << "system_memory_info_.size: " << system_memory_info_.size();
  CHECK_FAIL_RETURN_UNEXPECTED(start_index <= end_index,
                               "Expected start_index <= end_index. Got start_index: " + std::to_string(start_index) +
                                 " end_index: " + std::to_string(end_index));
  CHECK_FAIL_RETURN_UNEXPECTED(
    end_index <= system_memory_info_.size(),
    "Expected end_index <= system_memory_info_.size(). Got end_index: " + std::to_string(end_index) +
      " system_memory_info_.size: " + std::to_string(system_memory_info_.size()));
  if (metric == SystemMemoryMetric::kMemoryTotal) {
    (void)std::transform(system_memory_info_.begin() + start_index, system_memory_info_.begin() + end_index,
                         std::back_inserter(*result),
                         [&](const SystemMemInfo &info) { return static_cast<float>(info.total_mem); });
  } else if (metric == SystemMemoryMetric::kMemoryAvailable) {
    (void)std::transform(system_memory_info_.begin() + start_index, system_memory_info_.begin() + end_index,
                         std::back_inserter(*result),
                         [&](const SystemMemInfo &info) { return static_cast<float>(info.available_mem); });
  } else if (metric == SystemMemoryMetric::kMemoryUsed) {
    (void)std::transform(system_memory_info_.begin() + start_index, system_memory_info_.begin() + end_index,
                         std::back_inserter(*result),
                         [&](const SystemMemInfo &info) { return static_cast<float>(info.used_mem); });
  }
  return Status::OK();
}

Status CpuSampler::GetSystemMemoryInfo(SystemMemoryMetric metric, uint64_t start_index, uint64_t end_index,
                                       std::vector<float> *result) {
  return (sys_info_.GetSystemMemInfo(metric, start_index, end_index, result));
}
}  // namespace dataset
}  // namespace mindspore
