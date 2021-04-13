/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/engine/perf/cpu_sampling.h"
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
#include <sys/syscall.h>
#endif
#include <math.h>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include "minddata/dataset/api/python/pybind_conversion.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/util/path.h"

using json = nlohmann::json;
namespace mindspore {
namespace dataset {
bool BaseCpu::fetched_all_process_shared = false;
std::unordered_map<int32_t, std::vector<pid_t>> BaseCpu::op_process_shared = {};

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
#define USING_LINUX
#endif

BaseCpu::BaseCpu() {
  pre_cpu_stat_.user_stat_ = 0;
  pre_cpu_stat_.sys_stat_ = 0;
  pre_cpu_stat_.io_stat_ = 0;
  pre_cpu_stat_.idle_stat_ = 0;
  pre_cpu_stat_.total_stat_ = 0;
  fetched_all_process = false;
  pre_fetched_state = false;
  cpu_processor_num_ = 0;
}

Status DeviceCpu::ParseCpuInfo(const std::string &str) {
  CpuStat cpu_stat;
  uint64_t nice = 0;
  uint64_t irq = 0;
  uint64_t softirq = 0;
  if (sscanf_s(str.c_str(), "%*s %lu %lu %lu %lu %lu %lu %lu", &cpu_stat.user_stat_, &nice, &cpu_stat.sys_stat_,
               &cpu_stat.idle_stat_, &cpu_stat.io_stat_, &irq, &softirq) == EOF) {
    return Status(StatusCode::kMDUnexpectedError, "Get device CPU failed.");
  }

  cpu_stat.total_stat_ =
    cpu_stat.user_stat_ + nice + cpu_stat.sys_stat_ + cpu_stat.idle_stat_ + cpu_stat.io_stat_ + irq + softirq;
  // Calculate the utilization from the second sampling
  if (!first_collect_) {
    CpuUtil info;
    info.user_utilization_ = floor((cpu_stat.user_stat_ - pre_cpu_stat_.user_stat_) * 1.0 /
                                     (cpu_stat.total_stat_ - pre_cpu_stat_.total_stat_) * 100 +
                                   0.5);
    info.sys_utilization_ = floor((cpu_stat.sys_stat_ - pre_cpu_stat_.sys_stat_) * 1.0 /
                                    (cpu_stat.total_stat_ - pre_cpu_stat_.total_stat_) * 100 +
                                  0.5);
    info.io_utilization_ = floor((cpu_stat.io_stat_ - pre_cpu_stat_.io_stat_) * 1.0 /
                                   (cpu_stat.total_stat_ - pre_cpu_stat_.total_stat_) * 100 +
                                 0.5);
    info.idle_utilization_ = floor((cpu_stat.idle_stat_ - pre_cpu_stat_.idle_stat_) * 1.0 /
                                     (cpu_stat.total_stat_ - pre_cpu_stat_.total_stat_) * 100 +
                                   0.5);
    cpu_util_.emplace_back(info);
  }
  pre_cpu_stat_.user_stat_ = cpu_stat.user_stat_;
  pre_cpu_stat_.sys_stat_ = cpu_stat.sys_stat_;
  pre_cpu_stat_.io_stat_ = cpu_stat.io_stat_;
  pre_cpu_stat_.idle_stat_ = cpu_stat.idle_stat_;
  pre_cpu_stat_.total_stat_ = cpu_stat.total_stat_;

  return Status::OK();
}

Status DeviceCpu::ParseCtxt(const std::string &str) {
  uint64_t ctxt;
  if (sscanf_s(str.c_str(), "%*s %lu", &ctxt) == EOF) {
    return Status(StatusCode::kMDUnexpectedError, "Get context switch count failed.");
  }
  // Calculate the utilization from the second sampling
  if (!first_collect_) {
    context_switch_count_.push_back(ctxt - pre_context_switch_count_);
  }
  pre_context_switch_count_ = ctxt;
  return Status::OK();
}

Status DeviceCpu::ParseRunningProcess(const std::string &str) {
  uint32_t running_process;
  if (sscanf_s(str.c_str(), "%*s %ud", &running_process) == EOF) {
    return Status(StatusCode::kMDUnexpectedError, "Get context switch count failed.");
  }
  // Drop the first value in order to collect same amount of CPU utilization
  if (!first_collect_) {
    running_process_.push_back(running_process);
  }

  return Status::OK();
}

Status DeviceCpu::Collect(const ExecutionTree *tree) {
  std::ifstream file("/proc/stat");
  if (!file.is_open()) {
    MS_LOG(INFO) << "Open CPU file failed when collect CPU information";
    return Status::OK();
  }
  bool first_line = true;
  std::string line;
  while (getline(file, line)) {
    if (first_line) {
      first_line = false;
      RETURN_IF_NOT_OK(ParseCpuInfo(line));
    }
    if (line.find("ctxt") != std::string::npos) {
      RETURN_IF_NOT_OK(ParseCtxt(line));
    }
    if (line.find("procs_running") != std::string::npos) {
      RETURN_IF_NOT_OK(ParseRunningProcess(line));
    }
  }
  file.close();

  first_collect_ = false;
  return Status::OK();
}
Status DeviceCpu::Analyze(std::string *const name, double *utilization, std::string *const extra_message) {
  name->clear();
  name->append("device_info");
  int total_samples = cpu_util_.size();
  int sum = 0;
  // Only analyze the middle half of the samples
  // Starting and ending may be impacted by startup or ending pipeline activities
  int start_analyze = total_samples / 4;
  int end_analyze = total_samples - start_analyze;

  for (int i = start_analyze; i < end_analyze; i++) {
    sum += cpu_util_[i].user_utilization_;
    sum += cpu_util_[i].sys_utilization_;
  }

  // Note device utilization is already in range of 0-1, so don't
  // need to divide by number of CPUS
  if ((end_analyze - start_analyze) > 0) {
    *utilization = sum / (end_analyze - start_analyze);
  }
  return Status::OK();
}

Status DeviceCpu::SaveToFile(const std::string &file_path) {
  Path path = Path(file_path);
  json output;
  if (path.Exists()) {
    MS_LOG(DEBUG) << file_path << " exists already";
    std::ifstream file(file_path);
    file >> output;
  } else {
    output["sampling_interval"] = GlobalContext::config_manager()->monitor_sampling_interval();
  }

  std::vector<int8_t> user_util;
  std::transform(cpu_util_.begin(), cpu_util_.end(), std::back_inserter(user_util),
                 [&](const CpuUtil &info) { return info.user_utilization_; });
  std::vector<int8_t> sys_util;
  std::transform(cpu_util_.begin(), cpu_util_.end(), std::back_inserter(sys_util),
                 [&](const CpuUtil &info) { return info.sys_utilization_; });
  std::vector<int8_t> io_util;
  std::transform(cpu_util_.begin(), cpu_util_.end(), std::back_inserter(io_util),
                 [&](const CpuUtil &info) { return info.io_utilization_; });
  std::vector<int8_t> idle_util;
  std::transform(cpu_util_.begin(), cpu_util_.end(), std::back_inserter(idle_util),
                 [&](const CpuUtil &info) { return info.idle_utilization_; });

  output["device_info"] = {{"user_utilization", user_util},
                           {"sys_utilization", sys_util},
                           {"io_utilization", io_util},
                           {"idle_utilization", idle_util},
                           {"runable_processes", running_process_},
                           {"context_switch_count", context_switch_count_}};

  // Discard the content of the file when opening.
  std::ofstream os(file_path, std::ios::trunc);
  os << output;

  MS_LOG(INFO) << "Save device CPU success.";
  return Status::OK();
}

Status OperatorCpu::ParseCpuInfo(int32_t op_id, int64_t thread_id,
                                 std::unordered_map<int32_t, std::unordered_map<int64_t, CpuOpStat>> *op_stat) {
  pid_t pid = 0;
#if defined(USING_LINUX)
  pid = syscall(SYS_getpid);
#endif
  std::string stat_path = "/proc/" + std::to_string(pid) + "/task/" + std::to_string(thread_id) + "/stat";

  // Judge whether file exist first
  Path temp_path(stat_path);
  if (!temp_path.Exists()) {
    (*op_stat)[op_id][thread_id].user_stat_ = 0;
    (*op_stat)[op_id][thread_id].sys_stat_ = 0;
    return Status(StatusCode::kMDFileNotExist);
  }

  std::ifstream file(stat_path);
  if (!file.is_open()) {
    MS_LOG(INFO) << "Open CPU file failed when collect CPU information";
    return Status::OK();
  }
  std::string str;
  getline(file, str);
  uint64_t utime;
  uint64_t stime;
  if (sscanf_s(str.c_str(), "%*d %*s %*s %*lu %*lu %*lu %*lu %*lu %*lu %*lu %*lu %*lu %*lu %lu %lu", &utime, &stime) ==
      EOF) {
    file.close();
    return Status(StatusCode::kMDUnexpectedError, "Get device CPU failed.");
  }
  file.close();
  (*op_stat)[op_id][thread_id].user_stat_ = utime;
  (*op_stat)[op_id][thread_id].sys_stat_ = stime;

  return Status::OK();
}

Status OperatorCpu::GetTotalCpuTime(uint64_t *total_stat) {
  std::ifstream file("/proc/stat");
  if (!file.is_open()) {
    MS_LOG(INFO) << "Open CPU file failed when collect CPU information";
    return Status::OK();
  }
  std::string str;
  getline(file, str);
  uint64_t user = 0, sys = 0, idle = 0, iowait = 0, nice = 0, irq = 0, softirq = 0;
  if (sscanf_s(str.c_str(), "%*s %lu %lu %lu %lu %lu %lu %lu", &user, &nice, &sys, &idle, &iowait, &irq, &softirq) ==
      EOF) {
    file.close();
    return Status(StatusCode::kMDUnexpectedError, "Get device CPU failed.");
  }
  file.close();
  *total_stat = user + nice + sys + idle + iowait + irq + softirq;

  return Status::OK();
}

Status OperatorCpu::Collect(const ExecutionTree *tree) {
  if (first_collect_) {
    for (auto iter = tree->begin(); iter != tree->end(); ++iter) {
      id_count_++;
      op_name[iter->id()] = iter->NameWithID();
      op_parallel_workers[iter->id()] = iter->num_workers();
    }
#if defined(USING_LINUX)
    cpu_processor_num_ = get_nprocs_conf();
#endif
  }

  // Obtain the op and thread mapping
  op_thread.clear();
  List<Task> allTasks = tree->AllTasks()->GetTask();
  for (auto &task1 : allTasks) {
    int32_t op_id = task1.get_operator_id();
    op_thread[op_id].emplace_back(task1.get_linux_id());
  }

  // add process id into op_thread
  if (!fetched_all_process) {
    {
      py::gil_scoped_acquire gil_acquire;
      py::module ds = py::module::import("mindspore.dataset.engine.datasets");
      py::tuple process_info = ds.attr("_get_operator_process")();
      py::dict sub_process = py::reinterpret_borrow<py::dict>(process_info[0]);
      fetched_all_process = py::reinterpret_borrow<py::bool_>(process_info[1]);
      // parse dict value
      op_process = toIntMap(sub_process);
      BaseCpu::op_process_shared = op_process;
      BaseCpu::fetched_all_process_shared = fetched_all_process;
    }

    // judge whether there is device_que operator, if so operator id may need increase by one, temp use directly
    for (auto item : op_process) {
      if (!item.second.empty()) {
        if (op_thread.find(item.first) != op_thread.end()) {
          op_thread[item.first].insert(op_thread[item.first].end(), item.second.begin(), item.second.end());
        } else {
          op_thread[item.first] = item.second;
        }
      }
    }
  }

  uint64_t total_stat_;
  RETURN_IF_NOT_OK(GetTotalCpuTime(&total_stat_));
  std::vector<CpuOpUtil> cpu_step_util_;
  std::unordered_map<int32_t, std::unordered_map<int64_t, CpuOpStat>> op_stat_;

  if (!first_collect_) {
    // obtain all the op id in current tasks
    std::vector<int32_t> total_op_id;
    for (auto iter = op_thread.begin(); iter != op_thread.end(); iter++) {
      total_op_id.emplace_back(iter->first);
    }

    // iter all the op, and obtain the CPU utilization of each operator
    for (auto op_id = -1; op_id < id_count_; op_id++) {
      float user_util = 0, sys_util = 0;
      auto iter = std::find(total_op_id.begin(), total_op_id.end(), op_id);
      if (iter != total_op_id.end()) {
        for (auto thread_id : op_thread[op_id]) {
          if (ParseCpuInfo(op_id, thread_id, &op_stat_) == Status::OK()) {
            user_util += (op_stat_[op_id][thread_id].user_stat_ - pre_op_stat_[op_id][thread_id].user_stat_) * 1.0 /
                         (total_stat_ - pre_total_stat_) * 100;
            sys_util += (op_stat_[op_id][thread_id].sys_stat_ - pre_op_stat_[op_id][thread_id].sys_stat_) * 1.0 /
                        (total_stat_ - pre_total_stat_) * 100;
          }
        }
      }
      CpuOpUtil info;
      info.op_id = op_id;
      info.sys_utilization_ = sys_util;
      info.user_utilization_ = user_util;
      cpu_step_util_.emplace_back(info);
    }
    cpu_op_util_.emplace_back(cpu_step_util_);
  } else {
    // mainly obtain the init CPU execute time in first collect
    for (auto iter = op_thread.begin(); iter != op_thread.end(); iter++) {
      int32_t op_id = iter->first;
      for (auto thread_id : iter->second) {
        ParseCpuInfo(op_id, thread_id, &op_stat_);
      }
    }
  }

  // copy current op_stat into pre_op_stat
  pre_op_stat_ = op_stat_;
  pre_total_stat_ = total_stat_;

  first_collect_ = false;
  return Status::OK();
}

Status OperatorCpu::Analyze(std::string *const name, double *utilization, std::string *const extra_message) {
  int total_samples = cpu_op_util_.size();

  // Only analyze the middle half of the samples
  // Starting and ending may be impacted by startup or ending pipeline activities
  int start_analyze = total_samples / 4;
  int end_analyze = total_samples - start_analyze;
  double op_util = 0;
  *utilization = 0;

  // start loop from 0 was as don't want to analyze op -1
  for (auto op_id = 0; op_id < id_count_; op_id++) {
    int sum = 0;
    int index = op_id + 1;
    for (int i = start_analyze; i < end_analyze; i++) {
      sum += cpu_op_util_[i][index].user_utilization_;
      sum += cpu_op_util_[i][index].sys_utilization_;
    }
    if ((end_analyze - start_analyze) > 0) {
      op_util = 1.0 * sum * cpu_processor_num_ / (op_parallel_workers[op_id] * (end_analyze - start_analyze));
    }
    if (op_util > *utilization) {
      *utilization = op_util;
      name->clear();
      name->append(op_name[op_id]);
    }
    extra_message->append(op_name[op_id] + " utiliization per thread: " + std::to_string(op_util) + "% (" +
                          std::to_string(op_parallel_workers[op_id]) + " parallel_workers);  ");
  }
  return Status::OK();
}

Status OperatorCpu::SaveToFile(const std::string &file_path) {
  Path path = Path(file_path);
  json output;
  if (path.Exists()) {
    MS_LOG(DEBUG) << file_path << "already exist.";
    std::ifstream file(file_path);
    file >> output;
  }

  uint8_t index = 0;
  json OpWriter;
  for (auto op_id = -1; op_id < id_count_; op_id++) {
    std::vector<uint16_t> user_util;
    std::vector<uint16_t> sys_util;
    std::transform(
      cpu_op_util_.begin(), cpu_op_util_.end(), std::back_inserter(user_util),
      [&](const std::vector<CpuOpUtil> &info) { return int16_t(info[index].user_utilization_ * cpu_processor_num_); });
    std::transform(
      cpu_op_util_.begin(), cpu_op_util_.end(), std::back_inserter(sys_util),
      [&](const std::vector<CpuOpUtil> &info) { return int16_t(info[index].sys_utilization_ * cpu_processor_num_); });

    json per_op_info = {{"metrics", {{"user_utilization", user_util}, {"sys_utilization", sys_util}}},
                        {"op_id", op_id}};
    OpWriter.emplace_back(per_op_info);
    index++;
  }
  output["op_info"] = OpWriter;

  // Discard the content of the file when opening.
  std::ofstream os(file_path, std::ios::trunc);
  os << output;

  MS_LOG(INFO) << "Save device CPU success.";
  return Status::OK();
}

Status ProcessCpu::ParseCpuInfo() {
  uint64_t total_stat_;
  RETURN_IF_NOT_OK(GetTotalCpuTime(&total_stat_));

  if (!pre_fetched_state) {
    process_id.clear();
    pid_t main_pid = 0;
#if defined(USING_LINUX)
    main_pid = syscall(SYS_getpid);
#endif
    process_id.emplace_back(main_pid);
    op_process = BaseCpu::op_process_shared;
    fetched_all_process = BaseCpu::fetched_all_process_shared;
    for (auto item : op_process) {
      for (auto id : item.second) {
        process_id.emplace_back(id);
      }
    }
  }

  float user_util = 0, sys_util = 0;
  for (auto pid : process_id) {
    std::string stat_path = "/proc/" + std::to_string(pid) + "/stat";

    std::ifstream file(stat_path);
    if (!file.is_open()) {
      MS_LOG(INFO) << "Open CPU file failed when collect CPU information";
      continue;
    }
    std::string str;
    getline(file, str);
    uint64_t user = 0, sys = 0;
    if (sscanf_s(str.c_str(), "%*d %*s %*s %*lu %*lu %*lu %*lu %*lu %*lu %*lu %*lu %*lu %*lu %lu %lu", &user, &sys) ==
        EOF) {
      file.close();
      return Status(StatusCode::kMDUnexpectedError, "Get device CPU failed.");
    }
    file.close();

    // Calculate the utilization from the second sampling
    if (!first_collect_ && (pre_process_stat_.find(pid) != pre_process_stat_.end())) {
      user_util += (user - pre_process_stat_[pid].user_stat_) * 1.0 / (total_stat_ - pre_total_stat_) * 100;
      sys_util += (sys - pre_process_stat_[pid].sys_stat_) * 1.0 / (total_stat_ - pre_total_stat_) * 100;
    }
    pre_process_stat_[pid].user_stat_ = user;
    pre_process_stat_[pid].sys_stat_ = sys;
  }
  if (!first_collect_) {
    CpuProcessUtil info;
    info.user_utilization_ = user_util;
    info.sys_utilization_ = sys_util;
    process_util_.emplace_back(info);
  }
  pre_total_stat_ = total_stat_;
  first_collect_ = false;
  pre_fetched_state = fetched_all_process;
  return Status::OK();
}

Status ProcessCpu::GetTotalCpuTime(uint64_t *total_stat) {
  std::ifstream file("/proc/stat");
  if (!file.is_open()) {
    MS_LOG(INFO) << "Open CPU file failed when collect CPU information";
    return Status::OK();
  }
  std::string str;
  getline(file, str);
  uint64_t user = 0, sys = 0, idle = 0, iowait = 0, nice = 0, irq = 0, softirq = 0;
  if (sscanf_s(str.c_str(), "%*s %lu %lu %lu %lu %lu %lu %lu", &user, &nice, &sys, &idle, &iowait, &irq, &softirq) ==
      EOF) {
    file.close();
    return Status(StatusCode::kMDUnexpectedError, "Get device CPU failed.");
  }
  file.close();
  *total_stat = user + nice + sys + idle + iowait + irq + softirq;

  return Status::OK();
}

Status ProcessCpu::Collect(const ExecutionTree *tree) {
  if (first_collect_) {
#if defined(USING_LINUX)
    cpu_processor_num_ = get_nprocs_conf();
#endif
  }
  RETURN_IF_NOT_OK(ParseCpuInfo());

  return Status::OK();
}

Status ProcessCpu::Analyze(std::string *const name, double *utilization, std::string *const extra_message) {
  name->clear();
  name->append("process_info");
  int total_samples = process_util_.size();
  int sum = 0;
  // Only analyze the middle half of the samples
  // Starting and ending may be impacted by startup or ending pipeline activities
  int start_analyze = total_samples / 4;
  int end_analyze = total_samples - start_analyze;

  for (int i = start_analyze; i < end_analyze; i++) {
    sum += process_util_[i].user_utilization_;
    sum += process_util_[i].sys_utilization_;
  }

  if ((end_analyze - start_analyze) > 0) {
    *utilization = sum / (end_analyze - start_analyze);
  }
  return Status::OK();
}

Status ProcessCpu::SaveToFile(const std::string &file_path) {
  Path path = Path(file_path);
  json output;
  if (path.Exists()) {
    MS_LOG(DEBUG) << file_path << "already exist.";
    std::ifstream file(file_path);
    file >> output;
  } else {
    output["sampling_interval"] = GlobalContext::config_manager()->monitor_sampling_interval();
  }

  std::vector<int16_t> user_util;
  std::transform(process_util_.begin(), process_util_.end(), std::back_inserter(user_util),
                 [&](const CpuProcessUtil &info) { return uint16_t(info.user_utilization_ * cpu_processor_num_); });
  std::vector<int16_t> sys_util;
  std::transform(process_util_.begin(), process_util_.end(), std::back_inserter(sys_util),
                 [&](const CpuProcessUtil &info) { return uint16_t(info.sys_utilization_ * cpu_processor_num_); });

  output["process_info"] = {{"user_utilization", user_util}, {"sys_utilization", sys_util}};
  output["cpu_processor_num"] = cpu_processor_num_;
  // Discard the content of the file when opening.
  std::ofstream os(file_path, std::ios::trunc);
  os << output;

  MS_LOG(INFO) << "Save process CPU success.";
  return Status::OK();
}

Status CpuSampling::CollectTimeStamp() {
  time_stamp_.emplace_back(ProfilingTime::GetCurMilliSecond());
  return Status::OK();
}

// Sample action
Status CpuSampling::Sample() {
  // Collect cpu information
  for (auto cpu : cpu_) {
    RETURN_IF_NOT_OK(cpu->Collect(this->tree_));
  }

  // Collect time stamp
  RETURN_IF_NOT_OK(CollectTimeStamp());
  return Status::OK();
}

Status CpuSampling::SaveTimeStampToFile() {
  // Save time stamp to json file
  // If the file is already exist, simply add the data to corresponding field.
  Path path = Path(file_path_);
  json output;
  if (path.Exists()) {
    std::ifstream file(file_path_);
    file >> output;
  }
  output["time_stamp"] = time_stamp_;
  std::ofstream os(file_path_, std::ios::trunc);
  os << output;

  return Status::OK();
}

Status CpuSampling::SaveSamplingItervalToFile() {
  // If the file is already exist, simply add the data to corresponding field.
  Path path = Path(file_path_);
  json output;
  if (path.Exists()) {
    std::ifstream file(file_path_);
    file >> output;
  }
  output["sampling_interval"] = GlobalContext::config_manager()->monitor_sampling_interval();
  std::ofstream os(file_path_, std::ios::trunc);
  os << output;

  return Status::OK();
}

// Analyze profiling data and output warning messages
Status CpuSampling::Analyze() {
  std::string name;
  double utilization = 0;

  // Keep track of specific information returned by differentn CPU sampling types
  double total_utilization = 0;
  double max_op_utilization = 0;
  std::string max_op_name;
  std::string detailed_op_cpu_message;

  // Save cpu information to json file
  for (auto cpu : cpu_) {
    std::string extra_message;
    RETURN_IF_NOT_OK(cpu->Analyze(&name, &utilization, &extra_message));
    if (name == "device_info") {
      total_utilization = utilization;
    } else if (name != "process_info") {
      max_op_utilization = utilization;
      max_op_name = name;
      detailed_op_cpu_message = extra_message;
    }
  }
  if ((total_utilization < 90) && (max_op_utilization > 80)) {
    MS_LOG(WARNING) << "Operator " << max_op_name << " is using " << max_op_utilization << "% CPU per thread.  "
                    << "This operator may benefit from increasing num_parallel_workers."
                    << "Full Operator CPU utiliization for all operators: " << detailed_op_cpu_message << std::endl;
  }
  return Status::OK();
}

// Save profiling data to file
Status CpuSampling::SaveToFile() {
  // Save time stamp to json file
  RETURN_IF_NOT_OK(SaveTimeStampToFile());

  // Save time stamp to json file
  RETURN_IF_NOT_OK(SaveSamplingItervalToFile());

  // Save cpu information to json file
  for (auto cpu : cpu_) {
    RETURN_IF_NOT_OK(cpu->SaveToFile(file_path_));
  }

  return Status::OK();
}

Status CpuSampling::Init(const std::string &dir_path, const std::string &device_id) {
  file_path_ = (Path(dir_path) / Path("minddata_cpu_utilization_" + device_id + ".json")).toString();
  std::shared_ptr<DeviceCpu> device_cpu = std::make_shared<DeviceCpu>();
  std::shared_ptr<OperatorCpu> operator_cpu = std::make_shared<OperatorCpu>();
  std::shared_ptr<ProcessCpu> process_cpu = std::make_shared<ProcessCpu>();
  cpu_.push_back(device_cpu);
  cpu_.push_back(operator_cpu);
  cpu_.push_back(process_cpu);
  return Status::OK();
}

Status CpuSampling::ChangeFileMode() {
  if (chmod(common::SafeCStr(file_path_), S_IRUSR | S_IWUSR) == -1) {
    std::string err_str = "Change file mode failed," + file_path_;
    return Status(StatusCode::kMDUnexpectedError, err_str);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
