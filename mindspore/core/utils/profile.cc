/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#include "utils/profile.h"
#include <chrono>
#include <numeric>
#include <cstdio>
#include <sstream>
#include <iomanip>
#include <vector>
#include <list>
#include <utility>
#include <cfloat>
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace {
constexpr size_t TIME_INFO_PREFIX_NUM_LEN = 4;
constexpr char kProcessStatusFileName[] = "/proc/self/status";
constexpr auto kLineMaxSize = 1024;
constexpr auto kInvalid = -1;
static bool kRecordMemoryFlag = common::GetEnv("MS_DEV_RECORD_MEMORY") == "1";
const auto kVmRSS = "VmRSS";

void PrintProfile(std::ostringstream &oss, const TimeInfo &time_info, int indent = 0,
                  std::map<std::string, double> *sums = nullptr, const std::string &prefix = "");

void PrintTimeInfoMap(std::ostringstream &oss, const TimeInfoMap &dict, int indent = 0,
                      std::map<std::string, double> *sums = nullptr, const std::string &prefix = "") {
  size_t count = 0;
  for (const auto &iter : dict) {
    count++;
    if (iter.second == nullptr) {
      continue;
    }
    // indent by multiples of 4 spaces.
    if (iter.first.size() < TIME_INFO_PREFIX_NUM_LEN) {
      MS_LOG(EXCEPTION) << "In TimeInfoMap, the " << count << "th string key is " << iter.first
                        << ", but the length is less than " << TIME_INFO_PREFIX_NUM_LEN;
    }
    auto name = iter.first.substr(TIME_INFO_PREFIX_NUM_LEN);
    oss << std::setw(indent * 4) << ""
        << "[" << name << "]: " << iter.second->time_;
    if (iter.second->dict_ != nullptr) {
      oss << ", [" << iter.second->dict_->size() << "]";
    }
    oss << "\n";

    std::string newPrefix = prefix;
    if (iter.first.find("Cycle ") == std::string::npos) {
      newPrefix = prefix.empty() ? iter.first : prefix + "." + iter.first;
    }
    PrintProfile(oss, *iter.second, indent + 1, sums, newPrefix);
    if (iter.second->dict_ == nullptr) {
      (*sums)[newPrefix] += iter.second->time_;
    }
  }
}

void PrintProfile(std::ostringstream &oss, const TimeInfo &time_info, int indent, std::map<std::string, double> *sums,
                  const std::string &prefix) {
  bool need_free = false;
  if (sums == nullptr) {
    sums = new (std::nothrow) std::map<std::string, double>();
    if (sums == nullptr) {
      MS_LOG(ERROR) << "memory allocation failed";
      return;
    }
    need_free = true;
  }

  // indent by multiples of 4 spaces.
  if (indent == 0) {
    oss << "\nTotalTime = " << time_info.time_;
    if (time_info.dict_ != nullptr) {
      oss << ", [" << time_info.dict_->size() << "]";
    }
    oss << "\n";
  }

  if (time_info.dict_ != nullptr) {
    PrintTimeInfoMap(oss, *time_info.dict_, indent, sums, prefix);
  }

  // print time percentage info
  if (need_free) {
    double total = 0.0;
    for (auto iter = sums->begin(); iter != sums->end(); ++iter) {
      total += iter->second;
    }
    oss << "Sums\n";
    if (total >= 0.0 + DBL_EPSILON) {
      for (auto &iter : *sums) {
        std::string name = iter.first;
        name.erase(0, TIME_INFO_PREFIX_NUM_LEN);
        std::size_t pos = 0;
        while ((pos = name.find('.', pos)) != std::string::npos) {
          pos++;
          name.erase(pos, TIME_INFO_PREFIX_NUM_LEN);
        }
        oss << "    " << std::left << std::setw(36) << name << " : " << std::right << std::setw(12) << std::fixed
            << std::setprecision(6) << iter.second << "s : " << std::right << std::setw(5) << std::fixed
            << std::setprecision(2) << (iter.second / total) * 100 << "%\n";
      }
    }
    delete sums;
  }
}
}  // namespace

double GetTime(void) {
  auto now = std::chrono::steady_clock::now();
  std::chrono::duration<double> d = now.time_since_epoch();
  return d.count();
}

TimeInfo::~TimeInfo() {
  if (dict_ == nullptr) {
    return;
  }
  for (auto iter = dict_->begin(); iter != dict_->end(); ++iter) {
    delete iter->second;
    iter->second = nullptr;
  }
  delete dict_;
  dict_ = nullptr;
}

ProfileBase::ProfileBase() : context_("", this) {
  ctx_ptr_ = &context_;
  context_.parent_ = nullptr;
}

ProfileBase::~ProfileBase() {
  context_.parent_ = nullptr;
  if (context_.time_info_ != nullptr) {
    delete context_.time_info_;
    context_.time_info_ = nullptr;
  }
  ctx_ptr_ = nullptr;
}

void Profile::Print(void) {
  if (ctx_ptr_ == nullptr || ctx_ptr_->time_info_ == nullptr) {
    return;
  }
  std::ostringstream oss;
  PrintProfile(oss, *ctx_ptr_->time_info_);
  // Here use printf to output profile info, not use MS_LOG(INFO) since when open log, it affects performance
  std::cout << oss.str() << std::endl;
}

// Start a step in the current context with the given name.
// Name must be unique otherwise the previous record will be overwritten.
ProfContext *Profile::Step(const std::string &name) {
  ctx_ptr_ = new (std::nothrow) ProfContext(name, this);
  if (ctx_ptr_ == nullptr) {
    MS_LOG(ERROR) << "memory allocation failed";
    return nullptr;
  }
  return ctx_ptr_;
}

// Creates subcontext for a repeated action.
// Count should be monotonically increasing.
ProfContext *Profile::Lap(int count) {
  std::ostringstream oss;
  oss << "Cycle " << count;
  ctx_ptr_ = new (std::nothrow) ProfContext(oss.str(), this);
  if (ctx_ptr_ == nullptr) {
    MS_LOG(ERROR) << "memory allocation failed";
    return nullptr;
  }
  return ctx_ptr_;
}

void Profile::Pop(void) noexcept {
  if (ctx_ptr_ == nullptr) {
    return;
  }
  ctx_ptr_ = ctx_ptr_->parent_;
}

ProfContext::ProfContext(const std::string &name, ProfileBase *const prof)
    : name_(name), prof_(prof), time_info_(nullptr) {
  // Initialize a subcontext.
  if (prof == nullptr || IsTopContext()) {
    parent_ = nullptr;
  } else {
    parent_ = prof->ctx_ptr_;
  }
}

ProfContext::~ProfContext() {
  // top level context
  if (parent_ == nullptr || IsTopContext()) {
    if (time_info_ != nullptr) {
      delete time_info_;
    }
  } else {
    parent_->Insert(name_, time_info_);
    if (prof_ != nullptr) {
      prof_->Pop();
    }
  }

  time_info_ = nullptr;
  prof_ = nullptr;
  parent_ = nullptr;
}

void ProfContext::SetTime(double time) noexcept {
  if (time_info_ == nullptr) {
    time_info_ = new (std::nothrow) TimeInfo(time);
    if (time_info_ == nullptr) {
      MS_LOG(ERROR) << "memory allocation failed";
      return;
    }
  }
  time_info_->time_ = time;
}

void ProfContext::Insert(const std::string &name, const TimeInfo *time) noexcept {
  if (time_info_ == nullptr) {
    time_info_ = new (std::nothrow) TimeInfo();
    if (time_info_ == nullptr) {
      MS_LOG(ERROR) << "memory allocation failed";
      delete time;
      time = nullptr;
      return;
    }
  }

  if (time_info_->dict_ == nullptr) {
    time_info_->dict_ = new (std::nothrow) TimeInfoMap();
    if (time_info_->dict_ == nullptr) {
      MS_LOG(ERROR) << "memory allocation failed";
      delete time;
      time = nullptr;
      delete time_info_;
      time_info_ = nullptr;
      return;
    }
  }

  std::stringstream ss;
  ss << std::setw(TIME_INFO_PREFIX_NUM_LEN) << std::setfill('0') << time_info_->actionNum_;
  std::string sorted_name(ss.str() + name);
  time_info_->actionNum_++;
  auto iter = time_info_->dict_->find(sorted_name);
  // if contains item with same name, delete it
  if (iter != time_info_->dict_->end()) {
    delete iter->second;
    iter->second = nullptr;
    (void)time_info_->dict_->erase(iter);
  }
  (*time_info_->dict_)[sorted_name] = time;
}

bool ProfContext::IsTopContext() const noexcept { return (prof_ != nullptr) && (this == &prof_->context_); }

ProfTransaction::ProfTransaction(const ProfileBase *prof) { ctx_ = (prof != nullptr ? prof->ctx_ptr_ : nullptr); }

ProfTransaction::~ProfTransaction() {
  if (ctx_ != nullptr && !ctx_->IsTopContext()) {
    delete ctx_;
  }
  ctx_ = nullptr;
}

DumpTime &DumpTime::GetInstance() {
  static DumpTime instance;
  return instance;
}

void DumpTime::Record(const std::string &step_name, const double time, const bool is_start) {
  file_ss_ << "    {" << std::endl;
  file_ss_ << "        \"name\": "
           << "\"" << step_name << "\"," << std::endl;
  file_ss_ << "        \"cat\": "
           << "\"FUNCTION\"," << std::endl;
  if (is_start) {
    file_ss_ << "        \"ph\": "
             << "\"B\"," << std::endl;
  } else {
    file_ss_ << "        \"ph\": "
             << "\"E\"," << std::endl;
  }
  file_ss_ << "        \"ts\": " << std::setprecision(16) << time * 1000000 << "," << std::endl;
  file_ss_ << "        \"pid\": "
           << "1" << std::endl;
  file_ss_ << "    }" << std::endl;
  file_ss_ << "    ," << std::endl;
}

void DumpTime::Save() {
  try {
    file_out_.open(file_path_, std::ios::trunc | std::ios::out);
  } catch (const std::exception &e) {
    MS_LOG(EXCEPTION) << "Cannot open file in " << (file_path_);
  }
  file_out_ << "{\n";
  file_out_ << "    \"traceEvents\": [" << std::endl;
  file_ss_ >> file_out_.rdbuf();
  constexpr int offset = -7;
  (void)file_out_.seekp(offset, std::ios::end);
  file_out_ << "    ]" << std::endl << "    ,\n";
  file_out_ << "    \"displayTimeUnit\": \"ms\"" << std::endl;
  file_out_ << "}";
  file_out_.close();
}

struct TimeInfoGroup {
  double total_time = 0.0;
  int total_count = 0;
  std::list<std::map<std::string, TimeStat>::const_iterator> items;
};

static void PrintTimeStat(std::ostringstream &oss, const TimeInfoGroup &group, const std::string &prefix) {
  oss << "------[" << prefix << "] " << std::setw(10) << std::fixed << std::setprecision(6) << group.total_time
      << std::setw(6) << group.total_count << "\n";
  for (const auto &iter : group.items) {
    oss << std::setw(5) << std::fixed << std::setprecision(2) << (iter->second.time_ / group.total_time) * 100
        << "% : " << std::setw(12) << std::fixed << std::setprecision(6) << iter->second.time_ << "s : " << std::setw(6)
        << iter->second.count_ << ": " << iter->first << "\n";
  }
}

void MsProfile::Print() {
  GetProfile()->Print();
  std::vector<std::string> items = {"substitution.",          "renormalize.", "replace.", "match.",
                                    "func_graph_cloner_run.", "meta_graph.",  "manager.", "pynative"};
  std::vector<TimeInfoGroup> groups(items.size() + 1);
  const auto &stat = GetSingleton().time_stat_;
  // group all time infos
  for (auto iter = stat.cbegin(); iter != stat.cend(); ++iter) {
    auto matched_idx = items.size();
    for (size_t i = 0; i < items.size(); ++i) {
      if (iter->first.find(items[i]) != std::string::npos) {
        matched_idx = i;
        break;
      }
    }
    groups[matched_idx].total_time += iter->second.time_;
    groups[matched_idx].total_count += iter->second.count_;
    groups[matched_idx].items.push_back(iter);
  }
  std::ostringstream oss;
  for (size_t i = 0; i < groups.size(); ++i) {
    std::string prefix = (i < items.size() ? items[i] : std::string("others."));
    PrintTimeStat(oss, groups[i], prefix);
  }
  // Here use printf to output profile info, not use MS_LOG(INFO) since when open log, it affects performance
  std::cout << "\nTime group info:\n" << oss.str() << std::endl;
}

ProcessStatus &ProcessStatus::GetInstance() {
  static ProcessStatus recorder;
  return recorder;
}

int64_t ProcessStatus::GetMemoryCost(const std::string &key) const {
#if defined(_WIN32) || defined(_WIN64) || defined(__APPLE__)
  return kInvalid;
#else
  if (!kRecordMemoryFlag) {
    return kInvalid;
  }

  FILE *file = fopen(kProcessStatusFileName, "r");
  if (file == nullptr) {
    MS_LOG(ERROR) << "Get process status file failed.";
    return 0;
  }

  int64_t mem_size = 0;
  char buf[kLineMaxSize] = {0};
  while (fgets(buf, kLineMaxSize, file)) {
    // Get mem title.
    std::string line(buf);
    auto title_end_pos = line.find(":");
    auto title = line.substr(0, title_end_pos);
    // Get mem size.
    if (title == key) {
      auto mem_size_end_pos = line.find_last_of(" ");
      auto mem_size_begin_pos = line.find_last_of(" ", mem_size_end_pos - 1);
      if ((mem_size_end_pos != std::string::npos) && (mem_size_begin_pos != std::string::npos)) {
        auto mem_size_string = line.substr(mem_size_begin_pos, mem_size_end_pos - mem_size_begin_pos);
        mem_size = std::atol(mem_size_string.c_str());
      }
      break;
    }
    if (memset_s(buf, kLineMaxSize, 0, kLineMaxSize) != EOK) {
      MS_LOG(ERROR) << "Set process status file failed.";
      (void)fclose(file);
      return 0;
    }
  }
  (void)fclose(file);
  return mem_size;
#endif
}

void ProcessStatus::RecordStart(const std::string &step_name) {
  if (!kRecordMemoryFlag) {
    return;
  }
  MemoryInfo memory_info;
  memory_info.name = step_name;
  memory_info.depth = stack_.size();
  memory_info.start_memory = GetMemoryCost(kVmRSS);
  (void)stack_.emplace_back(memory_info);
}

void ProcessStatus::RecordEnd() {
  if (!kRecordMemoryFlag) {
    return;
  }
  if (stack_.empty()) {
    MS_EXCEPTION(ValueError) << "ProcessStatus stack is empty";
  }
  auto memory_info = stack_.back();
  memory_info.end_memory = GetMemoryCost(kVmRSS);
  (void)memory_used_.emplace_back(memory_info);
  stack_.pop_back();
}

void ProcessStatus::Print() {
  if (!kRecordMemoryFlag) {
    return;
  }
  std::ostringstream oss;
  constexpr auto indent = 2;
  for (auto item : memory_used_) {
    std::string spaces(item.depth * indent, ' ');
    oss << spaces << "[" << item.name << "]: " << item.start_memory << "KB -> " << item.end_memory
        << "KB. Increased: " << item.end_memory - item.start_memory << "KB\n";
  }
  std::cout << "\nMemory increase info:\n" << oss.str() << std::endl;
  Clear();
}

void ProcessStatus::Clear() {
  memory_used_.clear();
  stack_.clear();
}
}  // namespace mindspore
