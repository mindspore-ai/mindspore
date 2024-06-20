/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "include/common/profiler.h"
#include <functional>
#include <iomanip>
#include <utility>
#include "utils/file_utils.h"
#include "include/common/utils/utils.h"
#include "include/common/debug/common.h"
#ifdef ENABLE_DEBUGGER
#include "include/backend/debug/profiler/profiling.h"
#include "common/debug/profiler/profiling_framework_data.h"
#endif

namespace mindspore {
namespace runtime {
static const int kPrecisionDigits = 2;
static const double kNsToUs = 1000;

// The string of json file.
static const char kJsonName[] = "name";
static const char kJsonPh[] = "ph";
static const char kJsonPid[] = "pid";
static const char kJsonTid[] = "tid";
static const char kJsonTs[] = "ts";
static const char kJsonDur[] = "dur";
static const char kJsonPhX[] = "X";
static const char kJsonArgs[] = "args";
static const char kJsonFlowId[] = "flow_id";
static const char kJsonPyStack[] = "py-stack";
static const char kNameFlow[] = "flow";

// The env of runtime profiler.
static const char kEnableRuntimeProfiler[] = "MS_ENABLE_RUNTIME_PROFILER";
static const char kRuntimeProfilerTopNum[] = "MS_ENABLE_PROFILER_TOP_NUM";

// Save file name.
static const char kJsonFileName[] = "RuntimeProfilerJson";
static const char kSummaryInfoFileName[] = "RuntimeProfilerSummary";
static const char kDetailInfoFileName[] = "RuntimeProfilerDetail";

namespace {
std::string GetRealPathName(const std::string &name) {
  auto path_name = GetSaveGraphsPathName(name);
  auto real_path = mindspore::Common::CreatePrefixPath(path_name);
  if (!real_path.has_value()) {
    MS_LOG(ERROR) << "Get real path failed, path: " << path_name;
    return ("./" + name);
  }
  return real_path.value();
}

uint64_t GetClockTimeNs() {
  auto ts = std::chrono::system_clock::now();
  int64_t system_t = std::chrono::duration_cast<std::chrono::nanoseconds>(ts.time_since_epoch()).count();
  return static_cast<uint64_t>(system_t);
}
}  // namespace

ProfilerRecorder::ProfilerRecorder(ProfilerModule module, ProfilerEvent event, const std::string &op_name,
                                   bool is_inner_event, bool need_py_stack, uint64_t flow_id) {
  auto &profiler = ProfilerAnalyzer::GetInstance();
  if (!profiler.profiler_enable()) {
    return;
  }
  data_ = std::make_unique<Data>(module, event, profiler.GetBriefName(op_name),
                                 !profiler.enable_by_env() && need_py_stack ? GetPythonStackStr_() : std::string(),
                                 profiler.GetTimeStamp(), flow_id, is_inner_event);
}

ProfilerRecorder::~ProfilerRecorder() {
  auto &profiler = ProfilerAnalyzer::GetInstance();
  if (!profiler.profiler_enable()) {
    return;
  }
  if (data_ == nullptr) {
    return;
  }
  profiler.RecordData(std::make_shared<ProfilerData>(data_->module_, data_->event_, data_->op_name_,
                                                     data_->is_inner_event_, data_->start_time_,
                                                     profiler.GetTimeStamp(), data_->flow_id_, data_->py_stack_));
}

PythonProfilerRecorder::PythonProfilerRecorder(const std::string &record_name)
    : start_time_(0),
      record_name_(record_name),
      module_(ProfilerModule::kPython),
      event_(ProfilerEvent::kPythonObserved) {}

void PythonProfilerRecorder::record_start() {
  if (runtime::ProfilerAnalyzer::GetInstance().profiler_enable()) {
    start_time_ = runtime::ProfilerAnalyzer::GetInstance().GetTimeStamp();
  }
}

void PythonProfilerRecorder::record_end() {
  if (runtime::ProfilerAnalyzer::GetInstance().profiler_enable()) {
    auto end_time = runtime::ProfilerAnalyzer::GetInstance().GetTimeStamp();
    runtime::ProfilerAnalyzer::GetInstance().RecordData(
      std::make_shared<runtime::ProfilerData>(module_, event_, record_name_, false, start_time_, end_time));
  }
}

ProfilerStageRecorder::ProfilerStageRecorder(ProfilerStage stage) {
  if (!ProfilerAnalyzer::GetInstance().profiler_enable()) {
    return;
  }
  start_time_ = ProfilerAnalyzer::GetInstance().GetTimeStamp();
  stage_ = stage;
}

ProfilerStageRecorder::~ProfilerStageRecorder() {
  if (!ProfilerAnalyzer::GetInstance().profiler_enable()) {
    return;
  }
  ProfilerAnalyzer::GetInstance().RecordData(
    std::make_shared<runtime::ProfilerData>(stage_, start_time_, ProfilerAnalyzer::GetInstance().GetTimeStamp()));
}

ProfilerAnalyzer &ProfilerAnalyzer::GetInstance() noexcept {
  static ProfilerAnalyzer instance{};
  return instance;
}

void ProfilerAnalyzer::Initialize() {
  if (init_) {
    return;
  }
  std::unique_lock<SpinLock> lock(data_mutex_);
  init_ = true;

  if (common::GetEnv(kEnableRuntimeProfiler) != "1") {
    return;
  }
  profiler_enable_ = true;
  auto top_num_env = common::GetEnv(kRuntimeProfilerTopNum);
  if (top_num_env != std::string()) {
    try {
      show_top_num_ = stoi(top_num_env);
    } catch (const std::exception &e) {
      MS_LOG(EXCEPTION) << "Invalid argument: " << e.what() << " when parse " << top_num_env;
    }
  }

  auto now_time = std::to_string(GetTimeStamp());
  json_file_name_ = GetRealPathName(kJsonFileName + now_time + ".json");
  summary_info_file_name_ = GetRealPathName(kSummaryInfoFileName + now_time + ".csv");
  detail_info_file_name_ = GetRealPathName(kDetailInfoFileName + now_time + ".csv");
}

bool ProfilerAnalyzer::profiler_enable() const { return profiler_enable_ || mi_profiler_enable_; }

void ProfilerAnalyzer::SetThreadIdToName(const std::thread::id &id, const std::string &name) {
  std::unique_lock<SpinLock> lock(data_mutex_);
  thread_id_to_name_[id] = name;
}

void ProfilerAnalyzer::ProcessData() {
  for (const auto &[step_info_ptr, span] : data_line_) {
    step_time_ = step_info_ptr->step_time_;
    // Process module overlapping data.
    ProcessModuleSummaryData(span);
    // Process data.
    for (auto &data : span) {
      SaveJsonData(data);
      AnalyzeSummaryData(data);
    }
    AddPythonSummaryData(step_info_ptr->step_time_);
    // Dump data.
    DumpDetailData(step_info_ptr->step_, span);
    DumpSummaryData(step_info_ptr->step_);
    // Clear temp data.
    module_total_time_ = 0ull;
    module_infos_.clear();
    stage_infos_.clear();
  }
}

void ProfilerAnalyzer::Clear() noexcept {
  std::unique_lock<SpinLock> lock(data_mutex_);
  if (!init_ || !profiler_enable_ || data_line_.empty()) {
    return;
  }
  ProcessData();

  // Dump json data.
  DumpJsonData();
#ifdef WITH_BACKEND
  json_infos_.clear();
#endif
  data_.clear();
  data_line_.clear();
  init_ = false;
}

void ProfilerAnalyzer::EnableMiProfile() { mi_profiler_enable_ = true; }

uint64_t ProfilerAnalyzer::GetTimeStamp() const noexcept {
  if (profiler_enable_) {
    return GetClockTimeNs();
  }
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__) && \
  defined(ENABLE_DEBUGGER)
  return profiler::GetClockSyscnt();
#else
  return 0;
#endif
}

// For example: ScopeName(XX/XX/ReLU-op1) --> BriefName(ReLU)
std::string ProfilerAnalyzer::GetBriefName(const std::string &scope_name) const {
  auto first_index = scope_name.rfind('/');
  auto second_index = scope_name.rfind("-op");
  if ((first_index != std::string::npos) && (second_index != std::string::npos) &&
      (first_index + 1 < scope_name.size()) && (first_index + 1 < second_index)) {
    return scope_name.substr(first_index + 1, second_index - first_index - 1);
  }
  return scope_name;
}

void ProfilerAnalyzer::RecordData(const ProfilerDataPtr &data) noexcept {
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
  MS_EXCEPTION_IF_NULL(data);
  std::unique_lock<SpinLock> lock(data_mutex_);
  if (profiler_enable_) {
    (void)data_.emplace_back(data);
  }
  if (mi_profiler_enable_) {
#if defined(ENABLE_DEBUGGER)
    auto ascend_profiler = mindspore::profiler::Profiler::GetInstance(kAscendDevice);
    if (ascend_profiler != nullptr && ascend_profiler->EnableHostStack()) {
      profiler::ascend::ProfilingFrameworkData::RecordHostProfile(data, step_);
    }
#endif
  }
#endif
}

void ProfilerAnalyzer::RecordFlowData(uint64_t flow_id) {
  if (!ProfilerAnalyzer::GetInstance().profiler_enable()) {
    return;
  }
  ProfilerAnalyzer::GetInstance().RecordData(std::make_shared<ProfilerData>(
    ProfilerModule::kDefault, ProfilerEvent::kDefault, kNameFlow, true, ProfilerAnalyzer::GetInstance().GetTimeStamp(),
    ProfilerAnalyzer::GetInstance().GetTimeStamp(), flow_id));
}

void ProfilerAnalyzer::StartStep() {
  Initialize();
  if (!profiler_enable()) {
    return;
  }

  std::unique_lock<SpinLock> lock(data_mutex_);
  ++step_;
  // Reset the saved data.
  data_.clear();
  step_start_time_ = GetTimeStamp();
}

void ProfilerAnalyzer::ProcessModuleSummaryData(const ProfilerDataSpan &span) {
  if (span.empty()) {
    return;
  }

  // Use multimap as start_time_ may be same.
  std::map<ProfilerModule, std::multimap<uint64_t, ProfilerDataPtr>> ordered_data;
  for (auto &data : span) {
    (void)ordered_data[data->module_].emplace(data->start_time_, data);
  }
  for (const auto &data_item : ordered_data) {
    ProfilerDataPtr last_data = nullptr;
    for (const auto &[start_time, data] : data_item.second) {
      // Skip stage data and inner event data.
      if (data->is_stage_ || data->is_inner_event_) {
        continue;
      }
      // last_data is null or current range is not in last range, add current range and update last_data.
      if (last_data == nullptr || start_time >= last_data->end_time_) {
        AnalyzeModuleSummaryData(data);
        last_data = data;
        continue;
      }
      // Current range is in last range, just skip.
      if (data->end_time_ <= last_data->end_time_) {
        auto old_dur_time = data->dur_time_;
        data->dur_time_ = 0;
        AnalyzeModuleSummaryData(data);
        data->dur_time_ = old_dur_time;
        continue;
      }
      // Process overlapping range of current range, data need deep copy.
      auto data_ptr = std::make_shared<ProfilerData>(*data);
      data_ptr->start_time_ = last_data->end_time_;
      data_ptr->dur_time_ = data_ptr->end_time_ - data_ptr->start_time_;
      AnalyzeModuleSummaryData(data);
      last_data = data_ptr;
    }
  }
}

void ProfilerAnalyzer::EndStep() {
  if (!profiler_enable()) {
    return;
  }

  std::unique_lock<SpinLock> lock(data_mutex_);
  if (data_.empty()) {
    return;
  }

#ifdef WITH_BACKEND
  step_time_ = GetTimeStamp() - step_start_time_;
#endif
  (void)data_line_.emplace_back(std::make_shared<StepInfo>(step_, step_time_), std::move(data_));
}

void ProfilerAnalyzer::SaveJsonData(const ProfilerDataPtr &data) {
  MS_EXCEPTION_IF_NULL(data);
  nlohmann::json json_data;
  if (data->is_stage_) {
    json_data[kJsonName] = kProfilerStageString.at(data->stage_);
  } else {
    json_data[kJsonName] =
      kProfilerModuleString.at(data->module_) + "::" + kProfilerEventString.at(data->event_) + "::" + data->op_name_;
  }
  json_data[kJsonPh] = kJsonPhX;
  json_data[kJsonPid] = std::to_string(data->pid_);
  json_data[kJsonTid] = std::to_string(data->tid_);
  json_data[kJsonTs] = static_cast<double>(data->start_time_) / kNsToUs;
  json_data[kJsonDur] = static_cast<double>(data->dur_time_) / kNsToUs;
  nlohmann::json args;
  args[kJsonFlowId] = data->flow_id_;
  if (!data->py_stack_.empty()) {
    args[kJsonPyStack] = data->py_stack_;
  }
  json_data[kJsonArgs] = args;

  (void)json_infos_.emplace_back(json_data);
}

void ProfilerAnalyzer::AddPythonSummaryData(const uint64_t span_time) {
  uint64_t python_time = span_time;
  (void)std::for_each(stage_infos_.begin(), stage_infos_.end(),
                      [&python_time](const std::pair<ProfilerStage, ProfilerStatisticsInfoPtr> &iter) {
                        python_time -= iter.second->total_time_;
                      });
  auto stage_info = std::make_shared<ProfilerStatisticsInfo>(kProfilerStageString.at(ProfilerStage::kPython), false);
  stage_info->AccumulateTime(python_time);
  (void)stage_infos_.emplace(ProfilerStage::kPython, stage_info);

  auto module_info = std::make_shared<ProfilerModuleInfo>();
  module_info->module_statistics_info_ =
    std::make_shared<ProfilerStatisticsInfo>(kProfilerModuleString.at(ProfilerModule::kPython));
  module_info->module_statistics_info_->AccumulateTime(python_time);
  module_total_time_ += python_time;
  (void)module_infos_.emplace(ProfilerModule::kPython, module_info);
}

void ProfilerAnalyzer::AnalyzeSummaryData(const ProfilerDataPtr &data) {
  if (data->is_stage_) {
    AnalyzeStageSummaryData(data);
  } else {
    AnalyzeEventSummaryData(data);
  }
}

void ProfilerAnalyzer::AnalyzeStageSummaryData(const ProfilerDataPtr &data) {
  MS_EXCEPTION_IF_NULL(data);
  if (stage_infos_.count(data->stage_) > 0) {
    auto &stage_info = stage_infos_[data->stage_];
    MS_EXCEPTION_IF_NULL(stage_info);
    stage_info->AccumulateTime(data->dur_time_);
  } else {
    auto stage_info =
      std::make_shared<ProfilerStatisticsInfo>(kProfilerStageString.at(data->stage_), data->is_inner_event_);
    stage_info->AccumulateTime(data->dur_time_);
    (void)stage_infos_.emplace(data->stage_, stage_info);
  }
}

void ProfilerAnalyzer::AnalyzeModuleSummaryData(const ProfilerDataPtr &data) {
  MS_EXCEPTION_IF_NULL(data);
  if (module_infos_.count(data->module_) == 0) {
    auto module_info_ptr = std::make_shared<ProfilerModuleInfo>();
    module_info_ptr->module_statistics_info_ =
      std::make_shared<ProfilerStatisticsInfo>(kProfilerModuleString.at(data->module_));
    (void)module_infos_.emplace(data->module_, module_info_ptr);
  }

  auto &module_info_ptr = module_infos_[data->module_];
  module_info_ptr->module_statistics_info_->AccumulateTime(data->dur_time_);
  module_total_time_ += data->dur_time_;
}

void ProfilerAnalyzer::AnalyzeEventSummaryData(const ProfilerDataPtr &data) {
  MS_EXCEPTION_IF_NULL(data);
  if (module_infos_.count(data->module_) == 0) {
    MS_LOG(INFO) << "Summarize Unknown module : " << data->module_ << ", will skip current data.";
    return;
  }

  auto &module_info_ptr = module_infos_[data->module_];
  auto event_infos_ptr = &(module_info_ptr->event_infos_);
  if (event_infos_ptr->count(data->event_) == 0) {
    auto event_info_ptr = std::make_shared<ProfilerEventInfo>();
    event_info_ptr->event_statistics_info_ =
      std::make_shared<ProfilerStatisticsInfo>(kProfilerEventString.at(data->event_), data->is_inner_event_);
    (void)event_infos_ptr->emplace(data->event_, event_info_ptr);
  }

  auto &event_info_ptr = (*event_infos_ptr)[data->event_];
  event_info_ptr->event_statistics_info_->AccumulateTime(data->dur_time_);
  AnalyzeOpSummaryData(&event_info_ptr->op_infos_, data);
}

void ProfilerAnalyzer::AnalyzeOpSummaryData(mindspore::HashMap<std::string, ProfilerStatisticsInfoPtr> *const op_infos,
                                            const ProfilerDataPtr &data) {
  MS_EXCEPTION_IF_NULL(op_infos);
  MS_EXCEPTION_IF_NULL(data);
  if (op_infos->count(data->op_name_) > 0) {
    auto &op_info = (*op_infos)[data->op_name_];
    MS_EXCEPTION_IF_NULL(op_info);
    return op_info->AccumulateTime(data->dur_time_);
  }

  auto op_info = std::make_shared<ProfilerStatisticsInfo>(data->op_name_, data->is_inner_event_);
  op_info->AccumulateTime(data->dur_time_);
  (void)op_infos->emplace(data->op_name_, op_info);
}

void ProfilerAnalyzer::DumpJsonData() const {
  ChangeFileMode(json_file_name_, S_IWUSR);
  std::ofstream ofs(json_file_name_, std::ofstream::app);
  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "Open file [" << json_file_name_ << "] failed!";
    return;
  }
  ofs << json_infos_.dump();
  ChangeFileMode(json_file_name_, S_IRUSR);
}

void ProfilerAnalyzer::DumpDetailData(const size_t step, const ProfilerDataSpan &span) const {
  ChangeFileMode(detail_info_file_name_, S_IWUSR);
  std::ofstream ofs(detail_info_file_name_, std::ofstream::app);
  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "Open file [" << detail_info_file_name_ << "] failed!";
    return;
  }

  ofs << "[Step:" << step << " step_time:" << step_time_ << "us, module_total_time:" << module_total_time_ << "us]\n";
  for (auto &data : span) {
    MS_EXCEPTION_IF_NULL(data);
    std::string title_name = data->is_stage_ ? ("stage:" + kProfilerStageString.at(data->stage_))
                                             : ("module:" + kProfilerModuleString.at(data->module_));
    ofs << title_name << ", event:" << kProfilerEventString.at(data->event_) << ", op:" << data->op_name_
        << ", start_time:" << data->start_time_ << ", end_time:" << data->end_time_ << ", dur_time:," << data->dur_time_
        << ",us, tid:" << std::to_string(data->tid_) << ", pid:" << data->pid_ << "\n";
  }
  ofs << "\n";

  ChangeFileMode(detail_info_file_name_, S_IRUSR);
}

void ProfilerAnalyzer::DumpSummaryData(const size_t step) const {
  // Fill the summary info.
  std::stringstream string_stream;
  string_stream << "[Step:" << step << ", step_time:" << step_time_ << "us, module_total_time:" << module_total_time_
                << "us]\n";
  DumpModuleSummaryData(string_stream);
  DumpStageSummaryData(string_stream);
  std::cout << string_stream.str() << std::endl;

  ChangeFileMode(summary_info_file_name_, S_IWUSR);
  std::ofstream ofs(summary_info_file_name_, std::ofstream::app);
  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "Open file [" << summary_info_file_name_ << "] failed!";
    return;
  }
  ofs << string_stream.str();
  ChangeFileMode(summary_info_file_name_, S_IRUSR);
}

void ProfilerAnalyzer::DumpStageSummaryData(std::stringstream &string_stream) const {
  // Order module info by total time.
  std::multimap<uint64_t, ProfilerStatisticsInfo *, std::greater_equal<uint64_t>> order_stage_infos;
  for (auto &stage_info : stage_infos_) {
    auto &stage_statistics_info = stage_info.second;
    MS_EXCEPTION_IF_NULL(stage_statistics_info);
    stage_statistics_info->Average();
    stage_statistics_info->CalculatePercent(step_time_);
    (void)order_stage_infos.emplace(stage_statistics_info->total_time_, stage_statistics_info.get());
  }

  string_stream << "==========================================[Stage]==========================================\n";
  for (auto &order_stage_info : order_stage_infos) {
    auto &stage_statistics_info = order_stage_info.second;
    string_stream << "Stage:" << stage_statistics_info->name_ << std::fixed << std::setprecision(kPrecisionDigits)
                  << ", total_time:" << stage_statistics_info->total_time_
                  << "us, average_time:" << stage_statistics_info->average_time_
                  << "us, total_count:" << stage_statistics_info->count_
                  << ", percent:" << stage_statistics_info->percent_ << "%\n";
    string_stream << "\n";
  }

  string_stream << "\n";
}

void ProfilerAnalyzer::DumpModuleSummaryData(std::stringstream &string_stream) const {
  // Order module info by total time.
  std::multimap<uint64_t, ProfilerModuleInfo *, std::greater_equal<uint64_t>> order_module_infos;
  for (auto &module_info : module_infos_) {
    MS_EXCEPTION_IF_NULL(module_info.second);
    auto &module_statistics_info = module_info.second->module_statistics_info_;
    MS_EXCEPTION_IF_NULL(module_statistics_info);
    module_statistics_info->Average();
    module_statistics_info->CalculatePercent(module_total_time_);
    (void)order_module_infos.emplace(module_statistics_info->total_time_, module_info.second.get());
  }

  string_stream << "==========================================[Module]==========================================\n";
  for (auto &order_module_info : order_module_infos) {
    auto &module_statistics_info = order_module_info.second->module_statistics_info_;
    string_stream << "Module:" << module_statistics_info->name_ << std::fixed << std::setprecision(kPrecisionDigits)
                  << ", total_time:" << module_statistics_info->total_time_
                  << "us, average_time:" << module_statistics_info->average_time_
                  << "us, total_count:" << module_statistics_info->count_
                  << ", percent:" << module_statistics_info->percent_ << "%\n";
    DumpEventSummaryData(order_module_info.second->event_infos_, string_stream);
  }

  string_stream << "\n";
}

void ProfilerAnalyzer::DumpEventSummaryData(const std::map<ProfilerEvent, ProfilerEventInfoPtr> &event_infos,
                                            std::stringstream &string_stream) const {
  // Order event info by total time.
  std::multimap<uint64_t, ProfilerEventInfo *, std::greater_equal<uint64_t>> order_event_infos;
  std::multimap<uint64_t, ProfilerEventInfo *, std::greater_equal<uint64_t>> order_inner_event_infos;
  for (auto &event_info : event_infos) {
    MS_EXCEPTION_IF_NULL(event_info.second);
    auto &event_statistics_info = event_info.second->event_statistics_info_;
    MS_EXCEPTION_IF_NULL(event_statistics_info);
    event_statistics_info->Average();
    event_statistics_info->CalculatePercent(module_total_time_);
    if (event_statistics_info->is_inner_info_) {
      (void)order_inner_event_infos.emplace(event_statistics_info->total_time_, event_info.second.get());
    } else {
      (void)order_event_infos.emplace(event_statistics_info->total_time_, event_info.second.get());
    }
  }

  for (auto &order_event_info : order_event_infos) {
    auto &event_statistics_info = order_event_info.second->event_statistics_info_;
    string_stream << "  Event:" << event_statistics_info->name_ << std::fixed << std::setprecision(kPrecisionDigits)
                  << ", total_time:" << event_statistics_info->total_time_
                  << "us, average_time:" << event_statistics_info->average_time_
                  << "us, total_count:" << event_statistics_info->count_
                  << ", percent:" << event_statistics_info->percent_ << "%\n";
    DumpOpSummaryData(order_event_info.second->op_infos_, string_stream);
  }

  // Inner event.
  for (auto &order_inner_event_info : order_inner_event_infos) {
    auto &event_statistics_info = order_inner_event_info.second->event_statistics_info_;
    string_stream << "  EventInner:" << event_statistics_info->name_ << std::fixed
                  << std::setprecision(kPrecisionDigits) << ", total_time:" << event_statistics_info->total_time_
                  << "us, average_time:" << event_statistics_info->average_time_
                  << "us, total_count:" << event_statistics_info->count_ << "\n";
    DumpOpSummaryData(order_inner_event_info.second->op_infos_, string_stream);
  }

  string_stream << "\n";
}

void ProfilerAnalyzer::DumpOpSummaryData(const mindspore::HashMap<std::string, ProfilerStatisticsInfoPtr> &op_infos,
                                         std::stringstream &string_stream) const {
  if (show_top_num_ == 0) {
    return;
  }

  // Order op info by total time and average time.
  std::multimap<uint64_t, ProfilerStatisticsInfo *, std::greater_equal<uint64_t>> total_time_order_op_infos;
  std::multimap<double, ProfilerStatisticsInfo *, std::greater_equal<double>> average_time_order_op_infos;
  for (auto &op_info : op_infos) {
    auto &op_statistics_info = op_info.second;
    MS_EXCEPTION_IF_NULL(op_statistics_info);
    op_statistics_info->Average();
    op_statistics_info->CalculatePercent(module_total_time_);
    (void)total_time_order_op_infos.emplace(op_statistics_info->total_time_, op_statistics_info.get());
    (void)average_time_order_op_infos.emplace(op_statistics_info->average_time_, op_statistics_info.get());
  }

  // Show the op info by the total time top num.
  string_stream << "    Total time top " << show_top_num_ << " op:\n";
  int show_num = 0;
  for (auto &order_op_info : total_time_order_op_infos) {
    if (++show_num > show_top_num_) {
      break;
    }
    auto &op_statistics_info = order_op_info.second;
    string_stream << "      Op:" << op_statistics_info->name_ << std::fixed << std::setprecision(kPrecisionDigits)
                  << ", total_time:" << op_statistics_info->total_time_
                  << "us, average_time:" << op_statistics_info->average_time_
                  << "us, total_count:" << op_statistics_info->count_;
    if (op_statistics_info->is_inner_info_) {
      string_stream << "\n";
    } else {
      string_stream << ", percent:" << op_statistics_info->percent_ << "%\n";
    }
  }

  // Show the op info by the average time top num.
  string_stream << "    Average time top " << show_top_num_ << " op:\n";
  show_num = 0;
  for (auto &order_op_info : average_time_order_op_infos) {
    if (++show_num > show_top_num_) {
      break;
    }
    auto &op_statistics_info = order_op_info.second;
    string_stream << "      Op:" << op_statistics_info->name_ << std::fixed << std::setprecision(kPrecisionDigits)
                  << ", average_time:" << op_statistics_info->average_time_
                  << "us, total_time:" << op_statistics_info->total_time_
                  << "us, total_count:" << op_statistics_info->count_;
    if (op_statistics_info->is_inner_info_) {
      string_stream << "\n";
    } else {
      string_stream << ", percent:" << op_statistics_info->percent_ << "%\n";
    }
  }

  string_stream << "\n";
}
}  // namespace runtime
}  // namespace mindspore
