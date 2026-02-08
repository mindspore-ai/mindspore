/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_PROFILER_PROFILER_H_
#define MINDSPORE_CCSRC_RUNTIME_PROFILER_PROFILER_H_

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
#include <sys/syscall.h>
#endif
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <tuple>

#include "nlohmann/json.hpp"
#include "utils/ms_utils.h"
#include "utils/hash_map.h"
#include "utils/log_adapter.h"
#include "utils/convert_utils_base.h"
#include "include/utils/visible.h"
#include "async/spinlock.h"
#include "tools/profiler/profiling.h"

namespace mindspore {
namespace runtime {
static const char kDefaultOpName[] = "Default";
static const size_t kPercent = 100;

enum class ProfilerStage {
  kDefault,
  kPython,
  kCapture,
  kRunGraph,
  kRunGrad,
  kRunOp,
  kAsnumpy,
  kCompileGradGraph,
  kWaitPipeline,
  kSyncStream,
};

enum class ProfilerModule { kDefault, kGraphExecutorPy, kRuntime, kPynative, kKernel, kPython, kCapture, kOther };

enum class ProfilerEvent {
  kDefault,
  kKernelInfer,
  kKernelResize,
  kKernelInferAndResize,
  kKernelLaunch,
  kKernelLaunckCallback,
  kKernelUpdate,
  kKernelPrepareData,
  kGraphLaunch,
  kInputProcess,
  kOutputProcess,
  kWaitTaskFinish,
  kPreLaunch,
  kPostLaunch,
  kSendOutput,
  kMemoryAlloc,
  kMemoryFree,
  kCopyData,
  kStreamSync,
  kProcessMultiStream,
  kWaitKernelsInferFinish,
  kWaitKernelsResizeFinish,
  kWaitKernelsLaunchFinish,

  // Inner event is not counted in the total time.
  kKernelInferInner,
  kKernelInferDataSync,
  kKernelResizeInner,
  kKernelLaunchInner,
  kBackendGraphRunInner,

  // PyNative Pipeline
  kRunOp,
  kPyNativeFrontendTask,
  kPyNativeBackendTask,
  kPyNativeDeviceTask,
  kPyNativeLaunchTask,
  kPyNativeBpropTask,
  kWait,
  // PyNative inner Event
  kPyNativeGilAcquire,
  kPyNativeCast,
  kPyNativeInfer,
  kPyNativeOpCompile,
  kPyNativeGradExpander,
  kPyNativeGradUpdateSens,
  kPyNativeGradClearTopCell,
  kPyNativeGradClearAutoGradCell,
  // PyBoost
  kPyBoostInferOutput,
  kPyBoostInferByOpDef,
  kPyBoostCreateOutputTensor,
  kPyBoostDeviceTask,
  kPyBoostMallocInput,
  kPyBoostMallocOutput,
  kPyBoostLaunchAclnn,
  kPyBoostLaunchAtb,
  // pybind api
  kPyNativeNewGraph,
  kPyNativeEndGraph,
  // Python
  kPythonObserved,
  // Capture Event
  kCaptureRunGraph,
  kCaptureProcess,
  kCaptureCompile,
  kCaptureGuard,
  // AclNN
  kAclnnHitCacheStage1,
  kAclnnHitCacheStage2,
  kAclnnMissCacheStage1,
  kAclnnMissCacheStage2,
  kAclnnUpdateAddress,
  kAclnnRunOp,
  // NoGraph grad
  kRunExpanderFunc,
  kEmitOp,
  kExecute,
  kReleaseResource,
  kNativeFunc,
};

static const std::map<ProfilerStage, std::string> kProfilerStageString = {
  {ProfilerStage::kDefault, "Default"},           {ProfilerStage::kPython, "Python"},
  {ProfilerStage::kCapture, "Capture"},           {ProfilerStage::kRunGraph, "RunGraph"},
  {ProfilerStage::kRunGrad, "RunGrad"},           {ProfilerStage::kRunOp, "RunOp"},
  {ProfilerStage::kAsnumpy, "Asnumpy"},           {ProfilerStage::kCompileGradGraph, "CompileGradGraph"},
  {ProfilerStage::kWaitPipeline, "WaitPipeline"}, {ProfilerStage::kSyncStream, "SyncStream"},
};

static const std::map<ProfilerModule, std::string> kProfilerModuleString = {
  {ProfilerModule::kDefault, "Default"},
  {ProfilerModule::kGraphExecutorPy, "GraphExecutorPy"},
  {ProfilerModule::kRuntime, "RuntimeFramework"},
  {ProfilerModule::kPynative, "PynativeFramework"},
  {ProfilerModule::kKernel, "Kernel"},
  {ProfilerModule::kPython, "Python"},
  {ProfilerModule::kCapture, "Capture"},
  {ProfilerModule::kOther, "Other"},
};

static const std::map<ProfilerEvent, std::string> kProfilerEventString = {
  {ProfilerEvent::kDefault, "Default"},
  {ProfilerEvent::kKernelInfer, "KernelInfer"},
  {ProfilerEvent::kKernelResize, "KernelResize"},
  {ProfilerEvent::kKernelInferAndResize, "KernelInferAndResize"},
  {ProfilerEvent::kKernelLaunch, "KernelLaunch"},
  {ProfilerEvent::kKernelLaunckCallback, "KernelLaunchCallback"},
  {ProfilerEvent::kKernelUpdate, "KernelUpdate"},
  {ProfilerEvent::kKernelPrepareData, "KernelPrepareData"},
  {ProfilerEvent::kGraphLaunch, "GraphLaunch"},
  {ProfilerEvent::kInputProcess, "InputProcess"},
  {ProfilerEvent::kOutputProcess, "OutputProcess"},
  {ProfilerEvent::kWaitTaskFinish, "WaitTaskFinish"},
  {ProfilerEvent::kPreLaunch, "PreLaunch"},
  {ProfilerEvent::kPostLaunch, "PostLaunch"},
  {ProfilerEvent::kSendOutput, "SendOutput"},
  {ProfilerEvent::kMemoryAlloc, "MemoryAlloc"},
  {ProfilerEvent::kMemoryFree, "MemoryFree"},
  {ProfilerEvent::kCopyData, "CopyData"},
  {ProfilerEvent::kStreamSync, "StreamSync"},
  {ProfilerEvent::kProcessMultiStream, "ProcessMultiStream"},
  {ProfilerEvent::kWaitKernelsInferFinish, "WaitKernelsInferFinish"},
  {ProfilerEvent::kWaitKernelsResizeFinish, "WaitKernelsResizeFinish"},
  {ProfilerEvent::kWaitKernelsLaunchFinish, "WaitKernelsLaunchFinish"},
  // Inner event.
  {ProfilerEvent::kKernelInferInner, "KernelInferInner"},
  {ProfilerEvent::kKernelResizeInner, "KernelResizeInner"},
  {ProfilerEvent::kKernelInferDataSync, "KernelInferDataSync"},
  {ProfilerEvent::kKernelLaunchInner, "KernelLaunchInner"},
  {ProfilerEvent::kBackendGraphRunInner, "BackendGraphRunInner"},
  // PyNative events
  {ProfilerEvent::kRunOp, "RunOp"},
  {ProfilerEvent::kPyNativeFrontendTask, "FrontendTask"},
  {ProfilerEvent::kPyNativeBackendTask, "BackendTask"},
  {ProfilerEvent::kPyNativeDeviceTask, "DeviceTask"},
  {ProfilerEvent::kPyNativeLaunchTask, "LaunchTask"},
  {ProfilerEvent::kPyNativeBpropTask, "BpropTask"},
  {ProfilerEvent::kWait, "Wait"},
  {ProfilerEvent::kPyNativeGilAcquire, "AcquireGil"},
  {ProfilerEvent::kPyNativeCast, "PyNativeCast"},
  {ProfilerEvent::kPyNativeInfer, "PyNativeInfer"},
  {ProfilerEvent::kPyNativeOpCompile, "OpCompile"},
  {ProfilerEvent::kPyNativeGradExpander, "Expander"},
  {ProfilerEvent::kPyNativeGradUpdateSens, "UpdateSens"},
  {ProfilerEvent::kPyNativeGradClearTopCell, "ClearTopCell"},
  {ProfilerEvent::kPyNativeGradClearAutoGradCell, "ClearAutoGradCell"},
  // PyBoost events
  {ProfilerEvent::kPyBoostInferOutput, "InferOutput"},
  {ProfilerEvent::kPyBoostInferByOpDef, "InferByOpDef"},
  {ProfilerEvent::kPyBoostCreateOutputTensor, "CreateOutputTensor"},
  {ProfilerEvent::kPyBoostDeviceTask, "DeviceTask"},
  {ProfilerEvent::kPyBoostMallocInput, "MallocInput"},
  {ProfilerEvent::kPyBoostMallocOutput, "MallocOutput"},
  {ProfilerEvent::kPyBoostLaunchAclnn, "LaunchAclnn"},
  {ProfilerEvent::kPyBoostLaunchAtb, "LaunchAtb"},
  // pybind api
  {ProfilerEvent::kPyNativeNewGraph, "new_graph"},
  {ProfilerEvent::kPyNativeEndGraph, "end_graph"},
  // python events
  {ProfilerEvent::kPythonObserved, "PythonObserved"},
  // Capture events
  {ProfilerEvent::kCaptureRunGraph, "CaptureRunGraph"},
  {ProfilerEvent::kCaptureProcess, "CaptureProcess"},
  {ProfilerEvent::kCaptureCompile, "CaptureCompile"},
  {ProfilerEvent::kCaptureGuard, "CaptureGuard"},
  // AclNN
  {ProfilerEvent::kAclnnHitCacheStage1, "AclnnHitCacheStage1"},
  {ProfilerEvent::kAclnnHitCacheStage2, "AclnnHitCacheStage2"},
  {ProfilerEvent::kAclnnMissCacheStage1, "AclnnMissCacheStage1"},
  {ProfilerEvent::kAclnnMissCacheStage2, "AclnnMissCacheStage2"},
  {ProfilerEvent::kAclnnUpdateAddress, "AclnnUpdateAddress"},
  {ProfilerEvent::kAclnnRunOp, "AclnnRunOp"},
  // Run Grad fun events
  {ProfilerEvent::kRunExpanderFunc, "RunExpanderFunc"},
  {ProfilerEvent::kEmitOp, "EmitOp"},
  {ProfilerEvent::kReleaseResource, "ReleaseResource"},
  {ProfilerEvent::kExecute, "Execute"},
  {ProfilerEvent::kNativeFunc, "NativeFunc"},
};

#define PROFILER_START(start_time)                                                            \
  do {                                                                                        \
    if (MS_UNLIKELY(mindspore::runtime::ProfilerAnalyzer::GetInstance().profiler_enable())) { \
      start_time = mindspore::runtime::ProfilerAnalyzer::GetInstance().GetTimeStamp();        \
    }                                                                                         \
  } while (0);

// Match PROFILER_START to use.
#define PROFILER_END(start_time, module, event, op_name, is_inner_event, ...)                                          \
  [&](const auto &...args) {                                                                                           \
    if (MS_UNLIKELY(mindspore::runtime::ProfilerAnalyzer::GetInstance().profiler_enable())) {                          \
      auto end_time = mindspore::runtime::ProfilerAnalyzer::GetInstance().GetTimeStamp();                              \
      auto brief_name = mindspore::runtime::ProfilerAnalyzer::GetInstance().GetBriefName(op_name);                     \
      if constexpr (sizeof...(args) > 0) {                                                                             \
        static auto ascend_profiler = mindspore::profiler::Profiler::GetInstance(kAscendDevice);                       \
        if (ascend_profiler != nullptr && ascend_profiler->EnableRecordShapes()) {                                     \
          auto tuple_params = std::forward_as_tuple(args...);                                                          \
          auto input_tensor = std::get<0>(tuple_params); /* std::vector<KernelTensor *> */                             \
          std::string input_shape = "";                                                                                \
          std::string input_type = "";                                                                                 \
          std::vector<ShapeVector> input_shapes;                                                                       \
          std::vector<std::string> input_types;                                                                        \
          for (auto &tensor : input_tensor) {                                                                          \
            input_shapes.emplace_back(tensor->GetShapeVector());                                                       \
            input_types.emplace_back(tensor->GetType()->ToString());                                                   \
          }                                                                                                            \
          mindspore::runtime::ProfilerAnalyzer::GetInstance().RecordShapesData(brief_name, input_shapes, input_types); \
        }                                                                                                              \
      }                                                                                                                \
      mindspore::runtime::ProfilerAnalyzer::GetInstance().RecordData(                                                  \
        std::make_shared<mindspore::runtime::ProfilerData>(module, event, op_name, brief_name, is_inner_event,         \
                                                           start_time, end_time));                                     \
    }                                                                                                                  \
  }(__VA_ARGS__)

// Match PROFILER_START to use.
#define PROFILER_STAGE_END(start_time, stage)                                             \
  do {                                                                                    \
    if (mindspore::runtime::ProfilerAnalyzer::GetInstance().profiler_enable()) {          \
      auto end_time = mindspore::runtime::ProfilerAnalyzer::GetInstance().GetTimeStamp(); \
      mindspore::runtime::ProfilerAnalyzer::GetInstance().RecordData(                     \
        std::make_shared<mindspore::runtime::ProfilerData>(stage, start_time, end_time)); \
    }                                                                                     \
  } while (0);

// Record the profiler data by the constructor and destructor of this class.
class PROFILER_EXPORT ProfilerRecorder {
 public:
  ProfilerRecorder(ProfilerModule module, ProfilerEvent event, const std::string &op_name, bool is_inner_event = false,
                   bool need_py_stack = false, uint64_t flow_id = UINT64_MAX);
  ~ProfilerRecorder();

  struct Data {
    Data(ProfilerModule module, ProfilerEvent event, std::string op_full_name, std::string op_name,
         std::string py_stack, uint64_t start_time, uint64_t flow_id, bool is_inner_event)
        : module_(module),
          event_(event),
          op_full_name_(std::move(op_full_name)),
          op_name_(std::move(op_name)),
          py_stack_(std::move(py_stack)),
          start_time_(start_time),
          flow_id_(flow_id),
          is_inner_event_(is_inner_event) {}
    ProfilerModule module_;
    ProfilerEvent event_;
    std::string op_full_name_;
    std::string op_name_;
    std::string py_stack_;
    uint64_t start_time_;
    uint64_t flow_id_;
    bool is_inner_event_;
  };

  inline static const std::string kNoName{};

 private:
  std::unique_ptr<Data> data_{nullptr};
};

class PROFILER_EXPORT PythonProfilerRecorder {
 public:
  explicit PythonProfilerRecorder(const std::string &record_name);
  ~PythonProfilerRecorder() = default;

  void record_start();
  void record_end();

 private:
  uint64_t start_time_;
  std::string record_name_;
  ProfilerModule module_;
  ProfilerEvent event_;
};

class PROFILER_EXPORT ProfilerStageRecorder {
 public:
  explicit ProfilerStageRecorder(ProfilerStage stage);
  ~ProfilerStageRecorder();

 private:
  ProfilerStage stage_{ProfilerStage::kDefault};
  uint64_t start_time_{0};
};

struct StepInfo {
  StepInfo(size_t step, uint64_t step_time) : step_(step), step_time_(step_time) {}
  const size_t step_;
  const uint64_t step_time_;
};

using StepInfoPtr = std::shared_ptr<StepInfo>;

struct ProfilerData {
  ProfilerData(const std::string &module, const std::string &event, const std::string &op_name, uint64_t start_time,
               uint64_t end_time, int8_t level, const std::map<std::string, std::string> &custom_info = {})
      : is_graph_data_(true),
        module_graph_(module),
        event_graph_(event),
        op_name_(op_name),
        start_time_(start_time),
        end_time_(end_time),
        level_(level),
        custom_info_(custom_info),
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
        tid_(LongToUlong(syscall(SYS_gettid))),
#else
        tid_(0),
#endif
        pid_(getpid()) {
  }

  ProfilerData(ProfilerModule module, ProfilerEvent event, const std::string &op_full_name, const std::string &op_name,
               bool is_inner_event, uint64_t start_time, uint64_t end_time, uint64_t flow_id = UINT64_MAX,
               std::string py_stack = "")
      : is_stage_(false),
        stage_(ProfilerStage::kDefault),
        module_(module),
        event_(event),
        op_full_name_(op_full_name),
        op_name_(op_name),
        is_inner_event_(is_inner_event),
        start_time_(start_time),
        end_time_(end_time),
        dur_time_(end_time - start_time),
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
        tid_(LongToUlong(syscall(SYS_gettid))),
#else
        tid_(0),
#endif
        pid_(getpid()),
        flow_id_(flow_id),
        py_stack_(std::move(py_stack)) {
  }

  ProfilerData(ProfilerStage stage, uint64_t start_time, uint64_t end_time)
      : is_stage_(true),
        stage_(stage),
        module_(ProfilerModule::kDefault),
        event_(ProfilerEvent::kDefault),
        op_full_name_(""),
        op_name_(""),
        is_inner_event_(false),
        start_time_(start_time),
        end_time_(end_time) {
    dur_time_ = end_time - start_time;
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
    tid_ = LongToUlong(syscall(SYS_gettid));
#else
    tid_ = 0;
#endif
    pid_ = getpid();
  }

  ProfilerData(const ProfilerData &other)
      : is_stage_(other.is_stage_),
        stage_(other.stage_),
        module_(other.module_),
        event_(other.event_),
        op_full_name_(other.op_full_name_),
        op_name_(other.op_name_),
        is_inner_event_(other.is_inner_event_),
        start_time_(other.start_time_),
        end_time_(other.end_time_),
        dur_time_(other.dur_time_),
        tid_(other.tid_),
        pid_(other.pid_),
        flow_id_(other.flow_id_) {}

  ProfilerData &operator=(const ProfilerData &other) {
    if (this == &other) {
      return *this;
    }

    is_stage_ = other.is_stage_;
    stage_ = other.stage_;
    module_ = other.module_;
    event_ = other.event_;
    op_full_name_ = other.op_full_name_;
    op_name_ = other.op_name_;
    is_inner_event_ = other.is_inner_event_;
    start_time_ = other.start_time_;
    end_time_ = other.end_time_;
    dur_time_ = other.dur_time_;
    tid_ = other.tid_;
    pid_ = other.pid_;
    flow_id_ = other.flow_id_;
    return *this;
  }

  bool is_graph_data_{false};
  bool is_stage_{false};
  ProfilerStage stage_{ProfilerStage::kDefault};
  ProfilerModule module_{ProfilerModule::kDefault};
  ProfilerEvent event_{ProfilerEvent::kDefault};
  std::string module_graph_{};
  std::string event_graph_{};
  std::string op_full_name_{};
  std::string op_name_{};
  bool is_inner_event_{false};
  uint64_t start_time_{0L};
  uint64_t end_time_{0L};
  uint64_t dur_time_{0L};
  int8_t level_{-1};
  std::map<std::string, std::string> custom_info_{};
  uint64_t tid_{};
  int32_t pid_{0};
  uint64_t flow_id_{UINT64_MAX};
  std::string py_stack_{};
};
using ProfilerDataPtr = std::shared_ptr<ProfilerData>;

struct ProfilerFlowEventData {};

struct ProfilerStatisticsInfo {
  explicit ProfilerStatisticsInfo(const std::string &name, bool is_inner_info = false)
      : name_(name), is_inner_info_(is_inner_info), count_(0), total_time_(0), average_time_(0), percent_(0) {}
  std::string name_;
  bool is_inner_info_;
  size_t count_;
  uint64_t total_time_;
  double average_time_;
  double percent_;

  void AccumulateTime(uint64_t time) {
    total_time_ += time;
    ++count_;
  }
  void Average() {
    MS_EXCEPTION_IF_ZERO("count", count_);
    average_time_ = static_cast<double>(total_time_) / count_;
  }
  void CalculatePercent(uint64_t step_total_time) {
    MS_EXCEPTION_IF_ZERO("step_total_time", step_total_time);
    percent_ = (static_cast<double>(total_time_) / step_total_time) * kPercent;
  }
};
using ProfilerStatisticsInfoPtr = std::shared_ptr<ProfilerStatisticsInfo>;

struct ProfilerEventInfo {
  ProfilerStatisticsInfoPtr event_statistics_info_;
  mindspore::HashMap<std::string, ProfilerStatisticsInfoPtr> op_infos_;
};
using ProfilerEventInfoPtr = std::shared_ptr<ProfilerEventInfo>;

struct ProfilerModuleInfo {
  ProfilerStatisticsInfoPtr module_statistics_info_;
  std::map<ProfilerEvent, ProfilerEventInfoPtr> event_infos_;
};
using ProfilerModuleInfoPtr = std::shared_ptr<ProfilerModuleInfo>;

using ProfilerDataSpan = std::list<ProfilerDataPtr>;
class PROFILER_EXPORT ProfilerAnalyzer {
 public:
  static ProfilerAnalyzer &GetInstance() noexcept;

  // Increase step at the step begin.
  void StartStep();
  // Analyze and output profiler data at the step end.
  void EndStep();

  void Clear() noexcept;

  // Enable MindInsight Profile
  void EnableMiProfile();

  // Disable MindInsight Profile
  void DisableMiProfile();

  // The used by ProfilerRecorder to record data.
  bool profiler_enable() const;
  bool enable_by_env() const { return profiler_enable_; }
  void RecordData(const ProfilerDataPtr &data) noexcept;
  void RecordShapesData(const std::string &op_name, const std::vector<std::vector<int64_t>> &input_shapes,
                        const std::vector<std::string> &input_types) noexcept;
  void RecordFlowData(uint64_t flow_id);
  uint64_t GetTimeStamp() const noexcept;
  std::string GetBriefName(const std::string &scope_name) const;

  void ProcessModuleSummaryData(const ProfilerDataSpan &span);

  const size_t step() const { return step_; }
  void set_step_time(uint64_t step_time) { step_time_ = step_time; }
  void set_profiler_enable(bool profiler_enable) { profiler_enable_ = profiler_enable; }
  const ProfilerDataSpan &data() const { return data_; }
  const std::list<std::pair<StepInfoPtr, ProfilerDataSpan>> data_line() const { return data_line_; }
  const nlohmann::json &json_infos() const { return json_infos_; }
  const std::map<ProfilerModule, ProfilerModuleInfoPtr> &module_infos() const { return module_infos_; }
  const std::map<ProfilerStage, ProfilerStatisticsInfoPtr> &stage_infos() const { return stage_infos_; }
  void SetThreadIdToName(const std::thread::id &id, const std::string &name);

 private:
  ProfilerAnalyzer() = default;
  ~ProfilerAnalyzer() { Clear(); }
  DISABLE_COPY_AND_ASSIGN(ProfilerAnalyzer);

  void Initialize();
  void ProcessData();

  // Process data.
  void SaveJsonData(const ProfilerDataPtr &data);
  void AnalyzeSummaryData(const ProfilerDataPtr &data);
  void AnalyzeStageSummaryData(const ProfilerDataPtr &data);
  void AnalyzeModuleSummaryData(const ProfilerDataPtr &data);
  void AnalyzeEventSummaryData(const ProfilerDataPtr &data);
  void AnalyzeOpSummaryData(mindspore::HashMap<std::string, ProfilerStatisticsInfoPtr> *const op_infos,
                            const ProfilerDataPtr &data);
  void AddPythonSummaryData(const uint64_t span_time);

  // Dump data.
  void DumpJsonData() const;
  void DumpDetailData(const size_t step, const ProfilerDataSpan &span) const;
  void DumpSummaryData(const size_t step) const;
  void DumpStageSummaryData(std::stringstream &string_stream) const;
  void DumpModuleSummaryData(std::stringstream &string_stream) const;
  void DumpEventSummaryData(const std::map<ProfilerEvent, ProfilerEventInfoPtr> &event_infos,
                            std::stringstream &string_stream) const;
  void DumpOpSummaryData(const mindspore::HashMap<std::string, ProfilerStatisticsInfoPtr> &op_infos,
                         std::stringstream &string_stream) const;

  // The relevant members of step.
  size_t step_{0};
  uint64_t step_time_{0};
  uint64_t step_start_time_{0};
  uint64_t module_total_time_{0};
  ProfilerDataSpan data_;
  // Container list for all data_ points.
  std::list<std::pair<StepInfoPtr, ProfilerDataSpan>> data_line_;
  SpinLock data_mutex_;
  nlohmann::json json_infos_;
  // The data analyzed level is module-->event-->op, these data would not be cleared in unit test.
  std::map<ProfilerModule, ProfilerModuleInfoPtr> module_infos_;
  std::map<ProfilerStage, ProfilerStatisticsInfoPtr> stage_infos_;

  std::map<std::thread::id, std::string> thread_id_to_name_;

  // Save file name.
  std::string json_file_name_;
  std::string summary_info_file_name_;
  std::string detail_info_file_name_;

  // The relevant members of init.
  int show_top_num_{0};
  bool profiler_enable_{false};
  bool mi_profiler_enable_{false};
  bool init_{false};
};
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_PROFILER_PROFILER_H_
