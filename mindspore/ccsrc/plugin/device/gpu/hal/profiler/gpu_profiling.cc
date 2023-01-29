/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/hal/profiler/gpu_profiling.h"

#ifndef _WIN32
#include <cxxabi.h>
#else
#include <typeinfo>
#include <Windows.h>
#endif
#include <chrono>
#include <cmath>
#include <ctime>
#include <thread>
#include <sstream>
#include "plugin/device/gpu/hal/profiler/cupti_interface.h"
#include "plugin/device/gpu/hal/profiler/gpu_data_saver.h"
#include "include/common/pybind_api/api_register.h"
#include "utils/log_adapter.h"
#include "include/common/utils/utils.h"
#include "utils/profile.h"
#include "utils/ms_context.h"

#ifdef _MSC_VER
namespace {
static int check_align(size_t align) {
  constexpr int step = 2;
  for (size_t i = sizeof(void *); i != 0; i *= step) {
    if (align == i) {
      return 0;
    }
  }
  return EINVAL;
}

int posix_memalign(void **ptr, size_t align, size_t size) {
  if (check_align(align)) {
    return EINVAL;
  }
  if (ptr == nullptr) {
    return EINVAL;
  }

  int saved_errno = errno;
  void *p = _aligned_malloc(size, align);
  if (p == nullptr) {
    errno = saved_errno;
    return ENOMEM;
  }

  *ptr = p;
  return 0;
}
}  // namespace
#endif

namespace mindspore {
namespace profiler {
namespace gpu {
namespace {
PROFILER_REG(kGPUDevice, GPUProfiler);
}  // namespace
const size_t BUF_SIZE = 32 * 1024;
const size_t ALIGN_SIZE = 8;
#define CHECK_CUPTI_RET_WITH_ERROR(expression, message)                                          \
  if ((expression) != CUPTI_SUCCESS) {                                                           \
    const char *errstr;                                                                          \
    CuptiGetResultString(expression, &errstr);                                                   \
    MS_LOG(ERROR) << "CUPTI Error:" << errstr << " function:" << (message)                       \
                  << ". You may not have access to the NVIDIA GPU performance counters on "      \
                  << "the target device. Please use the root account to run profiling or "       \
                  << "configure permissions. If there is still the problem, please refer to the" \
                  << " GPU performance tuning document on the official website of mindinsight."; \
  }

#define CHECK_CUPTI_RET_WITH_EXCEPT(expression, message)                        \
  if ((expression) != CUPTI_SUCCESS) {                                          \
    const char *errstr;                                                         \
    CuptiGetResultString(expression, &errstr);                                  \
    MS_LOG(EXCEPTION) << "CUPTI Error:" << errstr << " function:" << (message); \
  }
#define CHECK_CUDA_RET_WITH_ERROR(expression, message)                                     \
  do {                                                                                     \
    cudaError_t status = (expression);                                                     \
    if (status != cudaSuccess) {                                                           \
      MS_LOG(ERROR) << "CUDA Error: " << (message) << " | Error Number: " << status << " " \
                    << cudaGetErrorString(status);                                         \
    }                                                                                      \
  } while (0)
#define PROFILER_ERROR_IF_NULLPTR(ptr)                           \
  do {                                                           \
    if ((ptr) == nullptr) {                                      \
      MS_LOG(ERROR) << ": The pointer[" << #ptr << "] is null."; \
      return;                                                    \
    }                                                            \
  } while (0)

int32_t GetThreadID() {
#ifndef _MSC_VER
  uint32_t thread_id = static_cast<uint32_t>(pthread_self());
  return thread_id;
#else
  std::thread::id tid = std::this_thread::get_id();
  std::stringstream ss;
  ss << tid;
  int32_t idx = 0;
  ss >> idx;
  return idx;
#endif
}

uint32_t GetStreamID(const CUcontext context, const void *stream) {
  uint32_t stream_id = 0;
  if (stream != nullptr) {
    CHECK_CUPTI_RET_WITH_ERROR(CuptiGetStreamId(context, (CUstream)stream, &stream_id), "CuptiGetStreamId");
    if (CuptiGetStreamId(context, (CUstream)stream, &stream_id) != CUPTI_SUCCESS) {
      MS_LOG(ERROR) << "Training process unexpectedly stopped, profiling data cannot be write to file"
                    << "To obtain the profiling data, do not interrupt the training process.";
    }
  }
  return stream_id;
}

uint64_t GetCUPTITimeStamp() {
  uint64_t time_stamp = 0l;
  CHECK_CUPTI_RET_WITH_ERROR(CuptiGetTimestamp(&time_stamp), "CuptiGetTimestamp");
  return time_stamp;
}

uint64_t GetHostTimeStamp() {
  auto cur_sys_clock = std::chrono::system_clock::now();
  uint64_t cur_time_stamp =
    std::chrono::duration_cast<std::chrono::nanoseconds>(cur_sys_clock.time_since_epoch()).count();
  return cur_time_stamp;
}

std::string GetKernelFunc(const char *name) {
#ifndef _MSC_VER
  const char *demangled_name = abi::__cxa_demangle(name, nullptr, nullptr, nullptr);
#else
  const char *demangled_name = typeid(name).name();
#endif
  if (demangled_name != nullptr) {
    return demangled_name;
  } else {
    return name;
  }
}

bool IsMemcpyAsyncEvent(CUpti_CallbackId cb_id) {
  switch (cb_id) {
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoHAsync_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DAsync_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DAsync_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoAAsync_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyPeerAsync:
      return true;
    default:
      return false;
  }
  return false;
}

bool IsMemcpySyncEvent(CUpti_CallbackId cb_id) {
  switch (cb_id) {
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpy:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoH_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoD_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoA_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoA_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpy2D_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DUnaligned_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpy3D_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoA_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyPeer:
      return true;
    default:
      return false;
  }
  return false;
}

void CUPTIApiExit(const std::shared_ptr<GPUProfiler> &gpu_profiler_inst, CUpti_CallbackId cb_id,
                  const CUpti_CallbackData *cb_data) {
  uint64_t start_timestamp = *cb_data->correlationData;
  uint64_t end_timestamp = GetCUPTITimeStamp();
  switch (cb_id) {
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel:
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernelMultiDevice:
      gpu_profiler_inst->EventHandleProcess(cb_id, cb_data, "cuLaunchKernel", start_timestamp, end_timestamp);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuMemAlloc:
    case CUPTI_DRIVER_TRACE_CBID_cuMemAlloc_v2:
      gpu_profiler_inst->EventHandleProcess(cb_id, cb_data, "cuMemAlloc", start_timestamp, end_timestamp);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuEventCreate:
    case CUPTI_DRIVER_TRACE_CBID_cuEventDestroy_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuEventRecord:
    case CUPTI_DRIVER_TRACE_CBID_cuEventSynchronize:
    case CUPTI_DRIVER_TRACE_CBID_cuEventElapsedTime:
    // In some cases, the callback of cuctxsetcurrent is only exist
    // without entry, so this callback is ignored
    case CUPTI_DRIVER_TRACE_CBID_cuCtxSetCurrent:
      break;
    default:
      gpu_profiler_inst->EventHandleProcess(cb_id, cb_data, "others_api", start_timestamp, end_timestamp);
      break;
  }
  if (IsMemcpyAsyncEvent(cb_id) || IsMemcpySyncEvent(cb_id)) {
    gpu_profiler_inst->EventHandleProcess(cb_id, cb_data, "cuMemcpy", start_timestamp, end_timestamp);
  }
}

void CUPTICallBackFunc(void *user_data, CUpti_CallbackDomain domain, CUpti_CallbackId cb_id,
                       const CUpti_CallbackData *cb_data) {
  if (domain != CUPTI_CB_DOMAIN_DRIVER_API) {
    return;
  }
  auto gpu_profiler_inst = GPUProfiler::GetInstance();
  PROFILER_ERROR_IF_NULLPTR(gpu_profiler_inst);
  if (!gpu_profiler_inst->GetEnableFlag()) {
    return;
  }

  PROFILER_ERROR_IF_NULLPTR(cb_data);
  if (cb_data->context == nullptr) {
    MS_LOG(DEBUG) << "Callback data context is null , correlation Id:" << cb_data->correlationId
                  << " callback id:" << cb_id;
    return;
  }

  if (cb_data->callbackSite == CUPTI_API_ENTER) {
    *cb_data->correlationData = GetCUPTITimeStamp();
  } else if (cb_data->callbackSite == CUPTI_API_EXIT) {
    CUPTIApiExit(gpu_profiler_inst, cb_id, cb_data);
  }
}

std::string GetKernelFuncName(std::string kernel_name) {
  // remove the return type name (void) in kernel_name.
  std::string search_pattern("void ");
  auto func_name_begin_iter = kernel_name.find(search_pattern);
  if (func_name_begin_iter == kernel_name.npos) {
    func_name_begin_iter = 0;
  } else {
    func_name_begin_iter += search_pattern.length();
  }
  return kernel_name.substr(func_name_begin_iter);
}

std::shared_ptr<GPUProfiler> GPUProfiler::GetInstance() {
  auto instance = Profiler::GetInstance(kGPUDevice);
  MS_EXCEPTION_IF_NULL(instance);
  return std::dynamic_pointer_cast<GPUProfiler>(instance);
}

void GPUProfiler::SyncEnable(const bool enable_flag) {
  MS_LOG(INFO) << "GPU Profiler synchronous enable flag:" << enable_flag;
  sync_enable_flag_ = enable_flag;
}

void GPUProfiler::StepProfilingEnable(const bool enable_flag) {
  MS_LOG(INFO) << "GPU Profiler enable flag:" << enable_flag;
  CHECK_CUPTI_RET_WITH_ERROR(CuptiActivityFlushAll(0), "CuptiActivityFlushAll");
  enable_flag_ = enable_flag;
}

void GPUProfiler::FixOpNameByCorrelationId(Event *event) {
  PROFILER_ERROR_IF_NULLPTR(event);
  if (event->api_type != CUPTIApiType::kActivity) {
    return;
  }
  auto iter = op_name_map_.find(event->correlation_id);
  if (iter != op_name_map_.end()) {
    event->op_name = std::move(iter->second);
  }
}

void GPUProfiler::AddEvent(Event &&event) {
  // protect callback concurrency for driver api and activity
  std::unique_lock<std::mutex> lock(event_mutex_);
  switch (event.api_type) {
    case CUPTIApiType::kCallback: {
      if (cupti_callback_events_count_ < max_cupti_callback_events_) {
        events_.emplace_back(std::move(event));
        cupti_callback_events_count_++;
      } else {
        cupti_callback_events_drop_count_++;
      }
      break;
    }
    case CUPTIApiType::kActivity: {
      if (cupti_activity_events_count_ < max_cupti_activity_events_) {
        events_.emplace_back(std::move(event));
        cupti_activity_events_count_++;
      } else {
        cupti_activity_events_drop_count_++;
      }
      break;
    }
    default:
      break;
  }
}

void GPUProfiler::EventLog(const Event &event) {
  MS_LOG(DEBUG) << "GPUProfiler"
                << ",\"kernel_name:" << event.kernel_name << "\",kernel_type:" << event.kernel_type
                << ",api_type:" << static_cast<int>(event.api_type) << ",start_time_stamp:" << event.start_time_stamp
                << ",end_time_stamp:" << event.end_time_stamp << ",cost:,"
                << (event.end_time_stamp - event.start_time_stamp) / kTimeUnit << ",op_name:" << event.op_name
                << ",device_id:" << event.device_id << ",correlation_id:" << event.correlation_id
                << ",thread_id:" << event.thread_id << ",context_id:" << event.context_id
                << ",stream_id:" << event.stream_id << ",cb_id:" << event.cb_id;
}

void GPUProfiler::ProcessEvents() {
  for (Event &event : events_) {
    if (event.op_name.empty()) {
      FixOpNameByCorrelationId(&event);
    }

    EventLog(event);

    if (event.op_name.empty() || event.cb_id == CUPTI_DRIVER_TRACE_CBID_cuStreamSynchronize) {
      continue;
    }

    auto iter = op_info_map_.find(event.op_name);
    if (iter != op_info_map_.end()) {
      switch (event.api_type) {
        case CUPTIApiType::kCallback: {
          iter->second.op_kernel_api_count += 1;
          // The time unit from ns to us
          iter->second.cupti_api_call_time += (event.end_time_stamp - event.start_time_stamp) / kTimeUnit;
          break;
        }
        case CUPTIApiType::kActivity: {
          iter->second.op_kernel_count += 1;
          // The time unit from ns to us
          iter->second.cupti_activity_time += (event.end_time_stamp - event.start_time_stamp) / kTimeUnit;
          break;
        }
        default:
          break;
      }
    }
  }
}

void GPUProfiler::OpsParser() {
  MS_LOG(INFO) << "Count the number of events size:" << events_.size()
               << " callback api:" << cupti_callback_events_count_ << " activity:" << cupti_activity_events_count_;

  if (cupti_activity_events_drop_count_ > 0 || cupti_callback_events_drop_count_ > 0) {
    MS_LOG(WARNING)
      << "The total number of events exceeded the profiler's processing capacity, some events were discarded."
      << " activity api events:" << cupti_activity_events_drop_count_
      << " callback api events:" << cupti_callback_events_drop_count_;
  }

  if (events_.size() == 0) {
    return;
  }

  ProcessEvents();
  MS_LOG(DEBUG) << "GPU_profiler, op_name, op_count , kernel_count, kernel_api_count,|"
                   ",cupti_activity_total_time, cupti_api_call_total_time, op_host_cost_total_time,|"
                   ",cupti_activity_average_time,cupti_api_call_average_time, op_host_cost_average_time"
                << std::endl;

  std::vector<std::pair<std::string, OpInfo>> order_vec(op_info_map_.begin(), op_info_map_.end());

  auto cmp_func = [](const std::pair<std::string, OpInfo> &a, const std::pair<std::string, OpInfo> &b) {
    return a.second.cupti_activity_time > b.second.cupti_activity_time;
  };
  std::sort(order_vec.begin(), order_vec.end(), cmp_func);

  for (auto iter = order_vec.begin(); iter != order_vec.end(); iter++) {
    if (iter->second.op_count == 0) {
      MS_LOG(ERROR) << "The num of operations can not be 0.";
      return;
    }
    MS_LOG(DEBUG) << "GPU_profiler"
                  << "," << iter->first << "," << iter->second.op_count << "," << iter->second.op_kernel_count << ","
                  << iter->second.op_kernel_api_count << ","
                  << "|," << iter->second.cupti_activity_time << "," << iter->second.cupti_api_call_time << ","
                  << iter->second.op_host_cost_time << ","
                  << "|," << round(iter->second.cupti_activity_time / iter->second.op_count) << ","
                  << round(iter->second.cupti_api_call_time / iter->second.op_count) << ","
                  << round(iter->second.op_host_cost_time / iter->second.op_count) << std::endl;
  }
}

void GPUProfiler::EventHandleProcess(CUpti_CallbackId cbid, const CUpti_CallbackData *cbdata,
                                     const std::string &typestring, uint64_t startTimestamp, uint64_t endTimestamp) {
  Event event;
  uint32_t device_id = -1;
  CuptiGetDeviceId(cbdata->context, &device_id);
  event.kernel_name = cbdata->symbolName ? GetKernelFunc(cbdata->symbolName) : cbdata->functionName;
  event.kernel_name = GetKernelFuncName(event.kernel_name);
  event.kernel_type = typestring;
  event.api_type = CUPTIApiType::kCallback;
  event.start_time_stamp = startTimestamp;
  event.end_time_stamp = endTimestamp;
  event.op_name = op_name_;
  event.device_id = device_id;
  event.correlation_id = cbdata->correlationId;
  event.thread_id = GetThreadID();
  event.context_id = cbdata->contextUid;
  event.stream_id = GetStreamID(cbdata->context, stream_);
  event.cb_id = cbid;
  op_name_map_[event.correlation_id] = event.op_name;
  AddEvent(std::move(event));
}

void CUPTIAPI ActivityAllocBuffer(uint8_t **buffer, size_t *size, size_t *maxNumRecords);

void CUPTIAPI ActivityProcessBuffer(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize);

void GPUProfiler::Init(const std::string &profiling_path, uint32_t device_id, const std::string &profiling_options) {
  MS_LOG(INFO) << "Initialize GPU Profiling";
  if (subscriber_ != nullptr) {
    StopCUPTI();
    MS_LOG(EXCEPTION)
      << "Repeated initialization, Please check whether you have created the Profiler object multiple times";
  }
  CHECK_CUPTI_RET_WITH_EXCEPT(CuptiSubscribe(&subscriber_, (CUpti_CallbackFunc)CUPTICallBackFunc, this),
                              "CuptiSubscribe");
  CHECK_CUPTI_RET_WITH_EXCEPT(CuptiEnableDomain(1, subscriber_, CUPTI_CB_DOMAIN_DRIVER_API), "CuptiEnableDomain");

  activities_enable_.emplace_back(CUPTI_ACTIVITY_KIND_MEMCPY);
  activities_enable_.emplace_back(CUPTI_ACTIVITY_KIND_MEMCPY2);
  activities_enable_.emplace_back(CUPTI_ACTIVITY_KIND_KERNEL);

  for (std::vector<CUpti_ActivityKind>::iterator it = activities_enable_.begin(); it != activities_enable_.end();
       ++it) {
    CHECK_CUPTI_RET_WITH_EXCEPT(CuptiActivityEnable(*it), "CuptiActivityEnable");
  }

  CHECK_CUPTI_RET_WITH_EXCEPT(CuptiActivityRegisterCallbacks(ActivityAllocBuffer, ActivityProcessBuffer),
                              "CuptiActivityRegisterCallbacks");

  base_time_.gpu_start_time = GetCUPTITimeStamp();
  base_time_.host_start_time = GetHostTimeStamp();
  base_time_.host_start_monotonic_raw_time = GetHostMonoTimeStamp();

  profile_data_path_ = profiling_path;
  MS_LOG(INFO) << "GPU start time(ns):" << base_time_.gpu_start_time
               << " Host start time(ns):" << base_time_.host_start_time << " profile data path: " << profile_data_path_;
  is_init_ = true;
}

void GPUProfiler::SetRunTimeData(const std::string &op_name, void *stream) {
  auto iter = op_info_map_.find(op_name);
  if (iter != op_info_map_.end()) {
    iter->second.op_count += 1;
  } else {
    OpInfo op_info;
    op_info.op_name = op_name;
    op_info.stream = stream;
    op_info.op_count = 1;
    op_info_map_[op_name] = op_info;
  }
  op_name_ = op_name;
  stream_ = stream;
}

void GPUProfiler::OpDataProducerBegin(const std::string op_name, void *stream) {
  if (sync_enable_flag_) {
    CHECK_CUDA_RET_WITH_ERROR(cudaEventCreate(&op_event_start_), "cudaEventCreate  op event start failed");
    CHECK_CUDA_RET_WITH_ERROR(cudaEventCreate(&op_event_stop_), "cudaEventCreate op event stop failed");
    CHECK_CUDA_RET_WITH_ERROR(cudaEventRecord(op_event_start_, (CUstream)stream_),
                              "cudaEventRecord op event start failed");
    op_host_time_start_ = GetHostTimeStamp();
    op_cupti_time_start_ = GetCUPTITimeStamp();
  } else {
    op_host_time_start_ = GetHostTimeStamp();
    op_cupti_time_start_ = GetCUPTITimeStamp();
  }
  SetRunTimeData(op_name, stream);

  if (MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
    RecordOneStepStartEndInfo(op_name);
  }
}

void GPUProfiler::SingleOpLaunchTimeProcess(float op_time_elapsed) {
  auto launch_end_time = GetTime();
  double launch_start_time = launch_end_time - op_time_elapsed / kTimeUnit / kTimeUnit;
  SetSingleOpLaunchTime(std::make_pair(launch_start_time, launch_end_time));
}

void GPUProfiler::OpDataProducerEnd() {
  float op_time_elapsed = 0;
  if (sync_enable_flag_) {
    CHECK_CUDA_RET_WITH_ERROR(cudaEventRecord(op_event_stop_, (CUstream)stream_),
                              "cudaEventRecord op event stop failed");
    CHECK_CUDA_RET_WITH_ERROR(cudaEventSynchronize(op_event_start_), "cudaEventSynchronize op event start failed");
    CHECK_CUDA_RET_WITH_ERROR(cudaEventSynchronize(op_event_stop_), "cudaEventSynchronize op event stop failed");
    CHECK_CUDA_RET_WITH_ERROR(cudaEventElapsedTime(&op_time_elapsed, op_event_start_, op_event_stop_),
                              "cudaEventElapsedTime failed");
    CHECK_CUDA_RET_WITH_ERROR(cudaEventDestroy(op_event_start_), "cudaEventDestroy  op event start failed");
    CHECK_CUDA_RET_WITH_ERROR(cudaEventDestroy(op_event_stop_), "cudaEventDestroy  op event stop failed");
    op_time_elapsed = op_time_elapsed * kTimeUnit;
    op_host_time_stop_ = GetHostTimeStamp();
    SingleOpLaunchTimeProcess(op_time_elapsed);
  } else {
    op_host_time_stop_ = GetHostTimeStamp();
    op_time_elapsed = (op_host_time_stop_ - op_host_time_start_) / kTimeUnit;
    SingleOpLaunchTimeProcess(op_time_elapsed);
  }
  MS_LOG(DEBUG) << "Host Time Elapsed(us)," << op_name_ << "," << op_time_elapsed;
  Profiler::SetRunTimeData(op_name_, op_time_elapsed);
  Profiler::SetRunTimeData(op_name_, op_cupti_time_start_, op_time_elapsed);
}

void GPUProfiler::StopCUPTI() {
  if (subscriber_ != nullptr) {
    CHECK_CUPTI_RET_WITH_ERROR(CuptiUnsubscribe(subscriber_), "CuptiUnsubscribe");
    CHECK_CUPTI_RET_WITH_ERROR(CuptiActivityFlushAll(0), "CuptiActivityFlushAll");
    for (std::vector<CUpti_ActivityKind>::iterator it = activities_enable_.begin(); it != activities_enable_.end();
         ++it) {
      CHECK_CUPTI_RET_WITH_ERROR(CuptiActivityDisable(*it), "CuptiActivityDisable");
    }
    subscriber_ = nullptr;
    CHECK_CUPTI_RET_WITH_ERROR(CuptiFinalize(), "CuptiFinalize");
  }
}

void GPUProfiler::Stop() {
  MS_LOG(INFO) << "Stop GPU Profiling";
  StopCUPTI();
  OpsParser();
  SaveProfileData();
  ClearInst();
}

void GPUProfiler::SaveExtraProfileData() {
  for (auto op : profiling_op_) {
    op.second->SaveProfilingData();
  }
  MS_LOG(INFO) << "Save extra profiling data end.";
}

void GPUProfiler::SaveProfileData() {
  if (profile_data_path_.empty()) {
    MS_LOG(WARNING) << "Profile data path is empty, skip save profile data.";
  } else {
    GpuDataSaver dataSaver(step_trace_op_name_, all_step_start_end_info_);
    dataSaver.ParseOpInfo(op_info_map_);
    dataSaver.ParseEvent(events_);
    dataSaver.WriteFile(profile_data_path_, base_time_);
    if (!all_kernel_info_.empty()) {
      dataSaver.WriteFrameWork(profile_data_path_, all_kernel_info_);
    }
    SaveExtraProfileData();
  }
}

void GPUProfiler::RecordFrameWorkInfo(const CNodePtr &kernel) {
  auto op_name = kernel->fullname_with_scope();
  auto begin_iter = op_name.rfind('/') + 1;
  auto end_iter = op_name.rfind('-');
  if (begin_iter != std::string::npos && end_iter != std::string::npos && begin_iter < end_iter) {
    cur_kernel_info_.op_type = op_name.substr(begin_iter, end_iter - begin_iter);
    cur_kernel_info_.op_name = op_name.substr(begin_iter, op_name.length() - begin_iter);
  }
  for (uint32_t i = 0; i < (uint32_t)kernel->inputs().size(); i++) {
    if (kernel->input(i)->Shape() != nullptr) {
      cur_kernel_input_info_.input_id = i;
      cur_kernel_input_info_.shape = kernel->input(i)->Shape()->ToString();
      cur_kernel_info_.cur_kernel_all_inputs_info.push_back(cur_kernel_input_info_);
    }
  }
  all_kernel_info_.push_back(cur_kernel_info_);
  cur_kernel_info_.cur_kernel_all_inputs_info.clear();
}

void GPUProfiler::ClearInst() {
  op_info_map_.clear();
  op_name_map_.clear();
  events_.clear();
  activities_enable_.clear();
  all_step_start_end_info_.clear();
  step_start_end_info_vector_.clear();
  all_kernel_info_.clear();
  is_init_ = false;
  enable_flag_ = false;
  sync_enable_flag_ = true;
  data_process_enable_ = false;
  init_flag_ = false;
  enable_flag_ = false;
  has_find_ = false;
  cupti_callback_events_count_ = 0l;
  cupti_callback_events_drop_count_ = 0l;
  cupti_activity_events_count_ = 0l;
  cupti_activity_events_drop_count_ = 0l;
}

void CUPTIAPI ActivityAllocBuffer(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
  auto gpu_profiler_inst = GPUProfiler::GetInstance();
  if (gpu_profiler_inst == nullptr) {
    MS_LOG(ERROR) << "GPU profiler instance is nullptr";
    return;
  }
  gpu_profiler_inst->AllocBuffer(buffer, size, maxNumRecords);
}

void CUPTIAPI ActivityProcessBuffer(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize) {
  PROFILER_ERROR_IF_NULLPTR(buffer);
  auto gpu_profiler_inst = GPUProfiler::GetInstance();
  if (gpu_profiler_inst == nullptr) {
    MS_LOG(ERROR) << "GPU profiler instance is nullptr";
    return;
  }
  gpu_profiler_inst->ProcessBuffer(ctx, streamId, buffer, size, validSize);
}

void ProcessActivityMemcpyRecord(Event *profilingData, CUpti_Activity *record,
                                 CUpti_ActivityMemcpy *cupti_activity_memcpy) {
  switch (cupti_activity_memcpy->copyKind) {
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
      profilingData->activity_type = ActivityType::kMemcpyH2D;
      profilingData->kernel_name = "MemcpyH2D";
      break;
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
      profilingData->activity_type = ActivityType::kMemcpyD2H;
      profilingData->kernel_name = "MemcpyD2H";
      break;
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
      profilingData->activity_type = ActivityType::kMemcpyH2A;
      profilingData->kernel_name = "MemcpyH2A";
      break;
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
      profilingData->activity_type = ActivityType::kMemcpyA2H;
      profilingData->kernel_name = "MemcpyA2H";
      break;
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
      profilingData->activity_type = ActivityType::kMemcpyA2D;
      profilingData->kernel_name = "MemcpyA2D";
      break;
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
      profilingData->activity_type = ActivityType::kMemcpyD2A;
      profilingData->kernel_name = "MemcpyD2A";
      break;
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
      profilingData->activity_type = ActivityType::kMemcpyD2D;
      profilingData->kernel_name = "MemcpyD2D";
      break;
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
      profilingData->activity_type = ActivityType::kMemcpyH2H;
      profilingData->kernel_name = "MemcpyH2H";
      break;
    case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP:
      profilingData->activity_type = ActivityType::kMemcpyP2P;
      profilingData->kernel_name = "MemcpyP2P";
      break;
    default:
      profilingData->activity_type = ActivityType::kMemcpyUnknown;
      profilingData->kernel_name = "MemcpyUnknown";
      break;
  }
}

void HandleActivityMemcpyRecord(Event *profilingData, CUpti_Activity *record) {
  CUpti_ActivityMemcpy *cupti_activity_memcpy = reinterpret_cast<CUpti_ActivityMemcpy *>(record);
  ProcessActivityMemcpyRecord(profilingData, record, cupti_activity_memcpy);

  profilingData->kernel_type = "cuMemcpy";
  profilingData->api_type = CUPTIApiType::kActivity;
  profilingData->start_time_stamp = cupti_activity_memcpy->start;
  profilingData->end_time_stamp = cupti_activity_memcpy->end;
  profilingData->device_id = cupti_activity_memcpy->deviceId;
  profilingData->context_id = cupti_activity_memcpy->contextId;
  profilingData->stream_id = cupti_activity_memcpy->streamId;
  profilingData->correlation_id = cupti_activity_memcpy->correlationId;
  profilingData->memcpy_info.bytes = cupti_activity_memcpy->bytes;
  profilingData->memcpy_info.src_kind = cupti_activity_memcpy->srcKind;
  profilingData->memcpy_info.dst_kind = cupti_activity_memcpy->dstKind;
}

void HandleActivityMemcpy2Record(Event *profilingData, CUpti_Activity *record) {
  CUpti_ActivityMemcpy2 *memcpyP2P = reinterpret_cast<CUpti_ActivityMemcpy2 *>(record);
  profilingData->activity_type = ActivityType::kMemcpyP2P;
  profilingData->kernel_name = "MemcpyP2P";
  profilingData->kernel_type = "cuMemcpy";
  profilingData->api_type = CUPTIApiType::kActivity;
  profilingData->start_time_stamp = memcpyP2P->start;
  profilingData->end_time_stamp = memcpyP2P->end;
  profilingData->device_id = memcpyP2P->deviceId;
  profilingData->context_id = memcpyP2P->contextId;
  profilingData->stream_id = memcpyP2P->streamId;
  profilingData->correlation_id = memcpyP2P->correlationId;
  profilingData->memcpy_info.bytes = memcpyP2P->bytes;
  profilingData->memcpy_info.src_kind = memcpyP2P->srcKind;
  profilingData->memcpy_info.dst_kind = memcpyP2P->dstKind;
}

void HandleActivityMemsetRecord(Event *profilingData, CUpti_Activity *record) {
  CUpti_ActivityMemset *cupti_activity_memset = reinterpret_cast<CUpti_ActivityMemset *>(record);
  profilingData->activity_type = ActivityType::kMemset;
  profilingData->kernel_name = "MemorySet";
  profilingData->api_type = CUPTIApiType::kActivity;
  profilingData->start_time_stamp = cupti_activity_memset->start;
  profilingData->end_time_stamp = cupti_activity_memset->end;
  profilingData->device_id = cupti_activity_memset->deviceId;
  profilingData->context_id = cupti_activity_memset->contextId;
  profilingData->stream_id = cupti_activity_memset->streamId;
  profilingData->correlation_id = cupti_activity_memset->correlationId;
  profilingData->memcpy_info.bytes = cupti_activity_memset->bytes;
}

void HandleActivityKernelRecord(Event *profilingData, CUpti_Activity *record) {
  CUpti_ActivityKernel4 *kernel = reinterpret_cast<CUpti_ActivityKernel4 *>(record);
  profilingData->activity_type = ActivityType::kKernel;
  profilingData->api_type = CUPTIApiType::kActivity;
  profilingData->kernel_name = GetKernelFunc(kernel->name);
  profilingData->kernel_name = GetKernelFuncName(profilingData->kernel_name);
  profilingData->kernel_type = "cuLaunchKernel";
  profilingData->start_time_stamp = kernel->start;
  profilingData->end_time_stamp = kernel->end;
  profilingData->device_id = kernel->deviceId;
  profilingData->context_id = kernel->contextId;
  profilingData->stream_id = kernel->streamId;
  profilingData->correlation_id = kernel->correlationId;
  profilingData->kernel_info.registers_per_thread = kernel->registersPerThread;
  profilingData->kernel_info.static_shared_memory = kernel->staticSharedMemory;
  profilingData->kernel_info.dynamic_shared_memory = kernel->dynamicSharedMemory;
  profilingData->kernel_info.block_x = kernel->blockX;
  profilingData->kernel_info.block_y = kernel->blockY;
  profilingData->kernel_info.block_z = kernel->blockZ;
  profilingData->kernel_info.grid_x = kernel->gridX;
  profilingData->kernel_info.grid_y = kernel->gridY;
  profilingData->kernel_info.grid_z = kernel->gridZ;
}

void GPUProfiler::HandleActivityRecord(CUpti_Activity *record) {
  PROFILER_ERROR_IF_NULLPTR(record);
  Event profilingData;
  profilingData.cb_id = 0;
  switch (record->kind) {
    case CUPTI_ACTIVITY_KIND_MEMCPY: {
      HandleActivityMemcpyRecord(&profilingData, record);
      break;
    }
    case CUPTI_ACTIVITY_KIND_MEMCPY2: {
      HandleActivityMemcpy2Record(&profilingData, record);
      break;
    }
    case CUPTI_ACTIVITY_KIND_MEMSET: {
      HandleActivityMemsetRecord(&profilingData, record);
      break;
    }
    case CUPTI_ACTIVITY_KIND_KERNEL:
    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
      HandleActivityKernelRecord(&profilingData, record);
      break;
    }
    default:
      MS_LOG(WARNING) << "Unknown activity type!";
      return;
  }

  AddEvent(std::move(profilingData));
}

void GPUProfiler::SetStepTraceOpName(ProfilingTraceInfo trace_op_name) { step_trace_op_name_ = trace_op_name; }

void GPUProfiler::RegisterProfilingOp(std::shared_ptr<ProfilingOp> node) {
  PROFILER_ERROR_IF_NULLPTR(node);
  if (profiling_op_.find(node->Name()) != profiling_op_.end()) {
    return;
  }
  node->Init();
  profiling_op_[node->Name()] = node;
}

void CUPTIAPI GPUProfiler::AllocBuffer(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
  PROFILER_ERROR_IF_NULLPTR(size);
  PROFILER_ERROR_IF_NULLPTR(maxNumRecords);
  int stat = posix_memalign(reinterpret_cast<void **>(buffer), ALIGN_SIZE, BUF_SIZE);
  if (stat) {
    MS_LOG(ERROR) << "Out of memory, activity buffer alloc failed.";
    return;
  }
  MS_LOG(DEBUG) << "Alloc activity buffer, buffer size: " << BUF_SIZE;
  *size = BUF_SIZE;
  *maxNumRecords = 0;
}

void CUPTIAPI GPUProfiler::ProcessBuffer(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size,
                                         size_t validSize) {
  if (!enable_flag_) {
    MS_LOG(DEBUG) << "Profiler is not enable, skip to process activity record.";
    free(buffer);
    return;
  }
  CUptiResult status;
  CUpti_Activity *record = NULL;

  MS_LOG(DEBUG) << "Process activity buffer, valid size:" << validSize << ",Stream ID:" << streamId;
  if (validSize > 0) {
    do {
      status = CuptiActivityGetNextRecord(buffer, validSize, &record);
      if (status == CUPTI_SUCCESS) {
        HandleActivityRecord(record);
      } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
        break;
      } else {
        CHECK_CUPTI_RET_WITH_ERROR(status, "CuptiActivityGetNextRecord");
      }
    } while (1);

    // report any records dropped from the queue
    size_t dropped;
    CHECK_CUPTI_RET_WITH_ERROR(CuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped),
                               "CuptiActivityGetNumDroppedRecords");
    if (dropped != 0) {
      MS_LOG(INFO) << "Dropped " << (unsigned int)dropped << " activity records\n";
    }
  }

  free(buffer);
}
}  // namespace gpu
}  // namespace profiler
}  // namespace mindspore
