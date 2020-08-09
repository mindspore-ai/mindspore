/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include <cxxabi.h>
#include <cmath>
#include <chrono>
#include "profiler/device/gpu/gpu_profiling.h"
#include "profiler/device/gpu/cupti_interface.h"
#include "utils/log_adapter.h"
#include "pybind_api/api_register.h"

namespace mindspore {
namespace profiler {
namespace gpu {
#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define CHECK_CUPTI_RET_WITH_ERROR(expression, message)                   \
  if (expression != CUPTI_SUCCESS) {                                      \
    const char *errstr;                                                   \
    CuptiGetResultString(expression, &errstr);                            \
    MS_LOG(ERROR) << "CUPTI Error:" << errstr << " function:" << message; \
  }

#define CHECK_CUPTI_RET_WITH_EXCEPT(expression, message)                      \
  if (expression != CUPTI_SUCCESS) {                                          \
    const char *errstr;                                                       \
    CuptiGetResultString(expression, &errstr);                                \
    MS_LOG(EXCEPTION) << "CUPTI Error:" << errstr << " function:" << message; \
  }
#define CHECK_CUDA_RET_WITH_ERROR(expression, message)                                   \
  {                                                                                      \
    cudaError_t status = (expression);                                                   \
    if (status != cudaSuccess) {                                                         \
      MS_LOG(ERROR) << "CUDA Error: " << message << " | Error Number: " << status << " " \
                    << cudaGetErrorString(status);                                       \
    }                                                                                    \
  }
#define PROFILER_ERROR_IF_NULLPTR(ptr)                           \
  do {                                                           \
    if ((ptr) == nullptr) {                                      \
      MS_LOG(ERROR) << ": The pointer[" << #ptr << "] is null."; \
      return;                                                    \
    }                                                            \
  } while (0)

std::shared_ptr<GPUProfiler> GPUProfiler::profiler_inst_ = nullptr;

int32_t GetThreadID() {
  int32_t thread_id = 0;
  thread_id = static_cast<int32_t>(pthread_self());
  return thread_id;
}

uint32_t GetStreamID(const CUcontext context, const void *stream) {
  uint32_t stream_id = 0;
  if (stream != nullptr) {
    CHECK_CUPTI_RET_WITH_ERROR(CuptiGetStreamId(context, (CUstream)stream, &stream_id), "CuptiGetStreamId");
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
  char *demangledName = abi::__cxa_demangle(name, nullptr, nullptr, nullptr);
  if (demangledName != nullptr) {
    return demangledName;
  } else {
    return name;
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
    MS_LOG(DEBUG) << "callback data context is null , correlation Id:" << cb_data->correlationId
                  << " callback id:" << cb_id;
    return;
  }

  uint64_t start_timestamp;
  uint64_t end_timestamp;

  if (cb_data->callbackSite == CUPTI_API_ENTER) {
    *cb_data->correlationData = GetCUPTITimeStamp();

  } else if (cb_data->callbackSite == CUPTI_API_EXIT) {
    start_timestamp = *cb_data->correlationData;
    end_timestamp = GetCUPTITimeStamp();

    switch (cb_id) {
      case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
      case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel:
      case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernelMultiDevice:
        gpu_profiler_inst->EventHandleProcess(cb_id, cb_data, "cuLaunchKernel", start_timestamp, end_timestamp);
        break;
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
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoHAsync_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DAsync_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DAsync_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoAAsync_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyPeer:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyPeerAsync:
        gpu_profiler_inst->EventHandleProcess(cb_id, cb_data, "cuMemcpy", start_timestamp, end_timestamp);
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
  }
}

std::shared_ptr<GPUProfiler> GPUProfiler::GetInstance() {
  if (profiler_inst_ == nullptr) {
    profiler_inst_ = std::shared_ptr<GPUProfiler>(new (std::nothrow) GPUProfiler());
  }
  return profiler_inst_;
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

void fillActivityInfo(OpInfo *opInfo, const Event &event) {
  if (event.api_type != CUPTIApiType::kActivity) {
    return;
  }
  switch (event.activity_type) {
    case ActivityType::kKernel:
      opInfo->kernel_info.registers_per_thread = event.kernel_info.registers_per_thread;
      opInfo->kernel_info.static_shared_memory = event.kernel_info.static_shared_memory;
      opInfo->kernel_info.dynamic_shared_memory = event.kernel_info.dynamic_shared_memory;
      opInfo->kernel_info.block_x = event.kernel_info.block_x;
      opInfo->kernel_info.block_y = event.kernel_info.block_y;
      opInfo->kernel_info.block_z = event.kernel_info.block_z;
      opInfo->kernel_info.grid_x = event.kernel_info.grid_x;
      opInfo->kernel_info.grid_y = event.kernel_info.grid_y;
      opInfo->kernel_info.grid_z = event.kernel_info.grid_z;
      break;
    case ActivityType::kMemcpyH2D:
    case ActivityType::kMemcpyD2H:
    case ActivityType::kMemcpyH2A:
    case ActivityType::kMemcpyA2H:
    case ActivityType::kMemcpyA2D:
    case ActivityType::kMemcpyD2A:
    case ActivityType::kMemcpyP2P:
    case ActivityType::kMemcpyH2H:
    case ActivityType::kMemset:
    case ActivityType::kMemcpyUnknown:
      opInfo->memcpy_info.bytes = event.memcpy_info.bytes;
    default:
      break;
  }
}

void GPUProfiler::OpsParser() {
  MS_LOG(INFO) << "Count the number of events size:" << events_.size()
               << " callback api:" << cupti_callback_events_count_ << " activity:" << cupti_activity_events_count_;

  if (cupti_activity_events_drop_count_ > 0 || cupti_callback_events_drop_count_ > 0) {
    MS_LOG(WARNING)
      << "The total number of events exceeded the profiler's processing capacity, Some events were discarded."
      << " callback api events:" << cupti_activity_events_drop_count_
      << " activity api events:" << cupti_callback_events_drop_count_;
  }

  if (events_.size() == 0) {
    return;
  }

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
          fillActivityInfo(&iter->second, event);
          break;
        }
        default:
          break;
      }
    }
  }

  MS_LOG(INFO) << "GPU_profiler, op_name, op_count , kernel_count, kernel_api_count,|"
                  ",cupti_activity_total_time, cupti_api_call_total_time, op_host_cost_total_time,|"
                  ",cupti_activity_average_time,cupti_api_call_average_time, op_host_cost_average_time,|"
                  ",mem_bytes,registers_per_thread,static_shared_memory,dynamic_shared_memory"
                  ",block_x,block_y,block_z,grid_x,grid_y,grid_z"
               << std::endl;

  std::vector<std::pair<std::string, OpInfo>> order_vec(op_info_map_.begin(), op_info_map_.end());

  auto cmp_func = [](const std::pair<std::string, OpInfo> &a, const std::pair<std::string, OpInfo> &b) {
    return a.second.cupti_activity_time > b.second.cupti_activity_time;
  };
  std::sort(order_vec.begin(), order_vec.end(), cmp_func);

  for (auto iter = order_vec.begin(); iter != order_vec.end(); iter++) {
    MS_LOG(INFO) << "GPU_profiler"
                 << "," << iter->first << "," << iter->second.op_count << "," << iter->second.op_kernel_count << ","
                 << iter->second.op_kernel_api_count << ","
                 << "|," << iter->second.cupti_activity_time << "," << iter->second.cupti_api_call_time << ","
                 << round(iter->second.op_host_cost_time) << ","
                 << "|," << round(iter->second.cupti_activity_time / iter->second.op_count) << ","
                 << round(iter->second.cupti_api_call_time / iter->second.op_count) << ","
                 << round(iter->second.op_host_cost_time / iter->second.op_count) << ","
                 << "|," << iter->second.memcpy_info.bytes << "," << iter->second.kernel_info.registers_per_thread
                 << "," << iter->second.kernel_info.static_shared_memory << ","
                 << iter->second.kernel_info.dynamic_shared_memory << "," << iter->second.kernel_info.block_x << ","
                 << iter->second.kernel_info.block_y << "," << iter->second.kernel_info.block_z << ","
                 << iter->second.kernel_info.grid_x << "," << iter->second.kernel_info.grid_y << ","
                 << iter->second.kernel_info.grid_z << std::endl;
  }
}

void GPUProfiler::EventHandleProcess(CUpti_CallbackId cbid, const CUpti_CallbackData *cbdata,
                                     const std::string &typestring, uint64_t startTimestamp, uint64_t endTimestamp) {
  Event event;
  uint32_t device_id = -1;
  CuptiGetDeviceId(cbdata->context, &device_id);
  event.kernel_name = cbdata->symbolName ? GetKernelFunc(cbdata->symbolName) : cbdata->functionName;
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

void GPUProfiler::Init(const std::string &profileDataPath = "") {
  MS_LOG(INFO) << "Initialize GPU Profiling";
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

  profile_data_path_ = profileDataPath;
  MS_LOG(INFO) << "GPU start time(ns):" << base_time_.gpu_start_time
               << " Host start time(ns):" << base_time_.host_start_time << " profile data path: " << profile_data_path_;
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

void GPUProfiler::SetRunTimeData(const std::string &op_name, const float time_elapsed) {
  auto iter = op_info_map_.find(op_name);
  if (iter != op_info_map_.end()) {
    // The time unit is ms ,convert to us
    iter->second.op_host_cost_time += time_elapsed;
  }
}

void GPUProfiler::OpDataProducerBegin(const std::string op_name, void *stream) {
  if (sync_enable_flag_) {
    CHECK_CUDA_RET_WITH_ERROR(cudaEventCreate(&op_event_start_), "cudaEventCreate  op event start failed");
    CHECK_CUDA_RET_WITH_ERROR(cudaEventCreate(&op_event_stop_), "cudaEventCreate op event stop failed");
    CHECK_CUDA_RET_WITH_ERROR(cudaEventRecord(op_event_start_, (CUstream)stream_),
                              "cudaEventRecord op event start failed");
  } else {
    op_host_time_start_ = GetHostTimeStamp();
  }
  SetRunTimeData(op_name, stream);
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
  } else {
    op_host_time_stop_ = GetHostTimeStamp();
    op_time_elapsed = (op_host_time_stop_ - op_host_time_start_) / kTimeUnit;
  }
  SetRunTimeData(op_name_, op_time_elapsed);
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
  }
}

void GPUProfiler::Stop() {
  MS_LOG(INFO) << "Stop GPU Profiling";
  StopCUPTI();
  OpsParser();
  SaveProfileData();
}

void GPUProfiler::SaveProfileData() {
  if (profile_data_path_.empty()) {
    MS_LOG(WARNING) << "profile_data_path is empty, skip save profile data.";
    return;
  }
  op_info_map_.clear();
  op_name_map_.clear();
  events_.clear();
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
  GPUProfiler::GetInstance()->ProcessBuffer(ctx, streamId, buffer, size, validSize);
}

void HandleActivityMemcpyRecord(Event *profillingData, CUpti_Activity *record) {
  CUpti_ActivityMemcpy *memcpy = reinterpret_cast<CUpti_ActivityMemcpy *>(record);
  switch (memcpy->copyKind) {
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
      profillingData->activity_type = ActivityType::kMemcpyH2D;
      profillingData->kernel_name = "MemcpyH2D";
      break;
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
      profillingData->activity_type = ActivityType::kMemcpyD2H;
      profillingData->kernel_name = "MemcpyD2H";
      break;
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
      profillingData->activity_type = ActivityType::kMemcpyH2A;
      profillingData->kernel_name = "MemcpyH2A";
      break;
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
      profillingData->activity_type = ActivityType::kMemcpyA2H;
      profillingData->kernel_name = "MemcpyA2H";
      break;
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
      profillingData->activity_type = ActivityType::kMemcpyA2D;
      profillingData->kernel_name = "MemcpyA2D";
      break;
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
      profillingData->activity_type = ActivityType::kMemcpyD2A;
      profillingData->kernel_name = "MemcpyD2A";
      break;
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
      profillingData->activity_type = ActivityType::kMemcpyD2D;
      profillingData->kernel_name = "MemcpyD2D";
      break;
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
      profillingData->activity_type = ActivityType::kMemcpyH2H;
      profillingData->kernel_name = "MemcpyH2H";
      break;
    case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP:
      profillingData->activity_type = ActivityType::kMemcpyP2P;
      profillingData->kernel_name = "MemcpyP2P";
      break;
    default:
      profillingData->activity_type = ActivityType::kMemcpyUnknown;
      profillingData->kernel_name = "MemcpyUnknown";
      break;
  }
  profillingData->kernel_type = "cuMemcpy";
  profillingData->api_type = CUPTIApiType::kActivity;
  profillingData->start_time_stamp = memcpy->start;
  profillingData->end_time_stamp = memcpy->end;
  profillingData->device_id = memcpy->deviceId;
  profillingData->context_id = memcpy->contextId;
  profillingData->stream_id = memcpy->streamId;
  profillingData->correlation_id = memcpy->correlationId;
  profillingData->memcpy_info.bytes = memcpy->bytes;
  profillingData->memcpy_info.src_kind = memcpy->srcKind;
  profillingData->memcpy_info.dst_kind = memcpy->dstKind;
}

void HandleActivityMemcpy2Record(Event *profillingData, CUpti_Activity *record) {
  CUpti_ActivityMemcpy2 *memcpyP2P = reinterpret_cast<CUpti_ActivityMemcpy2 *>(record);
  profillingData->activity_type = ActivityType::kMemcpyP2P;
  profillingData->kernel_name = "MemcpyP2P";
  profillingData->kernel_type = "cuMemcpy";
  profillingData->api_type = CUPTIApiType::kActivity;
  profillingData->start_time_stamp = memcpyP2P->start;
  profillingData->end_time_stamp = memcpyP2P->end;
  profillingData->device_id = memcpyP2P->deviceId;
  profillingData->context_id = memcpyP2P->contextId;
  profillingData->stream_id = memcpyP2P->streamId;
  profillingData->correlation_id = memcpyP2P->correlationId;
  profillingData->memcpy_info.bytes = memcpyP2P->bytes;
  profillingData->memcpy_info.src_kind = memcpyP2P->srcKind;
  profillingData->memcpy_info.dst_kind = memcpyP2P->dstKind;
}

void HandleActivityMemsetRecord(Event *profillingData, CUpti_Activity *record) {
  CUpti_ActivityMemset *memset = reinterpret_cast<CUpti_ActivityMemset *>(record);
  profillingData->activity_type = ActivityType::kMemset;
  profillingData->kernel_name = "MemorySet";
  profillingData->api_type = CUPTIApiType::kActivity;
  profillingData->start_time_stamp = memset->start;
  profillingData->end_time_stamp = memset->end;
  profillingData->device_id = memset->deviceId;
  profillingData->context_id = memset->contextId;
  profillingData->stream_id = memset->streamId;
  profillingData->correlation_id = memset->correlationId;
  profillingData->memcpy_info.bytes = memset->bytes;
}

void HandleActivityKernelRecord(Event *profillingData, CUpti_Activity *record) {
  CUpti_ActivityKernel4 *kernel = reinterpret_cast<CUpti_ActivityKernel4 *>(record);
  profillingData->activity_type = ActivityType::kKernel;
  profillingData->api_type = CUPTIApiType::kActivity;
  profillingData->kernel_name = GetKernelFunc(kernel->name);
  profillingData->kernel_type = "cuLaunchKernel";
  profillingData->start_time_stamp = kernel->start;
  profillingData->end_time_stamp = kernel->end;
  profillingData->device_id = kernel->deviceId;
  profillingData->context_id = kernel->contextId;
  profillingData->stream_id = kernel->streamId;
  profillingData->correlation_id = kernel->correlationId;
  profillingData->kernel_info.registers_per_thread = kernel->registersPerThread;
  profillingData->kernel_info.static_shared_memory = kernel->staticSharedMemory;
  profillingData->kernel_info.dynamic_shared_memory = kernel->dynamicSharedMemory;
  profillingData->kernel_info.block_x = kernel->blockX;
  profillingData->kernel_info.block_y = kernel->blockY;
  profillingData->kernel_info.block_z = kernel->blockZ;
  profillingData->kernel_info.grid_x = kernel->gridX;
  profillingData->kernel_info.grid_y = kernel->gridY;
  profillingData->kernel_info.grid_z = kernel->gridZ;
}

void GPUProfiler::HandleActivityRecord(CUpti_Activity *record) {
  PROFILER_ERROR_IF_NULLPTR(record);
  Event profillingData;
  profillingData.cb_id = 0;
  switch (record->kind) {
    case CUPTI_ACTIVITY_KIND_MEMCPY: {
      HandleActivityMemcpyRecord(&profillingData, record);
      break;
    }
    case CUPTI_ACTIVITY_KIND_MEMCPY2: {
      HandleActivityMemcpy2Record(&profillingData, record);
      break;
    }
    case CUPTI_ACTIVITY_KIND_MEMSET: {
      HandleActivityMemsetRecord(&profillingData, record);
      break;
    }
    case CUPTI_ACTIVITY_KIND_KERNEL:
    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
      HandleActivityKernelRecord(&profillingData, record);
      break;
    }
    default:
      MS_LOG(WARNING) << "unknown activity type!";
      return;
  }

  AddEvent(std::move(profillingData));
}

void CUPTIAPI GPUProfiler::AllocBuffer(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
  int stat = posix_memalign(reinterpret_cast<void **>(buffer), ALIGN_SIZE, BUF_SIZE);
  if (stat) {
    MS_LOG(ERROR) << "Out of memory, activity buffer alloc failed.";
    return;
  }

  *size = BUF_SIZE;
  *maxNumRecords = 0;
}

void CUPTIAPI GPUProfiler::ProcessBuffer(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size,
                                         size_t validSize) {
  if (!enable_flag_) {
    free(buffer);
    return;
  }
  CUptiResult status;
  CUpti_Activity *record = NULL;

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

REGISTER_PYBIND_DEFINE(GPUProfiler_, ([](const py::module *m) {
                         (void)py::class_<GPUProfiler, std::shared_ptr<GPUProfiler>>(*m, "GPUProfiler")
                           .def_static("get_instance", &GPUProfiler::GetInstance, "GPUProfiler get_instance.")
                           .def("init", &GPUProfiler::Init, py::arg("profile_data_path"), "init")
                           .def("stop", &GPUProfiler::Stop, "stop")
                           .def("step_profiling_enable", &GPUProfiler::StepProfilingEnable, py::arg("enable_flag"),
                                "enable or disable step profiling")
                           .def("sync_enable", &GPUProfiler::SyncEnable, py::arg("enable_flag"),
                                "enable or disable synchronization profiling");
                       }));

}  // namespace gpu
}  // namespace profiler
}  // namespace mindspore
