/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/gpu/kernel/data/dataset_iterator_kernel.h"
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include "include/common/utils/convert_utils.h"
#include "plugin/device/gpu/kernel/data/dataset_utils.h"
#include "kernel/common_utils.h"

#ifndef ENABLE_SECURITY
#include "plugin/device/gpu/hal/profiler/gpu_profiling.h"
#endif
#include "include/backend/data_queue/data_queue_mgr.h"
#include "plugin/device/gpu/hal/device/gpu_common.h"
#ifdef ENABLE_DUMP_IR
#include "include/common/debug/rdr/recorder_manager.h"
#endif

namespace mindspore {
namespace kernel {
using mindspore::device::DataQueueMgr;

DatasetIteratorKernelMod::DatasetIteratorKernelMod()
    : is_opened_(false), profiling_enable_(false), profiling_op_(nullptr) {}

DatasetIteratorKernelMod::~DatasetIteratorKernelMod() { DataQueueMgr::GetInstance().Close(queue_name_); }

bool DatasetIteratorKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  for (size_t i = 0; i < outputs.size(); ++i) {
    dynamic_shape_ |= IsDynamic(outputs[i]->GetShapeVector());
  }
  queue_name_ = GetValue<std::string>(primitive_->GetAttr("shared_name"));
  std::vector<std::vector<int>> shapes;
  std::vector<TypePtr> type_ptrs;
  GetShapeAndType(primitive_, &shapes, &type_ptrs);
  for (auto item : type_ptrs) {
    MS_EXCEPTION_IF_NULL(item);
  }
  std::transform(type_ptrs.begin(), type_ptrs.end(), std::back_inserter(types_),
                 [](const TypePtr &value) { return value->type_id(); });
  if (dynamic_shape_) {
    output_size_list_ = std::vector<size_t>(shapes.size(), 0);
  } else {
    for (size_t i = 0; i < shapes.size(); i++) {
      int unit = UnitSizeInBytes(type_ptrs[i]->type_id());
      int nums = ElementNums(shapes[i]);
      int bytes = unit * nums;
      output_size_list_.push_back(bytes);
    }
  }

  is_need_retrieve_output_shape_ = true;

#ifndef ENABLE_SECURITY
  auto profiler_inst = profiler::gpu::GPUProfiler::GetInstance();
  MS_EXCEPTION_IF_NULL(profiler_inst);
  if (profiler_inst->IsInitialized()) {
    std::string path = profiler_inst->ProfileDataPath();
    profiling_op_ = std::make_shared<GetNextProfiling>(path);
    MS_EXCEPTION_IF_NULL(profiling_op_);
    profiler_inst->RegisterProfilingOp(profiling_op_);
  }
#endif
  return true;
}

int DatasetIteratorKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  if (dynamic_shape_) {
    device::UpdateGetNextNode(primitive_, inputs, outputs, &output_size_list_);
  }
  return KRET_OK;
}

const uint32_t log_interval_step = 30;  // log info in each 30s when getnext timeout

bool DatasetIteratorKernelMod::ReadDevice(std::vector<DataQueueItem> *data) {
  uint64_t start_time_stamp = 0;
  uint32_t queue_size = 0;
#ifndef ENABLE_SECURITY
  auto profiler_inst = profiler::gpu::GPUProfiler::GetInstance();
  MS_EXCEPTION_IF_NULL(profiler_inst);
#endif
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  uint32_t op_timeout = ms_context->get_param<uint32_t>(MS_CTX_OP_TIMEOUT);
  uint32_t time_cost = 0;
  uint32_t log_interval = log_interval_step;
  while (true) {
#ifndef ENABLE_SECURITY
    profiling_enable_ = profiler_inst->GetDataProcessEnableFlag();
    if (profiling_enable_) {
      start_time_stamp = profiling_op_->GetTimeStamp();
      queue_size = DataQueueMgr::GetInstance().Size(queue_name_);
    }
#endif
    time_t start_time = time(nullptr);
    auto ret = DataQueueMgr::GetInstance().Front(queue_name_, data);
    if (ret == device::DataQueueStatus::SUCCESS) {
#ifndef ENABLE_SECURITY
      if (profiling_enable_) {
        uint64_t end_time_stamp = profiling_op_->GetTimeStamp();
        profiling_op_->RecordData(queue_size, start_time_stamp, end_time_stamp);
      }
#endif
      break;
    }
    if (ret == device::DataQueueStatus::ERROR_INPUT) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', get dynamic-shape data from queue " << queue_name_
                        << ", you need to call network.set_inputs() to "
                           "configure dynamic dims of input data before running the network";
    }
    if (ret == device::DataQueueStatus::TIMEOUT) {
      time_cost += (time(nullptr) - start_time);
      if (op_timeout == 0 || time_cost < op_timeout) {
        if (time_cost > log_interval) {
          MS_LOG(INFO) << "The op_timeout is: " << std::to_string(op_timeout) << ", continue waiting for data...";
          log_interval += log_interval_step;
        }
        continue;
      } else {
#ifdef ENABLE_DUMP_IR
        mindspore::RDR::TriggerAll();
#endif
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', get data timeout. Queue name: " << queue_name_;
      }
    }
#ifndef ENABLE_SECURITY
    if (profiling_enable_) {
      uint64_t end_time_stamp = profiling_op_->GetTimeStamp();
      profiling_op_->RecordData(queue_size, start_time_stamp, end_time_stamp);
    }
#endif
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', get data failed, errcode " << ret
                  << ", queue name: " << queue_name_;
    return false;
  }
  return true;
}

bool DatasetIteratorKernelMod::Launch(const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &,
                                      const std::vector<KernelTensor *> &outputs, void *stream) {
  if (!is_opened_) {
    auto ret = DataQueueMgr::GetInstance().Open(queue_name_);
    if (ret != device::DataQueueStatus::SUCCESS) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', gpu Queue(" << queue_name_ << ") Open Failed: " << ret;
    }
    is_opened_ = true;
  }

  if (!ReadDevice(&output_data_)) {
    return false;
  }

  for (size_t i = 0; i < output_data_.size(); i++) {
    void *output_addr = GetDeviceAddress<void>(outputs, i);
    auto device_addr = output_data_[i].device_addr;
    auto data_len = output_data_[i].data_len;
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(output_addr, device_addr, data_len, cudaMemcpyDeviceToDevice,
                                                       reinterpret_cast<cudaStream_t>(stream)),
                                       "Cuda Memcpy Failed");
  }
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)),
                                     "cudaStreamSynchronize failed");
  (void)DataQueueMgr::GetInstance().Pop(queue_name_);
  return true;
}

void DatasetIteratorKernelMod::UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                                        const std::vector<KernelTensor *> &outputs) {
  std::vector<ShapeVector> shapes;
  size_t idx = 0;
  for (const auto &item : output_data_) {
    ShapeVector shape;
    std::transform(item.shapes.begin(), item.shapes.end(), std::back_inserter(shape), LongToSize);
    shapes.push_back(shape);
    outputs[idx]->SetShapeVector(shape);
    outputs[idx]->set_size(output_size_list_[idx]);
    ++idx;
  }
}
}  // namespace kernel
}  // namespace mindspore
