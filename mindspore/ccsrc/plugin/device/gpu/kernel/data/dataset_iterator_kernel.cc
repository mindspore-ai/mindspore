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
#include "include/common/utils/convert_utils.h"
#include "plugin/device/gpu/kernel/data/dataset_utils.h"
#include "kernel/common_utils.h"

#ifndef ENABLE_SECURITY
#include "profiler/device/gpu/gpu_profiling.h"
#endif
#include "plugin/device/gpu/hal/device/gpu_buffer_mgr.h"
#include "plugin/device/gpu/hal/device/gpu_common.h"
#ifdef ENABLE_DUMP_IR
#include "debug/rdr/running_data_recorder.h"
#endif

namespace mindspore {
namespace kernel {
using mindspore::device::GpuBufferMgr;

DatasetIteratorKernelMod::DatasetIteratorKernelMod()
    : handle_(GpuBufferMgr::INVALID_HANDLE), profiling_enable_(false), profiling_op_(nullptr) {}

DatasetIteratorKernelMod::~DatasetIteratorKernelMod() { GpuBufferMgr::GetInstance().Close(handle_); }

bool DatasetIteratorKernelMod::Init(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_node_ = kernel_node;
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  queue_name_ = GetAttr<std::string>(kernel_node, "shared_name");
  std::vector<std::vector<int>> shapes;
  std::vector<TypePtr> type_ptrs;
  GetShapeAndType(kernel_node, &shapes, &type_ptrs);
  for (auto item : type_ptrs) {
    MS_EXCEPTION_IF_NULL(item);
  }
  std::transform(type_ptrs.begin(), type_ptrs.end(), std::back_inserter(types_),
                 [](const TypePtr &value) { return value->type_id(); });

  for (size_t i = 0; i < shapes.size(); i++) {
    int unit = UnitSizeInBytes(type_ptrs[i]->type_id());
    int nums = ElementNums(shapes[i]);
    int bytes = unit * nums;
    output_size_list_.push_back(bytes);
  }

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

void DatasetIteratorKernelMod::InitSizeLists() { return; }

bool DatasetIteratorKernelMod::ReadDevice(std::vector<DataItemGpu> *data) {
  uint64_t start_time_stamp = 0;
  uint32_t queue_size = 0;
#ifndef ENABLE_SECURITY
  auto profiler_inst = profiler::gpu::GPUProfiler::GetInstance();
  MS_EXCEPTION_IF_NULL(profiler_inst);
#endif
  int repeat = 0;
  while (true) {
#ifndef ENABLE_SECURITY
    profiling_enable_ = profiler_inst->GetEnableFlag();
    if (profiling_enable_) {
      start_time_stamp = profiling_op_->GetTimeStamp();
      queue_size = GpuBufferMgr::GetInstance().Size(handle_);
    }
#endif
    auto ret = GpuBufferMgr::GetInstance().Front(handle_, data);
    if (ret == device::SUCCESS) {
#ifndef ENABLE_SECURITY
      if (profiling_enable_) {
        uint64_t end_time_stamp = profiling_op_->GetTimeStamp();
        profiling_op_->RecordData(queue_size, start_time_stamp, end_time_stamp);
      }
#endif
      break;
    }

    if (ret == device::TIMEOUT) {
      repeat++;
      if (repeat < 10) {
        MS_LOG(INFO) << "Waiting for data...(" << repeat << " / 10)";
        continue;
      } else {
#ifdef ENABLE_DUMP_IR
        mindspore::RDR::TriggerAll();
#endif
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', get data timeout";
      }
    }
#ifndef ENABLE_SECURITY
    if (profiling_enable_) {
      uint64_t end_time_stamp = profiling_op_->GetTimeStamp();
      profiling_op_->RecordData(queue_size, start_time_stamp, end_time_stamp);
    }
#endif
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', get data failed, errcode " << ret;
    return false;
  }
  return true;
}

bool DatasetIteratorKernelMod::Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                      const std::vector<AddressPtr> &outputs, void *stream) {
  if (handle_ == GpuBufferMgr::INVALID_HANDLE) {
    handle_ = GpuBufferMgr::GetInstance().Open(0, queue_name_, output_size_list_);
    if (handle_ == GpuBufferMgr::INVALID_HANDLE) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', gpu Queue(" << queue_name_ << ") Open Failed";
    }
  }

  if (!ReadDevice(&output_data_)) {
    return false;
  }

  for (size_t i = 0; i < output_data_.size(); i++) {
    void *output_addr = GetDeviceAddress<void>(outputs, i);
    auto device_addr = output_data_[i].device_addr_;
    auto data_len = output_data_[i].data_len_;
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(output_addr, device_addr, data_len, cudaMemcpyDeviceToDevice,
                                               reinterpret_cast<cudaStream_t>(stream)),
                               "Cuda Memcpy Failed");
  }

  CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_, cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)),
                             "cudaStreamSynchronize failed");
  (void)GpuBufferMgr::GetInstance().Pop(handle_);
  return true;
}

void DatasetIteratorKernelMod::UpdateOp() {
  std::vector<std::vector<size_t>> shapes;
  for (const auto &item : output_data_) {
    std::vector<size_t> shape;
    std::transform(item.shapes_.begin(), item.shapes_.end(), std::back_inserter(shape), LongToSize);
    shapes.push_back(shape);
  }
  common::AnfAlgo::SetOutputInferTypeAndShape(types_, shapes, kernel_node_.lock().get());
}
}  // namespace kernel
}  // namespace mindspore
