/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "backend/kernel_compiler/gpu/data/dataset_iterator_kernel.h"

#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include "backend/kernel_compiler/gpu/data/dataset_utils.h"
#include "backend/kernel_compiler/common_utils.h"
#ifndef ENABLE_SECURITY
#include "profiler/device/gpu/gpu_profiling.h"
#endif
#include "runtime/device/gpu/gpu_buffer_mgr.h"
#include "runtime/device/gpu/gpu_common.h"
#ifdef ENABLE_DUMP_IR
#include "debug/rdr/running_data_recorder.h"
#endif

namespace mindspore {
namespace kernel {
using mindspore::device::GpuBufferMgr;
using mindspore::device::HandleMgr;

DatasetIteratorKernel::DatasetIteratorKernel()
    : handle_(HandleMgr::INVALID_HANDLE), total_bytes_(0), profiling_enable_(false), profiling_op_(nullptr) {}

DatasetIteratorKernel::~DatasetIteratorKernel() { GpuBufferMgr::GetInstance().Close(handle_); }

void DatasetIteratorKernel::ReleaseResource() {
  GpuBufferMgr::GetInstance().Close(handle_);
  handle_ = HandleMgr::INVALID_HANDLE;
}

const std::vector<size_t> &DatasetIteratorKernel::GetInputSizeList() const { return input_size_list_; }

const std::vector<size_t> &DatasetIteratorKernel::GetOutputSizeList() const { return output_size_list_; }

const std::vector<size_t> &DatasetIteratorKernel::GetWorkspaceSizeList() const { return workspace_size_list_; }

bool DatasetIteratorKernel::Init(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_node_ = kernel_node;
  queue_name_ = GetAttr<std::string>(kernel_node, "shared_name");
  std::vector<std::vector<int>> shapes;
  std::vector<TypePtr> types;
  GetShapeAndType(kernel_node, &shapes, &types);
  for (auto item : types) {
    MS_EXCEPTION_IF_NULL(item);
  }
  for (size_t i = 0; i < shapes.size(); i++) {
    int unit = UnitSizeInBytes(types[i]->type_id());
    int nums = ElementNums(shapes[i]);
    int bytes = unit * nums;
    output_size_list_.push_back(bytes);
    total_bytes_ += bytes;
  }

  handle_ = GpuBufferMgr::GetInstance().Open(0, queue_name_, output_size_list_);
  if (handle_ == HandleMgr::INVALID_HANDLE) {
    MS_LOG(EXCEPTION) << "Gpu Queue(" << queue_name_ << ") Open Failed";
  }

#ifndef ENABLE_SECURITY
  auto profiler_inst = profiler::gpu::GPUProfiler::GetInstance();
  MS_EXCEPTION_IF_NULL(profiler_inst);
  profiling_enable_ = profiler_inst->GetEnableFlag();
  if (profiling_enable_) {
    std::string path = profiler_inst->ProfileDataPath();
    profiling_op_ = std::make_shared<GetNextProfiling>(path);
    MS_EXCEPTION_IF_NULL(profiling_op_);
    profiler_inst->RegisterProfilingOp(profiling_op_);
  }
#endif
  return true;
}

void DatasetIteratorKernel::InitSizeLists() { return; }

bool DatasetIteratorKernel::ReadDevice(void **addr, size_t *len) {
  uint64_t start_time_stamp = 0;
  uint32_t queue_size = 0;

  int repeat = 0;
  while (true) {
    if (profiling_enable_) {
      start_time_stamp = profiling_op_->GetTimeStamp();
      queue_size = GpuBufferMgr::GetInstance().Size(handle_);
    }
    auto ret = GpuBufferMgr::GetInstance().Front(handle_, addr, len);
    if (ret == device::SUCCESS) {
      if (profiling_enable_) {
        uint64_t end_time_stamp = profiling_op_->GetTimeStamp();
        profiling_op_->RecordData(queue_size, start_time_stamp, end_time_stamp);
      }
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
        MS_LOG(EXCEPTION) << "Get data timeout";
      }
    }

    if (profiling_enable_) {
      uint64_t end_time_stamp = profiling_op_->GetTimeStamp();
      profiling_op_->RecordData(queue_size, start_time_stamp, end_time_stamp);
    }
    MS_LOG(ERROR) << "Get data failed, errcode " << ret;
    return false;
  }
  return true;
}

bool DatasetIteratorKernel::Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                   const std::vector<AddressPtr> &outputs, void *stream) {
  if (handle_ == HandleMgr::INVALID_HANDLE) {
    handle_ = GpuBufferMgr::GetInstance().Open(0, queue_name_, output_size_list_);
    if (handle_ == HandleMgr::INVALID_HANDLE) {
      MS_LOG(EXCEPTION) << "Gpu Queue(" << queue_name_ << ") Open Failed";
    }
  }

  void *addr = nullptr;
  size_t len = 0;
  if (!ReadDevice(&addr, &len)) {
    return false;
  }
  if (total_bytes_ != len) {
    MS_LOG(ERROR) << "Dataset front error. read: " << len << ", expect: " << total_bytes_ << ", ";
    return false;
  }

  for (size_t i = 0; i < output_size_list_.size(); i++) {
    void *output_addr = GetDeviceAddress<void>(outputs, i);
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(output_addr, addr, output_size_list_[i], cudaMemcpyDeviceToDevice,
                                               reinterpret_cast<cudaStream_t>(stream)),
                               "Cuda Memcpy Failed");
    addr = reinterpret_cast<unsigned char *>(addr) + output_size_list_[i];
  }

  CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_, cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)),
                             "cudaStreamSynchronize failed");
  (void)GpuBufferMgr::GetInstance().Pop(handle_);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
