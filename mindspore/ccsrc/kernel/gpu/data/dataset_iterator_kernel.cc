/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/data/dataset_iterator_kernel.h"
#include <cuda_runtime_api.h>
#include <string>
#include <vector>
#include "device/gpu/gpu_buffer_mgr.h"
#include "device/gpu/gpu_common.h"
#include "kernel/gpu/data/dataset_utils.h"

namespace mindspore {
namespace kernel {
using mindspore::device::GpuBufferMgr;
using mindspore::device::HandleMgr;

DatasetIteratorKernel::DatasetIteratorKernel() : handle_(HandleMgr::INVALID_HANDLE), total_bytes_(0) {}

DatasetIteratorKernel::~DatasetIteratorKernel() { GpuBufferMgr::GetInstance().Close(handle_); }

const std::vector<size_t> &DatasetIteratorKernel::GetInputSizeList() const { return input_size_list_; }

const std::vector<size_t> &DatasetIteratorKernel::GetOutputSizeList() const { return output_size_list_; }

const std::vector<size_t> &DatasetIteratorKernel::GetWorkspaceSizeList() const { return workspace_size_list_; }

bool DatasetIteratorKernel::Init(const CNodePtr &kernel_node) {
  queue_name_ = GetAttr<std::string>(kernel_node, "shared_name");
  auto shapes = GetAttr<const std::vector<std::vector<int>>>(kernel_node, "shapes");
  auto types = GetAttr<const std::vector<TypePtr>>(kernel_node, "types");
  if (shapes.size() != types.size()) {
    MS_LOG(EXCEPTION) << "Invalid shapes: " << shapes << ", types: " << types;
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

  return true;
}

void DatasetIteratorKernel::InitSizeLists() { return; }

bool DatasetIteratorKernel::Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                   const std::vector<AddressPtr> &outputs, void *stream) {
  void *addr = nullptr;
  size_t len = 0;

  int repeat = 0;
  while (true) {
    auto ret = GpuBufferMgr::GetInstance().Front(handle_, &addr, &len);
    if (ret == device::SUCCESS) {
      break;
    }

    if (ret == device::TIMEOUT) {
      repeat++;
      if (repeat < 10) {
        MS_LOG(INFO) << "Waiting for data...(" << repeat << " / 10)";
        continue;
      } else {
        MS_LOG(ERROR) << "Get data timeout";
        return false;
      }
    }

    MS_LOG(ERROR) << "Get data failed, errcode " << ret;
    return false;
  }

  if (total_bytes_ != len) {
    MS_LOG(ERROR) << "Dataset front error. read: " << len << ", expect: " << total_bytes_ << ", ";
    return false;
  }

  for (size_t i = 0; i < output_size_list_.size(); i++) {
    void *output_addr = GetDeviceAddress<void>(outputs, i);
    CHECK_CUDA_RET_WITH_EXCEPT(cudaMemcpyAsync(output_addr, addr, output_size_list_[i], cudaMemcpyDeviceToDevice,
                                               reinterpret_cast<cudaStream_t>(stream)),
                               "Cuda Memcpy Failed");
    addr = reinterpret_cast<unsigned char *>(addr) + output_size_list_[i];
  }

  CHECK_CUDA_RET_WITH_EXCEPT(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)),
                             "cudaStreamSynchronize failed");
  (void)GpuBufferMgr::GetInstance().Pop(handle_);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
