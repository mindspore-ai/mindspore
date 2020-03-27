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

namespace mindspore {
namespace kernel {
using mindspore::device::GpuBufferMgr;
using mindspore::device::HandleMgr;

DatasetIteratorKernel::DatasetIteratorKernel()
    : output_num_(0), handle_(HandleMgr::INVALID_HANDLE), feature_size_(0), label_size_(0) {}

DatasetIteratorKernel::~DatasetIteratorKernel() { GpuBufferMgr::GetInstance().Close(handle_); }

const std::vector<size_t> &DatasetIteratorKernel::GetInputSizeList() const { return input_size_list_; }

const std::vector<size_t> &DatasetIteratorKernel::GetOutputSizeList() const { return output_size_list_; }

const std::vector<size_t> &DatasetIteratorKernel::GetWorkspaceSizeList() const { return workspace_size_list_; }

size_t DatasetIteratorKernel::TensorSize(std::vector<int> &shape) const {
  if (shape.size() == 0) {
    return 0;
  }

  int size = 1;
  for (size_t i = 0; i < shape.size(); i++) {
    size *= shape[i];
  }

  return IntToSize(size);
}

bool DatasetIteratorKernel::Init(const CNodePtr &kernel_node) {
  output_num_ = GetAttr<int>(kernel_node, "output_num");
  queue_name_ = GetAttr<std::string>(kernel_node, "shared_name");
  auto shapes = GetAttr<const std::vector<std::vector<int>>>(kernel_node, "shapes");
  auto data_num = shapes.size();
  if (data_num != 2) {
    MS_LOG(EXCEPTION) << "Invalid Shapes " << data_num;
  }

  auto &feature_Shapes = shapes[0];
  auto size = TensorSize(feature_Shapes);
  feature_size_ = size * sizeof(float);

  auto types = GetAttr<const std::vector<TypePtr>>(kernel_node, "types");
  if ((types[1]->type_id() != kNumberTypeInt32) && (types[1]->type_id() != kNumberTypeInt64)) {
    MS_LOG(EXCEPTION) << "Invalid types " << types[1]->type_id();
  }

  size_t label_unit = (types[1]->type_id() == kNumberTypeInt32) ? sizeof(int32_t) : sizeof(int64_t);
  size = TensorSize(shapes[1]);
  label_size_ = size * label_unit;

  InitSizeLists();

  handle_ = GpuBufferMgr::GetInstance().Open(0, queue_name_, feature_size_, label_size_);
  if (handle_ == HandleMgr::INVALID_HANDLE) {
    MS_LOG(EXCEPTION) << "Gpu Queue(" << queue_name_ << ") Open Failed: feature_size(" << feature_size_
                      << "), label_size(" << label_size_ << ")";
  }

  return true;
}

void DatasetIteratorKernel::InitSizeLists() {
  output_size_list_.push_back(feature_size_);
  output_size_list_.push_back(label_size_);
}

bool DatasetIteratorKernel::Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                   const std::vector<AddressPtr> &outputs, uintptr_t) {
  void *feature_addr{nullptr}, *label_addr{nullptr};
  size_t feature_size{0}, label_size{0};

  int repeat = 0;
  while (true) {
    auto ret = GpuBufferMgr::GetInstance().Front(handle_, &feature_addr, &feature_size, &label_addr, &label_size);
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

  if (feature_size != feature_size_ || label_size != label_size_) {
    MS_LOG(ERROR) << "DatasetIteratorKernel: Front Error: " << feature_addr << ", " << feature_size << ", "
                  << label_addr << ", " << label_size;
    return false;
  }

  CHECK_CUDA_RET_WITH_EXCEPT(cudaMemcpy(outputs[0]->addr, feature_addr, feature_size, cudaMemcpyDeviceToDevice),
                             "Cuda Memcpy Failed");
  CHECK_CUDA_RET_WITH_EXCEPT(cudaMemcpy(outputs[1]->addr, label_addr, label_size, cudaMemcpyDeviceToDevice),
                             "Cuda Memcpy Failed");

  (void)GpuBufferMgr::GetInstance().Pop(handle_);

  return true;
}
}  // namespace kernel
}  // namespace mindspore
