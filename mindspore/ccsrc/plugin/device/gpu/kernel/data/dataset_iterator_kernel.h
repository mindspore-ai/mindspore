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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DATA_DATASET_ITERATOR_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DATA_DATASET_ITERATOR_GPU_KERNEL_H_

#include <memory>
#include <string>
#include <vector>
#include <map>
#include "plugin/device/gpu/kernel/data/dataset_profiling.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "include/backend/data_queue/blocking_queue.h"
namespace mindspore {
namespace kernel {
using mindspore::device::DataQueueItem;

class DatasetIteratorKernelMod : public DeprecatedNativeGpuKernelMod {
 public:
  DatasetIteratorKernelMod();
  ~DatasetIteratorKernelMod();

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;
  bool Init(const CNodePtr &kernel_node) override;
  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &others) override;

 protected:
  void InitSizeLists() override;
  void SyncOutputShape() override;

 private:
  bool ReadDevice(std::vector<DataQueueItem> *data);
  std::string queue_name_;
  bool is_opened_;
  bool profiling_enable_;
  std::shared_ptr<GetNextProfiling> profiling_op_;
  std::vector<TypeId> types_;
  std::vector<DataQueueItem> output_data_;
  bool dynamic_shape_;
};

MS_REG_GPU_KERNEL(GetNext, DatasetIteratorKernelMod)
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DATA_DATASET_ITERATOR_GPU_KERNEL_H_
