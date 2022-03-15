/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_BATCHOSPACE_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_BATCHOSPACE_KERNEL_H_

#include <vector>
#include <string>
#include <memory>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/batchtospace_helper.h"

namespace mindspore {
namespace kernel {
template <typename T>
class BatchToSpaceGpuKernelMod : public NativeGpuKernelMod {
 public:
  BatchToSpaceGpuKernelMod() { ResetResource(); }
  ~BatchToSpaceGpuKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    std::vector<void *> input_addrs = ConvertPtrs(inputs);
    std::vector<void *> work_addrs = ConvertPtrs(workspace);
    std::vector<void *> output_addrs = ConvertPtrs(outputs);
    int flag = helper_ptr_->Process(input_addrs, output_addrs, work_addrs, stream_ptr);
    if (flag != 0) {
      return false;
    }
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);

    helper_ptr_ = std::make_unique<cukernel::BatchToSpaceHelperGpuKernel<T>>(kernel_name_);
    helper_ptr_->ResetResource();

    std::vector<std::vector<size_t>> input_shapes;
    std::vector<std::vector<size_t>> output_shapes;
    auto input_shape = AnfAlgo::GetInputDeviceShapeAdaptively(kernel_node, 0);
    auto output_shape = AnfAlgo::GetOutputDeviceShapeAdaptively(kernel_node, 0);
    input_shapes.emplace_back(input_shape);
    output_shapes.emplace_back(output_shape);
    attr_.block_size = GetAttr<int64_t>(kernel_node, "block_size");
    attr_.crops = GetAttr<std::vector<std::vector<int64_t>>>(kernel_node, "crops");
    attr_.input_shape = input_shape;
    int flag = helper_ptr_->CheckKernelParam(&attr_);
    if (flag != 0) {
      return false;
    }

    flag = helper_ptr_->CalMemSize(input_shapes, output_shapes);
    if (flag != 0) {
      return false;
    }
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_ = helper_ptr_->GetInputSizeList();
    output_size_list_ = helper_ptr_->GetOutputSizeList();
  }

 private:
  std::string kernel_name_;
  std::unique_ptr<cukernel::BatchToSpaceHelperGpuKernel<T>> helper_ptr_ = nullptr;
  cukernel::BatchToSpaceAttr attr_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_BATCHOSPACE_KERNEL_H_
