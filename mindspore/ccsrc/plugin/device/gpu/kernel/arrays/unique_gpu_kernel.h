/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_UNIQUE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_UNIQUE_GPU_KERNEL_H_

#include <vector>
#include <memory>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/unique_helper.h"
namespace mindspore {
namespace kernel {
template <typename T, typename S>
class UniqueGpuKernelMod : public NativeGpuKernelMod {
 public:
  UniqueGpuKernelMod() { ResetResource(); }
  ~UniqueGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    stream_ptr_ = stream_ptr;
    std::vector<void *> input_ptrs = ConvertPtrs(inputs);
    std::vector<void *> work_ptrs = ConvertPtrs(workspace);
    std::vector<void *> output_ptrs = ConvertPtrs(outputs);
    if (helper_ptr_->Process(input_ptrs, output_ptrs, work_ptrs, stream_ptr) != 0) {
      return false;
    }
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    helper_ptr_ = std::make_unique<cukernel::UniqueHelperGpuKernel<T, S>>(kernel_name);
    helper_ptr_->ResetResource();
    std::vector<std::vector<size_t>> input_shapes;
    std::vector<std::vector<size_t>> output_shapes;
    std::vector<size_t> shape = AnfAlgo::GetInputDeviceShapeAdaptively(kernel_node, 0);
    is_null_input_ = CHECK_SHAPE_NULL(shape, kernel_name, "input");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    input_shapes.emplace_back(shape);
    helper_ptr_->CalMemSize(input_shapes, output_shapes);
    InitSizeLists();
    return true;
  }

  void UpdateOp() override {
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_, cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_ptr_)),
                               "cudaStreamSynchronized failed");
    std::vector<TypeId> type_ids;
    std::vector<std::vector<size_t>> shapes;
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node_.lock());
    for (size_t i = 0; i < output_num; ++i) {
      std::vector<size_t> shape = common::AnfAlgo::GetOutputInferShape(kernel_node_.lock(), i);
      if (i == 0) {
        shape[0] = helper_ptr_->GetOutSize();
      }
      TypeId type_id = common::AnfAlgo::GetOutputInferDataType(kernel_node_.lock(), i);
      type_ids.emplace_back(type_id);
      shapes.emplace_back(shape);
    }
    common::AnfAlgo::SetOutputInferTypeAndShape(type_ids, shapes, kernel_node_.lock().get());
  }

  void ResetResource() noexcept override {
    is_null_input_ = false;
    stream_ptr_ = nullptr;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    input_size_list_ = helper_ptr_->GetInputSizeList();
    output_size_list_ = helper_ptr_->GetOutputSizeList();
    workspace_size_list_ = helper_ptr_->GetWorkSizeList();
  }

 private:
  void *stream_ptr_;
  bool is_null_input_;
  std::unique_ptr<cukernel::UniqueHelperGpuKernel<T, S>> helper_ptr_ = nullptr;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_UNIQUE_GPU_KERNEL_H_
