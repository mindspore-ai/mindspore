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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_UNSORTED_SEGMENT_SUM_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_UNSORTED_SEGMENT_SUM_GPU_KERNEL_H_

#include <vector>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/unsorted_segment_sum.cuh"

namespace mindspore {
namespace kernel {
template <typename T, typename S>
class UnsortedSegmentSumGpuKernel : public GpuKernel {
 public:
  UnsortedSegmentSumGpuKernel() { ResetResource(); }
  ~UnsortedSegmentSumGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    S *indices_addr = GetDeviceAddress<S>(inputs, 1);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);

    CHECK_CUDA_RET_WITH_EXCEPT(
      kernel_node_, cudaMemsetAsync(output_addr, 0, outputs[0]->size, reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemSet Failed");
    UnsortedSegmentSum(input_dim0_, input_dim1_, output_dim0_, output_dim1_, input_addr, indices_addr, output_addr,
                       reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    auto input_shapes = AnfAlgo::GetInputRealDeviceShapeIfExist(kernel_node, 0);
    auto ids_shapes = AnfAlgo::GetInputRealDeviceShapeIfExist(kernel_node, 1);
    auto output_shapes = AnfAlgo::GetOutputRealDeviceShapeIfExist(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(input_shapes) || CHECK_NULL_INPUT(ids_shapes) || CHECK_NULL_INPUT(output_shapes);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'UnsortedSegmentSumGpuKernel', input or output is null";
      InitSizeLists();
      return true;
    }

    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num == 3) {
      MS_LOG(INFO) << "UnsortedSegmentSum Kernel Input count is 3 - dynamic mode";
    } else {
      MS_LOG(INFO) << "UnsortedSegmentSum Kernel Input count is 2";
    }

    auto axis = ids_shapes.size();
    for (size_t i = 0; i < input_shapes.size(); i++) {
      if (i < axis) {
        input_dim0_ *= input_shapes[i];
      } else {
        input_dim1_ *= input_shapes[i];
      }
    }

    output_dim0_ = output_shapes[0];
    for (size_t j = 1; j < output_shapes.size(); j++) {
      output_dim1_ *= output_shapes[j];
    }

    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    input_dim0_ = 1;
    input_dim1_ = 1;
    output_dim0_ = 1;
    output_dim1_ = 1;
    is_null_input_ = false;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_dim0_ * input_dim1_ * sizeof(T));
    input_size_list_.push_back(input_dim0_ * sizeof(S));
    output_size_list_.push_back(output_dim0_ * output_dim1_ * sizeof(T));
  }

 private:
  size_t input_dim0_;
  size_t input_dim1_;
  size_t output_dim0_;
  size_t output_dim1_;
  bool is_null_input_;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_UNSORTED_SEGMENT_SUM_GPU_KERNEL_H_
