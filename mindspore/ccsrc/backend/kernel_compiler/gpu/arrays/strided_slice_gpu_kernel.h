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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_STRIDED_SLICE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_STRIDED_SLICE_GPU_KERNEL_H_

#include <vector>
#include <bitset>
#include <algorithm>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/arrays/strided_slice_gpu_common.h"
#include "backend/kernel_compiler/gpu/cuda_impl/slice_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class StridedSliceGpuKernel : public GpuKernel, public StridedSliceGpuCommon {
 public:
  StridedSliceGpuKernel() = default;
  ~StridedSliceGpuKernel() override = default;
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (null_output_) {
      return true;
    }

    T *input = GetDeviceAddress<T>(inputs, 0);
    T *output = GetDeviceAddress<T>(outputs, 0);
    StridedSlice(input_shape_, begin_, strides_, output_shape_, input, output,
                 reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    input_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    null_output_ = CHECK_NULL_INPUT(input_shape_);
    if (null_output_) {
      MS_LOG(WARNING) << "For 'StridedSliceGpuKernel', input is null";
      InitSizeLists();
      return true;
    }
    if (input_shape_.size() > MAX_DIMS) {
      MS_LOG(ERROR) << "StridedSlice support dims no more than " << MAX_DIMS << ", but the input shape is "
                    << input_shape_.size();
      return false;
    }

    CollectInfo(kernel_node);
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    size_t size = sizeof(T);
    for (size_t i = 0; i < input_shape_.size(); i++) {
      size *= input_shape_[i];
    }
    input_size_list_.push_back(size);

    size_t size1 = sizeof(T);
    for (size_t i = 0; i < output_shape_.size(); i++) {
      size1 *= output_shape_[i];
    }
    output_size_list_.push_back(size1);
  }
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_STRIDED_SLICE_GPU_KERNEL_H_
