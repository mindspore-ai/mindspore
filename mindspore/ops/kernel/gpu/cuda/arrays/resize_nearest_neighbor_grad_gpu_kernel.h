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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_RESIZE_NEAREST_NEIGHBOR_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_RESIZE_NEAREST_NEIGHBOR_GRAD_GPU_KERNEL_H_

#include <algorithm>
#include <map>
#include <vector>
#include "utils/check_convert_utils.h"
#include "kernel/gpu/cuda_impl/cuda_ops/resize_nearest_neighbor_grad_impl.cuh"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T, typename S = int64_t>
class ResizeNearestNeighborGradGpuKernelMod : public NativeGpuKernelMod {
 public:
  ResizeNearestNeighborGradGpuKernelMod() {}
  ~ResizeNearestNeighborGradGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    T *input = GetDeviceAddress<T>(inputs, 0);
    MS_EXCEPTION_IF_NULL(input);
    T *output = GetDeviceAddress<T>(outputs, 0);
    MS_EXCEPTION_IF_NULL(output);
    float *work = GetDeviceAddress<float>(workspace, kIndex0);
    if (is_fp16) {
      MS_EXCEPTION_IF_NULL(work);
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
        cudaMemsetAsync(work, 0, workspace[kIndex0]->size(), reinterpret_cast<cudaStream_t>(stream_ptr)),
        "For ResizeNearestNeighborGrad, wrok cudaMemsetAsync failed.");
    } else {
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
        cudaMemsetAsync(output, 0, outputs[kIndex0]->size(), reinterpret_cast<cudaStream_t>(stream_ptr)),
        "For ResizeNearestNeighborGrad, output cudaMemsetAsync failed.");
    }
    float h_scale = Scaling(output_shape_[kIndex2], input_shape_[kIndex2], align_corners_);
    float w_scale = Scaling(output_shape_[kIndex3], input_shape_[kIndex3], align_corners_);
    auto status = CalResizeNearestNeighborGrad<T>(
      work, input_size_, input, input_shape_[kIndex0], input_shape_[kIndex1], input_shape_[kIndex2],
      input_shape_[kIndex3], output, output_shape_[kIndex0], output_shape_[kIndex1], output_shape_[kIndex2],
      output_shape_[kIndex3], align_corners_, h_scale, w_scale, reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
    return true;
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    auto out = outputs.at(kIndex0);
    MS_EXCEPTION_IF_NULL(out);
    auto o_type = out->dtype_id();
    is_fp16 = (o_type == kNumberTypeFloat16);
    return true;
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    auto ret = KernelMod::Resize(inputs, outputs);
    if (ret != KRET_OK) {
      return ret;
    }

    auto input_shape = inputs.at(kIndex0)->GetShapeVector();
    shape_size_ = input_shape.size();
    input_size_ = SizeToInt(SizeOf(input_shape));

    input_shape_.clear();
    for (size_t i = 0; i < shape_size_; i++) {
      if (input_shape[i] == 0) {
        MS_LOG(ERROR) << "For '" << kernel_name_ << "', the shape of input at " << i << " index cannot be 0, "
                      << "but got " << input_shape[i];
        return KRET_RESIZE_FAILED;
      }
      input_shape_.push_back(LongToInt(input_shape[i]));
    }

    auto output_shape = outputs[kIndex0]->GetShapeVector();
    output_shape_.clear();
    for (size_t i = 0; i < shape_size_; i++) {
      if (output_shape[i] == 0) {
        MS_LOG(ERROR) << "For '" << kernel_name_ << "', the shape of output at " << i << " index cannot be 0, "
                      << "but got " << output_shape[i];
        return KRET_RESIZE_FAILED;
      }
      output_shape_.push_back(LongToInt(output_shape[i]));
    }

    work_size_ = 0;
    if (is_fp16) {
      auto o_num = SizeOf(output_shape);
      work_size_ = o_num * sizeof(float);
    }
    workspace_size_list_.push_back(work_size_);
    align_corners_ = inputs.at(kIndex2)->GetValueWithCheck<bool>();
    return KRET_OK;
  }

 private:
  float Scaling(const int in_size, const int out_size, bool align_corners) {
    return (align_corners && out_size > 1) ? (in_size - 1) / static_cast<float>(out_size - 1)
                                           : in_size / static_cast<float>(out_size);
  }

  bool is_fp16{false};
  bool align_corners_{false};
  std::vector<int> input_shape_;
  std::vector<int> output_shape_;
  int input_size_{0};
  size_t input_num_{0};
  size_t shape_size_{0};
  size_t work_size_{0};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_RESIZE_NEAREST_NEIGHBOR_GRAD_GPU_KERNEL_H_
