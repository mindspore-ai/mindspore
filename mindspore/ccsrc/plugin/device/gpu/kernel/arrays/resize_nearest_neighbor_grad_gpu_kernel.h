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

#include <map>
#include <vector>
#include <algorithm>
#include "mindspore/core/utils/check_convert_utils.h"
#include "mindspore/core/ops/grad/resize_nearest_neighbor_grad.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/resize_nearest_neighbor_grad_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr size_t kNumOne = 1;
constexpr size_t kNumTwo = 2;
constexpr auto kInputNum = "inputs number";
template <typename T, typename S = int64_t>
class ResizeNearestNeighborGradGpuKernelMod : public NativeGpuKernelMod {
 public:
  ResizeNearestNeighborGradGpuKernelMod()
      : align_corners_(false),
        is_null_input_(false),
        shape_size_(0),
        input_size_(0),
        output_size_(0),
        workspace_size_(0),
        input_num_(0) {}
  ~ResizeNearestNeighborGradGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input = GetDeviceAddress<T>(inputs, 0);
    T *output = GetDeviceAddress<T>(outputs, 0);
    int input_size = SizeToInt(input_size_ / sizeof(T));
    float h_scale = Scaling(output_shape_[kIndex2], input_shape_[kIndex2], align_corners_);
    float w_scale = Scaling(output_shape_[kIndex3], input_shape_[kIndex3], align_corners_);
    CalResizeNearestNeighborGrad(input_size, input, input_shape_[kIndex0], input_shape_[kIndex1], input_shape_[kIndex2],
                                 input_shape_[kIndex3], output, output_shape_[kIndex0], output_shape_[kIndex1],
                                 output_shape_[kIndex2], output_shape_[kIndex3], align_corners_, h_scale, w_scale,
                                 reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override {
    MS_ERROR_IF_NULL(base_operator);
    kernel_name_ = base_operator->name();
    (void)CheckAndConvertUtils::CheckInteger(kInputNum, SizeToLong(inputs.size()), kLessEqual, kNumTwo, kernel_name_);
    (void)CheckAndConvertUtils::CheckInteger(kInputNum, SizeToLong(outputs.size()), kEqual, kNumOne, kernel_name_);
    auto prim = base_operator->GetPrim();
    MS_EXCEPTION_IF_NULL(prim);
    align_corners_ = GetValue<bool>(prim->GetAttr("align_corners"));
    return true;
  }

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override {
    auto ret = KernelMod::Resize(base_operator, inputs, outputs);
    if (ret != KRET_OK) {
      return ret;
    }
    auto input_shape = inputs[kIndex0]->GetShapeVector();
    shape_size_ = input_shape.size();
    auto output_shape = outputs[kIndex0]->GetShapeVector();
    if (shape_size_ != RESIZENEARESTNEIGHBORGRAD_DIMENSION) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of input must be "
                    << RESIZENEARESTNEIGHBORGRAD_DIMENSION << ", but got " << shape_size_;
      return KRET_RESIZE_FAILED;
    }
    if (shape_size_ != output_shape.size()) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', the dimension of input and output must be the same, but got the dimension of input: "
                    << shape_size_ << ", the dimension of output: " << output_shape.size();
      return KRET_RESIZE_FAILED;
    }

    input_shape_.clear();
    for (size_t i = 0; i < shape_size_; i++) {
      if (input_shape[i] == 0) {
        MS_LOG(ERROR) << "For '" << kernel_name_ << "', the shape of input at " << i << " index cannot be 0, "
                      << "but got " << input_shape[i];
        return KRET_RESIZE_FAILED;
      }
      input_shape_.push_back(LongToInt(input_shape[i]));
    }

    output_shape_.clear();
    input_size_ = sizeof(T) * SizeOf(input_shape);

    output_shape_.clear();
    for (size_t i = 0; i < shape_size_; i++) {
      if (output_shape[i] == 0) {
        MS_LOG(ERROR) << "For '" << kernel_name_ << "', the shape of output at " << i << " index cannot be 0, "
                      << "but got " << output_shape[i];
        return KRET_RESIZE_FAILED;
      }
      output_shape_.push_back(LongToInt(output_shape[i]));
    }
    output_size_ = sizeof(T) * SizeOf(output_shape);
    return KRET_OK;
  }

  void DestroyResource() noexcept override {
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
    input_shape_.clear();
    output_shape_.clear();
    align_corners_ = false;
    is_null_input_ = false;
    shape_size_ = 0;
    input_size_ = 0;
    output_size_ = 0;
    workspace_size_ = 0;
    input_num_ = 0;
  }

 private:
  float Scaling(const int in_size, const int out_size, bool align_corners) {
    return (align_corners && out_size > 1) ? (in_size - 1) / static_cast<float>(out_size - 1)
                                           : in_size / static_cast<float>(out_size);
  }

  bool align_corners_{false};
  bool is_null_input_{false};
  size_t shape_size_{0};
  std::vector<int> input_shape_;
  std::vector<int> output_shape_;
  size_t input_size_{0};
  size_t output_size_{0};
  size_t workspace_size_{0};
  size_t input_num_{0};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_RESIZE_NEAREST_NEIGHBOR_GRAD_GPU_KERNEL_H_
