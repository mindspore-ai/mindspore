/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_GRID_SAMPLER_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_GRID_SAMPLER_GPU_KERNEL_H_
#include <cmath>
#include <map>
#include <set>
#include <algorithm>
#include <vector>
#include <string>
#include <random>
#include <limits>
#include <functional>
#include "mindspore/core/ops/grid_sampler_2d.h"
#include "mindspore/core/ops/grid_sampler_3d.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/grid_sampler_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class GridSampler2DGpuKernelMod : public NativeGpuKernelMod {
 public:
  GridSampler2DGpuKernelMod() { ResetResource(); }
  ~GridSampler2DGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspaces,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input_addr = GetDeviceAddress<T>(inputs, kIndex0);
    T *grid_addr = GetDeviceAddress<T>(inputs, kIndex1);
    T *output_addr = GetDeviceAddress<T>(outputs, kIndex0);
    GridSampler2D(size_, input_addr, grid_addr, output_addr, input_shape_, grid_shape_, output_shape_, input_stride_,
                  grid_stride_, output_stride_, interpolation_mode_, padding_mode_, align_corners_,
                  reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override {
    auto kernel_ptr = std::dynamic_pointer_cast<ops::GridSampler2D>(base_operator);
    if (kernel_ptr == nullptr) {
      MS_EXCEPTION(ValueError) << "For primitive[GridSampler2D], cast op from BaseOperator to GridSampler2D failed.";
      return false;
    }
    kernel_name_ = kernel_ptr->name();
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), kGridSamplerInputNum, kernel_name_);
    CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kGridSamplerOutputNum, kernel_name_);
    interpolation_mode_ = kGridSamplerInterpolationMap[kernel_ptr->get_interpolation_mode()];
    padding_mode_ = kGridSamplerPaddingMap[kernel_ptr->get_padding_mode()];
    align_corners_ = kernel_ptr->get_align_corners();
    return true;
  }

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) {
    int ret = KernelMod::Resize(base_operator, inputs, outputs);
    if (ret != 0) {
      return ret;
    }
    if (input_size_list_.size() != kGridSamplerInputNum) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "' input size must be equal " << kGridSamplerInputNum << ".";
      return KRET_RESIZE_FAILED;
    }
    auto convert_int64_shape_to_sizet_shape = [=](std::vector<int64_t> int64_shape) -> std::vector<size_t> {
      std::vector<size_t> size_t_shape;
      (void)std::transform(int64_shape.begin(), int64_shape.end(), std::back_inserter(size_t_shape), LongToSize);
      return size_t_shape;
    };
    input_shape_ = convert_int64_shape_to_sizet_shape(inputs[kIndex0]->GetShapeVector());
    grid_shape_ = convert_int64_shape_to_sizet_shape(inputs[kIndex1]->GetShapeVector());
    output_shape_ = convert_int64_shape_to_sizet_shape(outputs[kIndex0]->GetShapeVector());

    if (input_shape_.empty()) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'input' must be at 4-D, but got scalar or None.";
      return KRET_RESIZE_FAILED;
    }

    if (grid_shape_.empty()) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'grid' must be at 4-D, but got scalar or None.";
      return KRET_RESIZE_FAILED;
    }

    if (output_shape_.empty()) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'output' must be at 4-D, but got scalar or None.";
      return KRET_RESIZE_FAILED;
    }

    size_t stride_tmp = 1;
    auto stride_compute = [&](std::vector<size_t> &stride, std::vector<size_t> shape) {
      for (int i = 3; i > -static_cast<int>(1); i--) {
        (void)stride.insert(stride.begin(), stride_tmp);
        stride_tmp *= shape[static_cast<size_t>(i)];
      }
      stride_tmp = 1;
    };
    stride_compute(input_stride_, input_shape_);
    stride_compute(grid_stride_, grid_shape_);
    stride_compute(output_stride_, output_shape_);
    size_ = input_shape_[kIndex0] * grid_shape_[kIndex1] * grid_shape_[kIndex2] * grid_shape_[kIndex3];
    return KRET_OK;
  }

  void ResetResource() noexcept {
    input_shape_.clear();
    grid_shape_.clear();
    output_shape_.clear();
    input_stride_.clear();
    grid_stride_.clear();
    output_stride_.clear();
    size_ = 0;
    interpolation_mode_ = GridSamplerInterpolationMode::BILINEAR;
    padding_mode_ = GridSamplerPaddingMode::ZEROS;
    align_corners_ = false;
    is_null_input_ = false;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 private:
  size_t size_;
  std::vector<size_t> input_shape_;
  std::vector<size_t> grid_shape_;
  std::vector<size_t> output_shape_;
  std::vector<size_t> input_stride_;
  std::vector<size_t> grid_stride_;
  std::vector<size_t> output_stride_;
  GridSamplerInterpolationMode interpolation_mode_;
  GridSamplerPaddingMode padding_mode_;
  bool align_corners_;
  bool is_null_input_;
};

template <typename T>
class GridSampler3DGpuKernelMod : public NativeGpuKernelMod {
 public:
  GridSampler3DGpuKernelMod() { ResetResource(); }
  ~GridSampler3DGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspaces,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input_addr = GetDeviceAddress<T>(inputs, kIndex0);
    T *grid_addr = GetDeviceAddress<T>(inputs, kIndex1);
    T *output_addr = GetDeviceAddress<T>(outputs, kIndex0);
    GridSampler3D(size_, input_addr, grid_addr, output_addr, input_shape_, grid_shape_, output_shape_, input_stride_,
                  grid_stride_, output_stride_, interpolation_mode_, padding_mode_, align_corners_,
                  reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override {
    auto kernel_ptr = std::dynamic_pointer_cast<ops::GridSampler3D>(base_operator);
    if (kernel_ptr == nullptr) {
      MS_EXCEPTION(ValueError) << "For primitive[GridSampler3D], cast op from BaseOperator to GridSampler3D failed.";
      return false;
    }
    kernel_name_ = kernel_ptr->name();
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), kGridSamplerInputNum, kernel_name_);
    CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kGridSamplerOutputNum, kernel_name_);
    interpolation_mode_ = kGridSamplerInterpolationMap[kernel_ptr->get_interpolation_mode()];
    padding_mode_ = kGridSamplerPaddingMap[kernel_ptr->get_padding_mode()];
    align_corners_ = kernel_ptr->get_align_corners();
    return true;
  }

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) {
    int ret = KernelMod::Resize(base_operator, inputs, outputs);
    if (ret != 0) {
      return ret;
    }
    if (input_size_list_.size() != kGridSamplerInputNum) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "' input size must be equal " << kGridSamplerInputNum << ".";
      return KRET_RESIZE_FAILED;
    }
    auto convert_int64_shape_to_sizet_shape = [=](std::vector<int64_t> int64_shape) -> std::vector<size_t> {
      std::vector<size_t> size_t_shape;
      (void)std::transform(int64_shape.begin(), int64_shape.end(), std::back_inserter(size_t_shape), LongToSize);
      return size_t_shape;
    };
    input_shape_ = convert_int64_shape_to_sizet_shape(inputs[kIndex0]->GetShapeVector());
    grid_shape_ = convert_int64_shape_to_sizet_shape(inputs[kIndex1]->GetShapeVector());
    output_shape_ = convert_int64_shape_to_sizet_shape(outputs[kIndex0]->GetShapeVector());

    if (input_shape_.empty()) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'input' must be at 5-D, but got scalar or None.";
      return KRET_RESIZE_FAILED;
    }

    if (grid_shape_.empty()) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'grid' must be at 5-D, but got scalar or None.";
      return KRET_RESIZE_FAILED;
    }

    if (output_shape_.empty()) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'output' must be at 5-D, but got scalar or None.";
      return KRET_RESIZE_FAILED;
    }

    size_t stride_tmp = 1;
    auto stride_compute = [&](std::vector<size_t> &stride, std::vector<size_t> shape) {
      for (int i = 4; i > -static_cast<int>(1); i--) {
        (void)stride.insert(stride.begin(), stride_tmp);
        stride_tmp *= shape[static_cast<size_t>(i)];
      }
      stride_tmp = 1;
    };
    stride_compute(input_stride_, input_shape_);
    stride_compute(grid_stride_, grid_shape_);
    stride_compute(output_stride_, output_shape_);
    size_ = input_shape_[kIndex0] * grid_shape_[kIndex1] * grid_shape_[kIndex2] * grid_shape_[kIndex3];
    return KRET_OK;
  }

  void ResetResource() noexcept {
    input_shape_.clear();
    grid_shape_.clear();
    output_shape_.clear();
    input_stride_.clear();
    grid_stride_.clear();
    output_stride_.clear();
    size_ = 0;
    interpolation_mode_ = GridSamplerInterpolationMode::BILINEAR;
    padding_mode_ = GridSamplerPaddingMode::ZEROS;
    align_corners_ = false;
    is_null_input_ = false;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 private:
  std::vector<size_t> input_shape_;
  std::vector<size_t> grid_shape_;
  std::vector<size_t> output_shape_;
  std::vector<size_t> input_stride_;
  std::vector<size_t> grid_stride_;
  std::vector<size_t> output_stride_;
  size_t size_;
  GridSamplerInterpolationMode interpolation_mode_;
  GridSamplerPaddingMode padding_mode_;
  bool align_corners_;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_GRID_SAMPLER_GPU_KERNEL_H_
