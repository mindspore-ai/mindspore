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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_GRIDSAMPLER_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_GRIDSAMPLER_GRAD_GPU_KERNEL_H_

#include <map>
#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include <utility>
#include "mindspore/core/ops/ops_func_impl/grid_sampler_2d_grad.h"
#include "mindspore/core/ops/ops_func_impl/grid_sampler_3d_grad.h"
#include "mindspore/core/ops/op_enum.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/grid_sampler_grad_impl.cuh"

namespace mindspore {
namespace kernel {
class GridSampler2DGradKernelMod : public NativeGpuKernelMod {
 public:
  GridSampler2DGradKernelMod() { ResetResource(); }
  ~GridSampler2DGradKernelMod() override = default;

  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs,
                    void *stream_ptr) {
    if (is_null_input_) {
      return true;
    }
    interpolation_mode_ = static_cast<GridSamplerInterpolationMode>(inputs[kIndex3]->GetValueWithCheck<int64_t>());
    padding_mode_ = static_cast<GridSamplerPaddingMode>(inputs[kIndex4]->GetValueWithCheck<int64_t>());
    align_corners_ = inputs[kIndex5]->GetValueWithCheck<bool>();
    T *grad_addr = GetDeviceAddress<T>(inputs, kIndex0);
    T *input_addr = GetDeviceAddress<T>(inputs, kIndex1);
    T *grid_addr = GetDeviceAddress<T>(inputs, kIndex2);
    T *dinput_addr = GetDeviceAddress<T>(outputs, kIndex0);
    T *dgrid_addr = GetDeviceAddress<T>(outputs, kIndex1);

    auto status = GridSampler2DGrad(
      size_, dinput_size_, dgrid_size_, grad_addr, input_addr, grid_addr, dinput_addr, dgrid_addr, grad_shape_,
      input_shape_, grid_shape_, dinput_shape_, dgrid_shape_, grad_stride_, input_stride_, grid_stride_, dinput_stride_,
      dgrid_stride_, interpolation_mode_, padding_mode_, align_corners_, reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
    return true;
  }

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), kGridSamplerGradInputNum, kernel_name_);
    CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kGridSamplerGradOutputNum, kernel_name_);
    kernel_func_(this, inputs, outputs, stream_ptr);
    return true;
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), kGridSamplerGradInputNum, kernel_name_);
    CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kGridSamplerGradOutputNum, kernel_name_);
    auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
    auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
    if (!is_match) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
      return false;
    }
    kernel_func_ = func_list_[index].second;
    return true;
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
    int ret = KernelMod::Resize(inputs, outputs);
    if (ret != 0) {
      return ret;
    }
    auto convert_int64_shape_to_sizet_shape = [=](std::vector<int64_t> int64_shape) -> std::vector<size_t> {
      std::vector<size_t> size_t_shape;
      (void)std::transform(int64_shape.begin(), int64_shape.end(), std::back_inserter(size_t_shape), LongToSize);
      return size_t_shape;
    };
    grad_shape_ = convert_int64_shape_to_sizet_shape(inputs[kIndex0]->GetShapeVector());
    input_shape_ = convert_int64_shape_to_sizet_shape(inputs[kIndex1]->GetShapeVector());
    grid_shape_ = convert_int64_shape_to_sizet_shape(inputs[kIndex2]->GetShapeVector());
    dinput_shape_ = convert_int64_shape_to_sizet_shape(outputs[kIndex0]->GetShapeVector());
    dgrid_shape_ = convert_int64_shape_to_sizet_shape(outputs[kIndex1]->GetShapeVector());

    if (grad_shape_.empty()) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'grad' must be at 4-D, but got scalar or None.";
      return KRET_RESIZE_FAILED;
    }

    if (input_shape_.empty()) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'input' must be at 4-D, but got scalar or None.";
      return KRET_RESIZE_FAILED;
    }

    if (grid_shape_.empty()) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'grid' must be at 4-D, but got scalar or None.";
      return KRET_RESIZE_FAILED;
    }

    if (dinput_shape_.empty()) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'dinput' must be at 4-D, but got scalar or None.";
      return KRET_RESIZE_FAILED;
    }

    if (dgrid_shape_.empty()) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'dgrid' must be at 4-D, but got scalar or None.";
      return KRET_RESIZE_FAILED;
    }

    size_t stride_tmp = 1;
    auto stride_compute = [&](std::vector<size_t> &stride, std::vector<size_t> shape) {
      for (int i = 3; i > -static_cast<int>(1); i--) {
        stride.insert(stride.begin(), stride_tmp);
        stride_tmp *= shape[i];
      }
      stride_tmp = 1;
    };
    grad_stride_.clear();
    input_stride_.clear();
    grid_stride_.clear();
    dinput_stride_.clear();
    dgrid_stride_.clear();
    stride_compute(grad_stride_, grad_shape_);
    stride_compute(input_stride_, input_shape_);
    stride_compute(grid_stride_, grid_shape_);
    stride_compute(dinput_stride_, dinput_shape_);
    stride_compute(dgrid_stride_, dgrid_shape_);
    size_ = input_shape_[kIndex0] * grid_shape_[kIndex1] * grid_shape_[kIndex2];
    dinput_size_ = GetTensorSize(dinput_shape_);
    dgrid_size_ = GetTensorSize(dgrid_shape_);
    return KRET_OK;
  }

  void ResetResource() noexcept {
    grad_shape_.clear();
    input_shape_.clear();
    grid_shape_.clear();
    dinput_shape_.clear();
    dgrid_shape_.clear();
    grad_stride_.clear();
    input_stride_.clear();
    grid_stride_.clear();
    dinput_stride_.clear();
    dgrid_stride_.clear();
    interpolation_mode_ = GridSamplerInterpolationMode::BILINEAR;
    padding_mode_ = GridSamplerPaddingMode::ZEROS;
    align_corners_ = false;
    is_null_input_ = false;
    size_ = 0;
    dinput_size_ = 0;
    dgrid_size_ = 0;
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  using KernelFunc = std::function<void(GridSampler2DGradKernelMod *, const std::vector<KernelTensor *> &,
                                        const std::vector<KernelTensor *> &, void *)>;
  KernelFunc kernel_func_{};
  static std::vector<std::pair<KernelAttr, KernelFunc>> func_list_;
  std::vector<size_t> grad_shape_;
  std::vector<size_t> input_shape_;
  std::vector<size_t> grid_shape_;
  std::vector<size_t> dinput_shape_;
  std::vector<size_t> dgrid_shape_;
  std::vector<size_t> grad_stride_;
  std::vector<size_t> input_stride_;
  std::vector<size_t> grid_stride_;
  std::vector<size_t> dinput_stride_;
  std::vector<size_t> dgrid_stride_;
  GridSamplerInterpolationMode interpolation_mode_;
  GridSamplerPaddingMode padding_mode_;
  bool align_corners_;
  bool is_null_input_;
  size_t size_;
  size_t dinput_size_;
  size_t dgrid_size_;
};

class GridSampler3DGradKernelMod : public NativeGpuKernelMod {
 public:
  GridSampler3DGradKernelMod() { ResetResource(); }
  ~GridSampler3DGradKernelMod() override = default;

  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs,
                    void *stream_ptr) {
    if (is_null_input_) {
      return true;
    }
    T *grad_addr = GetDeviceAddress<T>(inputs, kIndex0);
    T *input_addr = GetDeviceAddress<T>(inputs, kIndex1);
    T *grid_addr = GetDeviceAddress<T>(inputs, kIndex2);
    T *dinput_addr = GetDeviceAddress<T>(outputs, kIndex0);
    T *dgrid_addr = GetDeviceAddress<T>(outputs, kIndex1);
    interpolation_mode_ = static_cast<GridSamplerInterpolationMode>(inputs[kIndex3]->GetValueWithCheck<int64_t>());
    padding_mode_ = static_cast<GridSamplerPaddingMode>(inputs[kIndex4]->GetValueWithCheck<int64_t>());
    align_corners_ = inputs[kIndex5]->GetValueWithCheck<bool>();
    auto status = GridSampler3DGrad(
      size_, dinput_size_, dgrid_size_, grad_addr, input_addr, grid_addr, dinput_addr, dgrid_addr, grad_shape_,
      input_shape_, grid_shape_, dinput_shape_, dgrid_shape_, grad_stride_, input_stride_, grid_stride_, dinput_stride_,
      dgrid_stride_, interpolation_mode_, padding_mode_, align_corners_, reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
    return true;
  }

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), kGridSamplerGradInputNum, kernel_name_);
    CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kGridSamplerGradOutputNum, kernel_name_);
    kernel_func_(this, inputs, outputs, stream_ptr);
    return true;
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), kGridSamplerGradInputNum, kernel_name_);
    CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kGridSamplerGradOutputNum, kernel_name_);
    auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
    auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
    if (!is_match) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
      return false;
    }
    kernel_func_ = func_list_[index].second;
    return true;
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
    int ret = KernelMod::Resize(inputs, outputs);
    if (ret != 0) {
      return ret;
    }

    auto convert_int64_shape_to_sizet_shape = [=](std::vector<int64_t> int64_shape) -> std::vector<size_t> {
      std::vector<size_t> size_t_shape;
      (void)std::transform(int64_shape.begin(), int64_shape.end(), std::back_inserter(size_t_shape), LongToSize);
      return size_t_shape;
    };
    grad_shape_ = convert_int64_shape_to_sizet_shape(inputs[kIndex0]->GetShapeVector());
    input_shape_ = convert_int64_shape_to_sizet_shape(inputs[kIndex1]->GetShapeVector());
    grid_shape_ = convert_int64_shape_to_sizet_shape(inputs[kIndex2]->GetShapeVector());
    dinput_shape_ = convert_int64_shape_to_sizet_shape(outputs[kIndex0]->GetShapeVector());
    dgrid_shape_ = convert_int64_shape_to_sizet_shape(outputs[kIndex1]->GetShapeVector());

    if (grad_shape_.empty()) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'grad' must be at 5-D, but got scalar or None.";
      return KRET_RESIZE_FAILED;
    }

    if (input_shape_.empty()) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'input' must be at 5-D, but got scalar or None.";
      return KRET_RESIZE_FAILED;
    }

    if (grid_shape_.empty()) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'grid' must be at 5-D, but got scalar or None.";
      return KRET_RESIZE_FAILED;
    }

    if (dinput_shape_.empty()) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'dinput' must be at 5-D, but got scalar or None.";
      return KRET_RESIZE_FAILED;
    }

    if (dgrid_shape_.empty()) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'dgrid' must be at 5-D, but got scalar or None.";
      return KRET_RESIZE_FAILED;
    }

    size_t stride_tmp = 1;
    auto stride_compute = [&](std::vector<size_t> &stride, std::vector<size_t> shape) {
      for (int i = 4; i > -static_cast<int>(1); i--) {
        stride.insert(stride.begin(), stride_tmp);
        stride_tmp *= shape[i];
      }
      stride_tmp = 1;
    };
    grad_stride_.clear();
    input_stride_.clear();
    grid_stride_.clear();
    dinput_stride_.clear();
    dgrid_stride_.clear();
    stride_compute(grad_stride_, grad_shape_);
    stride_compute(input_stride_, input_shape_);
    stride_compute(grid_stride_, grid_shape_);
    stride_compute(dinput_stride_, dinput_shape_);
    stride_compute(dgrid_stride_, dgrid_shape_);
    size_ = input_shape_[kIndex0] * grid_shape_[kIndex1] * grid_shape_[kIndex2] * grid_shape_[kIndex3];
    dinput_size_ = GetTensorSize(dinput_shape_);
    dgrid_size_ = GetTensorSize(dgrid_shape_);
    return KRET_OK;
  }

  void ResetResource() noexcept {
    grad_shape_.clear();
    input_shape_.clear();
    grid_shape_.clear();
    dinput_shape_.clear();
    dgrid_shape_.clear();
    grad_stride_.clear();
    input_stride_.clear();
    grid_stride_.clear();
    dinput_stride_.clear();
    dgrid_stride_.clear();
    interpolation_mode_ = GridSamplerInterpolationMode::BILINEAR;
    padding_mode_ = GridSamplerPaddingMode::ZEROS;
    align_corners_ = false;
    is_null_input_ = false;
    size_ = 0;
    dinput_size_ = 0;
    dgrid_size_ = 0;
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  using KernelFunc = std::function<void(GridSampler3DGradKernelMod *, const std::vector<KernelTensor *> &,
                                        const std::vector<KernelTensor *> &, void *)>;
  KernelFunc kernel_func_{};
  static std::vector<std::pair<KernelAttr, KernelFunc>> func_list_;
  std::vector<size_t> grad_shape_;
  std::vector<size_t> input_shape_;
  std::vector<size_t> grid_shape_;
  std::vector<size_t> dinput_shape_;
  std::vector<size_t> dgrid_shape_;
  std::vector<size_t> grad_stride_;
  std::vector<size_t> input_stride_;
  std::vector<size_t> grid_stride_;
  std::vector<size_t> dinput_stride_;
  std::vector<size_t> dgrid_stride_;
  GridSamplerInterpolationMode interpolation_mode_;
  GridSamplerPaddingMode padding_mode_;
  bool align_corners_;
  bool is_null_input_;
  size_t size_;
  size_t dinput_size_;
  size_t dgrid_size_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_GRIDSAMPLER_GRAD_GPU_KERNEL_H_
