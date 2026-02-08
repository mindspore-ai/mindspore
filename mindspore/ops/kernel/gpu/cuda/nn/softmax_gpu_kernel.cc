/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/cuda/nn/softmax_gpu_kernel.h"
#include "kernel/gpu/cuda_impl/cuda_ops/transpose_impl.cuh"

namespace mindspore {
namespace kernel {
namespace {
int64_t MaybeWrapDim(int64_t dim, int64_t dim_post_expr, const std::string &kernel_name) {
  int64_t min = -dim_post_expr;
  int64_t max = dim_post_expr - 1;
  if (dim < min || dim > max) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the value of 'axis' must be in range [-" << dim_post_expr << ", "
                      << dim_post_expr << "), but got " << dim;
  }
  if (dim < 0) {
    dim += dim_post_expr;
  }
  return dim;
}
}  // namespace
bool SoftmaxGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  is_log_softmax_ = kernel_name_ == "LogSoftmax";

  return true;
}

size_t SoftmaxGpuKernelMod::GetAccAxis(KernelTensor *axis_kernel_tensor) const noexcept {
  std::vector<int64_t> axis;
  if (is_log_softmax_) {
    axis.push_back(axis_kernel_tensor->GetValueWithCheck<int64_t>());
  } else {
    axis = axis_kernel_tensor->GetValueWithCheck<std::vector<int64_t>>();
    // axis size must be 1
    if (axis.size() != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'axis' cannot be equal to 0, but got "
                        << axis.size();
    }
  }
  // check axis value
  auto axis_acc = static_cast<size_t>(MaybeWrapDim(axis[0], shape_size_, kernel_name_));
  return axis_acc;
}

int SoftmaxGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  // input, workspace and output will be assign in InitSizeLists.
  const auto &input_shape = inputs[kIndex0]->GetShapeVector();
  auto input_element_num =
    std::accumulate(input_shape.begin(), input_shape.end(), size_t(1), std::multiplies<size_t>());
  is_null_input_ = (input_element_num == 0);
  if (is_null_input_) {
    return KRET_OK;
  }

  ResetResource();
  (void)std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(shape_),
                       [](const int64_t &value) { return LongToInt(value); });
  shape_size_ = input_shape.size();
  axis_acc_ = GetAccAxis(inputs[kIndex1]);
  if (input_element_num > 0) {
    // calculate outer and inner size
    for (size_t i = 0; i < axis_acc_; ++i) {
      outer_size_ *= shape_[i];
    }
    for (size_t i = axis_acc_ + 1; i < shape_.size(); ++i) {
      inner_size_ *= shape_[i];
    }
  }

  return KRET_OK;
}

template <typename T>
bool SoftmaxGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &workspace,
                                       const std::vector<KernelTensor *> &outputs, void *stream_ptr) noexcept {
  if (is_null_input_) {
    return true;
  }

  T *input_addr = GetDeviceAddress<T>(inputs, 0);
  MS_ERROR_IF_NULL_W_RET_VAL(input_addr, false);
  T *output_addr = GetDeviceAddress<T>(outputs, 0);
  MS_ERROR_IF_NULL_W_RET_VAL(output_addr, false);

  // kernel function
  if (is_log_softmax_) {
    Softmax<T, true>(input_addr, output_addr, shape_[axis_acc_], outer_size_, inner_size_, device_id_,
                     reinterpret_cast<cudaStream_t>(stream_ptr));
  } else {
    Softmax<T, false>(input_addr, output_addr, shape_[axis_acc_], outer_size_, inner_size_, device_id_,
                      reinterpret_cast<cudaStream_t>(stream_ptr));
  }

  return true;
}

#define SOFTMAX_GPU_REG(MT, T)                                                                      \
  KernelAttr().AddInputAttr(MT).AddInputAttr(kObjectTypeTuple, kNumberTypeInt64).AddOutputAttr(MT), \
    &SoftmaxGpuKernelMod::LaunchKernel<T>

#define LOG_SOFTMAX_GPU_REG(MT, T)                                                                   \
  KernelAttr().AddInputAttr(MT).AddInputAttr(kObjectTypeNumber, kNumberTypeInt64).AddOutputAttr(MT), \
    &SoftmaxGpuKernelMod::LaunchKernel<T>

std::vector<std::pair<KernelAttr, SoftmaxGpuKernelMod::SoftmaxGpuLaunchFunc>> SoftmaxGpuKernelMod::func_list_ = {
  {SOFTMAX_GPU_REG(kNumberTypeFloat64, double)},    {SOFTMAX_GPU_REG(kNumberTypeFloat32, float)},
  {SOFTMAX_GPU_REG(kNumberTypeFloat16, half)},      {LOG_SOFTMAX_GPU_REG(kNumberTypeFloat64, double)},
  {LOG_SOFTMAX_GPU_REG(kNumberTypeFloat32, float)}, {LOG_SOFTMAX_GPU_REG(kNumberTypeFloat16, half)}};

std::vector<KernelAttr> SoftmaxGpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list;
  if (support_list.empty()) {
    (void)std::transform(
      func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
      [](const std::pair<KernelAttr, SoftmaxGpuKernelMod::SoftmaxGpuLaunchFunc> &pair) { return pair.first; });
  }
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Softmax, SoftmaxGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, LogSoftmax, SoftmaxGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
