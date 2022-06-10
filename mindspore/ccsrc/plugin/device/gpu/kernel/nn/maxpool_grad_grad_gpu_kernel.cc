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

#include "plugin/device/gpu/kernel/nn/maxpool_grad_grad_gpu_kernel.h"
#include <algorithm>
#include <functional>
#include "mindspore/core/ops/grad/max_pool_grad_grad.h"
#include "abstract/utils.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/maxpool_with_argmax_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/gather.cuh"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMaxPoolGradGradInputsNum = 3;
constexpr size_t kMaxPoolGradGradOutputsNum = 1;
}  // namespace

bool MaxPoolGradGradGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (inputs.size() != kMaxPoolGradGradInputsNum || outputs.size() != kMaxPoolGradGradOutputsNum) {
    MS_LOG(ERROR) << kernel_name_ << ": input and output size should be " << kMaxPoolGradGradInputsNum << " and "
                  << kMaxPoolGradGradOutputsNum << ", but get " << inputs.size() << " and " << outputs.size();
    return false;
  }

  auto kernel_ptr = std::dynamic_pointer_cast<ops::MaxPoolGradGrad>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "Cast MaxPoolGradGrad ops failed!";
    return false;
  }
  window_height_ = LongToInt(kernel_ptr->get_kernel_size()[kDim2]);
  window_width_ = LongToInt(kernel_ptr->get_kernel_size()[kDim3]);
  stride_height_ = LongToInt(kernel_ptr->get_strides()[kDim2]);
  stride_width_ = LongToInt(kernel_ptr->get_strides()[kDim3]);
  pad_mode_ = kernel_ptr->get_pad_mode();
  if (pad_mode_ != PadMode::SAME && pad_mode_ != PadMode::VALID) {
    MS_LOG(ERROR) << kernel_name_ << " only support pad mode same or valid, but get " << pad_mode_;
    return false;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int MaxPoolGradGradGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs,
                                        const std::map<uint32_t, tensor::TensorPtr> &) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != 0) {
    return ret;
  }
  if (input_size_list_.size() != kMaxPoolGradGradInputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' input size must be equal kMaxPoolGradGradInputsNum.";
    return KRET_RESIZE_FAILED;
  }
  auto input_shape = inputs[kIndex0]->GetShapeVector();
  batch_ = LongToInt(input_shape[kDim0]);
  channel_ = LongToInt(input_shape[kDim1]);
  input_height_ = LongToInt(input_shape[kDim2]);
  input_width_ = LongToInt(input_shape[kDim3]);
  auto output_shape = outputs[kIndex0]->GetShapeVector();
  output_height_ = LongToInt(output_shape[kDim2]);
  output_width_ = LongToInt(output_shape[kDim3]);

  // calculate pad
  MS_EXCEPTION_IF_ZERO("stride height", stride_height_);
  MS_EXCEPTION_IF_ZERO("stride width", stride_width_);
  if (pad_mode_ == PadMode::SAME) {
    int tmp_height = (input_height_ / stride_height_) * stride_height_ == input_height_
                       ? (input_height_ / stride_height_)
                       : (input_height_ / stride_height_) + 1;
    pad_height_ = std::max<int>(0, (tmp_height - 1) * stride_height_ + window_height_ - input_height_);

    int tmp_width = (input_width_ / stride_width_) * stride_width_ == input_width_ ? (input_width_ / stride_width_)
                                                                                   : (input_width_ / stride_width_) + 1;
    pad_width_ = std::max<int>(0, (tmp_width - 1) * stride_width_ + window_width_ - input_width_);
    pad_top_ = pad_height_ / 2;
    pad_left_ = pad_width_ / 2;
  }

  workspace_size_list_.clear();
  workspace_size_list_.push_back(input_size_list_[1]);
  auto index_size =
    std::accumulate(output_shape.begin(), output_shape.end(), sizeof(int32_t), std::multiplies<size_t>());
  workspace_size_list_.push_back(index_size);
  return KRET_OK;
}

template <typename T>
bool MaxPoolGradGradGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                               const std::vector<AddressPtr> &workspace,
                                               const std::vector<AddressPtr> &outputs) {
  T *input_addr = GetDeviceAddress<T>(inputs, kIndex0);
  T *output_addr = GetDeviceAddress<T>(workspace, kIndex0);
  auto *index_addr = GetDeviceAddress<int32_t>(workspace, kIndex1);
  CalMaxPoolWithArgmax<T, int32_t>(input_addr, batch_, channel_, input_height_, input_width_, window_height_,
                                   window_width_, stride_height_, stride_width_, pad_top_, pad_left_, output_height_,
                                   output_width_, output_addr, index_addr, device_id_,
                                   reinterpret_cast<cudaStream_t>(cuda_stream_));

  T *grad_addr = GetDeviceAddress<T>(inputs, kIndex2);
  T *dx = GetDeviceAddress<T>(outputs, kIndex0);
  size_t dim_before_axis = 1;
  size_t dim_at_axis_input = channel_ * input_height_ * input_width_;
  size_t dim_at_axis_output = channel_ * output_height_ * output_width_;
  size_t dim_after_axis = 1;
  for (int b = 0; b < batch_; b++) {
    int32_t *index_t = index_addr + b * channel_ * output_height_ * output_width_;
    T *grad_t = grad_addr + b * channel_ * input_height_ * input_width_;
    T *dx_t = dx + b * channel_ * output_height_ * output_height_;
    Gather<T, int32_t>(grad_t, index_t, dx_t, dim_before_axis, dim_at_axis_input, dim_at_axis_output, dim_after_axis,
                       reinterpret_cast<cudaStream_t>(cuda_stream_), device_id_);
  }
  return true;
}

std::vector<std::pair<KernelAttr, MaxPoolGradGradGpuKernelMod::MaxPoolGradGradFunc>>
  MaxPoolGradGradGpuKernelMod::func_list_ = {{KernelAttr()
                                                .AddInputAttr(kNumberTypeFloat16)
                                                .AddInputAttr(kNumberTypeFloat16)
                                                .AddInputAttr(kNumberTypeFloat16)
                                                .AddOutputAttr(kNumberTypeFloat16),
                                              &MaxPoolGradGradGpuKernelMod::LaunchKernel<half>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeFloat32)
                                                .AddInputAttr(kNumberTypeFloat32)
                                                .AddInputAttr(kNumberTypeFloat32)
                                                .AddOutputAttr(kNumberTypeFloat32),
                                              &MaxPoolGradGradGpuKernelMod::LaunchKernel<float>}};

std::vector<KernelAttr> MaxPoolGradGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MaxPoolGradGradFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, MaxPoolGradGrad, MaxPoolGradGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
