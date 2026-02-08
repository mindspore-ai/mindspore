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

#include "kernel/gpu/cuda/nn/maxpool_with_argmax_gpu_kernel.h"
#include <functional>
#include <memory>
#include "include/curand.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMaxPoolWithArgmaxInputsNum = 1;
constexpr size_t kMaxPoolWithArgmaxOutputsNum = 2;
constexpr size_t Index2 = 2;
constexpr size_t Index3 = 3;
}  // namespace

void MaxPoolWithArgmaxGpuKernelMod::SetPad() {
  MS_EXCEPTION_IF_ZERO("stride height", stride_height_);
  MS_EXCEPTION_IF_ZERO("stride width", stride_width_);

  int tmp_height = (input_height_ / stride_height_) * stride_height_ == input_height_
                     ? (input_height_ / stride_height_)
                     : (input_height_ / stride_height_) + 1;
  pad_height_ = std::max<int>(0, (tmp_height - 1) * stride_height_ + window_height_ - input_height_);

  int tmp_width = (input_width_ / stride_width_) * stride_width_ == input_width_ ? (input_width_ / stride_width_)
                                                                                 : (input_width_ / stride_width_) + 1;
  pad_width_ = std::max<int>(0, (tmp_width - 1) * stride_width_ + window_width_ - input_width_);
  pad_top_ = pad_height_ / Index2;
  pad_left_ = pad_width_ / Index2;
}

bool MaxPoolWithArgmaxGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  if (inputs.size() != kMaxPoolWithArgmaxInputsNum || outputs.size() != kMaxPoolWithArgmaxOutputsNum) {
    MS_LOG(EXCEPTION) << kernel_name_ << ": input and output size must be " << kMaxPoolWithArgmaxInputsNum << " and "
                      << kMaxPoolWithArgmaxOutputsNum << ", but get " << inputs.size() << " and " << outputs.size();
    return false;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto pair = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!pair.first) {
    MS_LOG(ERROR) << "'" << kernel_name_ << "' does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[pair.second].second;
  return true;
}

int MaxPoolWithArgmaxGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &outputs) {
  int ret = 0;
  if (ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::vector<int64_t>> output_shapes;
  std::vector<int64_t> input_shape = inputs[0]->GetShapeVector();
  std::vector<int64_t> output_shape = outputs[0]->GetShapeVector();

  is_null_input_ =
    CHECK_SHAPE_NULL(input_shape, kernel_name_, "input") || CHECK_SHAPE_NULL(output_shape, kernel_name_, "output");
  if (is_null_input_) {
    return KRET_RESIZE_FAILED;
  }
  if (input_shape.size() < kInputDimLowerLimit || output_shape.size() < kOutputDimLowerLimit) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input and output cannot be less than 4, but "
                      << "got the dimension of input: " << input_shape.size()
                      << ", the dimension of output: " << output_shape.size();
  }
  n_ = LongToInt(input_shape[kInputIndexForN]);
  c_ = LongToInt(input_shape[kInputIndexForC]);
  input_height_ = LongToInt(input_shape[kInputIndexForH]);
  input_width_ = LongToInt(input_shape[kInputIndexForW]);
  output_height_ = LongToInt(output_shape[kOutputIndexForH]);
  output_width_ = LongToInt(output_shape[kOutputIndexForW]);
  std::vector<int> window;
  std::vector<int64_t> window_me = GetValue<std::vector<int64_t>>(primitive_->GetAttr("kernel_size"));
  (void)std::transform(window_me.begin(), window_me.end(), std::back_inserter(window),
                       [](const int64_t &value) { return static_cast<int>(value); });
  if (window.size() < Index3) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'kernel_size' cannot be less than 3, but got "
                      << window.size();
  }
  window_height_ = window[1];
  window_width_ = window[Index2];

  std::vector<int> stride;
  std::vector<int64_t> stride_me = GetValue<std::vector<int64_t>>(primitive_->GetAttr("strides"));
  (void)std::transform(stride_me.begin(), stride_me.end(), std::back_inserter(stride),
                       [](const int64_t &value) { return static_cast<int>(value); });
  if (stride.size() < Index3) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'strides' cannot be less than 3, but got "
                      << stride.size();
  }
  stride_height_ = stride[1];
  stride_width_ = stride[Index2];
  auto mode_str = GetValue<std::string>(primitive_->GetAttr("pad_mode"));
  (void)std::transform(mode_str.begin(), mode_str.end(), mode_str.begin(), ::toupper);
  MS_EXCEPTION_IF_CHECK_FAIL((mode_str == "SAME" || mode_str == "VALID"),
                             "MaxPoolGradGrad only supports pad mode same or valid, but get " + mode_str);
  auto pad_mode_ = mode_str == "SAME" ? PadMode::SAME : PadMode::VALID;
  pad_top_ = 0;
  pad_left_ = 0;
  if (pad_mode_ == SAME) {
    SetPad();
  }
  return KRET_OK;
}
template <typename T, typename S>
bool MaxPoolWithArgmaxGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                 const std::vector<KernelTensor *> &workspace,
                                                 const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  T *input_addr = GetDeviceAddress<T>(inputs, 0);
  T *output_addr = GetDeviceAddress<T>(outputs, 0);
  S *index_addr = GetDeviceAddress<S>(outputs, 1);
  auto status = CalMaxPoolWithArgmax(input_addr, n_, c_, input_height_, input_width_, window_height_, window_width_,
                                     stride_height_, stride_width_, pad_top_, pad_left_, output_height_, output_width_,
                                     output_addr, index_addr, device_id_, reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}
bool MaxPoolWithArgmaxGpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &workspace,
                                           const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
}

std::vector<std::pair<KernelAttr, MaxPoolWithArgmaxGpuKernelMod::MaxPoolWithArgmaxFunc>>
  MaxPoolWithArgmaxGpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt32),
     &MaxPoolWithArgmaxGpuKernelMod::LaunchKernel<int8_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt32),
     &MaxPoolWithArgmaxGpuKernelMod::LaunchKernel<int16_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
     &MaxPoolWithArgmaxGpuKernelMod::LaunchKernel<int64_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt32),
     &MaxPoolWithArgmaxGpuKernelMod::LaunchKernel<uint8_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt32),
     &MaxPoolWithArgmaxGpuKernelMod::LaunchKernel<uint16_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt32),
     &MaxPoolWithArgmaxGpuKernelMod::LaunchKernel<uint32_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt32),
     &MaxPoolWithArgmaxGpuKernelMod::LaunchKernel<uint64_t, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32),
     &MaxPoolWithArgmaxGpuKernelMod::LaunchKernel<half, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
     &MaxPoolWithArgmaxGpuKernelMod::LaunchKernel<float, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt32),
     &MaxPoolWithArgmaxGpuKernelMod::LaunchKernel<double, int32_t>},
};
std::vector<KernelAttr> MaxPoolWithArgmaxGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MaxPoolWithArgmaxFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, MaxPoolWithArgmax, MaxPoolWithArgmaxGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
