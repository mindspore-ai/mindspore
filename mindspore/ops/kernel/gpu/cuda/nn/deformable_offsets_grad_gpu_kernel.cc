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
#include "kernel/gpu/cuda/nn/deformable_offsets_grad_gpu_kernel.h"

#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <utility>
#include <functional>
#include "mindspore/ops/infer/grad/deformable_offsets_grad.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputNum = 3;
constexpr size_t kOutputNum = 2;
constexpr size_t kInputShapeSize = 4;

constexpr size_t kGradIndex = 0;
constexpr size_t kXIndex = 1;
constexpr size_t kOffsetIndex = 2;
constexpr size_t kGradXIndex = 0;
constexpr size_t kGradOffsetIndex = 1;

auto constexpr kPadStr = "pads";
auto constexpr kStrideStr = "strides";
auto constexpr kDilationStr = "dilation";
auto constexpr kKernelSizeStr = "kernel size";
auto constexpr kInputXStr = "input_x";
auto constexpr kInputGradStr = "input_grad";

constexpr size_t kPadNum = 4;
constexpr size_t kStrideNum = 4;
constexpr size_t kDilationNum = 4;
constexpr size_t kKernelSizeNum = 2;

constexpr size_t kCIndexForNCHW = 1;
constexpr size_t kHIndexForNCHW = 2;
constexpr size_t kWIndexForNCHW = 3;
constexpr size_t kHIndexForNHWC = 1;
constexpr size_t kWIndexForNHWC = 2;
constexpr size_t kCIndexForNHWC = 3;

constexpr size_t kPadTopIndex = 0;
constexpr size_t kPadLeftIndex = 2;
constexpr size_t kStrideHIndex = 2;
constexpr size_t kStrideWIndex = 3;
constexpr size_t kDilationHIndex = 2;
constexpr size_t kDilationWIndex = 3;
constexpr size_t kKernelHIndex = 0;
constexpr size_t kKernelWIndex = 1;

void CheckSize(const std::string &kernel_name_, const std::string &dim_name, size_t expect, size_t actual) {
  if (actual != expect) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of '" << dim_name << "' must be " << expect
                      << ", but got " << actual;
  }
}
}  // namespace

bool DeformableOffsetsGradGpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &,
                                               const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  cuda_stream_ = reinterpret_cast<cudaStream_t>(stream_ptr);
  return kernel_func_(this, inputs, outputs);
}

bool DeformableOffsetsGradGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &outputs) {
  if (inputs.size() != kInputNum || outputs.size() != kOutputNum) {
    MS_LOG(ERROR) << kernel_name_ << ": input and output size should be " << kInputNum << " and " << kOutputNum
                  << ", but get " << inputs.size() << " and " << outputs.size();
    return false;
  }

  data_format_ = GetValue<std::string>(primitive_->GetAttr("format"));

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  type_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(0).dtype);
  return true;
}

void DeformableOffsetsGradGpuKernelMod::SetDims(const std::vector<KernelTensor *> &inputs,
                                                const std::vector<KernelTensor *> &outputs) {
  auto kernel_name_ = primitive_->name();
  dims_.deformable_group = LongToUint(GetValue<int64_t>(primitive_->GetAttr("deformable_groups")));
  if (dims_.deformable_group == 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', deformable group must be greater than 0.";
  }
  std::vector<int64_t> pad = GetValue<std::vector<int64_t>>(primitive_->GetAttr("pads"));
  CheckSize(kernel_name_, kPadStr, kPadNum, pad.size());
  dims_.pad_top = LongToUint(pad[kPadTopIndex]);
  dims_.pad_left = LongToUint(pad[kPadLeftIndex]);

  std::vector<int64_t> stride = GetValue<std::vector<int64_t>>(primitive_->GetAttr("strides"));
  CheckSize(kernel_name_, kStrideStr, kStrideNum, stride.size());
  dims_.stride_h = LongToUint(stride[kStrideHIndex]);
  dims_.stride_w = LongToUint(stride[kStrideWIndex]);

  std::vector<int64_t> dilation = GetValue<std::vector<int64_t>>(primitive_->GetAttr("dilations"));
  CheckSize(kernel_name_, kDilationStr, kDilationNum, dilation.size());
  dims_.dilation_h = LongToUint(dilation[kDilationHIndex]);
  dims_.dilation_w = LongToUint(dilation[kDilationWIndex]);

  std::vector<int64_t> ksize = GetValue<std::vector<int64_t>>(primitive_->GetAttr("ksize"));
  CheckSize(kernel_name_, kKernelSizeStr, kKernelSizeNum, ksize.size());
  dims_.kernel_h = LongToUint(ksize[kKernelHIndex]);
  dims_.kernel_w = LongToUint(ksize[kKernelWIndex]);
  if (dims_.kernel_h == 0 || dims_.kernel_w == 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'ksize' must be larger than 0.";
  }
  auto x_shape = inputs[kXIndex]->GetShapeVector();
  CheckSize(kernel_name_, kInputXStr, kInputShapeSize, x_shape.size());
  dims_.x_n = LongToUint(x_shape[0]);
  auto grad_shape = inputs[kGradIndex]->GetShapeVector();
  CheckSize(kernel_name_, kInputGradStr, kInputShapeSize, grad_shape.size());
  if (data_format_ == kOpFormat_NCHW) {
    dims_.grad_h = LongToUint(grad_shape[kHIndexForNCHW]);
    dims_.grad_w = LongToUint(grad_shape[kWIndexForNCHW]);
    dims_.x_h = LongToUint(x_shape[kHIndexForNCHW]);
    dims_.x_w = LongToUint(x_shape[kWIndexForNCHW]);
    dims_.deformable_group_channel = LongToUint(x_shape[kCIndexForNCHW]) / dims_.deformable_group;
  } else {
    dims_.grad_h = LongToUint(grad_shape[kHIndexForNHWC]);
    dims_.grad_w = LongToUint(grad_shape[kWIndexForNHWC]);
    dims_.x_h = LongToUint(x_shape[kHIndexForNHWC]);
    dims_.x_w = LongToUint(x_shape[kWIndexForNHWC]);
    dims_.deformable_group_channel = LongToUint(x_shape[kCIndexForNHWC]) / dims_.deformable_group;
  }
  dims_.offset_h = dims_.grad_h / dims_.kernel_h;
  dims_.offset_w = dims_.grad_w / dims_.kernel_w;

  auto grad_x_shape = outputs[kGradXIndex]->GetShapeVector();
  grad_x_size_ = std::accumulate(grad_x_shape.begin(), grad_x_shape.end(), type_size_, std::multiplies<size_t>());

  auto grad_offset_shape = outputs[kGradOffsetIndex]->GetShapeVector();
  grad_offset_size_ =
    std::accumulate(grad_offset_shape.begin(), grad_offset_shape.end(), type_size_, std::multiplies<size_t>());
}

int DeformableOffsetsGradGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    MS_LOG(ERROR) << kernel_name_ << " kernel mode resize failed.";
    return ret;
  }
  if (inputs.size() != kInputNum || output_size_list_.size() != kOutputNum) {
    MS_LOG(ERROR) << kernel_name_ << " resize : input and output size should be " << kInputNum << " and " << kOutputNum
                  << ", but got " << inputs.size() << " and " << output_size_list_.size();
    return KRET_RESIZE_FAILED;
  }
  SetDims(inputs, outputs);
  return KRET_OK;
}

template <typename T>
bool DeformableOffsetsGradGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                     const std::vector<KernelTensor *> &outputs) {
  T *grad_addr = GetDeviceAddress<T>(inputs, kGradIndex);
  T *x_addr = GetDeviceAddress<T>(inputs, kXIndex);
  T *offset_addr = GetDeviceAddress<T>(inputs, kOffsetIndex);
  T *grad_x_addr = GetDeviceAddress<T>(outputs, kGradXIndex);
  T *grad_offset_addr = GetDeviceAddress<T>(outputs, kGradOffsetIndex);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemsetAsync(grad_x_addr, 0, grad_x_size_, cuda_stream_),
                                     "Call cudaMemsetAsync grad_x failed");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemsetAsync(grad_offset_addr, 0, grad_offset_size_, cuda_stream_),
                                     "Call cudaMemsetAsync grad_x failed");
  uint dim_x_n = dims_.x_n;
  uint dim_x_h = dims_.x_h;
  uint dim_x_w = dims_.x_w;
  uint dim_offset_h = dims_.offset_h;
  uint dim_offset_w = dims_.offset_w;
  uint dim_kernel_h = dims_.kernel_h;
  uint dim_kernel_w = dims_.kernel_w;
  uint dim_pad_top = dims_.pad_top;
  uint dim_pad_left = dims_.pad_left;
  uint dim_stride_h = dims_.stride_h;
  uint dim_stride_w = dims_.stride_w;
  uint dim_dilation_h = dims_.dilation_h;
  uint dim_dilation_w = dims_.dilation_w;
  uint dim_deformable_group = dims_.deformable_group;
  uint dim_deformable_group_channel = dims_.deformable_group_channel;
  cudaError_t status = cudaErrorNotReady;
  if (data_format_ == kOpFormat_NCHW) {
    status = ApplyDeformableOffsetGrad(
      dim_x_n, dim_x_h, dim_x_w, dim_offset_h, dim_offset_w, dim_kernel_h, dim_kernel_w, dim_pad_top, dim_pad_left,
      dim_stride_h, dim_stride_w, dim_dilation_h, dim_dilation_w, dim_deformable_group, dim_deformable_group_channel,
      true, grad_addr, x_addr, offset_addr, grad_x_addr, grad_offset_addr, device_id_, cuda_stream_);
  } else {
    status = ApplyDeformableOffsetGrad(
      dim_x_n, dim_x_h, dim_x_w, dim_offset_h, dim_offset_w, dim_kernel_h, dim_kernel_w, dim_pad_top, dim_pad_left,
      dim_stride_h, dim_stride_w, dim_dilation_h, dim_dilation_w, dim_deformable_group, dim_deformable_group_channel,
      false, grad_addr, x_addr, offset_addr, grad_x_addr, grad_offset_addr, device_id_, cuda_stream_);
  }
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, DeformableOffsetsGradGpuKernelMod::KernelFunc>>
  DeformableOffsetsGradGpuKernelMod::func_list_ = {{KernelAttr()
                                                      .AddInputAttr(kNumberTypeFloat16)
                                                      .AddInputAttr(kNumberTypeFloat16)
                                                      .AddInputAttr(kNumberTypeFloat16)
                                                      .AddOutputAttr(kNumberTypeFloat16)
                                                      .AddOutputAttr(kNumberTypeFloat16),
                                                    &DeformableOffsetsGradGpuKernelMod::LaunchKernel<half>},
                                                   {KernelAttr()
                                                      .AddInputAttr(kNumberTypeFloat32)
                                                      .AddInputAttr(kNumberTypeFloat32)
                                                      .AddInputAttr(kNumberTypeFloat32)
                                                      .AddOutputAttr(kNumberTypeFloat32)
                                                      .AddOutputAttr(kNumberTypeFloat32),
                                                    &DeformableOffsetsGradGpuKernelMod::LaunchKernel<float>}};

std::vector<KernelAttr> DeformableOffsetsGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, KernelFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, DeformableOffsetsGrad, DeformableOffsetsGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
