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
#include "kernel/gpu/cuda/nn/deformable_offsets_gpu_kernel.h"

#include <memory>
#include <utility>
#include <algorithm>
#include "kernel/gpu/cuda_impl/cuda_ops/deformable_offsets_impl.cuh"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kTopPadIndex = 0;
constexpr size_t kLeftPadIndex = 2;
constexpr size_t kKernelSizeHIndex = 0;
constexpr size_t kKernelSizeWIndex = 1;
constexpr size_t kInputNum = 2;
constexpr size_t kOutputNum = 1;
constexpr size_t kStrideAttrNum = 4;
constexpr size_t kPadAttrNum = 4;
constexpr size_t kKernelSizeAttrNum = 2;
constexpr size_t kDilationAttrNum = 4;
}  // namespace

bool DeformableOffsetsGpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &workspace,
                                           const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
}

bool DeformableOffsetsGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  if (inputs.size() != kInputNum || outputs.size() != kOutputNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it should get two inputs and one output, but got " << inputs.size()
                  << "inputs and " << outputs.size() << " outputs";
    return false;
  }
  auto tensor_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(tensor_attr, GetOpSupport());
  if (!is_match) {
    return false;
  }
  kernel_func_ = func_list_[index].second;

  if (!CheckParam(primitive_)) {
    return false;
  }
  return true;
}

bool DeformableOffsetsGpuKernelMod::CheckParam(const PrimitivePtr &primitive_) {
  MS_EXCEPTION_IF_NULL(primitive_);
  auto kernel_name_ = primitive_->name();
  data_format_ = GetValue<std::string>(primitive_->GetAttr("format"));
  if (data_format_ == kOpFormat_NCHW) {
    n_axis_ = 0;
    c_axis_ = 1;
    h_axis_ = 2;
    w_axis_ = 3;
  } else {
    MS_LOG(ERROR) << kernel_name_ << " only supports input with format NCHW, but got format " << data_format_;
    return false;
  }
  const auto to_unsigned = [](const int64_t &value) { return LongToUint(value); };
  const auto &strides = GetValue<std::vector<int64_t>>(primitive_->GetAttr("strides"));
  std::transform(strides.begin(), strides.end(), std::back_inserter(strides_), to_unsigned);
  if (strides_.size() != kStrideAttrNum || strides_[n_axis_] != 1 || strides_[c_axis_] != 1) {
    MS_LOG(ERROR) << "Get invalid strides attr form " << kernel_name_
                  << ", strides should be a vector constructed by 4 integer and n&c dim should be 1, but got"
                  << strides_;
    return false;
  }
  const auto &pads = GetValue<std::vector<int64_t>>(primitive_->GetAttr("pads"));
  std::transform(pads.begin(), pads.end(), std::back_inserter(pads_), to_unsigned);
  if (pads_.size() != kPadAttrNum) {
    MS_LOG(ERROR) << "Get invalid pads attr form " << kernel_name_
                  << ", padding should be a vector constructed by 4 integer, but got" << pads_;
    return false;
  }
  const auto &kernel_size = GetValue<std::vector<int64_t>>(primitive_->GetAttr("ksize"));
  std::transform(kernel_size.begin(), kernel_size.end(), std::back_inserter(kernel_size_), to_unsigned);
  if (kernel_size_.size() != kKernelSizeAttrNum) {
    MS_LOG(ERROR) << "Get invalid ksize attr form " << kernel_name_
                  << ", ksize should be a vector constructed by 2 integer, but got" << kernel_size_;
    return false;
  }
  const auto &dilations = GetValue<std::vector<int64_t>>(primitive_->GetAttr("dilations"));
  std::transform(dilations.begin(), dilations.end(), std::back_inserter(dilations_), to_unsigned);
  if (dilations_.size() != kDilationAttrNum || dilations_[n_axis_] != 1 || dilations_[c_axis_] != 1) {
    MS_LOG(ERROR) << "Get invalid dilations attr form " << kernel_name_
                  << ", dilations should be a vector constructed by 4 integer and n&c dim should be 1, but got"
                  << dilations_;
    return false;
  }
  deformable_groups_ = static_cast<uint32_t>(GetValue<int64_t>(primitive_->GetAttr("deformable_groups")));
  if (deformable_groups_ == 0) {
    MS_LOG(ERROR) << kernel_name_ << "'s deformable_groups should greater than 0, but got " << deformable_groups_;
    return false;
  }
  modulated_ = GetValue<bool>(primitive_->GetAttr("modulated"));
  if (!modulated_) {
    MS_LOG(ERROR) << kernel_name_ << "only support v2, and the modulated should be true, but got false";
    return false;
  }
  return true;
}

int DeformableOffsetsGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &outputs) {
  if (inputs.size() != kInputNum || outputs.size() != kOutputNum) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", it should get two inputs and one output, but got "
                      << inputs.size() << " inputs and " << outputs.size() << "outputs.";
  }
  if (KernelMod::Resize(inputs, outputs) == KRET_UNKNOWN_SHAPE) {
    return KRET_UNKNOWN_SHAPE;
  }
  const auto &x_shape = inputs[0]->GetShapeVector();
  n_ = x_shape[n_axis_];
  c_ = x_shape[c_axis_];
  x_h_ = x_shape[h_axis_];
  x_w_ = x_shape[w_axis_];
  const auto &y_shape = outputs[0]->GetShapeVector();
  output_h_ = y_shape[h_axis_];
  output_w_ = y_shape[w_axis_];
  position_grid_num_ = output_w_ * output_h_;
  auto position_grid_size = position_grid_num_ * 2 * sizeof(int32_t);
  workspace_size_list_.emplace_back(position_grid_size);
  return KRET_OK;
}

template <class T>
bool DeformableOffsetsGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                 const std::vector<KernelTensor *> &workspace,
                                                 const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  int32_t *position_addr = GetDeviceAddress<int32_t>(workspace, 0);
  const size_t num = static_cast<size_t>(output_h_) * static_cast<size_t>(output_w_);
  cudaError_t status = cudaErrorNotReady;
  status = GenPositionGrid(kernel_size_[kKernelSizeHIndex], kernel_size_[kKernelSizeWIndex], strides_[h_axis_],
                           strides_[w_axis_], dilations_[h_axis_], dilations_[w_axis_], pads_[kLeftPadIndex],
                           pads_[kTopPadIndex], output_w_, num, position_addr, device_id_,
                           reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, kernel_name_);
  T *x_addr = GetDeviceAddress<T>(inputs, 0);
  T *offsets_addr = GetDeviceAddress<T>(inputs, 1);
  T *output_addr = GetDeviceAddress<T>(outputs, 0);
  status = DeformableOffsets(x_addr, offsets_addr, position_addr, n_, c_, x_h_, x_w_, deformable_groups_,
                             kernel_size_[kKernelSizeHIndex], kernel_size_[kKernelSizeWIndex], output_h_, output_w_,
                             output_addr, device_id_, reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, DeformableOffsetsGpuKernelMod::LaunchKernelFunc>>
  DeformableOffsetsGpuKernelMod::func_list_ = {{KernelAttr()
                                                  .AddInputAttr(kNumberTypeFloat32, kOpFormat_NCHW)
                                                  .AddInputAttr(kNumberTypeFloat32, kOpFormat_NCHW)
                                                  .AddOutputAttr(kNumberTypeFloat32, kOpFormat_NCHW),
                                                &DeformableOffsetsGpuKernelMod::LaunchKernel<float>},
                                               {KernelAttr()
                                                  .AddInputAttr(kNumberTypeFloat16, kOpFormat_NCHW)
                                                  .AddInputAttr(kNumberTypeFloat16, kOpFormat_NCHW)
                                                  .AddOutputAttr(kNumberTypeFloat16, kOpFormat_NCHW),
                                                &DeformableOffsetsGpuKernelMod::LaunchKernel<half>}};

std::vector<KernelAttr> DeformableOffsetsGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, LaunchKernelFunc> &item) { return item.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, DeformableOffsets, DeformableOffsetsGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
