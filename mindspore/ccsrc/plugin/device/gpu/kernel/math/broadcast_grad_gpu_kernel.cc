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

#include "plugin/device/gpu/kernel/math/broadcast_grad_gpu_kernel.h"
#include <memory>
#include <algorithm>
#include "mindspore/core/ops/grad/maximum_grad.h"
#include "mindspore/core/ops/grad/minimum_grad.h"

namespace mindspore {
namespace kernel {
namespace {
bool IsBroadcast(const std::vector<size_t> &lhs, const std::vector<size_t> &rhs) {
  if (lhs.size() != rhs.size()) {
    return true;
  }
  for (size_t i = 0; i < lhs.size(); i++) {
    if (lhs[i] != rhs[i]) {
      return true;
    }
  }
  return false;
}
}  // namespace

bool BroadcastOpGradGpuKernelMod::GetOpType() {
  const std::map<std::string, BroadcastGradOpType> kBroadcastTypeMap = {
    {prim::kPrimMaximumGrad->name(), BROADCAST_GRAD_TYPE_MAXIMUM},
    {prim::kPrimMinimumGrad->name(), BROADCAST_GRAD_TYPE_MINIMUM},
  };
  auto iter = kBroadcastTypeMap.find(kernel_name_);
  if (iter == kBroadcastTypeMap.end()) {
    MS_LOG(ERROR) << "For 'MaximumGrad' or 'MinimumGrad', it only support max and min grad, but got " << kernel_name_;
    return false;
  } else {
    op_type_ = iter->second;
  }
  return true;
}

bool BroadcastOpGradGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  if (!GetOpType()) {
    return false;
  }
  if (op_type_ == BROADCAST_GRAD_TYPE_MAXIMUM) {
    auto kernel_ptr = std::make_shared<ops::MaximumGrad>(base_operator->GetPrim());
    grad_x_ = kernel_ptr->get_grad_x();
    grad_y_ = kernel_ptr->get_grad_y();
  } else {
    auto kernel_ptr = std::make_shared<ops::MinimumGrad>(base_operator->GetPrim());
    grad_x_ = kernel_ptr->get_grad_x();
    grad_y_ = kernel_ptr->get_grad_y();
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int BroadcastOpGradGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs,
                                        const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  ResetResource();
  unit_size_ = GetTypeByte(TypeIdToType(inputs.at(kIndex0)->GetDtype()));
  std::vector<size_t> shape0;
  auto origin_shape0 = inputs.at(kIndex0)->GetShapeVector();
  (void)std::transform(origin_shape0.begin(), origin_shape0.end(), std::back_inserter(shape0), LongToSize);
  std::vector<size_t> shape1;
  auto origin_shape1 = inputs.at(kIndex1)->GetShapeVector();
  (void)std::transform(origin_shape1.begin(), origin_shape1.end(), std::back_inserter(shape1), LongToSize);
  std::vector<size_t> shape2;
  auto origin_shape2 = inputs.at(kIndex2)->GetShapeVector();
  (void)std::transform(origin_shape2.begin(), origin_shape2.end(), std::back_inserter(shape2), LongToSize);

  is_null_input_ = CHECK_SHAPE_NULL(shape0, kernel_name_, "input_0") ||
                   CHECK_SHAPE_NULL(shape1, kernel_name_, "input_1") ||
                   CHECK_SHAPE_NULL(shape2, kernel_name_, "input_2");
  if (is_null_input_) {
    return KRET_OK;
  }
  need_broadcast_ = IsBroadcast(shape0, shape1);
  if (need_broadcast_ && shape0.size() > kMaxShapeSize) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it's dimension of input cannot be greater than " << kMaxShapeSize
                  << ", but got " << shape0.size();
    return KRET_RESIZE_FAILED;
  }

  for (size_t i = 0; i < shape2.size(); i++) {
    if (need_broadcast_) {
      dy_shape_[i] = shape2[i];
    }
    output_num_ *= shape2[i];
  }

  int x0_offset = SizeToInt(shape2.size()) - SizeToInt(shape0.size());
  for (size_t i = 0; i < shape0.size(); i++) {
    if (need_broadcast_) {
      if ((SizeToInt(i) + x0_offset) >= 0 && (SizeToInt(i) + x0_offset) < kMaxShapeSize) {
        x0_shape_[i + x0_offset] = shape0[i];
      } else {
        auto index = i + x0_offset;
        MS_LOG(ERROR) << "For '" << kernel_name_ << "', it's dimension of input cannot be greater than "
                      << kMaxShapeSize << ", but got " << (index + 1);
        return KRET_RESIZE_FAILED;
      }
    }
    input0_num_ *= shape0[i];
  }
  int x1_offset = SizeToInt(shape2.size()) - SizeToInt(shape1.size());
  for (size_t i = 0; i < shape1.size(); i++) {
    if (need_broadcast_) {
      if ((SizeToInt(i) + x1_offset) >= 0 && (SizeToInt(i) + x1_offset) < kMaxShapeSize) {
        x1_shape_[i + x1_offset] = shape1[i];
      } else {
        auto index = i + x1_offset;
        MS_LOG(ERROR) << "For '" << kernel_name_ << "', it's dimension of input cannot be greater than "
                      << kMaxShapeSize << ", but got " << (index + 1);
        return KRET_RESIZE_FAILED;
      }
    }
    input1_num_ *= shape1[i];
  }
  InitSizeLists();
  return KRET_OK;
}

void BroadcastOpGradGpuKernelMod::InitSizeLists() {
  input_size_list_.emplace_back(input0_num_ * unit_size_);
  input_size_list_.emplace_back(input1_num_ * unit_size_);
  input_size_list_.emplace_back(output_num_ * unit_size_);
  output_size_list_.emplace_back(input0_num_ * unit_size_);
  output_size_list_.emplace_back(input1_num_ * unit_size_);
}

void BroadcastOpGradGpuKernelMod::ResetResource() noexcept {
  input0_num_ = 1;
  input1_num_ = 1;
  output_num_ = 1;
  std::fill(x0_shape_, x0_shape_ + kMaxShapeSize, 1);
  std::fill(x1_shape_, x1_shape_ + kMaxShapeSize, 1);
  std::fill(dy_shape_, dy_shape_ + kMaxShapeSize, 1);
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
}

template <typename T>
bool BroadcastOpGradGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,

                                               const std::vector<AddressPtr> &outputs) {
  auto x1 = GetDeviceAddress<T>(inputs, kIndex0);
  auto x2 = GetDeviceAddress<T>(inputs, kIndex1);
  auto dy = GetDeviceAddress<T>(inputs, kIndex2);
  auto dx1 = GetDeviceAddress<T>(outputs, kIndex0);
  auto dx2 = GetDeviceAddress<T>(outputs, kIndex1);

  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemsetAsync(dx1, 0, outputs[0]->size, reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "BroadcastOpGradGpuKernelMod cudaMemSet Failed");
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemsetAsync(dx2, 0, outputs[1]->size, reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "BroadcastOpGradGpuKernelMod cudaMemSet Failed");
  if (need_broadcast_) {
    BroadcastGrad(x0_shape_[kIndex0], x0_shape_[kIndex1], x0_shape_[kIndex2], x0_shape_[kIndex3], x1_shape_[kIndex0],
                  x1_shape_[kIndex1], x1_shape_[kIndex2], x1_shape_[kIndex3], dy_shape_[kIndex0], dy_shape_[kIndex1],
                  dy_shape_[kIndex2], dy_shape_[kIndex3], grad_x_, grad_y_, op_type_, x1, x2, dy, dx1, dx2, device_id_,
                  reinterpret_cast<cudaStream_t>(cuda_stream_));
  } else {
    NoBroadcastGrad(output_num_, grad_x_, grad_y_, op_type_, x1, x2, dy, dx1, dx2, device_id_,
                    reinterpret_cast<cudaStream_t>(cuda_stream_));
  }
  return true;
}

std::vector<std::pair<KernelAttr, BroadcastOpGradGpuKernelMod::BroadCastFunc>> BroadcastOpGradGpuKernelMod::func_list_ =
  {{KernelAttr()
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeInt32),
    &BroadcastOpGradGpuKernelMod::LaunchKernel<int>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt64),
    &BroadcastOpGradGpuKernelMod::LaunchKernel<int64_t>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeFloat16)
      .AddInputAttr(kNumberTypeFloat16)
      .AddInputAttr(kNumberTypeFloat16)
      .AddOutputAttr(kNumberTypeFloat16)
      .AddOutputAttr(kNumberTypeFloat16),
    &BroadcastOpGradGpuKernelMod::LaunchKernel<half>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeFloat32)
      .AddOutputAttr(kNumberTypeFloat32)
      .AddOutputAttr(kNumberTypeFloat32),
    &BroadcastOpGradGpuKernelMod::LaunchKernel<float>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeFloat64)
      .AddInputAttr(kNumberTypeFloat64)
      .AddInputAttr(kNumberTypeFloat64)
      .AddOutputAttr(kNumberTypeFloat64)
      .AddOutputAttr(kNumberTypeFloat64),
    &BroadcastOpGradGpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> BroadcastOpGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, BroadCastFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, MinimumGrad, BroadcastOpGradGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, MaximumGrad, BroadcastOpGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
