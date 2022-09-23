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
#include <functional>
#include "mindspore/core/ops/grad/maximum_grad.h"
#include "mindspore/core/ops/grad/minimum_grad.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMaxDim = 7;
}
bool BroadcastOpGradGpuKernelMod::GetOpType() {
  static const std::map<std::string, BroadcastGradOpType> broadcast_type_map = {
    {prim::kPrimMaximumGrad->name(), BROADCAST_GRAD_TYPE_MAXIMUM},
    {prim::kPrimMinimumGrad->name(), BROADCAST_GRAD_TYPE_MINIMUM},
  };
  auto iter = broadcast_type_map.find(kernel_name_);
  if (iter == broadcast_type_map.end()) {
    MS_LOG(ERROR) << "For " << kernel::Map2Str<std::map, BroadcastGradOpType>(broadcast_type_map)
                  << ", it only support max and min grad, but got " << kernel_name_;
    return false;
  }
  op_type_ = iter->second;
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
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  return true;
}

int BroadcastOpGradGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs,
                                        const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  x1_shape_ = LongVecToSizeVec(inputs.at(kIndex0)->GetShapeVector());
  x2_shape_ = LongVecToSizeVec(inputs.at(kIndex1)->GetShapeVector());
  dy_shape_ = LongVecToSizeVec(inputs.at(kIndex2)->GetShapeVector());
  output_num_ = std::accumulate(dy_shape_.begin(), dy_shape_.end(), size_t(1), std::multiplies<size_t>());
  is_null_input_ = CHECK_SHAPE_NULL(x1_shape_, kernel_name_, "x1") || CHECK_SHAPE_NULL(x2_shape_, kernel_name_, "x2") ||
                   CHECK_SHAPE_NULL(dy_shape_, kernel_name_, "dy");
  if (is_null_input_) {
    return KRET_OK;
  }
  need_broadcast_ = common::AnfAlgo::IsTensorBroadcast(x1_shape_, x2_shape_);
  // For x1_shape, x2_shape, dy_shape, it's validation has been done in core/ops/xxx.cc.
  // But we need check shape rank less equal to 7D.
  if (!broadcast_utils::AlignedBroadCastShape(kMaxDim, &dy_shape_, &x1_shape_, &x2_shape_)) {
    MS_LOG(ERROR)
      << "For '" << kernel_name_
      << "', it's dimension of input x1, x2 or dy shape less equal than 7D, which is invalid in gpu backend. ";
    return KRET_RESIZE_FAILED;
  }
  return KRET_OK;
}

template <typename T>
bool BroadcastOpGradGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                               const std::vector<AddressPtr> &outputs) {
  auto x1 = GetDeviceAddress<T>(inputs, kIndex0);
  auto x2 = GetDeviceAddress<T>(inputs, kIndex1);
  auto dy = GetDeviceAddress<T>(inputs, kIndex2);
  auto dx1 = GetDeviceAddress<T>(outputs, kIndex0);
  auto dx2 = GetDeviceAddress<T>(outputs, kIndex1);
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(cudaMemsetAsync(dx1, 0, outputs[kIndex0]->size, cuda_stream_),
                                    "BroadcastOpGradGpuKernelMod cudaMemSet Failed");
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(cudaMemsetAsync(dx2, 0, outputs[kIndex1]->size, cuda_stream_),
                                    "BroadcastOpGradGpuKernelMod cudaMemSet Failed");
  if (need_broadcast_) {
    BroadcastGrad(x1_shape_, x2_shape_, dy_shape_, output_num_, grad_x_, grad_y_, op_type_, x1, x2, dy, dx1, dx2,
                  device_id_, cuda_stream_);
  } else {
    NoBroadcastGrad(output_num_, grad_x_, grad_y_, op_type_, x1, x2, dy, dx1, dx2, device_id_, cuda_stream_);
  }
  return true;
}

const BroadcastOpGradGpuKernelMod::KernelFunc &BroadcastOpGradGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, BroadcastOpGradGpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr()
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
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddOutputAttr(kNumberTypeInt16)
       .AddOutputAttr(kNumberTypeInt16),
     &BroadcastOpGradGpuKernelMod::LaunchKernel<int16_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeUInt32)
       .AddOutputAttr(kNumberTypeUInt32)
       .AddOutputAttr(kNumberTypeUInt32),
     &BroadcastOpGradGpuKernelMod::LaunchKernel<uint32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kNumberTypeUInt64)
       .AddOutputAttr(kNumberTypeUInt64)
       .AddOutputAttr(kNumberTypeUInt64),
     &BroadcastOpGradGpuKernelMod::LaunchKernel<uint64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeUInt16)
       .AddOutputAttr(kNumberTypeUInt16)
       .AddOutputAttr(kNumberTypeUInt16),
     &BroadcastOpGradGpuKernelMod::LaunchKernel<uint16_t>},
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
     &BroadcastOpGradGpuKernelMod::LaunchKernel<double>},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, MinimumGrad, BroadcastOpGradGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, MaximumGrad, BroadcastOpGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
