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

#include "plugin/device/gpu/kernel/math/broadcast_grad_grad_gpu_kernel.h"
#include <memory>
#include <algorithm>
#include <functional>

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMaxDim = 7;
}
bool BroadcastOpGradGradGpuKernelMod::GetOpType() {
  static const std::map<std::string, BroadcastGradGradOpType> broadcast_grad_grad_op_type = {
    {prim::kPrimMaximumGradGrad->name(), BROADCAST_GRAD_GRAD_TYPE_MAXIMUM},
    {prim::kPrimMinimumGradGrad->name(), BROADCAST_GRAD_GRAD_TYPE_MINIMUM},
  };
  auto iter = broadcast_grad_grad_op_type.find(kernel_name_);
  if (iter == broadcast_grad_grad_op_type.end()) {
    MS_LOG(ERROR) << "For " << kernel::Map2Str<std::map, BroadcastGradGradOpType>(broadcast_grad_grad_op_type)
                  << ", it only support max and min grad grad, but got " << kernel_name_;
    return false;
  }
  op_type_ = iter->second;
  return true;
}

bool BroadcastOpGradGradGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                           const std::vector<KernelTensorPtr> &inputs,
                                           const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  if (!GetOpType()) {
    return false;
  }
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  return true;
}

int BroadcastOpGradGradGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                            const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs,
                                            const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto origin_x1_shape = inputs.at(kIndex0)->GetShapeVector();
  auto origin_dx1_shape = inputs.at(kIndex2)->GetShapeVector();
  if (origin_x1_shape != origin_dx1_shape) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it's input x2 shape is not equal to dx1 shape:  " << origin_x1_shape
                  << " vs " << origin_dx1_shape;
    return KRET_RESIZE_FAILED;
  }
  x1_shape_ = LongVecToSizeVec(origin_x1_shape);
  auto origin_x2_shape = inputs.at(kIndex1)->GetShapeVector();
  auto origin_dx2_shape = inputs.at(kIndex3)->GetShapeVector();
  if (origin_x2_shape != origin_dx2_shape) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it's input x2 shape is not equal to dx1 shape:  " << origin_x2_shape
                  << " vs " << origin_dx2_shape;
    return KRET_RESIZE_FAILED;
  }
  x2_shape_ = LongVecToSizeVec(origin_x2_shape);
  sopd_grad_shape_ = LongVecToSizeVec(outputs.at(kIndex2)->GetShapeVector());
  output_num_ = std::accumulate(sopd_grad_shape_.begin(), sopd_grad_shape_.end(), size_t(1), std::multiplies<size_t>());
  is_null_input_ = CHECK_SHAPE_NULL(x1_shape_, kernel_name_, "x1") || CHECK_SHAPE_NULL(x2_shape_, kernel_name_, "x2") ||
                   CHECK_SHAPE_NULL(sopd_grad_shape_, kernel_name_, "sopd_grad");
  if (is_null_input_) {
    return KRET_OK;
  }
  need_broadcast_ = common::AnfAlgo::IsTensorBroadcast(x1_shape_, x2_shape_);
  // For x1_shape, x2_shape, dy1_shape, it's validation has been done in core/ops/xxx.cc.
  // But we need check shape rank less equal to 7D.
  if (!broadcast_utils::AlignedBroadCastShape(kMaxDim, &sopd_grad_shape_, &x1_shape_, &x2_shape_)) {
    MS_LOG(ERROR)
      << "For '" << kernel_name_
      << "', it's dimension of input x1, x2 or dy shape less equal than 7D, which is invalid in gpu backend. ";
    return KRET_RESIZE_FAILED;
  }
  return KRET_OK;
}

template <typename T>
bool BroadcastOpGradGradGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                   const std::vector<AddressPtr> &,
                                                   const std::vector<AddressPtr> &outputs) {
  auto x1 = GetDeviceAddress<T>(inputs, kIndex0);
  auto x2 = GetDeviceAddress<T>(inputs, kIndex1);
  auto dy1 = GetDeviceAddress<T>(inputs, kIndex2);
  auto dy2 = GetDeviceAddress<T>(inputs, kIndex3);
  auto sopd_x1 = GetDeviceAddress<T>(outputs, kIndex0);
  auto sopd_x2 = GetDeviceAddress<T>(outputs, kIndex1);
  auto sopd_dout = GetDeviceAddress<T>(outputs, kIndex2);
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(cudaMemsetAsync(sopd_x1, 0, outputs[kIndex0]->size, cuda_stream_),
                                    "BroadcastOpGradGpuKernelMod cudaMemSet Failed");
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(cudaMemsetAsync(sopd_x2, 0, outputs[kIndex1]->size, cuda_stream_),
                                    "BroadcastOpGradGpuKernelMod cudaMemSet Failed");
  if (need_broadcast_) {
    BroadcastGradGrad(x1_shape_, x2_shape_, sopd_grad_shape_, output_num_, op_type_, x1, x2, dy1, dy2, sopd_dout,
                      device_id_, cuda_stream_);
  } else {
    NoBroadcastGradGrad(output_num_, op_type_, x1, x2, dy1, dy2, sopd_dout, device_id_, cuda_stream_);
  }
  return true;
}

const BroadcastOpGradGradGpuKernelMod::KernelFunc &BroadcastOpGradGradGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, BroadcastOpGradGradGpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32),
     &BroadcastOpGradGradGpuKernelMod::LaunchKernel<int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     &BroadcastOpGradGradGpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &BroadcastOpGradGradGpuKernelMod::LaunchKernel<half>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &BroadcastOpGradGradGpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64),
     &BroadcastOpGradGradGpuKernelMod::LaunchKernel<double>},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, MinimumGradGrad, BroadcastOpGradGradGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, MaximumGradGrad, BroadcastOpGradGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
