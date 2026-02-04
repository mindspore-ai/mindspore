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

#include "kernel/gpu/cuda/math/cdist_grad_gpu_kernel.h"
#include <utility>
#include <algorithm>

namespace mindspore {
namespace kernel {
constexpr size_t kCdistInputDimsMin = 2;
constexpr size_t kTwo = 2;
bool CdistGradGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  return true;
}

int CdistGradGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  ResetResource();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the kernel type should be in [float16, float32, float64], but got: " << kernel_attr << ".";
    return false;
  }
  kernel_func_ = func_list_[index].second;
  p_ = inputs[kIndex4]->GetValueWithCheck<float>();
  unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).dtype);
  std::vector<int64_t> grad_shape = inputs[kIndex0]->GetShapeVector();
  std::vector<int64_t> in_shape0 = inputs[kIndex1]->GetShapeVector();
  std::vector<int64_t> in_shape1 = inputs[kIndex2]->GetShapeVector();
  std::vector<int64_t> dist_shape = inputs[kIndex3]->GetShapeVector();
  std::vector<int64_t> output_shape = outputs[kIndex0]->GetShapeVector();
  auto in_shape_size = in_shape0.size();
  grad_size_ =
    std::accumulate(grad_shape.begin(), grad_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  input0_size_ =
    std::accumulate(in_shape0.begin(), in_shape0.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  input1_size_ =
    std::accumulate(in_shape1.begin(), in_shape1.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  dist_size_ =
    std::accumulate(dist_shape.begin(), dist_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  out_size_ =
    std::accumulate(output_shape.begin(), output_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  if (in_shape1.size() != in_shape_size || in_shape_size < kCdistInputDimsMin) {
    MS_LOG(ERROR) << "For " << kernel_name_ << ",invalid input shape, input0 shape size " << in_shape_size
                  << ", input1 shape size " << in_shape1.size();
    return KRET_RESIZE_FAILED;
  }
  for (size_t i = 0; i < in_shape_size - kCdistInputDimsMin; i++) {
    if (in_shape0[i] != in_shape1[i]) {
      MS_LOG(ERROR) << "invalid input shape, the batch shape of input0 must be the same as the shape of input1 ,but "
                       "got 'input0_shape["
                    << i << "]': " << in_shape0[i] << " and 'input1_shape[" << i << "]': " << in_shape1[i]
                    << ", kernel_name_ " << kernel_name_;
      return KRET_RESIZE_FAILED;
    }
  }
  batch_ = 0;
  for (size_t i = 0; i < in_shape_size - kCdistInputDimsMin; i++) {
    batch_ += in_shape0[i];
  }
  batch_ = (batch_ <= 0) ? 1 : batch_;

  r0_ = in_shape0[in_shape_size - kTwo];
  m_ = in_shape0[in_shape_size - 1];
  r1_ = in_shape1[in_shape_size - kTwo];

  l1_size_ = r0_ * m_;
  l2_size_ = r1_ * m_;

  output_size_list_.push_back(out_size_ * unit_size_);
  return KRET_OK;
}

template <typename T>
bool CdistGradGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &workspace,
                                         const std::vector<KernelTensor *> &outputs) {
  T *grad_start_ = GetDeviceAddress<T>(inputs, kIndex0);
  T *dist_start_ = GetDeviceAddress<T>(inputs, kIndex3);
  T *t1_start_ = GetDeviceAddress<T>(inputs, kIndex1);
  T *t2_start_ = GetDeviceAddress<T>(inputs, kIndex2);
  T *res_start_ = GetDeviceAddress<T>(outputs, kIndex0);

  auto status = CalCdistGrad(out_size_, l1_size_, l2_size_, grad_start_, dist_start_, t1_start_, t2_start_, res_start_,
                             m_, p_, r0_, r1_, batch_, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, kernel_name_);

  return true;
}

std::vector<std::pair<KernelAttr, CdistGradGpuKernelMod::CdistGradFunc>> CdistGradGpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &CdistGradGpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat64),
   &CdistGradGpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> CdistGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, CdistGradFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, CdistGrad, CdistGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
