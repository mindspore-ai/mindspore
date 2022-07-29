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

#include "plugin/device/gpu/kernel/math/cdist_grad_gpu_kernel.h"
#include <utility>
#include <algorithm>

namespace mindspore {
namespace kernel {
constexpr size_t kCdistInputDimsMin = 2;
constexpr size_t kTwo = 2;
constexpr size_t kThree = 3;
bool CdistGradGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr_ = std::dynamic_pointer_cast<ops::CdistGrad>(base_operator);
  kernel_name_ = kernel_ptr_->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the kernel type should be in [float16, float32, float64], but got: " << kernel_attr << ".";
    return false;
  }
  kernel_func_ = func_list_[index].second;
  p_ = kernel_ptr_->get_p();

  batch_ = 0;
  unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).first);
  return true;
}

int CdistGradGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &) {
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just return do nothing.
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  ResetResource();
  std::vector<int64_t> grad_shape = inputs[0]->GetShapeVector();
  std::vector<int64_t> in_shape0 = inputs[1]->GetShapeVector();
  std::vector<int64_t> in_shape1 = inputs[kTwo]->GetShapeVector();
  std::vector<int64_t> dist_shape = inputs[kThree]->GetShapeVector();
  std::vector<int64_t> output_shape = outputs[0]->GetShapeVector();
  auto in_shape_size = in_shape0.size();
  grad_size_ = std::accumulate(grad_shape.begin(), grad_shape.end(), 1, std::multiplies<int64_t>());
  input0_size_ = std::accumulate(in_shape0.begin(), in_shape0.end(), 1, std::multiplies<int64_t>());
  input1_size_ = std::accumulate(in_shape1.begin(), in_shape1.end(), 1, std::multiplies<int64_t>());
  dist_size_ = std::accumulate(dist_shape.begin(), dist_shape.end(), 1, std::multiplies<int64_t>());
  out_size_ = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int64_t>());
  if (in_shape1.size() != in_shape_size || in_shape_size < kCdistInputDimsMin) {
    MS_LOG(ERROR) << "For " << kernel_name_ << ",invalid input shape, input0 shape size " << in_shape_size
                  << ", input1 shape size " << in_shape1.size();
    return KRET_RESIZE_FAILED;
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

  input_size_list_.push_back(grad_size_ * unit_size_);
  input_size_list_.push_back(input0_size_ * unit_size_);
  input_size_list_.push_back(input1_size_ * unit_size_);
  input_size_list_.push_back(dist_size_ * unit_size_);
  output_size_list_.push_back(out_size_ * unit_size_);
  return KRET_OK;
}

template <typename T>
bool CdistGradGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &workspace,
                                         const std::vector<AddressPtr> &outputs) {
  T *grad_start_ = GetDeviceAddress<T>(inputs, 0);
  T *dist_start_ = GetDeviceAddress<T>(inputs, 3);
  T *t1_start_ = GetDeviceAddress<T>(inputs, 1);
  T *t2_start_ = GetDeviceAddress<T>(inputs, 2);
  T *res_start_ = GetDeviceAddress<T>(outputs, 0);

  CalCdistGrad(out_size_, l1_size_, l2_size_, grad_start_, dist_start_, t1_start_, t2_start_, res_start_, m_, p_, r0_,
               r1_, batch_, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));

  return true;
}

std::vector<std::pair<KernelAttr, CdistGradGpuKernelMod::CdistGradFunc>> CdistGradGpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &CdistGradGpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
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
