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
#include "plugin/device/gpu/kernel/nn/binary_cross_entropy_grad_kernel.h"
#include <map>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/loss_with_reduction_impl.cuh"
#include "ops/op_name.h"

namespace mindspore {
namespace kernel {
bool BinaryCrossEntropyGradGpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                                const std::vector<KernelTensor *> &workspace,
                                                const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel<half>(inputs, outputs, stream_ptr);
  } else if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs, stream_ptr);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of input must be float16 or float32, but got "
                      << TypeIdToType(dtype_)->ToString();
  }
  return true;
}

template <typename T>
void BinaryCrossEntropyGradGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                      const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  T *input_x = GetDeviceAddress<T>(inputs, kIndex0);
  T *input_y = GetDeviceAddress<T>(inputs, kIndex1);
  T *dloss = GetDeviceAddress<T>(inputs, kIndex2);
  T *weight = nullptr;
  if (inputs[kIndex3]->type_id() != kMetaTypeNone) {
    weight = GetDeviceAddress<T>(inputs, kIndex3);
  }
  auto reduction = static_cast<Reduction>(inputs[kIndex4]->GetValueWithCheck<int64_t>());
  if (reduction == Reduction::NONE) {
    reduction_ = ReductionMode::kNone;
  } else if (reduction == Reduction::MEAN) {
    reduction_ = ReductionMode::kMean;
  } else {
    reduction_ = ReductionMode::kSum;
  }
  T *dx = GetDeviceAddress<T>(outputs, kIndex0);
  if (input_size_ > 0) {
    auto status = BinaryCrossEntropyLossGrad(input_size_, reduction_, input_x, input_y, weight, dloss, dx,
                                             reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
  }
}

bool BinaryCrossEntropyGradGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto match = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!match.first) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
  }

  dtype_ = inputs[kIndex0]->dtype_id();
  return true;
}

int BinaryCrossEntropyGradGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    return ret;
  }
  auto input_shape = inputs[kIndex0]->GetShapeVector();
  input_size_ = SizeOf(input_shape);
  return KRET_OK;
}

std::vector<KernelAttr> BinaryCrossEntropyGradGpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> kernel_attr_list = {KernelAttr()
                                                       .AddInputAttr(kNumberTypeFloat16)
                                                       .AddInputAttr(kNumberTypeFloat16)
                                                       .AddInputAttr(kNumberTypeFloat16)
                                                       .AddOptionalInputAttr(kNumberTypeFloat16)
                                                       .AddInputAttr(kNumberTypeInt64)
                                                       .AddOutputAttr(kNumberTypeFloat16),
                                                     KernelAttr()
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddOptionalInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeInt64)
                                                       .AddOutputAttr(kNumberTypeFloat32)};
  return kernel_attr_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, BinaryCrossEntropyGrad, BinaryCrossEntropyGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
