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
#include "plugin/device/cpu/kernel/adaptive_max_pool_3d_grad_cpu_kernel.h"

#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
bool AdaptiveMaxPool3DGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                             const std::vector<KernelTensorPtr> &inputs,
                                             const std::vector<KernelTensorPtr> &outputs) {
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }

  return true;
}

int AdaptiveMaxPool3DGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                              const std::vector<KernelTensorPtr> &inputs,
                                              const std::vector<KernelTensorPtr> &outputs,
                                              const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  input_x_shape_ = inputs[1]->GetShapeVector();
  output_shape_ = input_x_shape_;
  const size_t dim_num = input_x_shape_.size();
  const size_t kDimNum4 = 4;
  const size_t kDimNum5 = 5;
  if (!(dim_num == kDimNum4 || dim_num == kDimNum5)) {
    MS_LOG(EXCEPTION) << "For AdaptiveMaxPool3DGrad, input dimensions should be equal to 4 or 5, but got " << dim_num
                      << ".";
  }

  input_grad_shape_ = inputs[0]->GetShapeVector();
  if (input_grad_shape_.size() != input_x_shape_.size()) {
    MS_LOG(EXCEPTION) << "For AdaptiveMaxPool3DGrad, input grad dimensions should be same as input dimensions, but got "
                      << input_grad_shape_.size() << ".";
  }

  input_argmax_shape_ = inputs[kIndex2]->GetShapeVector();
  if (input_argmax_shape_.size() != input_x_shape_.size()) {
    MS_LOG(EXCEPTION) << "For AdaptiveMaxPool3DGrad, input indice dimensions should be same as input "
                         "dimensions, but got "
                      << input_argmax_shape_.size() << ".";
  }
  if (input_argmax_shape_ != input_grad_shape_) {
    MS_LOG(EXCEPTION) << "For AdaptiveMaxPool3DGrad, input grad shape should be same as input argmax.";
  }

  return KRET_OK;
}

#define FUNTYPE const std::vector<std::pair<KernelAttr, AdaptiveMaxPool3DGradCpuKernelMod::KernelRunFunc>>
FUNTYPE &AdaptiveMaxPool3DGradCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, AdaptiveMaxPool3DGradCpuKernelMod::KernelRunFunc>> func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt8),
     &AdaptiveMaxPool3DGradCpuKernelMod::LaunchKernel<int8_t, int8_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt16),
     &AdaptiveMaxPool3DGradCpuKernelMod::LaunchKernel<int16_t, int16_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32),
     &AdaptiveMaxPool3DGradCpuKernelMod::LaunchKernel<int32_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt64),
     &AdaptiveMaxPool3DGradCpuKernelMod::LaunchKernel<int64_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeUInt8),
     &AdaptiveMaxPool3DGradCpuKernelMod::LaunchKernel<uint8_t, uint8_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeUInt16),
     &AdaptiveMaxPool3DGradCpuKernelMod::LaunchKernel<uint16_t, uint16_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeUInt32),
     &AdaptiveMaxPool3DGradCpuKernelMod::LaunchKernel<uint32_t, uint32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeUInt64),
     &AdaptiveMaxPool3DGradCpuKernelMod::LaunchKernel<uint64_t, uint64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat16),
     &AdaptiveMaxPool3DGradCpuKernelMod::LaunchKernelHalf},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32),
     &AdaptiveMaxPool3DGradCpuKernelMod::LaunchKernel<float, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat64),
     &AdaptiveMaxPool3DGradCpuKernelMod::LaunchKernel<double, double>}};
  return func_list_;
}

template <typename T1, typename T2>
bool AdaptiveMaxPool3DGradCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                     const std::vector<AddressPtr> &workspace,
                                                     const std::vector<AddressPtr> &outputs) {
  auto input_grad = reinterpret_cast<T1 *>(inputs[0]->addr);
  auto input_argmax = reinterpret_cast<int32_t *>(inputs[2]->addr);
  auto output = reinterpret_cast<T2 *>(outputs[0]->addr);
  const int64_t output_num_data = std::accumulate(output_shape_.begin(), output_shape_.end(), static_cast<size_t>(1),
                                                  [=](size_t a, size_t b) { return a * b; });
  const T2 data_zero = static_cast<T2>(0);
  for (int64_t i = 0; i < output_num_data; ++i) {
    output[i] = data_zero;
  }
  const int64_t output_stride = output_shape_.cend()[-1] * output_shape_.cend()[-2] * output_shape_.cend()[-3];
  const int64_t argmax_stride =
    input_argmax_shape_.cend()[-1] * input_argmax_shape_.cend()[-2] * input_argmax_shape_.cend()[-3];
  auto shard_adaptive_max_pool_3d_grad = [&](int64_t start, int64_t end) {
    for (int64_t n = start; n < end; ++n) {
      for (int64_t i = 0; i < argmax_stride; ++i) {
        int32_t maxp = input_argmax[i + n * argmax_stride] + n * output_stride;
        output[maxp] += static_cast<T2>(input_grad[i + n * argmax_stride]);
      }
    }
  };
  const int64_t batch = std::accumulate(input_argmax_shape_.begin(), input_argmax_shape_.end() - 3,
                                        static_cast<int64_t>(1), [=](int64_t a, int64_t b) { return a * b; });
  CPUKernelUtils::ParallelFor(shard_adaptive_max_pool_3d_grad, batch);
  return true;
}

bool AdaptiveMaxPool3DGradCpuKernelMod::LaunchKernelHalf(const std::vector<AddressPtr> &inputs,
                                                         const std::vector<AddressPtr> &workspace,
                                                         const std::vector<AddressPtr> &outputs) {
  auto input_grad = reinterpret_cast<Eigen::half *>(inputs[0]->addr);
  auto input_argmax = reinterpret_cast<int32_t *>(inputs[2]->addr);
  auto output = reinterpret_cast<Eigen::half *>(outputs[0]->addr);
  const int64_t output_num_data = std::accumulate(output_shape_.begin(), output_shape_.end(), static_cast<size_t>(1),
                                                  [=](size_t a, size_t b) { return a * b; });
  const float data_zero = static_cast<float>(0);
  float *output_tmp = static_cast<float *>(malloc(sizeof(float) * output_num_data));
  for (int64_t i = 0; i < output_num_data; ++i) {
    output_tmp[i] = data_zero;
  }
  const int64_t output_stride = output_shape_.cend()[-1] * output_shape_.cend()[-2] * output_shape_.cend()[-3];
  const int64_t argmax_stride =
    input_argmax_shape_.cend()[-1] * input_argmax_shape_.cend()[-2] * input_argmax_shape_.cend()[-3];
  auto shard_adaptive_max_pool_3d_grad = [&](int64_t start, int64_t end) {
    for (int64_t n = start; n < end; ++n) {
      for (int64_t i = 0; i < argmax_stride; ++i) {
        int32_t maxp = input_argmax[i + n * argmax_stride] + n * output_stride;
        output_tmp[maxp] += static_cast<float>(input_grad[i + n * argmax_stride]);
      }
    }
  };
  const int64_t batch = std::accumulate(input_argmax_shape_.begin(), input_argmax_shape_.end() - 3,
                                        static_cast<int64_t>(1), [=](int64_t a, int64_t b) { return a * b; });
  CPUKernelUtils::ParallelFor(shard_adaptive_max_pool_3d_grad, batch);
  auto shard_copy = [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; ++i) {
      output[i] = static_cast<Eigen::half>(output_tmp[i]);
    }
  };
  CPUKernelUtils::ParallelFor(shard_copy, output_num_data);
  free(output_tmp);
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, AdaptiveMaxPool3DGrad, AdaptiveMaxPool3DGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
