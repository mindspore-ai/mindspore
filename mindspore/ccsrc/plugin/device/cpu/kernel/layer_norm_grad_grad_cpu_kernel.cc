/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <functional>
#include <numeric>
#include <vector>
#include "plugin/device/cpu/kernel/layer_norm_grad_grad_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/nnacl/fp32/adam_fp32.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputSize = 8;
constexpr size_t kOutputSize = 3;
constexpr size_t kIdx0 = 0;
constexpr size_t kIdx3 = 3;
constexpr size_t kIdx4 = 4;
constexpr size_t kZero = 0;
constexpr size_t kMemMaxLen = 1e8;
}  // namespace

bool LayerNormGradGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  dtype_ = inputs[kIndex0]->GetDtype();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto match = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!match.first) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' it does not support this kernel type: " << kernel_attr;
    return false;
  }

  return true;
}

int LayerNormGradGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs,
                                          const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  input_shape_ = inputs[kIndex0]->GetDeviceShapeAdaptively();
  mean_shape_ = inputs[kIndex3]->GetDeviceShapeAdaptively();
  g_shape_ = inputs[kIndex4]->GetDeviceShapeAdaptively();
  return KRET_OK;
}

template <typename DATA_T>
bool calc_inv_std(DATA_T *input_var, DATA_T *inv_std, size_t mean_num) {
  for (size_t i = 0; i < mean_num; i++) {
    if (input_var[i] <= DATA_T(0)) {
      return false;
    }
    inv_std[i] = DATA_T(1) / sqrt(input_var[i]);
  }
  return true;
}

template <typename DATA_T>
bool shard_inner_mean(size_t start_idx, size_t end_idx, size_t g_num, DATA_T *sum1, DATA_T *sum2, DATA_T *sum3,
                      DATA_T *sum4, DATA_T *inv_std, DATA_T *input_d_dx, DATA_T *input_dy, DATA_T *input_gamma,
                      DATA_T *input_x, DATA_T *input_mean, DATA_T *x_hat, DATA_T *dy_gamma) {
  for (size_t i = start_idx; i < end_idx; i++) {
    if (g_num == 0) {
      return false;
    }
    size_t sum_idx = i / g_num;
    sum1[sum_idx] -= inv_std[sum_idx] * input_d_dx[i] / static_cast<DATA_T>(g_num);
    DATA_T cur_x_hat = (input_x[i] - input_mean[sum_idx]) * inv_std[sum_idx];
    x_hat[i] = cur_x_hat;
    sum2[sum_idx] -= cur_x_hat * inv_std[sum_idx] * input_d_dx[i] / static_cast<DATA_T>(g_num);
    size_t g_idx = i % g_num;
    DATA_T cur_dy_gamma = input_dy[i] * input_gamma[g_idx];
    dy_gamma[i] = cur_dy_gamma;
    sum3[sum_idx] += cur_dy_gamma / static_cast<DATA_T>(g_num);
    sum4[sum_idx] += cur_dy_gamma * cur_x_hat / static_cast<DATA_T>(g_num);
  }
  return true;
}

template <typename DATA_T>
bool shard_outer_mean(size_t start_idx, size_t end_idx, size_t g_num, DATA_T *sum2, DATA_T *sum3, DATA_T *sum4,
                      DATA_T *sum5, DATA_T *sum6, DATA_T *sum7, DATA_T *part3, DATA_T *inv_std, DATA_T *input_d_dx,
                      DATA_T *input_d_dg, DATA_T *x_hat, DATA_T *dy_gamma, DATA_T *input_dy, DATA_T *input_x,
                      DATA_T *input_mean) {
  for (size_t i = start_idx; i < end_idx; i++) {
    if (g_num == 0) {
      return false;
    }
    size_t g_idx = i % g_num;
    size_t sum_idx = i / g_num;
    DATA_T part_sum1 = dy_gamma[i] - sum3[sum_idx] - x_hat[i] * sum4[sum_idx];
    DATA_T part_sum2 =
      dy_gamma[i] * sum2[sum_idx] - sum4[sum_idx] * input_d_dx[i] * inv_std[sum_idx] + input_dy[i] * input_d_dg[g_idx];
    sum5[sum_idx] += input_d_dx[i] * part_sum1 / static_cast<DATA_T>(g_num);
    sum6[sum_idx] += (input_x[i] - input_mean[sum_idx]) * part_sum2 / static_cast<DATA_T>(g_num);
    DATA_T cur_part3 = inv_std[sum_idx] * part_sum2;
    part3[i] = cur_part3;
    sum7[sum_idx] -= cur_part3 / static_cast<DATA_T>(g_num);
  }
  return true;
}

template <typename DATA_T>
bool shard_input_prop(size_t start_idx, size_t end_idx, size_t g_num, DATA_T *sum1, DATA_T *sum2, DATA_T *sum5,
                      DATA_T *sum6, DATA_T *sum7, DATA_T *part3, DATA_T *inv_std, DATA_T *input_d_dx,
                      DATA_T *input_gamma, DATA_T *input_d_dg, DATA_T *input_d_db, DATA_T *x_hat, DATA_T *output_sopd_x,
                      DATA_T *output_sopd_dy) {
  for (size_t i = start_idx; i < end_idx; i++) {
    if (g_num == 0) {
      return false;
    }
    size_t g_idx = i % g_num;
    size_t sum_idx = i / g_num;
    DATA_T cur_part4 = -x_hat[i] * inv_std[sum_idx] * inv_std[sum_idx] * (sum5[sum_idx] + sum6[sum_idx]);
    output_sopd_x[i] = part3[i] + cur_part4 + sum7[sum_idx];
    DATA_T cur_part5 = input_gamma[g_idx] * input_d_dx[i] * inv_std[sum_idx];
    DATA_T cur_part6 = input_gamma[g_idx] * sum1[sum_idx];
    DATA_T cur_part7 = input_gamma[g_idx] * x_hat[i] * sum2[sum_idx];
    DATA_T cur_part8 = x_hat[i] * input_d_dg[g_idx];
    output_sopd_dy[i] = cur_part5 + cur_part6 + cur_part7 + cur_part8 + input_d_db[g_idx];
  }
  return true;
}

template <typename DATA_T>
bool shard_param_prop(size_t start_idx, size_t end_idx, size_t g_num, DATA_T *sum1, DATA_T *sum2, DATA_T *inv_std,
                      DATA_T *input_d_dx, DATA_T *x_hat, DATA_T *input_dy, DATA_T *output_sopd_g) {
  for (size_t i = start_idx; i < end_idx; i++) {
    if (g_num == 0) {
      return false;
    }
    size_t g_idx = i % g_num;
    size_t sum_idx = i / g_num;
    DATA_T cur_part9 = input_dy[i] * x_hat[i] * sum2[sum_idx];
    DATA_T cur_part10 = input_dy[i] * sum1[sum_idx];
    DATA_T cur_part11 = input_dy[i] * input_d_dx[i] * inv_std[sum_idx];
    output_sopd_g[g_idx] += cur_part9 + cur_part10 + cur_part11;
  }
  return true;
}

bool LayerNormGradGradCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                           const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputSize, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputSize, kernel_name_);
  if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of 'input_x' should be float16, float32 but got "
                      << dtype_;
  }
  return true;
}

template <typename DATA_T>
void LayerNormGradGradCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                 const std::vector<AddressPtr> &outputs) {
  // enter LayerNormGradGradCompute
  auto input_x = reinterpret_cast<DATA_T *>(inputs[0]->addr);
  MS_EXCEPTION_IF_NULL(input_x);
  auto input_dy = reinterpret_cast<DATA_T *>(inputs[1]->addr);
  MS_EXCEPTION_IF_NULL(input_dy);
  auto input_var = reinterpret_cast<DATA_T *>(inputs[2]->addr);
  MS_EXCEPTION_IF_NULL(input_var);
  auto input_mean = reinterpret_cast<DATA_T *>(inputs[3]->addr);
  MS_EXCEPTION_IF_NULL(input_mean);
  auto input_gamma = reinterpret_cast<DATA_T *>(inputs[4]->addr);
  MS_EXCEPTION_IF_NULL(input_gamma);
  auto input_d_dx = reinterpret_cast<DATA_T *>(inputs[5]->addr);
  MS_EXCEPTION_IF_NULL(input_d_dx);
  auto input_d_dg = reinterpret_cast<DATA_T *>(inputs[6]->addr);
  MS_EXCEPTION_IF_NULL(input_d_dg);
  auto input_d_db = reinterpret_cast<DATA_T *>(inputs[7]->addr);
  MS_EXCEPTION_IF_NULL(input_d_db);
  auto output_sopd_x = reinterpret_cast<DATA_T *>(outputs[0]->addr);
  MS_EXCEPTION_IF_NULL(output_sopd_x);
  auto output_sopd_dy = reinterpret_cast<DATA_T *>(outputs[1]->addr);
  MS_EXCEPTION_IF_NULL(output_sopd_dy);
  auto output_sopd_g = reinterpret_cast<DATA_T *>(outputs[2]->addr);
  MS_EXCEPTION_IF_NULL(output_sopd_g);
  size_t num =
    static_cast<size_t>(std::accumulate(input_shape_.cbegin(), input_shape_.cend(), 1, std::multiplies<int64_t>{}));
  size_t g_num =
    static_cast<size_t>(std::accumulate(g_shape_.cbegin(), g_shape_.cend(), 1, std::multiplies<int64_t>{}));
  size_t mean_num =
    static_cast<size_t>(std::accumulate(mean_shape_.cbegin(), mean_shape_.cend(), 1, std::multiplies<int64_t>{}));
  auto inv_std = std::make_unique<DATA_T[]>(mean_num);
  if (num == 0 || num > kMemMaxLen || mean_num == 0 || mean_num > kMemMaxLen) {
    MS_EXCEPTION(ValueError) << "memory allocation failed";
  }
  if (calc_inv_std<DATA_T>(input_var, inv_std.get(), mean_num) != true) {
    MS_EXCEPTION(ValueError) << "For LayerNormGradGrad, variance must be positive.";
  }
  auto x_hat = std::make_unique<DATA_T[]>(num);
  auto dy_gamma = std::make_unique<DATA_T[]>(num);
  auto sum1 = std::make_unique<DATA_T[]>(mean_num);
  std::fill_n(sum1.get(), mean_num, DATA_T(0));
  auto sum2 = std::make_unique<DATA_T[]>(mean_num);
  std::fill_n(sum2.get(), mean_num, DATA_T(0));
  auto sum3 = std::make_unique<DATA_T[]>(mean_num);
  std::fill_n(sum3.get(), mean_num, DATA_T(0));
  auto sum4 = std::make_unique<DATA_T[]>(mean_num);
  std::fill_n(sum4.get(), mean_num, DATA_T(0));
  shard_inner_mean<DATA_T>(0, num, g_num, sum1.get(), sum2.get(), sum3.get(), sum4.get(), inv_std.get(), input_d_dx,
                           input_dy, input_gamma, input_x, input_mean, x_hat.get(), dy_gamma.get());
  auto sum5 = std::make_unique<DATA_T[]>(mean_num);
  std::fill_n(sum5.get(), mean_num, DATA_T(0));
  auto sum6 = std::make_unique<DATA_T[]>(mean_num);
  std::fill_n(sum6.get(), mean_num, DATA_T(0));
  auto sum7 = std::make_unique<DATA_T[]>(mean_num);
  std::fill_n(sum7.get(), mean_num, DATA_T(0));
  auto part3 = std::make_unique<DATA_T[]>(num);
  shard_outer_mean<DATA_T>(0, num, g_num, sum2.get(), sum3.get(), sum4.get(), sum5.get(), sum6.get(), sum7.get(),
                           part3.get(), inv_std.get(), input_d_dx, input_d_dg, x_hat.get(), dy_gamma.get(), input_dy,
                           input_x, input_mean);
  shard_input_prop<DATA_T>(0, num, g_num, sum1.get(), sum2.get(), sum5.get(), sum6.get(), sum7.get(), part3.get(),
                           inv_std.get(), input_d_dx, input_gamma, input_d_dg, input_d_db, x_hat.get(), output_sopd_x,
                           output_sopd_dy);
  std::fill_n(output_sopd_g, g_num, DATA_T(0));
  shard_param_prop<DATA_T>(0, num, g_num, sum1.get(), sum2.get(), inv_std.get(), input_d_dx, x_hat.get(), input_dy,
                           output_sopd_g);
}

std::vector<KernelAttr> LayerNormGradGradCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr()
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeFloat32)
      .AddOutputAttr(kNumberTypeFloat32)
      .AddOutputAttr(kNumberTypeFloat32)
      .AddOutputAttr(kNumberTypeFloat32),
    KernelAttr()
      .AddInputAttr(kNumberTypeFloat16)
      .AddInputAttr(kNumberTypeFloat16)
      .AddInputAttr(kNumberTypeFloat16)
      .AddInputAttr(kNumberTypeFloat16)
      .AddInputAttr(kNumberTypeFloat16)
      .AddInputAttr(kNumberTypeFloat16)
      .AddInputAttr(kNumberTypeFloat16)
      .AddInputAttr(kNumberTypeFloat16)
      .AddOutputAttr(kNumberTypeFloat16)
      .AddOutputAttr(kNumberTypeFloat16)
      .AddOutputAttr(kNumberTypeFloat16),
  };

  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, LayerNormGradGrad, LayerNormGradGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
