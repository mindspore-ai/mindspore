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

#include "plugin/device/cpu/kernel/randperm_v2_cpu_kernel.h"
#include <random>
#include <map>
#include <vector>
#include <algorithm>
#include "Eigen/Core"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
const uint32_t kInputNum = 4;
const uint32_t kOutputNum = 1;
constexpr size_t kIndex0 = 0;
constexpr size_t kIndex1 = 1;
constexpr size_t kIndex2 = 2;
constexpr size_t kOutputIndex0 = 0;
constexpr size_t kOutputShapeLen = 1;
}  // namespace

bool RandpermV2CPUKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  MS_EXCEPTION_IF_NULL(outputs[kOutputIndex0]);
  output_type_ = outputs[kOutputIndex0]->dtype_id();
  if (seed_ < 0 && seed_ != -1) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', the attr seed must be greater than 0 or equal to 0 or -1, but got data: " << seed_
                             << ".";
  }
  return true;
}

int RandpermV2CPUKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  //  output
  output_shape_ = outputs[kOutputIndex0]->GetShapeVector();
  auto output_shape_len = output_shape_.size();
  if (output_shape_len != kOutputShapeLen) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', output is " << output_shape_len << ", but RandpermV2 supports only "
                  << kOutputShapeLen << "-D for output tensor.";
    return KRET_RESIZE_FAILED;
  }
  return KRET_OK;
}

bool RandpermV2CPUKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &workspace,
                                    const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  MS_EXCEPTION_IF_NULL(inputs[kIndex0]);
  MS_EXCEPTION_IF_NULL(inputs[kIndex1]);
  MS_EXCEPTION_IF_NULL(inputs[kIndex2]);
  MS_EXCEPTION_IF_NULL(outputs[kIndex0]);
  if (output_type_ == kNumberTypeInt32) {
    (void)LaunchKernel<int32_t>(inputs, outputs);
  } else if (output_type_ == kNumberTypeInt64) {
    (void)LaunchKernel<int64_t>(inputs, outputs);
  } else if (output_type_ == kNumberTypeInt16) {
    (void)LaunchKernel<int16_t>(inputs, outputs);
  } else if (output_type_ == kNumberTypeInt8) {
    (void)LaunchKernel<int8_t>(inputs, outputs);
  } else if (output_type_ == kNumberTypeUInt8) {
    (void)LaunchKernel<uint8_t>(inputs, outputs);
  } else if (output_type_ == kNumberTypeFloat32) {
    (void)LaunchKernel<float>(inputs, outputs);
  } else if (output_type_ == kNumberTypeFloat64) {
    (void)LaunchKernel<double>(inputs, outputs);
  } else if (output_type_ == kNumberTypeFloat16) {
    (void)LaunchKernelFp16(inputs, outputs);
  } else {
    MS_EXCEPTION(TypeError)
      << "For '" << kernel_name_
      << "', the output data type must be one of uint8, int8, int16, int32, int64, float, float16, double.";
  }
  return true;
}

template <typename T1>
bool RandpermV2CPUKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &outputs) {
  auto output = reinterpret_cast<T1 *>(outputs[kIndex0]->device_ptr());
  n_data_ = inputs[kIndex0]->template GetValueWithCheck<int64_t>();
  seed_ = inputs[kIndex1]->template GetValueWithCheck<int64_t>();
  offset_ = inputs[kIndex2]->template GetValueWithCheck<int64_t>();

  std::vector<T1> temp;
  std::random_device rd;
  size_t output_elem_num = outputs[kIndex0]->size() / sizeof(T1);
  int64_t final_seed = (offset_ != 0) ? offset_ : (seed_ != -1) ? seed_ : rd();
  std::mt19937 gen(final_seed);

  for (auto i = 0; i < static_cast<T1>(n_data_); i++) {
    temp.push_back(i);
  }
  shuffle(temp.begin(), temp.end(), gen);
  for (size_t j = 0; j < output_elem_num; j++) {
    *(output + j) = temp[j];
  }
  return true;
}

bool RandpermV2CPUKernelMod::LaunchKernelFp16(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &outputs) {
  auto output = reinterpret_cast<Eigen::half *>(outputs[kIndex0]->device_ptr());
  n_data_ = inputs[kIndex0]->template GetValueWithCheck<int64_t>();
  seed_ = inputs[kIndex1]->template GetValueWithCheck<int64_t>();
  offset_ = inputs[kIndex2]->template GetValueWithCheck<int64_t>();

  std::vector<float> temp;
  std::random_device rd;
  size_t output_elem_num = outputs[kIndex0]->size() / sizeof(float16);
  int64_t final_seed = (offset_ != 0) ? offset_ : (seed_ != -1) ? seed_ : rd();
  std::mt19937 gen(final_seed);

  for (auto i = 0; i < static_cast<float>(n_data_); i++) {
    temp.push_back(i);
  }
  shuffle(temp.begin(), temp.end(), gen);
  for (size_t j = 0; j < output_elem_num; j++) {
    *(output + j) = static_cast<Eigen::half>(temp[j]);
  }
  return true;
}

std::vector<KernelAttr> RandpermV2CPUKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list = {KernelAttr()
                                            .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                            .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                            .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                            .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                            .AddOutputAttr(kNumberTypeUInt8),
                                          KernelAttr()
                                            .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                            .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                            .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                            .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                            .AddOutputAttr(kNumberTypeFloat32),
                                          KernelAttr()
                                            .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                            .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                            .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                            .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                            .AddOutputAttr(kNumberTypeFloat16),
                                          KernelAttr()
                                            .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                            .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                            .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                            .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                            .AddOutputAttr(kNumberTypeFloat64),
                                          KernelAttr()
                                            .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                            .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                            .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                            .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                            .AddOutputAttr(kNumberTypeInt8),
                                          KernelAttr()
                                            .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                            .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                            .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                            .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                            .AddOutputAttr(kNumberTypeInt16),
                                          KernelAttr()
                                            .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                            .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                            .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                            .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                            .AddOutputAttr(kNumberTypeInt32),
                                          KernelAttr()
                                            .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                            .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                            .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                            .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                            .AddOutputAttr(kNumberTypeInt64)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, RandpermV2, RandpermV2CPUKernelMod);
}  // namespace kernel
}  // namespace mindspore
