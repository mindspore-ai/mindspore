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
#include "mindspore/core/ops/randperm_v2.h"

namespace mindspore {
namespace kernel {
namespace {
const uint32_t kInputNum = 3;
const uint32_t kOutputNum = 1;
constexpr size_t kOutputIndex0 = 0;
constexpr size_t kOutputShapeLen = 1;
}  // namespace

bool RandpermV2CPUKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  auto randperm_ptr = std::dynamic_pointer_cast<ops::RandpermV2>(base_operator);
  MS_EXCEPTION_IF_NULL(randperm_ptr);
  kernel_name_ = base_operator->GetPrim()->name();
  output_type_ = outputs[0]->GetDtype();
  int64_t layout_ = GetValue<int64_t>(randperm_ptr->GetAttr("layout"));
  if (seed_ < 0 && seed_ != -1) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', the attr seed must be greater than 0 or equal to 0 or -1, but got data: " << seed_
                             << ".";
  }
  if (layout_ < 0) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', the attr layout must be greater than or equal to 0, but got data: " << layout_
                             << ".";
  }
  return true;
}

int RandpermV2CPUKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs,
                                   const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
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

bool RandpermV2CPUKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                    const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
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
bool RandpermV2CPUKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                          const std::vector<AddressPtr> &outputs) {
  int64_t *n_tensor = reinterpret_cast<int64_t *>(inputs[0]->addr);
  MS_EXCEPTION_IF_NULL(n_tensor);
  n_data_ = reinterpret_cast<int64_t *>(inputs[0]->addr)[0];
  int64_t *seed_tensor = reinterpret_cast<int64_t *>(inputs[1]->addr);
  MS_EXCEPTION_IF_NULL(seed_tensor);
  seed_ = static_cast<int64_t *>(seed_tensor)[0];
  int64_t *offset_tensor = reinterpret_cast<int64_t *>(inputs[2]->addr);
  MS_EXCEPTION_IF_NULL(offset_tensor);
  offset_ = static_cast<int64_t *>(offset_tensor)[0];
  std::random_device rd;
  int64_t final_seed = (offset_ != 0) ? offset_ : (seed_ != -1) ? seed_ : rd();
  auto output = reinterpret_cast<T1 *>(outputs[0]->addr);
  size_t output_elem_num = outputs[0]->size / sizeof(T1);
  ShapeVector out_shape;
  out_shape.push_back(n_data_);
  std::vector<T1> temp;
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

bool RandpermV2CPUKernelMod::LaunchKernelFp16(const std::vector<AddressPtr> &inputs,
                                              const std::vector<AddressPtr> &outputs) {
  auto output = reinterpret_cast<Eigen::half *>(outputs[0]->addr);
  size_t output_elem_num = outputs[0]->size / sizeof(float16);
  int64_t *n_tensor = reinterpret_cast<int64_t *>(inputs[0]->addr);
  MS_EXCEPTION_IF_NULL(n_tensor);
  n_data_ = reinterpret_cast<int64_t *>(inputs[0]->addr)[0];
  int64_t *seed_tensor = reinterpret_cast<int64_t *>(inputs[1]->addr);
  MS_EXCEPTION_IF_NULL(seed_tensor);
  seed_ = static_cast<int64_t *>(seed_tensor)[0];
  int64_t *offset_tensor = reinterpret_cast<int64_t *>(inputs[2]->addr);
  MS_EXCEPTION_IF_NULL(offset_tensor);
  offset_ = static_cast<int64_t *>(offset_tensor)[0];
  std::random_device rd;
  int64_t final_seed = (offset_ != 0) ? offset_ : (seed_ != -1) ? seed_ : rd();
  ShapeVector out_shape;
  out_shape.push_back(n_data_);
  std::vector<float> temp;
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
                                            .AddInputAttr(kNumberTypeInt64)
                                            .AddInputAttr(kNumberTypeInt64)
                                            .AddInputAttr(kNumberTypeInt64)
                                            .AddOutputAttr(kNumberTypeUInt8),
                                          KernelAttr()
                                            .AddInputAttr(kNumberTypeInt64)
                                            .AddInputAttr(kNumberTypeInt64)
                                            .AddInputAttr(kNumberTypeInt64)
                                            .AddOutputAttr(kNumberTypeFloat32),
                                          KernelAttr()
                                            .AddInputAttr(kNumberTypeInt64)
                                            .AddInputAttr(kNumberTypeInt64)
                                            .AddInputAttr(kNumberTypeInt64)
                                            .AddOutputAttr(kNumberTypeFloat16),
                                          KernelAttr()
                                            .AddInputAttr(kNumberTypeInt64)
                                            .AddInputAttr(kNumberTypeInt64)
                                            .AddInputAttr(kNumberTypeInt64)
                                            .AddOutputAttr(kNumberTypeFloat64),
                                          KernelAttr()
                                            .AddInputAttr(kNumberTypeInt64)
                                            .AddInputAttr(kNumberTypeInt64)
                                            .AddInputAttr(kNumberTypeInt64)
                                            .AddOutputAttr(kNumberTypeInt8),
                                          KernelAttr()
                                            .AddInputAttr(kNumberTypeInt64)
                                            .AddInputAttr(kNumberTypeInt64)
                                            .AddInputAttr(kNumberTypeInt64)
                                            .AddOutputAttr(kNumberTypeInt16),
                                          KernelAttr()
                                            .AddInputAttr(kNumberTypeInt64)
                                            .AddInputAttr(kNumberTypeInt64)
                                            .AddInputAttr(kNumberTypeInt64)
                                            .AddOutputAttr(kNumberTypeInt32),
                                          KernelAttr()
                                            .AddInputAttr(kNumberTypeInt64)
                                            .AddInputAttr(kNumberTypeInt64)
                                            .AddInputAttr(kNumberTypeInt64)
                                            .AddOutputAttr(kNumberTypeInt64)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, RandpermV2, RandpermV2CPUKernelMod);
}  // namespace kernel
}  // namespace mindspore
