/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/assert_cpu_kernel.h"
#include <utility>
#include <algorithm>
#include "abstract/utils.h"
namespace mindspore {
namespace kernel {
void PrintDataInt8(KernelTensor *input, int summarize) {
  std::ostringstream oss;
  oss << "input data: [";
  int8_t *data = reinterpret_cast<int8_t *>(input->device_ptr());
  MS_EXCEPTION_IF_NULL(data);
  for (int j = 0; j < summarize - 1; j++) {
    oss << static_cast<int32_t>(data[j]) << " ";
  }
  oss << static_cast<int32_t>(data[summarize - 1]) << "]";
  std::cout << oss.str() << std::endl;
  return;
}

void PrintDataUInt8(KernelTensor *input, int summarize) {
  std::ostringstream oss;
  oss << "input data: [";
  uint8_t *data = reinterpret_cast<uint8_t *>(input->device_ptr());
  MS_EXCEPTION_IF_NULL(data);
  for (int j = 0; j < summarize - 1; j++) {
    oss << static_cast<uint32_t>(data[j]) << " ";
  }
  oss << static_cast<uint32_t>(data[summarize - 1]) << "]";
  std::cout << oss.str() << std::endl;
  return;
}

template <typename T>
void PrintData(KernelTensor *input, int summarize) {
  std::ostringstream oss;
  oss << "input data: [";
  T *data = reinterpret_cast<T *>(input->device_ptr());
  MS_EXCEPTION_IF_NULL(data);
  for (int j = 0; j < summarize - 1; j++) {
    oss << data[j] << " ";
  }
  oss << data[summarize - 1] << "]";
  std::cout << oss.str() << std::endl;
  return;
}

std::map<TypeId, AssertCpuKernelMod::AssertPrintFunc> AssertCpuKernelMod::func_map_ = {
  {kNumberTypeFloat16, PrintData<float16>}, {kNumberTypeFloat32, PrintData<float>},
  {kNumberTypeFloat64, PrintData<double>},  {kNumberTypeInt8, PrintDataInt8},
  {kNumberTypeInt16, PrintData<int16_t>},   {kNumberTypeInt32, PrintData<int32_t>},
  {kNumberTypeInt64, PrintData<int64_t>},   {kNumberTypeUInt8, PrintDataUInt8},
  {kNumberTypeUInt16, PrintData<uint16_t>}, {kNumberTypeUInt32, PrintData<uint32_t>},
  {kNumberTypeUInt64, PrintData<uint64_t>}, {kNumberTypeBool, PrintData<bool>}};

bool AssertCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  summarize_ = GetValue<int64_t>(primitive_->GetAttr(ops::kSummarize));
  return true;
}

int AssertCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    return ret;
  }
  auto inputs_size = inputs.size();
  kernel_funcs_.resize(inputs_size);
  summarizes_.resize(inputs_size);
  for (size_t i = 0; i < inputs_size; i++) {
    MS_EXCEPTION_IF_NULL(inputs[i]);
    auto input_type_id = inputs[i]->dtype_id();
    auto func_iter = func_map_.find(input_type_id);
    if (func_iter == func_map_.end()) {
      MS_LOG(ERROR) << "assert kernel does not support " << TypeIdToString(input_type_id);
      return KRET_RESIZE_FAILED;
    }
    kernel_funcs_[i] = func_iter->second;
    auto element = inputs[i]->size() / abstract::TypeIdSize(input_type_id);
    summarizes_[i] = static_cast<int>(std::min(static_cast<size_t>(summarize_), element));
  }

  return ret;
}

std::vector<KernelAttr> AssertCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {KernelAttr().AddSkipCheckAttr(true)};
  return support_list;
}

bool AssertCpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                                const std::vector<KernelTensor *> &outputs) {
  auto cond = static_cast<bool *>(inputs[0]->device_ptr());
  if (*cond) {
    return true;
  }

  std::cout << "For '" << kernel_name_ << "' condition is false." << std::endl;
  for (size_t i = 1; i < inputs.size(); i++) {
    MS_EXCEPTION_IF_NULL(inputs[i]);
    kernel_funcs_[i](std::move(inputs[i]), summarizes_[i]);
  }
  MS_LOG(EXCEPTION) << "assert failed";

  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Assert, AssertCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
