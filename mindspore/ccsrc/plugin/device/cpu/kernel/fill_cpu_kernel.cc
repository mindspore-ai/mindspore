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

#include <algorithm>
#include <memory>
#include <utility>
#include <complex>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/fill.h"
#include "plugin/device/cpu/kernel/fill_cpu_kernel.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kFillInputsNum = 2;
constexpr size_t kFillOutputsNum = 1;
}  // namespace

#define FILL_CPU_REG(MS_T, MS_U, MS_V, T) \
  { KernelAttr().AddInputAttr(MS_T).AddInputAttr(MS_U).AddOutputAttr(MS_V), &FillCpuKernelMod::LaunchKernel<T> }

bool FillCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                            const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  input0_dtype_ = inputs[kIndex0]->GetDtype();

  auto tensor_attr = GetKernelAttrFromTensors(inputs, outputs);
  x_type_id_ = tensor_attr.GetInputAttr(kIndex1).dtype;

  auto kernel_ptr = std::make_shared<ops::Fill>(base_operator->GetPrim());
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' "
                  << "cast Fill ops failed!";
    return false;
  }
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int FillCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs,
                             const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  return KRET_OK;
}

template <typename T>
bool FillCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                    const std::vector<kernel::AddressPtr> &workspace,
                                    const std::vector<kernel::AddressPtr> &outputs) {
  T value_data(0);
  if (x_type_id_ == kNumberTypeInt8) {
    value_data = static_cast<T>(*reinterpret_cast<int8_t *>(inputs[kIndex1]->addr));
  } else if (x_type_id_ == kNumberTypeInt16) {
    value_data = static_cast<T>(*reinterpret_cast<int16_t *>(inputs[kIndex1]->addr));
  } else if (x_type_id_ == kNumberTypeInt32) {
    value_data = static_cast<T>(*reinterpret_cast<int32_t *>(inputs[kIndex1]->addr));
  } else if (x_type_id_ == kNumberTypeInt64) {
    value_data = static_cast<T>(*reinterpret_cast<int64_t *>(inputs[kIndex1]->addr));
  } else if (x_type_id_ == kNumberTypeUInt8) {
    value_data = static_cast<T>(*reinterpret_cast<uint8_t *>(inputs[kIndex1]->addr));
  } else if (x_type_id_ == kNumberTypeUInt16) {
    value_data = static_cast<T>(*reinterpret_cast<uint16_t *>(inputs[kIndex1]->addr));
  } else if (x_type_id_ == kNumberTypeUInt32) {
    value_data = static_cast<T>(*reinterpret_cast<uint32_t *>(inputs[kIndex1]->addr));
  } else if (x_type_id_ == kNumberTypeUInt64) {
    value_data = static_cast<T>(*reinterpret_cast<uint64_t *>(inputs[kIndex1]->addr));
  } else if (x_type_id_ == kNumberTypeFloat16) {
    value_data = static_cast<T>(static_cast<float>(*reinterpret_cast<float16 *>(inputs[kIndex1]->addr)));
  } else if (x_type_id_ == kNumberTypeFloat32) {
    value_data = static_cast<T>(*reinterpret_cast<float *>(inputs[kIndex1]->addr));
  } else if (x_type_id_ == kNumberTypeFloat64) {
    value_data = static_cast<T>(*reinterpret_cast<double *>(inputs[kIndex1]->addr));
  } else if (x_type_id_ == kNumberTypeBool) {
    value_data = static_cast<T>(*reinterpret_cast<bool *>(inputs[kIndex1]->addr));
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' "
                  << "cannot convert datatype between complex and real number!";
  }
  const auto output = outputs[kIndex0];
  auto *output_data = reinterpret_cast<T *>(output->addr);
  size_t lens = static_cast<size_t>(output->size / sizeof(T));
  auto task = [output_data, value_data](const size_t start, const size_t end) {
    for (size_t i = start; i < end; i++) {
      output_data[i] = value_data;
    }
  };
  ParallelLaunchAutoSearch(task, lens, this, &parallel_search_info_);
  return true;
}

template <>
bool FillCpuKernelMod::LaunchKernel<std::complex<float>>(const std::vector<kernel::AddressPtr> &inputs,
                                                         const std::vector<kernel::AddressPtr> &workspace,
                                                         const std::vector<kernel::AddressPtr> &outputs) {
  std::complex<float> value_data{0, 0};
  if (x_type_id_ == kNumberTypeComplex64) {
    value_data = *reinterpret_cast<std::complex<float> *>(inputs[kIndex1]->addr);
  } else if (x_type_id_ == kNumberTypeComplex128) {
    value_data = static_cast<std::complex<float>>(*reinterpret_cast<std::complex<double> *>(inputs[kIndex1]->addr));
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' "
                  << "cannot convert datatype between complex and real number!";
  }
  const auto output = outputs[kIndex0];
  auto *output_data = reinterpret_cast<std::complex<float> *>(output->addr);
  size_t lens = static_cast<size_t>(output->size / sizeof(std::complex<float>));
  auto task = [output_data, value_data](const size_t start, const size_t end) {
    for (size_t i = start; i < end; i++) {
      output_data[i] = value_data;
    }
  };
  ParallelLaunchAutoSearch(task, lens, this, &parallel_search_info_);
  return true;
}

template <>
bool FillCpuKernelMod::LaunchKernel<std::complex<double>>(const std::vector<kernel::AddressPtr> &inputs,
                                                          const std::vector<kernel::AddressPtr> &workspace,
                                                          const std::vector<kernel::AddressPtr> &outputs) {
  std::complex<double> value_data{0, 0};
  if (x_type_id_ == kNumberTypeComplex64) {
    value_data = static_cast<std::complex<double>>(*reinterpret_cast<std::complex<float> *>(inputs[kIndex1]->addr));
  } else if (x_type_id_ == kNumberTypeComplex128) {
    value_data = *reinterpret_cast<std::complex<double> *>(inputs[kIndex1]->addr);
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' "
                  << "cannot convert datatype between complex and real number!";
  }
  const auto output = outputs[kIndex0];
  auto *output_data = reinterpret_cast<std::complex<double> *>(output->addr);
  size_t lens = static_cast<size_t>(output->size / sizeof(std::complex<double>));
  auto task = [output_data, value_data](const size_t start, const size_t end) {
    for (size_t i = start; i < end; i++) {
      output_data[i] = value_data;
    }
  };
  ParallelLaunchAutoSearch(task, lens, this, &parallel_search_info_);
  return true;
}

const std::vector<std::pair<KernelAttr, FillCpuKernelMod::KernelRunFunc>> &FillCpuKernelMod::GetFuncList() const {
  static std::vector<std::pair<KernelAttr, FillCpuKernelMod::KernelRunFunc>> func_list;
  std::vector<TypeId> shape_type_list = {kNumberTypeInt32, kNumberTypeInt64};
  std::vector<TypeId> value_type_list = {
    kNumberTypeInt8,    kNumberTypeInt16,   kNumberTypeInt32,     kNumberTypeInt64,     kNumberTypeFloat16,
    kNumberTypeFloat32, kNumberTypeFloat64, kNumberTypeUInt8,     kNumberTypeUInt16,    kNumberTypeUInt32,
    kNumberTypeUInt64,  kNumberTypeBool,    kNumberTypeComplex64, kNumberTypeComplex128};

  if (func_list.empty()) {
    std::pair<KernelAttr, FillCpuKernelMod::KernelRunFunc> type_pair;
    for (auto i : shape_type_list) {
      for (auto j : value_type_list) {
        for (auto k : value_type_list) {
          if (k == kNumberTypeInt8) {
            type_pair = FILL_CPU_REG(i, j, k, int8_t);
          } else if (k == kNumberTypeInt16) {
            type_pair = FILL_CPU_REG(i, j, k, int16_t);
          } else if (k == kNumberTypeInt32) {
            type_pair = FILL_CPU_REG(i, j, k, int32_t);
          } else if (k == kNumberTypeInt64) {
            type_pair = FILL_CPU_REG(i, j, k, int64_t);
          } else if (k == kNumberTypeFloat16) {
            type_pair = FILL_CPU_REG(i, j, k, float16);
          } else if (k == kNumberTypeFloat32) {
            type_pair = FILL_CPU_REG(i, j, k, float);
          } else if (k == kNumberTypeFloat64) {
            type_pair = FILL_CPU_REG(i, j, k, double);
          } else if (k == kNumberTypeUInt8) {
            type_pair = FILL_CPU_REG(i, j, k, uint8_t);
          } else if (k == kNumberTypeUInt16) {
            type_pair = FILL_CPU_REG(i, j, k, uint16_t);
          } else if (k == kNumberTypeUInt32) {
            type_pair = FILL_CPU_REG(i, j, k, uint32_t);
          } else if (k == kNumberTypeUInt64) {
            type_pair = FILL_CPU_REG(i, j, k, uint64_t);
          } else if (k == kNumberTypeBool) {
            type_pair = FILL_CPU_REG(i, j, k, bool);
          } else if (k == kNumberTypeComplex64) {
            type_pair = FILL_CPU_REG(i, j, k, std::complex<float>);
          } else if (k == kNumberTypeComplex128) {
            type_pair = FILL_CPU_REG(i, j, k, std::complex<double>);
          }
          func_list.emplace_back(type_pair);
        }
      }
    }
  }
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Fill, FillCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
