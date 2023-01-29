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

#include "plugin/device/cpu/kernel/index_put_cpu_kernel.h"

#include <algorithm>
#include <complex>
#include <functional>
#include <iostream>

#include "mindspore/core/ops/index_put.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kIndexPutInputsNum = 3;
constexpr size_t kIndexPutOutputsNum = 1;

#define INDEXPUT_LAUNCH_CASE(DTYPE, TYPE, DTYPE0, INPUTS, OUTPUTS) \
  case DTYPE: {                                                    \
    if ((DTYPE0) == kNumberTypeInt32) {                            \
      LaunchKernel<TYPE, int32_t>(INPUTS, OUTPUTS);                \
    } else {                                                       \
      LaunchKernel<TYPE, int64_t>(INPUTS, OUTPUTS);                \
    }                                                              \
    break;                                                         \
  }
}  // namespace

std::vector<std::vector<int64_t>> IndexPutCpuKernelMod::Transpose(const std::vector<std::vector<int64_t>> &A) {
  std::vector<std::vector<int64_t>> v;
  if (A.empty()) {
    return std::vector<std::vector<int64_t>>();
  }
  for (size_t i = 0; i < A[0].size(); ++i) {
    std::vector<int64_t> k;
    for (size_t j = 0; j < A.size(); ++j) {
      k.push_back(A[j][i]);
    }
    v.push_back(k);
  }
  return v;
}

int64_t IndexPutCpuKernelMod::Multiplicative(const std::vector<int64_t> &tensorshapes, int64_t start, int64_t end) {
  int64_t result = 1;
  for (int64_t i = start; i < end; i++) {
    result *= tensorshapes[i];
  }
  return result;
}

bool IndexPutCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::IndexPut>(base_operator);
  kernel_name_ = kernel_ptr->name();
  x1_shape_ = inputs[0]->GetShapeVector();
  auto type_id = inputs[0]->GetDtype();
  input_info_.push_back(type_id);
  x2_shape_ = inputs[1]->GetShapeVector();
  type_id = inputs[1]->GetDtype();
  input_info_.push_back(type_id);
  for (size_t i = 2; i < inputs.size(); i++) {
    indices_shape_.push_back(inputs[i]->GetShapeVector());
    type_id = inputs[i]->GetDtype();
    input_info_.push_back(type_id);
  }
  inputs_nums = inputs.size();
  accumulate = GetValue<int64_t>(base_operator->GetAttr("accumulate"));
  return true;
}

void IndexPutCpuKernelMod::CheckParams() {
  constexpr int indices_start_pos = 2;
  if (input_info_[0] != input_info_[1]) {
    MS_EXCEPTION(TypeError) << "For '" << kernel_name_
                            << "', the x1 and x2 must have the same type, but x1 "
                               "got type with "
                            << TypeIdLabel(input_info_[0]) << " and x2 got type with " << TypeIdLabel(input_info_[1])
                            << ".";
  }
  for (size_t i = indices_start_pos; i < inputs_nums; i++) {
    if (input_info_[i] != kNumberTypeInt32 && input_info_[i] != kNumberTypeInt64) {
      MS_EXCEPTION(TypeError) << "For '" << kernel_name_
                              << "', the tensors in indices should be the type of int32 or "
                                 "int64, but indices["
                              << i << "] got type with " << TypeIdLabel(input_info_[i]) << ".";
    }
  }
  if (x1_shape_.size() < indices_shape_.size()) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', rank(x1) must be greater than size(indices) but got "
                             << indices_shape_.size() << " vs " << x1_shape_.size() << ".";
  }
  if (x2_shape_.size() != 1) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', rank(x2) must be 1, but got " << x2_shape_.size() << ".";
  }
  int64_t maxnum = 0;
  for (size_t i = 0; i < indices_shape_.size(); i++) {
    if (indices_shape_[i].size() != 1) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', rank of indices[" << i << "] must be 1, but got "
                               << indices_shape_[i].size() << ".";
    }
    maxnum = (maxnum < indices_shape_[i][0]) ? indices_shape_[i][0] : maxnum;
  }
  for (size_t i = 0; i < indices_shape_.size(); i++) {
    if (indices_shape_[i][0] != 1 && indices_shape_[i][0] != maxnum) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                               << "', tensors of indices should be broadcastable, but indices[" << i << "].shape got "
                               << indices_shape_[i][0] << ".";
    }
  }
  bool x2_check = x2_shape_[0] != 1 && x2_shape_[0] != maxnum && x2_shape_[0] != x1_shape_[x1_shape_.size() - 1];
  if (x2_check) {
    MS_EXCEPTION(ValueError)
      << "For '" << kernel_name_
      << "', the size of x2 must be 1, the max size of the tensors in indices or x1.shape[-1], but got " << x2_shape_[0]
      << ".";
  }
  if (accumulate != 0 && accumulate != 1) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the values of accumulate should be 0 or 1, but got "
                             << accumulate << ".";
  }
}

template <typename T>
void IndexPutCpuKernelMod::ComputeNospecial(T *x2, size_t x2_nums, std::vector<std::vector<int64_t>> indices_value,
                                            T *y, int accumulate) {
  auto x1_shape = x1_shape_;
  size_t x1_shape_size = x1_shape.size();
  size_t idxli = indices_value.size();
  size_t idxcol = indices_value[0].size();
  if (x2_nums == 0) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', x2 input illegal, please check!";
  }
  for (size_t i = 0; i < idxli; ++i) {
    size_t offset = 0;
    for (size_t j = 0; j < idxcol; ++j) {
      offset += indices_value[i][j] * Multiplicative(x1_shape, j + 1, x1_shape_size);
    }
    size_t v_idx = i % x2_nums;
    y[offset] = (accumulate == 0) ? x2[v_idx] : y[offset] + x2[v_idx];
  }
}

template <typename T>
void IndexPutCpuKernelMod::ComputeSpecial(T *x2, size_t x2_nums, std::vector<std::vector<int64_t>> indices_value, T *y,
                                          int accumulate) {
  auto x1_shape = x1_shape_;
  size_t x1_shape_size = x1_shape.size();
  size_t idxli = indices_value.size();
  size_t idxcol = indices_value[0].size();
  size_t strides = Multiplicative(x1_shape, indices_value.size(), x1_shape_size);
  if (x2_nums == 0) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', x2 input illegal, please check!";
  }
  for (size_t i = 0; i < idxcol; i++) {
    size_t offset = 0;
    for (size_t j = 0; j < idxli; j++) {
      offset += indices_value[j][i] * Multiplicative(x1_shape, j + 1, x1_shape_size);
    }
    for (size_t j = 0; j < strides; j++) {
      size_t y_idx = offset + j;
      size_t v_idx = j % x2_nums;
      y[y_idx] = (accumulate == 0) ? x2[v_idx] : y[y_idx] + x2[v_idx];
    }
  }
}

template <typename T, typename T0>
bool IndexPutCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  auto *x1 = reinterpret_cast<T *>(inputs[0]->addr);
  auto *x2 = reinterpret_cast<T *>(inputs[1]->addr);
  auto *y = reinterpret_cast<T *>(outputs[0]->addr);
  size_t x1_nums =
    std::accumulate(x1_shape_.begin(), x1_shape_.end(), static_cast<size_t>(1), std::multiplies<size_t>());
  size_t x2_nums =
    std::accumulate(x2_shape_.begin(), x2_shape_.end(), static_cast<size_t>(1), std::multiplies<size_t>());
  constexpr size_t indices_start_pos = 2;
  std::vector<std::vector<int64_t>> indices_value;
  for (size_t i = indices_start_pos; i < inputs.size(); i++) {
    auto *linetensor = reinterpret_cast<T0 *>(inputs[i]->addr);
    std::vector<int64_t> iline;
    for (size_t j = 0; static_cast<int64_t>(j) < indices_shape_[i - indices_start_pos][0]; j++) {
      linetensor[j] = (linetensor[j] < 0) ? linetensor[j] + x1_shape_[i - indices_start_pos] : linetensor[j];
      if (linetensor[j] < 0) {
        MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', indices[" << i - indices_start_pos
                                 << "] input illegal "
                                 << ".";
      }
      if (linetensor[j] >= x1_shape_[i - indices_start_pos]) {
        MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', each element in indices[" << i
                                 << "] should be smaller than the value of x1.shape[" << i - indices_start_pos
                                 << "], but got " << linetensor[j] << " and got the value of x1.shape with "
                                 << x1_shape_[i - indices_start_pos] << ".";
      }
      iline.push_back(linetensor[j]);
    }
    indices_value.push_back(iline);
  }
  auto task = [&](size_t start, size_t end) {
    size_t length = (end - start) * sizeof(T);
    auto ret = memcpy_s(y + start, length, x1 + start, length);
    if (ret != 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memcpy_s error. Error no: " << ret << ".";
    }
  };
  ParallelLaunchAutoSearch(task, x1_nums, this, &parallel_search_info_);
  size_t maxl = 0;
  for (size_t i = 0; i < indices_value.size(); i++) {
    if (indices_value[i].size() > maxl) {
      maxl = indices_value[i].size();
    }
  }
  for (size_t i = 0; i < indices_value.size(); i++) {
    while (indices_value[i].size() != maxl) {
      indices_value[i].push_back(indices_value[i][0]);
    }
  }
  if (indices_value.size() == x1_shape_.size()) {
    std::vector<std::vector<int64_t>> rindices_value = Transpose(indices_value);
    (void)ComputeNospecial<T>(x2, x2_nums, rindices_value, y, accumulate);
  } else {
    (void)ComputeSpecial<T>(x2, x2_nums, indices_value, y, accumulate);
  }
  return true;
}

bool IndexPutCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                  const std::vector<AddressPtr> &outputs) {
  constexpr int indices_start_pos = 2;
  CheckParams();
  TypeId input_type = input_info_[0];
  TypeId indices_type = input_info_[indices_start_pos];
  switch (input_type) {
    INDEXPUT_LAUNCH_CASE(kNumberTypeFloat16, float16, indices_type, inputs, outputs)
    INDEXPUT_LAUNCH_CASE(kNumberTypeFloat32, float, indices_type, inputs, outputs)
    INDEXPUT_LAUNCH_CASE(kNumberTypeFloat64, double, indices_type, inputs, outputs)
    INDEXPUT_LAUNCH_CASE(kNumberTypeInt32, int32_t, indices_type, inputs, outputs)
    INDEXPUT_LAUNCH_CASE(kNumberTypeUInt8, uint8_t, indices_type, inputs, outputs)
    INDEXPUT_LAUNCH_CASE(kNumberTypeInt16, int16_t, indices_type, inputs, outputs)
    INDEXPUT_LAUNCH_CASE(kNumberTypeInt8, int8_t, indices_type, inputs, outputs)
    INDEXPUT_LAUNCH_CASE(kNumberTypeComplex64, std::complex<float>, indices_type, inputs, outputs)
    INDEXPUT_LAUNCH_CASE(kNumberTypeInt64, int64_t, indices_type, inputs, outputs)
    INDEXPUT_LAUNCH_CASE(kNumberTypeUInt16, uint16_t, indices_type, inputs, outputs)
    INDEXPUT_LAUNCH_CASE(kNumberTypeComplex128, std::complex<double>, indices_type, inputs, outputs)
    INDEXPUT_LAUNCH_CASE(kNumberTypeUInt32, uint32_t, indices_type, inputs, outputs)
    INDEXPUT_LAUNCH_CASE(kNumberTypeUInt64, uint64_t, indices_type, inputs, outputs)
    default:
      MS_LOG(EXCEPTION) << kernel_name_ << " does not support this kernel data type: " << TypeIdLabel(input_type)
                        << ".";
  }
  return true;
}

const std::vector<std::pair<KernelAttr, IndexPutCpuKernelMod::KernelRunFunc>> &IndexPutCpuKernelMod::GetFuncList()
  const {
  static const std::vector<std::pair<KernelAttr, IndexPutCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddSkipCheckAttr(true), &IndexPutCpuKernelMod::Launch},
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, IndexPut, IndexPutCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
