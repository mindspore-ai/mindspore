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
#include <cstdio>
#include <vector>
#include <complex>
#include <map>
#include "utils/ms_utils.h"
#include "plugin/device/cpu/kernel/sparse_concat_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kCOOTensorNum = 3;
constexpr size_t kSparseConcatOutputNum = 3;
constexpr size_t kAxis = 0;
constexpr size_t kSpInputIndicesStart = 0;
constexpr size_t kSpInputValuesStart = 1;
constexpr size_t kSpInputShapesStart = 2;
constexpr size_t kOutputIndicesStart = 0;
constexpr size_t kOutputValuesStart = 1;
constexpr size_t kOutputShapesStart = 2;
constexpr size_t kCOOElementNum = 3;
constexpr auto kConcatDim = "concat_dim";
}  // namespace

bool SparseConcatCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  auto prim = base_operator->GetPrim();
  MS_EXCEPTION_IF_NULL(prim);
  concat_dim_ = GetValue<int64_t>(prim->GetAttr(kConcatDim));
  kernel_name_ = base_operator->name();
  input_num_ = inputs.size();
  size_t N = input_num_ / kCOOTensorNum;
  values_dtype_ = inputs[N]->GetDtype();
  shapes_dtype_ = inputs[N * kSpInputShapesStart]->GetDtype();
  size_t min_input_mun = 6;
  size_t nocoo_input_num = 0;
  if (((input_num_ % kCOOElementNum) != nocoo_input_num) && (input_num_ < min_input_mun)) {
    MS_LOG(EXCEPTION) << "For op " << kernel_name_ << ". The input number is " << input_num_
                      << " but must be bigger than 4 and the number must be 3X+1(each COO have 3 input).";
  }
  size_t output_num = outputs.size();
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kSparseConcatOutputNum, kernel_name_);
  return true;
}

bool SparseConcatCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                      const std::vector<AddressPtr> &outputs) {
  switch (values_dtype_) {
    case kNumberTypeInt8:
      return LaunchFunc<int8_t>(inputs, workspace, outputs);
    case kNumberTypeUInt8:
      return LaunchFunc<uint8_t>(inputs, workspace, outputs);
    case kNumberTypeInt16:
      return LaunchFunc<int16_t>(inputs, workspace, outputs);
    case kNumberTypeUInt16:
      return LaunchFunc<uint16_t>(inputs, workspace, outputs);
    case kNumberTypeInt32:
      return LaunchFunc<int32_t>(inputs, workspace, outputs);
    case kNumberTypeUInt32:
      return LaunchFunc<uint32_t>(inputs, workspace, outputs);
    case kNumberTypeInt64:
      return LaunchFunc<int64_t>(inputs, workspace, outputs);
    case kNumberTypeUInt64:
      return LaunchFunc<uint64_t>(inputs, workspace, outputs);
    case kNumberTypeFloat16:
      return LaunchFunc<float16>(inputs, workspace, outputs);
    case kNumberTypeFloat32:
      return LaunchFunc<float>(inputs, workspace, outputs);
    case kNumberTypeBool:
      return LaunchFunc<bool>(inputs, workspace, outputs);
    case kNumberTypeFloat64:
      return LaunchFunc<double>(inputs, workspace, outputs);
    case kNumberTypeComplex64:
      return LaunchFunc<std::complex<float>>(inputs, workspace, outputs);
    case kNumberTypeComplex128:
      return LaunchFunc<std::complex<double>>(inputs, workspace, outputs);
    default:
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of input "
                        << TypeIdToType(values_dtype_)->ToString() << " not support.";
  }
  return false;
}

template <typename S>
bool SparseConcatCpuKernelMod::LaunchFunc(const std::vector<AddressPtr> &inputs,
                                          const std::vector<AddressPtr> &workspace,
                                          const std::vector<AddressPtr> &outputs) {
  switch (shapes_dtype_) {
    case kNumberTypeInt32:
      return LaunchKernel<S, int32_t>(inputs, workspace, outputs);
    case kNumberTypeInt64:
      return LaunchKernel<S, int64_t>(inputs, workspace, outputs);
    default:
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of COOTensor.shape "
                        << TypeIdToType(shapes_dtype_)->ToString() << " not support.";
  }
  return false;
}

int SparseConcatCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs,
                                     const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  return NativeCpuKernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
}

struct VmpByValue {
  bool operator()(const std::pair<size_t, int64_t> &lhs, const std::pair<size_t, int64_t> &rhs) {
    return lhs.second < rhs.second;
  }
};

template <typename S, typename T>
bool SparseConcatCpuKernelMod::SparseConcat(const std::vector<kernel::AddressPtr> &inputs,
                                            const std::vector<kernel::AddressPtr> &,
                                            const std::vector<kernel::AddressPtr> &outputs, const size_t shape_size) {
  auto output_indices = static_cast<int64_t *>(outputs[kOutputIndicesStart]->addr);
  auto output_values = static_cast<S *>(outputs[kOutputValuesStart]->addr);
  auto output_shape = static_cast<int64_t *>(outputs[kOutputShapesStart]->addr);
  auto input_coo_num = input_num_ / kCOOTensorNum;
  const auto &first_shape_ptr = reinterpret_cast<T *>(inputs[kSpInputShapesStart * input_coo_num]->addr);
  std::map<size_t, int64_t> dim_position_map = {};
  int shape_cnt = 0;
  std::vector<int64_t> in_indices = {};
  std::vector<S> in_values = {};
  if (shape_size == 0) {
    MS_LOG(EXCEPTION) << "For op " << kernel_name_
                      << "The input COO sparse tensor shape dims size is 0, should be bigger than 0.";
  }

  for (unsigned int i = 0; i < input_coo_num; i++) {
    const auto &indices_ptr = static_cast<int64_t *>(inputs[kSpInputIndicesStart * input_coo_num + i]->addr);
    const auto &values_ptr = static_cast<S *>(inputs[kSpInputValuesStart * input_coo_num + i]->addr);
    const auto &shape_ptr = static_cast<T *>(inputs[kOutputShapesStart * input_coo_num + i]->addr);
    auto cur_axis_shape = *(shape_ptr + concat_dim_);
    for (unsigned int j = 0; j < inputs[kSpInputIndicesStart * input_coo_num + i]->size / sizeof(int64_t); j++) {
      if (static_cast<int>(j % shape_size) == concat_dim_) {
        in_indices.push_back(*(indices_ptr + j) + shape_cnt);
      } else {
        in_indices.push_back(*(indices_ptr + j));
      }
    }
    for (unsigned int j = 0; j < inputs[kSpInputValuesStart * input_coo_num + i]->size / sizeof(S); j++) {
      in_values.push_back(*(values_ptr + j));
    }
    shape_cnt += cur_axis_shape;
  }

  for (size_t i = 0; i < shape_size; i++) {
    if (static_cast<int>(i) == concat_dim_) {
      output_shape[i] = shape_cnt;
    } else {
      output_shape[i] = first_shape_ptr[i];
    }
  }

  std::vector<int64_t> shape_sizes = {};
  int64_t low_shape_size = 1;
  // shape_sizes value: 1，shape[-1], shape[-1]*shape[-2],...,shape[-1]*...*shape[1]
  shape_sizes.push_back(low_shape_size);
  for (int i = shape_size - 1; i > 0; i--) {
    low_shape_size *= output_shape[i];
    shape_sizes.push_back(low_shape_size);
  }
  for (unsigned int i = 0; i < in_values.size(); i++) {
    int64_t dim_position = 0;
    for (size_t j = 0; j < shape_size; j++) {
      dim_position += in_indices[i * shape_size + j] * shape_sizes[shape_size - 1 - j];
    }
    dim_position_map.insert({i, dim_position});
  }
  std::vector<std::pair<int, int64_t>> dims_vec(dim_position_map.begin(), dim_position_map.end());
  sort(dims_vec.begin(), dims_vec.end(), VmpByValue());
  for (unsigned int i = 0; i < dims_vec.size(); i++) {
    auto out_number = dims_vec[i].first;
    for (size_t j = 0; j < shape_size; j++) {
      output_indices[i * shape_size + j] = in_indices[out_number * shape_size + j];
    }
    output_values[i] = in_values[out_number];
  }
  return true;
}

template <typename S, typename T>
bool SparseConcatCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                            const std::vector<kernel::AddressPtr> &workspace,
                                            const std::vector<kernel::AddressPtr> &outputs) {
  size_t size = input_num_ / 3;
  const auto &shape = reinterpret_cast<T *>(inputs[kSpInputShapesStart * size]->addr);
  size_t shape_size = inputs[kSpInputShapesStart * size]->size / sizeof(T);
  if ((concat_dim_ < (static_cast<int64_t>(shape_size) * (-1))) || (concat_dim_ >= static_cast<int64_t>(shape_size))) {
    MS_LOG(EXCEPTION) << "For op " << kernel_name_ << "Input concat_dim is error, concat_dim is " << concat_dim_
                      << " but COO tensor shape dim size is " << shape_size << " concat_dim value must be in range -"
                      << shape_size << " to " << (shape_size - 1) << ".";
  }
  concat_dim_ = (concat_dim_ < 0) ? (concat_dim_ + shape_size) : concat_dim_;
  for (unsigned int i = 0; i < size; i++) {
    const auto &temp_shape = reinterpret_cast<T *>(inputs[kSpInputShapesStart * size + i]->addr);
    if (shape_size != inputs[kSpInputShapesStart * size + i]->size / sizeof(T)) {
      MS_LOG(EXCEPTION) << "For op " << kernel_name_ << "The input COO sparse tensor shape dims is "
                        << inputs[kSpInputShapesStart * size + i]->size / sizeof(T)
                        << " is not equal with the first COO sparse tensor dims: " << shape_size << ".";
    }
    for (unsigned int j = 0; j < shape_size; j++) {
      if ((j != concat_dim_) && (shape[j] != temp_shape[j])) {
        MS_LOG(EXCEPTION) << "For op " << kernel_name_ << "The No." << i
                          << " input COO tensor shape size is incorrect. The No." << j << " shape is " << temp_shape[j]
                          << " not equal with first COO tensor shape: " << shape[j] << ".";
      }
    }
  }
  (void)SparseConcat<S, T>(inputs, workspace, outputs, shape_size);
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseConcat, SparseConcatCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
