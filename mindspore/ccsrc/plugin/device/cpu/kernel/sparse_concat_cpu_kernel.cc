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
std::pair<bool, size_t> SparseConcatCpuKernelMod::Match2OutputAttrFromNode(
  const KernelAttr &kernel_attr, const std::vector<KernelAttr> &kernel_attr_list) {
  for (size_t index = 0; index < kernel_attr_list.size(); ++index) {
    const auto &cur_kernel_attr = kernel_attr_list[index];
    if ((kernel_attr.GetOutputAttr(0).first == cur_kernel_attr.GetOutputAttr(0).first) &&
        (kernel_attr.GetOutputAttr(1).first == cur_kernel_attr.GetOutputAttr(1).first)) {
      return std::make_pair(true, index);
    }
  }
  return std::make_pair(false, 0);
}

bool SparseConcatCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  auto prim = base_operator->GetPrim();
  MS_EXCEPTION_IF_NULL(prim);
  concat_dim_ = GetValue<int64_t>(prim->GetAttr(kConcatDim));
  kernel_name_ = base_operator->name();
  input_num_ = inputs.size();
  size_t min_input_mun = 6;
  size_t nocoo_input_num = 0;
  if (((input_num_ % kCOOElementNum) != nocoo_input_num) && (input_num_ < min_input_mun)) {
    MS_LOG(EXCEPTION) << "For op " << kernel_name_ << ". The input number is " << input_num_
                      << " but must be bigger than 4 and the number must be 3X+1(each COO have 3 input).";
  }
  size_t output_num = outputs.size();
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kSparseConcatOutputNum, kernel_name_);

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = Match2OutputAttrFromNode(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "For op " << kernel_name_
                      << "SparseConcat does not support this kernel data type: " << kernel_attr
                      << "support kernel input type and format: " << GetOpSupport();
  }
  kernel_func_ = func_list_[index].second;
  return true;
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

template <typename T, typename S>
bool SparseConcatCpuKernelMod::SparseConcat(const std::vector<kernel::AddressPtr> &inputs,
                                            const std::vector<kernel::AddressPtr> &,
                                            const std::vector<kernel::AddressPtr> &outputs, const size_t shape_size,
                                            const int size) {
  auto output_indices = reinterpret_cast<T *>(outputs[kOutputIndicesStart]->addr);
  auto output_values = reinterpret_cast<S *>(outputs[kOutputValuesStart]->addr);
  auto output_shape = reinterpret_cast<int64_t *>(outputs[kOutputShapesStart]->addr);
  auto input_coo_num = input_num_ / kCOOTensorNum;
  const auto &first_shape_ptr = reinterpret_cast<int64_t *>(inputs[kSpInputShapesStart * input_coo_num]->addr);
  std::map<size_t, int64_t> dim_position_map = {};
  int shape_cnt = 0;
  std::vector<T> in_indices = {};
  std::vector<S> in_values = {};
  if (shape_size == 0) {
    MS_LOG(EXCEPTION) << "For op " << kernel_name_
                      << "The input COO sparse tensor shape dims size is 0, should be bigger than 0.";
  }

  for (unsigned int i = 0; i < input_coo_num; i++) {
    const auto &indices_ptr = reinterpret_cast<T *>(inputs[kSpInputIndicesStart * input_coo_num + i]->addr);
    const auto &values_ptr = reinterpret_cast<S *>(inputs[kSpInputValuesStart * input_coo_num + i]->addr);
    const auto &shape_ptr = reinterpret_cast<int64_t *>(inputs[kOutputShapesStart * input_coo_num + i]->addr);
    auto cur_axis_shape = *(shape_ptr + concat_dim_);
    for (unsigned int j = 0; j < inputs[kSpInputIndicesStart * input_coo_num + i]->size / sizeof(T); j++) {
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

template <typename T, typename S>
bool SparseConcatCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                            const std::vector<kernel::AddressPtr> &workspace,
                                            const std::vector<kernel::AddressPtr> &outputs) {
  size_t size = input_num_ / 3;
  const auto &shape = reinterpret_cast<int64_t *>(inputs[kSpInputShapesStart * size]->addr);
  size_t shape_size = inputs[kSpInputShapesStart * size]->size / sizeof(int64_t);
  if ((concat_dim_ < (static_cast<int64_t>(shape_size) * (-1))) || (concat_dim_ >= static_cast<int64_t>(shape_size))) {
    MS_LOG(EXCEPTION) << "For op " << kernel_name_ << "Input concat_dim is error, concat_dim is " << concat_dim_
                      << " but COO tensor shape dim size is " << shape_size << " concat_dim value must be in range -"
                      << shape_size << " to " << (shape_size - 1) << ".";
  }
  concat_dim_ = (concat_dim_ < 0) ? (concat_dim_ + shape_size) : concat_dim_;
  for (unsigned int i = 0; i < size; i++) {
    const auto &temp_shape = reinterpret_cast<int64_t *>(inputs[kSpInputShapesStart * size + i]->addr);
    if (shape_size != inputs[kSpInputShapesStart * size + i]->size / sizeof(int64_t)) {
      MS_LOG(EXCEPTION) << "For op " << kernel_name_ << "The input COO sparse tensor shape dims is "
                        << inputs[kSpInputShapesStart * size + i]->size / sizeof(int64_t)
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
  SparseConcat<T, S>(inputs, workspace, outputs, shape_size, size);
  return true;
}

#define ADD_INPUT_ATTR(ms_index_type, ms_value_type, ms_shape_type) \
  .AddInputAttr(ms_index_type)                                      \
    .AddInputAttr(ms_index_type)                                    \
    .AddInputAttr(ms_value_type)                                    \
    .AddInputAttr(ms_value_type)                                    \
    .AddInputAttr(ms_shape_type)                                    \
    .AddInputAttr(ms_shape_type)

#define CPU_SPARSE_CONCAT_KERNEL_REGISTER(ms_index_type, ms_value_type, ms_shape_type, index_type, value_type) \
  {                                                                                                            \
    KernelAttr() ADD_INPUT_ATTR(ms_index_type, ms_value_type, ms_shape_type)                                   \
      .AddOutputAttr(ms_index_type)                                                                            \
      .AddOutputAttr(ms_value_type)                                                                            \
      .AddOutputAttr(ms_shape_type),                                                                           \
      &SparseConcatCpuKernelMod::LaunchKernel<index_type, value_type>                                          \
  }

std::vector<std::pair<KernelAttr, SparseConcatCpuKernelMod::SparseConcatFunc>> SparseConcatCpuKernelMod::func_list_ = {
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeInt8, kNumberTypeInt64, int64_t, int8_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeUInt8, kNumberTypeInt64, int64_t, uint8_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeInt16, kNumberTypeInt64, int64_t, int16_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeUInt16, kNumberTypeInt64, int64_t, uint16_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt64, int64_t, int32_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeUInt32, kNumberTypeInt64, int64_t, uint32_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt64, int64_t, int64_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeUInt64, kNumberTypeInt64, int64_t, uint64_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeFloat32, kNumberTypeInt64, int64_t, float),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeFloat16, kNumberTypeInt64, int64_t, float16),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt16, kNumberTypeInt8, kNumberTypeInt64, int16_t, int8_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt16, kNumberTypeUInt8, kNumberTypeInt64, int16_t, uint8_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt16, kNumberTypeInt16, kNumberTypeInt64, int16_t, int16_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt16, kNumberTypeUInt16, kNumberTypeInt64, int16_t, uint16_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt16, kNumberTypeInt32, kNumberTypeInt64, int16_t, int32_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt16, kNumberTypeUInt32, kNumberTypeInt64, int16_t, uint32_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt16, kNumberTypeInt64, kNumberTypeInt64, int16_t, int64_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt16, kNumberTypeUInt64, kNumberTypeInt64, int16_t, uint64_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt16, kNumberTypeFloat32, kNumberTypeInt64, int16_t, float),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt16, kNumberTypeFloat16, kNumberTypeInt64, int16_t, float16),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt32, kNumberTypeInt8, kNumberTypeInt64, int32_t, int8_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt32, kNumberTypeUInt8, kNumberTypeInt64, int32_t, uint8_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt32, kNumberTypeInt16, kNumberTypeInt64, int32_t, int16_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt32, kNumberTypeUInt16, kNumberTypeInt64, int32_t, uint16_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt64, int32_t, int32_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt32, kNumberTypeUInt32, kNumberTypeInt64, int32_t, uint32_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt64, int32_t, int64_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt32, kNumberTypeUInt64, kNumberTypeInt64, int32_t, uint64_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt32, kNumberTypeFloat32, kNumberTypeInt64, int32_t, float),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt32, kNumberTypeFloat16, kNumberTypeInt64, int32_t, float16),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeInt8, kNumberTypeInt32, int64_t, int8_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeUInt8, kNumberTypeInt32, int64_t, uint8_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeInt16, kNumberTypeInt32, int64_t, int16_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeUInt16, kNumberTypeInt32, int64_t, uint16_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt32, int64_t, int32_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeUInt32, kNumberTypeInt32, int64_t, uint32_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt32, int64_t, int64_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeUInt64, kNumberTypeInt32, int64_t, uint64_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeFloat32, kNumberTypeInt32, int64_t, float),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeFloat16, kNumberTypeInt32, int64_t, float16),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt16, kNumberTypeInt8, kNumberTypeInt32, int16_t, int8_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt16, kNumberTypeUInt8, kNumberTypeInt32, int16_t, uint8_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt16, kNumberTypeInt16, kNumberTypeInt32, int16_t, int16_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt16, kNumberTypeUInt16, kNumberTypeInt32, int16_t, uint16_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt16, kNumberTypeInt32, kNumberTypeInt32, int16_t, int32_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt16, kNumberTypeUInt32, kNumberTypeInt32, int16_t, uint32_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt16, kNumberTypeInt64, kNumberTypeInt32, int16_t, int64_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt16, kNumberTypeUInt64, kNumberTypeInt32, int16_t, uint64_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt16, kNumberTypeFloat32, kNumberTypeInt32, int16_t, float),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt16, kNumberTypeFloat16, kNumberTypeInt32, int16_t, float16),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt32, kNumberTypeInt8, kNumberTypeInt32, int32_t, int8_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt32, kNumberTypeUInt8, kNumberTypeInt32, int32_t, uint8_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt32, kNumberTypeInt16, kNumberTypeInt32, int32_t, int16_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt32, kNumberTypeUInt16, kNumberTypeInt32, int32_t, uint16_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, int32_t, int32_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt32, kNumberTypeUInt32, kNumberTypeInt32, int32_t, uint32_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt32, int32_t, int64_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt32, kNumberTypeUInt64, kNumberTypeInt32, int32_t, uint64_t),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt32, kNumberTypeFloat32, kNumberTypeInt32, int32_t, float),
  CPU_SPARSE_CONCAT_KERNEL_REGISTER(kNumberTypeInt32, kNumberTypeFloat16, kNumberTypeInt32, int32_t, float16),
};

std::vector<KernelAttr> SparseConcatCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)transform(func_list_.begin(), func_list_.end(), back_inserter(support_list),
                  [](const std::pair<KernelAttr, SparseConcatFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseConcat, SparseConcatCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
