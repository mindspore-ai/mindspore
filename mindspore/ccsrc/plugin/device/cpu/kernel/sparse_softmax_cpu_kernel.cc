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

#include "plugin/device/cpu/kernel/sparse_softmax_cpu_kernel.h"
#include <algorithm>
#include <stack>
#include <memory>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kIndicesShapeSize = 2;
constexpr size_t kValuesShapeSize = 1;
constexpr size_t kShapeShapeSize = 1;
constexpr size_t kSparseSoftmaxInputsNum = 3;
constexpr size_t kSparseSoftmaxOutputsNum = 1;
constexpr size_t kShapeMinSize = 2;
constexpr size_t kinput_indices = 0;
constexpr size_t kinput_values = 1;
constexpr size_t kinput_shape = 2;
constexpr size_t kIndex0 = 0;
constexpr size_t kIndex1 = 1;
constexpr size_t kIndex2 = 2;

template <typename I>
inline bool CompareIndices(const I *a, const I *b, const size_t &len) {
  size_t i = 0;
  while (i < len) {
    if (a[i] != b[i]) {
      return a[i] > b[i];
    }
    ++i;
  }
  return true;
}

template <typename I, typename T>
inline void CopyIndicesAndValue(I *dst_indices_addr, T *dst_values_addr, const I *src_indices_addr,
                                const T *src_values_addr, const size_t &indices_size) {
  memcpy_s(dst_indices_addr, indices_size, src_indices_addr, indices_size);
  *dst_values_addr = *src_values_addr;
}

template <typename I, typename T>
inline int64_t Partition(I *__restrict indices_addr, T *__restrict values_addr, I *__restrict tmp_indices,
                         const size_t &indices_len, const int64_t &left, const int64_t &right) {
  int64_t i = left, j = right;
  T tmp_values = 0;
  const size_t indices_size = indices_len * sizeof(I);
#define INDICES_OFFSET_ADDR(addr, index, len) addr + index *len

  CopyIndicesAndValue(tmp_indices, &tmp_values, INDICES_OFFSET_ADDR(indices_addr, left, indices_len),
                      values_addr + left, indices_size);
  while (i < j) {
    while (i < j && CompareIndices(INDICES_OFFSET_ADDR(indices_addr, j, indices_len), tmp_indices, indices_len)) {
      --j;
    }
    CopyIndicesAndValue(INDICES_OFFSET_ADDR(indices_addr, i, indices_len), values_addr + i,
                        INDICES_OFFSET_ADDR(indices_addr, j, indices_len), values_addr + j, indices_size);
    while (i < j && !CompareIndices(INDICES_OFFSET_ADDR(indices_addr, i, indices_len), tmp_indices, indices_len)) {
      ++i;
    }
    CopyIndicesAndValue(INDICES_OFFSET_ADDR(indices_addr, j, indices_len), values_addr + j,
                        INDICES_OFFSET_ADDR(indices_addr, i, indices_len), values_addr + i, indices_size);
  }
  CopyIndicesAndValue(INDICES_OFFSET_ADDR(indices_addr, i, indices_len), values_addr + i, tmp_indices, &tmp_values,
                      indices_size);
  return i;
}
}  // namespace

template <typename I, typename T>
void SparseSoftmaxCpuKernelMod::QuickSortIndicesAndValues(I *__restrict indices_addr, T *__restrict values_addr,
                                                          const int64_t &left, const int64_t &right) {
  std::stack<int64_t> index_stk;
  index_stk.emplace(right);
  index_stk.emplace(left);
  const size_t indices_len = shape_size_;
  I *indices_buff = new I[indices_len];

  while (!index_stk.empty()) {
    int64_t i = index_stk.top();
    index_stk.pop();
    int64_t j = index_stk.top();
    index_stk.pop();
    if (i < j) {
      int64_t k = Partition(indices_addr, values_addr, indices_buff, indices_len, i, j);
      if (k > i) {
        index_stk.emplace(k - 1);
        index_stk.emplace(i);
      }
      if (j > k) {
        index_stk.emplace(j);
        index_stk.emplace(k + 1);
      }
    }
  }
  delete[] indices_buff;
}

bool SparseSoftmaxCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  return true;
}

int SparseSoftmaxCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs,
                                      const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto indices_shape = inputs.at(kIndex0)->GetShapeVector();
  auto values_shape = inputs.at(kIndex1)->GetShapeVector();
  values_size_ = LongToSize(values_shape[0]);
  auto shape_shape = inputs.at(kIndex2)->GetShapeVector();
  shape_size_ = LongToSize(shape_shape[0]);
  auto output_shape = outputs.at(kIndex0)->GetShapeVector();
  output_shape_ = Convert2SizeT(output_shape);
  if (indices_shape.size() != kIndicesShapeSize) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', it requires 'indices' should be a " << kIndicesShapeSize
                             << "-D Tensor, but got " << indices_shape.size() << "-D";
  }
  if (values_shape.size() != kValuesShapeSize || values_shape[0] != indices_shape[0]) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', it requires 'values' should be a 1-D Tensor and "
                                "the first dimension length should be equal to the first dimension length of "
                                "'indices', but got 'values' shape: "
                             << Vector2Str(values_shape) << " and 'indices' shape: " << Vector2Str(indices_shape);
  }
  if (shape_shape.size() != kShapeShapeSize || LongToSize(shape_shape[0]) < kShapeMinSize ||
      shape_shape[0] != indices_shape[1]) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', it requires 'shape' should be 1-D and more than 2 element, the element "
                                "should be equal to the second dimension length of 'indices', but "
                                "got 'shape' shape: "
                             << Vector2Str(shape_shape) << " and 'indices' shape: " << Vector2Str(indices_shape);
  }
  return KRET_OK;
}

template <typename I, typename T>
bool SparseSoftmaxCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                             const std::vector<kernel::AddressPtr> &,
                                             const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSparseSoftmaxInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSparseSoftmaxOutputsNum, kernel_name_);
  if (outputs[0]->size == 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', output memory size should be greater than 0, but got 0.";
  }
  auto ret = memset_s(outputs[0]->addr, outputs[0]->size, 0, outputs[0]->size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset output failed. Error no: " << ret;
  }
  auto *indices_addr = static_cast<I *>(inputs[kIndex0]->addr);
  auto *values_addr = static_cast<T *>(inputs[kIndex1]->addr);
  auto *output_addr = static_cast<T *>(outputs[kIndex0]->addr);
  const size_t indices_length = inputs[kIndex0]->size / sizeof(I);
  const size_t values_length = inputs[kIndex1]->size / sizeof(T);
  std::vector<T> exp_values;
  std::vector<size_t> index_values;
  std::vector<size_t> visited;

  QuickSortIndicesAndValues(indices_addr, values_addr, 0, SizeToLong(values_size_) - 1);

  for (size_t i = 0; i < values_size_; ++i) {
    visited.push_back(0);
  }
  T exp_sum = static_cast<T>(0);
  int equal_judge = 0;
  for (size_t i = 0; i < values_size_; ++i) {
    if (visited[i] == 1) {
      continue;
    }
    if (i >= values_length) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the size of 'values' "
                           "should be the same size as values memory length '"
                        << values_length << "'but got '" << i << "'.";
    }
    for (size_t j = i; j < values_size_; j++) {
      for (size_t k = 0; k < shape_size_ - 1; k++) {
        if (i * shape_size_ + k >= indices_length || j * shape_size_ + k >= indices_length) {
          MS_LOG(EXCEPTION) << "For '" << kernel_name_
                            << "', the size of 'indices' "
                               "should be the same size as indices memory length '"
                            << indices_length << "'but got '" << i << "'.";
        }
        size_t index_i = i * shape_size_ + k;
        size_t index_j = j * shape_size_ + k;
        if (indices_addr[index_i] == indices_addr[index_j]) {
          equal_judge = 1;
        } else {
          equal_judge = 0;
          break;
        }
      }
      if (equal_judge == 1) {
        visited[j] = 1;
        exp_values.push_back(exp(values_addr[j]));
        index_values.push_back(j);
      }
      equal_judge = 0;
    }
    for (size_t p = 0; p < exp_values.size(); ++p) {
      exp_sum += exp_values[p];
    }
    for (size_t q = 0; q < exp_values.size(); ++q) {
      output_addr[index_values[q]] = exp_values[q] / exp_sum;
    }
    exp_sum = 0;
    std::vector<T>().swap(exp_values);
    std::vector<size_t>().swap(index_values);
  }
  return true;
}
const std::vector<std::pair<KernelAttr, SparseSoftmaxCpuKernelMod::KernelRunFunc>>
  &SparseSoftmaxCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, SparseSoftmaxCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32),
     &SparseSoftmaxCpuKernelMod::LaunchKernel<int64_t, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat64),
     &SparseSoftmaxCpuKernelMod::LaunchKernel<int64_t, double>},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseSoftmax, SparseSoftmaxCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
