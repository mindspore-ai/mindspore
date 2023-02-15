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

#include "plugin/device/cpu/kernel/count_nonzero_cpu_kernel.h"

#include <string>
#include <vector>
#include <complex>
#include <memory>
#include <map>
#include <algorithm>
#include <utility>
#include <numeric>

#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/device/cpu/kernel/nnacl/fp32/mul_fp32.h"

namespace mindspore {
namespace kernel {
namespace {
const size_t kCountNonZeroInputsNum = 1;
const size_t kCountNonZeroOutputsNum = 1;

std::vector<int64_t> cnz_dims;
std::vector<int64_t> cnz_transposed_shape;
int64_t cnz_stride;

using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

// Class def of ParallelIterator.
class ParallelIterator {
 public:
  ParallelIterator(const std::vector<int64_t> &transposed_shape, const std::vector<int64_t> &dims,
                   const std::vector<int64_t> &input_shape);
  ~ParallelIterator() = default;
  void Next();
  void Set(int64_t pos);
  inline int64_t Get() const { return _pos; }

 private:
  int64_t _dimension{0};
  std::vector<int64_t> _coord;
  std::vector<int64_t> _shape;
  std::vector<int64_t> _strides;
  std::vector<int64_t> _back_strides;
  std::vector<int64_t> _dims;
  int64_t _pos{0};
};

ParallelIterator::ParallelIterator(const std::vector<int64_t> &transposed_shape, const std::vector<int64_t> &dims,
                                   const std::vector<int64_t> &input_shape)
    : _dimension(transposed_shape.size()),
      _coord(transposed_shape.size(), 0),
      _shape(transposed_shape),
      _strides(transposed_shape.size(), 1),
      _back_strides(transposed_shape.size(), 1),
      _dims(dims),
      _pos(0) {
  std::vector<int64_t> strides(_dimension, 1);
  for (int64_t i = _dimension - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * input_shape[i + 1];
  }
  for (int64_t i = _dimension - 1; i >= 0; --i) {
    _strides[i] = strides[_dims[i]];
    _back_strides[i] = (_shape[i] - 1) * _strides[i];
  }
}
void ParallelIterator::Set(int64_t pos) {
  for (int64_t i = _dimension - 1; i >= 0 && pos != 0; --i) {
    _coord[i] = pos % _shape[i];
    _pos += _coord[i] * _strides[i];
    pos /= _shape[i];
  }
}
void ParallelIterator::Next() {
  for (int64_t i = _dimension - 1; i >= 0; --i) {
    if (_coord[i] + 1 == _shape[i]) {
      _coord[i] = 0;
      _pos -= _back_strides[i];
    } else {
      _coord[i]++;
      _pos += _strides[i];
      break;
    }
  }
}
}  // namespace

template <class T>
struct is_complex_t : std::false_type {};
template <class T>
struct is_complex_t<std::complex<T>> : std::true_type {};

template <class T>
int64_t IsNonZero(T val, std::true_type) {
  return val.real() != 0 || val.imag() != 0 ? static_cast<int64_t>(1) : static_cast<int64_t>(0);
}
template <class T>
int64_t IsNonZero(T val, std::false_type) {
  return val != static_cast<T>(0) ? static_cast<int64_t>(1) : static_cast<int64_t>(0);
}

void CountNonZeroCpuKernelMod::ComputeCountParameter(void) {
  int64_t input_rank = x_shape_.size();
  std::vector<int64_t> dims = dims_;

  if (dims.size() == 0) {
    for (int64_t i = 0; i < input_rank; ++i) {
      dims.push_back(i);
    }
  }
  // Check dims in [-x_rank, x_rank)
  std::for_each(dims.begin(), dims.end(), [input_rank](auto &dim) { dim = dim < 0 ? dim + input_rank : dim; });
  std::sort(dims.begin(), dims.end());
  dims.erase(std::unique(dims.begin(), dims.end()), dims.end());

  int64_t stride_ = static_cast<int64_t>(1);
  std::vector<int64_t> axes_(input_rank);
  int64_t j = static_cast<int64_t>(0), k = static_cast<int64_t>(0);
  for (int64_t i = 0; i < input_rank; i++) {
    if (j == static_cast<int64_t>(dims.size()) || i != dims[j]) {
      axes_[k] = i;
      ++k;
    } else {
      stride_ *= x_shape_[i];
      ++j;
    }
  }

  for (auto &dim : dims) {
    axes_[k] = dim;
    ++k;
  }
  // Calculate transposed_shape using axes.
  // For example, if input_shape = (3, 4, 5, 6, 7), axes = [0, 2, 4, 1, 3],
  // then transposed_shape = (3, 5, 7) + (4, 6)
  std::vector<int64_t> transposed_shape_(input_rank);
  for (int64_t i = 0; i < input_rank; ++i) {
    transposed_shape_[i] = x_shape_[axes_[i]];
  }
  // Assign values.
  cnz_stride = stride_, cnz_transposed_shape = transposed_shape_, cnz_dims = axes_;
}

bool CountNonZeroCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  dims_ = GetValue<std::vector<int64_t>>(base_operator->GetAttr("dims"));
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << kernel_name_ << " does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int CountNonZeroCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs,
                                     const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  x_shape_ = inputs[0]->GetShapeVector();
  y_shape_ = outputs[0]->GetShapeVector();
  return KRET_OK;
}

template <typename T>
bool CountNonZeroCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                            const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kCountNonZeroInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kCountNonZeroOutputsNum, kernel_name_);
  auto *x = reinterpret_cast<T *>(inputs[0]->addr);
  auto *y = reinterpret_cast<int64_t *>(outputs[0]->addr);
  auto input_shape = x_shape_;
  int64_t input_nums = static_cast<int64_t>(inputs[0]->size / sizeof(T));
  int64_t data_nums = static_cast<int64_t>(outputs[0]->size / sizeof(int64_t));

  if (y_shape_.size() == 0) {
    (void)y_shape_.insert(y_shape_.begin(), 1);
  }
  auto output_size = SizeOf(y_shape_);

  auto count_nonzero_scalar_shard = [&](int64_t start, int64_t end) {
    y[0] = static_cast<int64_t>(0);
    for (int64_t i = start; i < end; ++i) {
      y[0] += IsNonZero<T>(x[i], is_complex_t<T>{});
    }
  };

  auto count_nonzero_shard = [&](int64_t start, int64_t end) {
    ParallelIterator iter(cnz_transposed_shape, cnz_dims, input_shape);
    iter.Set(start * cnz_stride);
    for (int64_t i = start; i < end; ++i) {
      int64_t reduce_initial = static_cast<int64_t>(0);
      for (int64_t j = 0; j < cnz_stride; ++j) {
        reduce_initial += IsNonZero<T>(x[iter.Get()], is_complex_t<T>{});
        iter.Next();
      }
      y[i] = reduce_initial;
    }
  };
  if (data_nums == 1) {
    ParallelLaunchAutoSearch(count_nonzero_scalar_shard, input_nums, this, &parallel_search_info_);
  } else {
    ComputeCountParameter();
    ParallelLaunchAutoSearch(count_nonzero_shard, output_size, this, &parallel_search_info_);
  }
  return true;
}

std::vector<std::pair<KernelAttr, CountNonZeroCpuKernelMod::CountNonZeroLaunchFunc>>
  CountNonZeroCpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt64),
     &CountNonZeroCpuKernelMod::LaunchKernel<float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt64),
     &CountNonZeroCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt64),
     &CountNonZeroCpuKernelMod::LaunchKernel<double>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt64),
     &CountNonZeroCpuKernelMod::LaunchKernel<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt64),
     &CountNonZeroCpuKernelMod::LaunchKernel<int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
     &CountNonZeroCpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &CountNonZeroCpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt64),
     &CountNonZeroCpuKernelMod::LaunchKernel<uint8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt64),
     &CountNonZeroCpuKernelMod::LaunchKernel<uint16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt64),
     &CountNonZeroCpuKernelMod::LaunchKernel<uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt64),
     &CountNonZeroCpuKernelMod::LaunchKernel<uint64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeInt64),
     &CountNonZeroCpuKernelMod::LaunchKernel<complex64>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeInt64),
     &CountNonZeroCpuKernelMod::LaunchKernel<complex128>}};

std::vector<KernelAttr> CountNonZeroCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, CountNonZeroLaunchFunc> &pair) { return pair.first; });

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, CountNonZero, CountNonZeroCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
