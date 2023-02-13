/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/cumsum_cpu_kernel.h"
#include "mindspore/core/ops/cumsum.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kCumSumInputsNum = 2;
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

template <typename T>
inline void LeftMove(const T *input, T *output, size_t dim0, size_t dim1, size_t dim2, size_t stride, size_t stride2,
                     size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    size_t k1 = i / dim2 % dim0;
    size_t k2 = i % dim2;
    size_t offset = k1 * stride + k2;
    for (size_t j = 0; j < dim1; ++j) {
      size_t read_index = j * stride2 + offset;
      if (j == 0) {
        output[read_index] = (T)0;
      } else {
        size_t read_index2 = (j - 1) * stride2 + offset;
        output[read_index] = input[read_index2];
      }
    }
  }
}

template <typename T>
inline void RightMove(const T *input, T *output, size_t dim0, size_t dim1, size_t dim2, size_t stride, size_t stride2,
                      size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    size_t k1 = i / dim2 % dim0;
    size_t k2 = i % dim2;
    size_t offset = k1 * stride + k2;
    for (int j = SizeToInt(dim1 - 1); j >= 0; --j) {
      size_t read_index = j * stride2 + offset;
      if (j == SizeToInt(dim1 - 1)) {
        output[read_index] = (T)0;
      } else {
        size_t read_index2 = (j + 1) * stride2 + offset;
        output[read_index] = input[read_index2];
      }
    }
  }
}

template <typename T>
inline void Copy(T *input, const T *output, size_t dim0, size_t dim1, size_t dim2, size_t stride, size_t stride2,
                 size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    size_t k1 = i / dim2 % dim0;
    size_t k2 = i % dim2;
    size_t offset = k1 * stride + k2;
    for (size_t j = 0; j < dim1; ++j) {
      size_t read_index = j * stride2 + offset;
      input[read_index] = output[read_index];
    }
  }
}

template <typename T>
inline void CumSumKernelReverse(const T *input, T *output, size_t dim0, size_t dim1, size_t dim2, size_t stride,
                                size_t stride2, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    size_t k1 = i / dim2 % dim0;
    size_t k2 = i % dim2;
    size_t offset = k1 * stride + k2;
    for (int j = SizeToInt(dim1 - 1); j >= 0; --j) {
      size_t read_index = j * stride2 + offset;
      if (j == SizeToInt(dim1 - 1)) {
        output[read_index] = input[read_index];
      } else {
        size_t read_index2 = (j + 1) * stride2 + offset;
        output[read_index] = output[read_index2] + input[read_index];
      }
    }
  }
}

template <typename T>
inline void CumSumKernel(const T *input, T *output, size_t dim0, size_t dim1, size_t dim2, size_t stride,
                         size_t stride2, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    size_t k1 = i / dim2 % dim0;
    size_t k2 = i % dim2;
    size_t offset = k1 * stride + k2;
    for (size_t j = 0; j < dim1; ++j) {
      size_t read_index = j * stride2 + offset;
      if (j == 0) {
        output[read_index] = input[read_index];
      } else {
        size_t read_index2 = (j - 1) * stride2 + offset;
        output[read_index] = output[read_index2] + input[read_index];
      }
    }
  }
}
}  // namespace

bool CumSumCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->GetPrim()->name();
  is_dynamic_shape_ = inputs[kIndex0]->IsDynamicShape();
  auto input_num = inputs.size();
  if (input_num != kCumSumInputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of inputs must be 2 or 3, but got " << input_num;
    return false;
  }

  auto dims_shape = inputs[kIndex0]->GetShapeVector();
  if (dims_shape.size() == 0) {
    MS_LOG(ERROR) << "Invalid input tensor shape: " << dims_shape.size();
    return false;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int CumSumCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs,
                               const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  auto shape = inputs.at(kIndex0)->GetShapeVector();
  shape_.clear();
  (void)std::transform(shape.begin(), shape.end(), std::back_inserter(shape_), LongToSize);
  auto kernel_ptr = std::make_shared<ops::CumSum>(base_operator->GetPrim());
  MS_EXCEPTION_IF_NULL(kernel_ptr);
  exclusive_ = kernel_ptr->get_exclusive();
  reverse_ = kernel_ptr->get_reverse();
  workspace_size_list_.push_back(input_size_list_.at(kIndex0));
  return KRET_OK;
}

void CumSumCpuKernelMod::Reshape() {
  int shape_size = SizeToInt(shape_.size());
  if (axis_ < -shape_size || axis_ >= shape_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the `axis` should be in [" << -shape_size << ", " << shape_size
                      << "), but got " << axis_;
  }
  axis_ = (axis_ < 0) ? axis_ + SizeToInt(shape_.size()) : axis_;
  dims_[kIndex0] = 1;
  dims_[kIndex1] = shape_[IntToSize(axis_)];
  dims_[kIndex2] = 1;
  for (size_t i = 0; i < IntToSize(axis_); i++) {
    dims_[kIndex0] *= shape_[i];
  }
  for (size_t i = IntToSize(axis_) + 1; i < shape_.size(); i++) {
    dims_[kIndex2] *= shape_[i];
  }
  stride_ = dims_[kIndex1] * dims_[kIndex2];
  stride2_ = dims_[kIndex2];
}

template <typename T>
bool CumSumCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                      const std::vector<AddressPtr> &outputs) {
  const auto *input = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  auto *ws = reinterpret_cast<T *>(workspace[kIndex0]->addr);
  auto output = reinterpret_cast<T *>(outputs[kIndex0]->addr);
  auto any = [](auto... args) -> bool { return ((args == nullptr) || ...); };
  if (any(input, ws, output)) {
    return false;
  }
  const auto &axis_addr = inputs.at(kIndex1);
  MS_EXCEPTION_IF_NULL(axis_addr);
  if (axis_addr->size == sizeof(int)) {
    axis_ = *reinterpret_cast<int *>(axis_addr->addr);
  } else if (axis_addr->size == sizeof(int64_t)) {
    axis_ = static_cast<int>(*reinterpret_cast<int64_t *>(axis_addr->addr));
  } else {
    MS_LOG(ERROR) << "The dtype of 'axis' should be int or int64";
    return false;
  }
  Reshape();
  if (dims_[kIndex1] == 0) {
    MS_LOG(ERROR) << "Invalid zero value. Please check resize input data.";
    return false;
  }

  // multithreading
  size_t lens = inputs[kIndex0]->size > 0 ? static_cast<size_t>(inputs[kIndex0]->size / sizeof(T)) : 1;
  auto task = [this, &input, &output, &ws](size_t start, size_t end) {
    start = start / dims_[kIndex1];
    end = end / dims_[kIndex1];
    if (exclusive_) {
      if (reverse_) {
        RightMove(input, output, dims_[kIndex0], dims_[kIndex1], dims_[kIndex2], stride_, stride2_, start, end);
        Copy(ws, output, dims_[kIndex0], dims_[kIndex1], dims_[kIndex2], stride_, stride2_, start, end);
        CumSumKernelReverse(ws, output, dims_[kIndex0], dims_[kIndex1], dims_[kIndex2], stride_, stride2_, start, end);
      } else {
        LeftMove(input, output, dims_[kIndex0], dims_[kIndex1], dims_[kIndex2], stride_, stride2_, start, end);
        Copy(ws, output, dims_[kIndex0], dims_[kIndex1], dims_[kIndex2], stride_, stride2_, start, end);
        CumSumKernel(ws, output, dims_[kIndex0], dims_[kIndex1], dims_[kIndex2], stride_, stride2_, start, end);
      }
    } else {
      if (reverse_) {
        CumSumKernelReverse(input, output, dims_[kIndex0], dims_[kIndex1], dims_[kIndex2], stride_, stride2_, start,
                            end);
      } else {
        CumSumKernel(input, output, dims_[kIndex0], dims_[kIndex1], dims_[kIndex2], stride_, stride2_, start, end);
      }
    }
  };
  ParallelLaunchAutoSearch(task, lens, this, &parallel_search_info_);
  return true;
}

std::vector<std::pair<KernelAttr, CumSumCpuKernelMod::CumSumLaunchFunc>> CumSumCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt8),
   &CumSumCpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt16),
   &CumSumCpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
   &CumSumCpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &CumSumCpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt8),
   &CumSumCpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt16),
   &CumSumCpuKernelMod::LaunchKernel<uint16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt32),
   &CumSumCpuKernelMod::LaunchKernel<uint32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt64),
   &CumSumCpuKernelMod::LaunchKernel<uint64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
   &CumSumCpuKernelMod::LaunchKernel<float16>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
   &CumSumCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
   &CumSumCpuKernelMod::LaunchKernel<double>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeComplex64),
   &CumSumCpuKernelMod::LaunchKernel<std::complex<float>>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeComplex128),
   &CumSumCpuKernelMod::LaunchKernel<std::complex<double>>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8),
   &CumSumCpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16),
   &CumSumCpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &CumSumCpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
   &CumSumCpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
   &CumSumCpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt16),
   &CumSumCpuKernelMod::LaunchKernel<uint16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt32),
   &CumSumCpuKernelMod::LaunchKernel<uint32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt64),
   &CumSumCpuKernelMod::LaunchKernel<uint64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
   &CumSumCpuKernelMod::LaunchKernel<float16>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
   &CumSumCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
   &CumSumCpuKernelMod::LaunchKernel<double>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeComplex64),
   &CumSumCpuKernelMod::LaunchKernel<std::complex<float>>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeComplex128),
   &CumSumCpuKernelMod::LaunchKernel<std::complex<double>>},
};

std::vector<KernelAttr> CumSumCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, CumSumCpuKernelMod::CumSumLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, CumSum, CumSumCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
