/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/correlate_cpu_kernel.h"
#include <functional>
#include <algorithm>
#include <utility>
#include <memory>
#include <complex>

namespace mindspore {
namespace kernel {
namespace {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
}  // namespace

template <typename T>
void CorrelateCpuKernelMod::CorrelatePad(T *source_array, T *padded_array, int64_t padded_array_size) {
  for (int64_t i = 0; i < padded_array_size; i++) {
    padded_array[i] = (T)0;
  }
  int start = 0;
  if (mode_type_ == mindspore::PadMode::FULL) {
    start = short_size_ - 1;
  } else if (mode_type_ == mindspore::PadMode::SAME) {
    start = short_size_ / 2;
  }
  for (int64_t i = 0; i < long_size_; i++) {
    padded_array[start + i] = source_array[i];
  }
}

bool CorrelateCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << kernel_name_ << " valid cpu kernel does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int CorrelateCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  std::vector<int64_t> a_shape = inputs[kIndex0]->GetShapeVector();
  std::vector<int64_t> v_shape = inputs[kIndex1]->GetShapeVector();
  auto mode = inputs[kIndex2]->GetValueWithCheck<int64_t>();
  mode_type_ = static_cast<mindspore::PadMode>(mode);
  int64_t a_dims = a_shape.size();
  int64_t v_dims = v_shape.size();
  if (a_dims != kIndex1 || v_dims != kIndex1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of 'a' and 'v' should be 1-D, but got 'a' at"
                      << a_dims << "-D and 'v' at " << v_dims << "-D.";
  }
  a_size_ = a_shape[0];
  v_size_ = v_shape[0];
  if (a_size_ == 0 || v_size_ == 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', input 'a' and 'v' should not be empty, but got 'a' at ("
                      << a_size_ << ") and 'v' at (" << v_dims << ").";
  }

  a_ge_v_ = a_size_ >= v_size_;
  if (a_ge_v_) {
    long_size_ = a_size_;
    short_size_ = v_size_;
  } else {
    long_size_ = v_size_;
    short_size_ = a_size_;
  }
  return KRET_OK;
}

template <typename T_in, typename T_out>
bool CorrelateCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                         const std::vector<kernel::KernelTensor *> &outputs) {
  T_in *a_array = reinterpret_cast<T_in *>(inputs[0]->device_ptr());
  T_in *v_array = reinterpret_cast<T_in *>(inputs[1]->device_ptr());
  T_out *out_array = reinterpret_cast<T_out *>(outputs[0]->device_ptr());

  // step0: cast input dtype to output dtype
  T_out *casted_a_array = static_cast<T_out *>(malloc(sizeof(T_out) * a_size_));
  if (casted_a_array == nullptr) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', malloc [casted_a_array] memory failed.";
    return false;
  }
  auto cast_a_task = [&a_array, &casted_a_array, this](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      casted_a_array[i] = static_cast<T_out>(a_array[i]);
    }
  };
  ParallelLaunchAutoSearch(cast_a_task, a_size_, this, &parallel_search_info_);

  T_out *casted_v_array = static_cast<T_out *>(malloc(sizeof(T_out) * v_size_));
  if (casted_v_array == nullptr) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', malloc [casted_v_array] memory failed.";
    return false;
  }
  auto cast_v_task = [&v_array, &casted_v_array, this](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      casted_v_array[i] = static_cast<T_out>(v_array[i]);
    }
  };
  ParallelLaunchAutoSearch(cast_v_task, v_size_, this, &parallel_search_info_);

  // step1: get padded array witch depend on mode
  int64_t out_size = long_size_ - short_size_ + 1;
  int64_t padded_long_size = long_size_;
  if (mode_type_ == mindspore::PadMode::SAME) {
    padded_long_size = long_size_ + short_size_ - 1;
    out_size = long_size_;
  } else if (mode_type_ == mindspore::PadMode::FULL) {
    padded_long_size = long_size_ + 2 * (short_size_ - 1);
    out_size = long_size_ + short_size_ - 1;
  }
  T_out *long_array = static_cast<T_out *>(malloc(sizeof(T_out) * padded_long_size));
  if (long_array == nullptr) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', malloc [long_array] memory failed.";
    return false;
  }
  T_out *short_array;
  if (a_ge_v_) {
    short_array = casted_v_array;
    CorrelatePad<T_out>(casted_a_array, long_array, padded_long_size);
  } else {
    short_array = casted_a_array;
    CorrelatePad<T_out>(casted_v_array, long_array, padded_long_size);
  }

  // step2: calculate convolution
  auto task = [&long_array, &short_array, &out_array, this](size_t start, size_t end) {
    for (size_t out_id = start; out_id < end; ++out_id) {
      T_out sum_temp = static_cast<T_out>(0);
      for (int64_t dot_id = 0; dot_id < short_size_; dot_id++) {
        sum_temp += long_array[out_id + dot_id] * short_array[dot_id];
      }
      out_array[out_id] = sum_temp;
    }
  };
  ParallelLaunchAutoSearch(task, out_size, this, &parallel_search_info_);

  // step3: if a is shorter than v, then we should reverse the result
  if (a_ge_v_ == false) {
    for (int i = 0; i < out_size / 2; i++) {
      std::swap(out_array[i], out_array[out_size - 1 - i]);
    }
  }

  free(long_array);
  free(casted_a_array);
  free(casted_v_array);

  long_array = nullptr;
  short_array = nullptr;
  casted_a_array = nullptr;
  casted_v_array = nullptr;
  return true;
}

template <typename T>
bool CorrelateCpuKernelMod::LaunchComplexKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                                const std::vector<kernel::KernelTensor *> &outputs) {
  T *a_array = reinterpret_cast<T *>(inputs[0]->device_ptr());
  T *v_array = reinterpret_cast<T *>(inputs[1]->device_ptr());
  T *out_array = reinterpret_cast<T *>(outputs[0]->device_ptr());
  MS_EXCEPTION_IF_NULL(a_array);
  MS_EXCEPTION_IF_NULL(v_array);
  MS_EXCEPTION_IF_NULL(out_array);

  // step0: get conjugate v
  T *conj_v_array = static_cast<T *>(malloc(sizeof(T) * v_size_));
  if (conj_v_array == nullptr) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', malloc [conj_v_array] memory failed.";
    return false;
  }
  auto conj_v_task = [&v_array, &conj_v_array, this](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      conj_v_array[i] = static_cast<T>(std::conj(v_array[i]));
    }
  };
  ParallelLaunchAutoSearch(conj_v_task, v_size_, this, &parallel_search_info_);

  // step1: get padded array witch depend on mode
  int64_t out_size = long_size_ - short_size_ + 1;
  int64_t padded_long_size = long_size_;
  if (mode_type_ == mindspore::PadMode::SAME) {
    padded_long_size = long_size_ + short_size_ - 1;
    out_size = long_size_;
  } else if (mode_type_ == mindspore::PadMode::FULL) {
    padded_long_size = long_size_ + 2 * (short_size_ - 1);
    out_size = long_size_ + short_size_ - 1;
  }
  T *long_array = static_cast<T *>(malloc(sizeof(T) * padded_long_size));
  if (long_array == nullptr) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', malloc [long_array] memory failed.";
    return false;
  }
  T *short_array;
  if (a_ge_v_) {
    short_array = conj_v_array;
    CorrelatePad<T>(a_array, long_array, padded_long_size);
  } else {
    short_array = a_array;
    CorrelatePad<T>(conj_v_array, long_array, padded_long_size);
  }

  // step2: calculate convolution
  auto task = [&long_array, &short_array, &out_array, this](size_t start, size_t end) {
    for (size_t out_id = start; out_id < end; ++out_id) {
      T sum_temp = static_cast<T>(0);
      for (int64_t dot_id = 0; dot_id < short_size_; dot_id++) {
        sum_temp += long_array[out_id + dot_id] * short_array[dot_id];
      }
      out_array[out_id] = sum_temp;
    }
  };
  ParallelLaunchAutoSearch(task, out_size, this, &parallel_search_info_);

  // step3: if a is shorter than v, then we should reverse the result
  if (a_ge_v_ == false) {
    for (int i = 0; i < out_size / 2; i++) {
      std::swap(out_array[i], out_array[out_size - 1 - i]);
    }
  }

  free(long_array);
  free(conj_v_array);
  long_array = nullptr;
  short_array = nullptr;
  conj_v_array = nullptr;

  return true;
}

#define CORRELATE_CPU_REG(T1, T2, T3, T4)                         \
  KernelAttr()                                                    \
    .AddInputAttr(T1)                                  /* a */    \
    .AddInputAttr(T1)                                  /* v */    \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64) /* mode */ \
    .AddOutputAttr(T2),                                           \
    &CorrelateCpuKernelMod::LaunchKernel<T3, T4>

#define CORRELATE_CPU_COMPLEX_REG(T1, T2)                         \
  KernelAttr()                                                    \
    .AddInputAttr(T1)                                  /* a */    \
    .AddInputAttr(T1)                                  /* v */    \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64) /* mode */ \
    .AddOutputAttr(T1),                                           \
    &CorrelateCpuKernelMod::LaunchComplexKernel<T2>

std::vector<std::pair<KernelAttr, CorrelateCpuKernelMod::CorrelateFunc>> CorrelateCpuKernelMod::func_list_ = {
  {CORRELATE_CPU_REG(kNumberTypeInt8, kNumberTypeFloat32, int8_t, float)},
  {CORRELATE_CPU_REG(kNumberTypeInt16, kNumberTypeFloat32, int16_t, float)},
  {CORRELATE_CPU_REG(kNumberTypeInt32, kNumberTypeFloat32, int32_t, float)},
  {CORRELATE_CPU_REG(kNumberTypeInt64, kNumberTypeFloat64, int64_t, double)},
  {CORRELATE_CPU_REG(kNumberTypeFloat16, kNumberTypeFloat16, float16, float16)},
  {CORRELATE_CPU_REG(kNumberTypeFloat32, kNumberTypeFloat32, float, float)},
  {CORRELATE_CPU_REG(kNumberTypeFloat64, kNumberTypeFloat64, double, double)},
  {CORRELATE_CPU_COMPLEX_REG(kNumberTypeComplex64, complex64)},
  {CORRELATE_CPU_COMPLEX_REG(kNumberTypeComplex128, complex128)},
};

std::vector<KernelAttr> CorrelateCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, CorrelateFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Correlate, CorrelateCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
