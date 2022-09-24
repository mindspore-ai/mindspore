/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include <cmath>
#include <algorithm>
#include "plugin/device/cpu/kernel/pdist_cpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kPdistInputsNum = 1;
constexpr size_t kPdistOutputsNum = 1;
constexpr float P_ZERO = 0.0;
constexpr float P_ONE = 1.0;
constexpr float P_TWO = 2.0;
constexpr int64_t GRAIN_SIZE = 2048;
constexpr float EPS = 1.0e-6;
}  // namespace

struct zdist_calc {
  static inline double map(const double &diff, const float &p) { return std::min(ceil(diff), 1.0); }
  static inline double red(const double &agg, const double &up) { return agg + up; }
  static inline double finish(const double &agg, const float &p) { return agg; }
};

struct odist_calc {
  static inline double map(const double &diff, const float &p) { return diff; }
  static inline double red(const double &agg, const double &up) { return agg + up; }
  static inline double finish(const double &agg, const float &p) { return agg; }
};

struct tdist_calc {
  static inline double map(const double &diff, const float &p) { return diff * diff; }
  static inline double red(const double &agg, const double &up) { return agg + up; }
  static inline double finish(const double &agg, const float &p) { return std::sqrt(agg); }
};

struct idist_calc {
  static inline double map(const double &diff, const float &p) { return diff; }
  static inline double red(const double &agg, const double &up) { return std::max(agg, up); }
  static inline double finish(const double &agg, const float &p) { return agg; }
};

struct pdist_calc {
  static inline double map(const double &diff, const float &p) { return std::pow(diff, p); }
  static inline double red(const double &agg, const double &up) { return agg + up; }
  static inline double finish(const double &agg, const float &p) { return std::pow(agg, 1.0 / p); }
};

bool PdistCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::Pdist>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast Pdist ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();
  p_ = kernel_ptr->get_p();

  auto input_shape = inputs[0]->GetShapeVector();
  auto input_dim_ = input_shape.size();
  h_ = input_shape[input_dim_ - kIndex2];
  w_ = input_shape[input_dim_ - kIndex1];
  dtype_ = inputs[0]->GetDtype();
  return true;
}

int PdistCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs,
                              const std::map<uint32_t, tensor::TensorPtr> &others) {
  if (NativeCpuKernelMod::Resize(base_operator, inputs, outputs, others) == KRET_RESIZE_FAILED) {
    MS_LOG(WARNING) << kernel_name_ << " reinit failed.";
    return KRET_RESIZE_FAILED;
  }
  return 0;
}

template <typename F, typename T>
bool PdistCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  if (h_ == 1) {
    return true;
  }
  auto output_size = outputs[0]->size / sizeof(T);
  const auto *input = GetDeviceAddress<T>(inputs, kIndex0);
  auto *output = GetDeviceAddress<T>(outputs, kIndex0);
  int64_t combs = h_ * (h_ - 1) / 2;
  int64_t one_size = h_ * w_;
  int64_t temp = one_size - w_;
  auto task = [this, input, output, combs, one_size, temp](size_t start, size_t end) {
    int64_t l = start / combs;
    int64_t k = start % combs;
    double h2 = h_ - .5;
    int64_t i = static_cast<int64_t>((h2 - std::sqrt(h2 * h2 - 2 * k - 1)));
    int64_t j = k - h_ * i + i * (i + 1) / 2 + i + 1;
    i = i * w_;
    j = j * w_;
    T *res = output + start;
    const T *const res_end = output + end;

    while (res != res_end) {
      const T *input_i = input + l * one_size + i;
      const T *input_j = input + l * one_size + j;
      double agg = 0;
      for (int64_t x = 0; x < w_; x++) {
        double a = static_cast<double>(*(input_i + x));
        double b = static_cast<double>(*(input_j + x));
        agg = F::red(agg, F::map(std::abs(a - b), p_));
      }
      *res = static_cast<T>(F::finish(agg, p_));
      res += 1;
      j += w_;
      if (j == one_size) {
        i += w_;
        j = i + w_;
        if (i == temp) {
          i = 0;
          j = w_;
          l += 1;
        }
      }
    }
  };
  ParallelLaunch(task, output_size, GRAIN_SIZE / w_, this);
  return true;
}

template <typename T>
void PdistCpuKernelMod::Apply_pdist(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  if (std::isinf(p_)) {
    LaunchKernel<idist_calc, T>(inputs, outputs);
  } else if (std::abs(p_ - P_ZERO) <= EPS * p_) {
    LaunchKernel<zdist_calc, T>(inputs, outputs);
  } else if (std::abs(p_ - P_ONE) <= EPS * p_) {
    LaunchKernel<odist_calc, T>(inputs, outputs);
  } else if (std::abs(p_ - P_TWO) <= EPS * p_) {
    LaunchKernel<tdist_calc, T>(inputs, outputs);
  } else {
    LaunchKernel<pdist_calc, T>(inputs, outputs);
  }
}

bool PdistCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                               const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kPdistInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kPdistOutputsNum, kernel_name_);
  if (dtype_ == kNumberTypeFloat64) {
    Apply_pdist<double>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    Apply_pdist<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat16) {
    Apply_pdist<float16>(inputs, outputs);
  } else {
    MS_LOG(ERROR) << "Pdist kernel does not support" << TypeIdToString(dtype_);
    return false;
  }
  return true;
}

std::vector<KernelAttr> PdistCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Pdist, PdistCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
