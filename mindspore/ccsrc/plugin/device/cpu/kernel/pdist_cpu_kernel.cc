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
constexpr size_t kPdistInputDimsMin = 2;
constexpr int64_t GRAIN_SIZE = 2048;
}  // namespace

template <typename T>
void PdistZeroNormalcompute(const T *in1, const T *in2, T *output, size_t col, float p) {
  double res = 0;
  for (size_t i = 0; i < col; i++) {
    res += (in1[i] != in2[1]);
  }
  *output = static_cast<T>(res);
}

template <typename T>
void PdistInfNormalcompute(const T *in1, const T *in2, T *output, size_t col, float p) {
  double res = 0;
  for (size_t i = 0; i < col; i++) {
    double x = static_cast<double>(in1[i]);
    double y = static_cast<double>(in2[i]);
    res = std::max(std::abs(x - y), res);
  }
  *output = static_cast<T>(res);
}

template <typename T>
void PdistOneNormalcompute(const T *in1, const T *in2, T *output, size_t col, float p) {
  double res = 0;
  for (size_t i = 0; i < col; i++) {
    double x = static_cast<double>(in1[i]);
    double y = static_cast<double>(in2[i]);
    res += std::abs(x - y);
  }
  *output = static_cast<T>(res);
}

template <typename T>
void PdistTwoNormalcompute(const T *in1, const T *in2, T *output, size_t col, float p) {
  double res = 0;
  for (size_t i = 0; i < col; i++) {
    double x = static_cast<double>(in1[i]);
    double y = static_cast<double>(in2[i]);
    auto temp = x - y;
    res += temp * temp;
  }
  *output = static_cast<T>(std::sqrt(res));
}

template <typename T>
void PdistPNormalcompute(const T *in1, const T *in2, T *output, size_t col, float p) {
  double res = 0;
  for (size_t i = 0; i < col; i++) {
    double x = static_cast<double>(in1[i]);
    double y = static_cast<double>(in2[i]);
    res += std::pow(std::abs(x - y), p);
  }
  res = std::pow(res, 1.0 / p);
  *output = static_cast<T>(res);
}

bool PdistCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::Pdist>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast Pdist ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();
  p_ = kernel_ptr->get_p();
  if (inputs.size() != kPdistInputsNum || outputs.size() != kPdistOutputsNum) {
    MS_LOG(ERROR) << kernel_name_ << ": input and output size should be " << kPdistInputsNum << " and "
                  << kPdistOutputsNum << ", but get " << inputs.size() << " and " << outputs.size();
    return false;
  }
  auto input_shape = inputs[0]->GetShapeVector();
  auto input_dim_ = input_shape.size();
  h_ = input_shape[input_dim_ - kIndex2];
  w_ = input_shape[input_dim_ - kIndex1];

  auto input_dtype_ = inputs[0]->GetDtype();
  switch (input_dtype_) {
    case kNumberTypeFloat64:
      kernel_func_ = &PdistCpuKernelMod::LaunchKernel<double>;
      break;
    case kNumberTypeFloat32:
      kernel_func_ = &PdistCpuKernelMod::LaunchKernel<float>;
      break;
    case kNumberTypeFloat16:
      kernel_func_ = &PdistCpuKernelMod::LaunchKernel<float16>;
      break;
    default:
      MS_LOG(ERROR) << "Pdist kernel does not support " << TypeIdToString(input_dtype_);
      return false;
  }
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

template <typename T>
bool PdistCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                     const std::vector<kernel::AddressPtr> &outputs) {
  auto input_size = inputs[0]->size / sizeof(T);
  auto output_size = outputs[0]->size / sizeof(T);
  const auto *input_start = GetDeviceAddress<T>(inputs, kIndex0);
  const auto *input_end = input_start + input_size;
  auto *output = GetDeviceAddress<T>(outputs, kIndex0);
  int64_t combs = h_ * (h_ - 1) / 2;
  int64_t one_size = h_ * w_;
  int64_t temp = one_size - w_;
  auto task = [this, input_start, input_end, output, combs, one_size, temp](size_t start, size_t end) {
    int64_t l = start / combs;
    int64_t k = start % combs;
    double h2 = h_ - .5;
    int64_t i = static_cast<int64_t>((h2 - sqrtf(h2 * h2 - 2 * k - 1)));
    int64_t j = k - h_ * i + i * (i + 1) / 2 + i + 1;
    i = i * w_;
    j = j * w_;
    T *res = output + start;
    const T *const res_end = output + end;

    while (res != res_end) {
      const T *input_i = input_start + l * one_size + i;
      const T *input_j = input_start + l * one_size + j;
      if (p_ == 0.0) {
        PdistZeroNormalcompute(input_i, input_j, res, w_, p_);
      } else if (p_ == 1.0) {
        PdistOneNormalcompute(input_i, input_j, res, w_, p_);
      } else if (p_ == 2.0) {
        PdistTwoNormalcompute(input_i, input_j, res, w_, p_);
      } else if (std::isinf(p_)) {
        PdistInfNormalcompute(input_i, input_j, res, w_, p_);
      } else {
        PdistPNormalcompute(input_i, input_j, res, w_, p_);
      }
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
