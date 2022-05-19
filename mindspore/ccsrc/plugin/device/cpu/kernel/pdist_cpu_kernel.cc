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
#include <functional>
#include <map>
#include <algorithm>
#include "plugin/device/cpu/kernel/pdist_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/pdist.h"
#include "abstract/utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kPdistInputsNum = 1;
constexpr size_t kPdistOutputsNum = 1;
}  // namespace

template <typename T>
void PdistZeroNormalcompute(const T *input, T *output, size_t start_x, size_t start_y, float p, size_t col,
                            size_t idx) {
  double res = 0;
  for (size_t i = 0; i < col; i++) {
    res += (input[start_x + i] == input[start_y + i]) ? 0 : 1;
  }
  output[idx] = static_cast<T>(res);
}

template <typename T>
void PdistInfNormalcompute(const T *input, T *output, size_t start_x, size_t start_y, float p, size_t col, size_t idx) {
  double res = 0;
  for (size_t i = 0; i < col; i++) {
    double x = static_cast<double>(input[start_x + i]);
    double y = static_cast<double>(input[start_y + i]);
    res = std::max(std::abs(x - y), res);
  }
  output[idx] = static_cast<T>(res);
}

template <typename T>
void PdistOneNormalcompute(const T *input, T *output, size_t start_x, size_t start_y, float p, size_t col, size_t idx) {
  double res = 0;
  for (size_t i = 0; i < col; i++) {
    double x = static_cast<double>(input[start_x + i]);
    double y = static_cast<double>(input[start_y + i]);
    res += std::abs(x - y);
  }
  output[idx] = static_cast<T>(res);
}

template <typename T>
void PdistNormalcompute(const T *input, T *output, size_t start_x, size_t start_y, float p, size_t col, size_t idx) {
  double res = 0;
  for (size_t i = 0; i < col; i++) {
    double x = static_cast<double>(input[start_x + i]);
    double y = static_cast<double>(input[start_y + i]);
    res += std::pow(std::abs(x - y), p);
  }
  res = std::pow(res, 1.0 / p);
  output[idx] = static_cast<T>(res);
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
  (void)std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(input_shape_), LongToSize);
  input_dim_ = input_shape_.size();
  input_size_ = std::accumulate(input_shape_.begin(), input_shape_.end(), 1, std::multiplies<size_t>());
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
  const auto *input = reinterpret_cast<T *>(inputs[0]->addr);
  auto output = reinterpret_cast<T *>(outputs[0]->addr);
  auto col = input_shape_[input_dim_ - 1];
  auto temp = input_shape_[input_dim_ - 1] * input_shape_[input_dim_ - 2];
  auto task = [this, &input, &output, col, temp](size_t start, size_t end) {
    size_t idx = 0;
    for (size_t i = start; i < end; i = i + temp) {
      for (size_t j = i; j < i + temp; j = j + col) {
        for (size_t k = j + col; k < i + temp; k = k + col) {
          if (p_ == 0.0) {
            PdistZeroNormalcompute(input, output, j, k, p_, col, idx);
          } else if (std::isinf(p_)) {
            PdistInfNormalcompute(input, output, j, k, p_, col, idx);
          } else if (p_ == 1.0) {
            PdistOneNormalcompute(input, output, j, k, p_, col, idx);
          } else {
            PdistNormalcompute(input, output, j, k, p_, col, idx);
          }
          idx++;
        }
      }
    }
  };
  ParallelLaunchAutoSearch(task, input_size_, this, &parallel_search_info_, pool_);
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
