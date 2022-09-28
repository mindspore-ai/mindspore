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
#include "plugin/device/cpu/kernel/pdist_grad_cpu_kernel.h"
#include <cmath>
#include <functional>
#include <map>
#include <algorithm>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "abstract/utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kPdistGradInputsNum = 3;
constexpr size_t kPdistGradOutputsNum = 1;
}  // namespace

template <typename T>
static inline T sign(T val) {
  return val > T{0.f} ? T{1.f} : T{-1.f};
}

template <typename T>
static inline T abs(T val) {
  return static_cast<T>(std::abs(static_cast<float>(val)));
}

template <typename T>
static inline T pow(T val, float p) {
  return static_cast<T>(std::pow(static_cast<float>(val), p));
}

template <typename T>
static inline T PdistOneNormalcompute(T diff, T grad, T dist, float p) {
  return grad * sign(diff);
}

template <typename T>
static inline T PdistInfNormalcompute(T diff, T grad, T dist, float p) {
  bool is_equal = false;
  if constexpr (std::is_same_v<T, double>) {
    is_equal = common::IsDoubleEqual(dist, abs(diff));
  } else if constexpr (std::is_same_v<T, float>) {
    is_equal = common::IsFloatEqual(dist, abs(diff));
  } else if constexpr (std::is_same_v<T, float16>) {
    is_equal = (dist == abs(diff));
  }
  if (is_equal) {
    return sign(diff) * grad;
  } else {
    return T{0.f};
  }
}

template <typename T>
static inline T PdistNormalcompute(T diff, T grad, T dist, float p) {
  bool is_equal = false;
  if constexpr (std::is_same_v<T, double>) {
    is_equal = common::IsDoubleEqual(dist, 0.0);
  } else if constexpr (std::is_same_v<T, float>) {
    is_equal = common::IsFloatEqual(dist, 0.0f);
  } else if constexpr (std::is_same_v<T, float16>) {
    is_equal = (dist == T{0.f});
  }
  if (is_equal) {
    return T{0.f};
  } else {
    return sign(diff) * pow(abs(diff), p - 1) * grad / pow(dist, p - 1);
  }
}

bool PdistGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto kernel_ptr = std::dynamic_pointer_cast<ops::PdistGrad>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "': cast PdistGrad ops failed!";
    return false;
  }
  p_ = kernel_ptr->get_p();
  if (inputs.size() != kPdistGradInputsNum || outputs.size() != kPdistGradOutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "': input and output size should be " << kPdistGradInputsNum << " and "
                  << kPdistGradOutputsNum << ", but get " << inputs.size() << " and " << outputs.size();
    return false;
  }
  auto x_shape = inputs[1]->GetShapeVector();
  (void)std::transform(x_shape.begin(), x_shape.end(), std::back_inserter(x_shape_), LongToSize);
  x_dim_ = static_cast<int64_t>(x_shape_.size());
  x_size_ = std::accumulate(x_shape_.begin(), x_shape_.end(), 1, std::multiplies<size_t>());
  auto x_dtype_ = inputs[1]->GetDtype();
  switch (x_dtype_) {
    case kNumberTypeFloat32:
      kernel_func_ = &PdistGradCpuKernelMod::LaunchKernel<float>;
      break;
    case kNumberTypeFloat16:
      kernel_func_ = &PdistGradCpuKernelMod::LaunchKernel<float16>;
      break;
    default:
      MS_LOG(ERROR) << "For '" << kernel_name_ << "': can not support the data type " << TypeIdToString(x_dtype_);
      return false;
  }
  return true;
}

int PdistGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &others) {
  if (NativeCpuKernelMod::Resize(base_operator, inputs, outputs, others) == KRET_RESIZE_FAILED) {
    MS_LOG(WARNING) << kernel_name_ << " reinit failed.";
    return KRET_RESIZE_FAILED;
  }
  return 0;
}

template <typename T>
bool PdistGradCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                         const std::vector<kernel::AddressPtr> &outputs) {
  T *grad = static_cast<T *>(inputs[0]->addr);
  T *x = static_cast<T *>(inputs[1]->addr);
  T *dist = static_cast<T *>(inputs[2]->addr);
  T *y = static_cast<T *>(outputs[0]->addr);
  memset(y, 0, x_size_ * sizeof(T));
  std::function<T(T diff, T grad, T dist, float p)> dist_func_ = PdistNormalcompute<T>;
  if (common::IsFloatEqual(p_, 0.0f)) {
    return true;
  } else if (common::IsFloatEqual(p_, 1.0f)) {
    dist_func_ = PdistOneNormalcompute<T>;
  } else if (std::isinf(p_)) {
    dist_func_ = PdistInfNormalcompute<T>;
  }
  int64_t col = x_shape_[x_dim_ - 1];
  int64_t temp = x_shape_[x_dim_ - 1] * x_shape_[x_dim_ - 2];
  auto task = [this, &grad, &x, &dist, &y, col, temp, &dist_func_](int64_t start, int64_t end) {
    for (int64_t m = start; m < end; m++) {
      int64_t index = 0;
      for (int64_t i = m; i < temp; i += col) {
        for (int64_t j = i + col; j < temp; j += col) {
          T diff = x[i] - x[j];

          bool is_equal = false;
          if constexpr (std::is_same_v<T, double>) {
            is_equal = common::IsDoubleEqual(diff, 0.0f);
          } else if constexpr (std::is_same_v<T, float>) {
            is_equal = common::IsFloatEqual(diff, 0.0f);
          } else if constexpr (std::is_same_v<T, float16>) {
            is_equal = (diff == T{0.f});
          }

          if (is_equal) {
            index++;
            continue;
          }
          T result = dist_func_(diff, grad[index], dist[index], p_);
          *(y + i) += result;
          *(y + j) -= result;
          index++;
        }
      }
    }
  };
  ParallelLaunchAutoSearch(task, col, this, &parallel_search_info_, pool_);
  return true;
}

std::vector<KernelAttr> PdistGradCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list = {KernelAttr()
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddOutputAttr(kNumberTypeFloat32)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, PdistGrad, PdistGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
