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
#include "kernel/cpu/native/adam_cpu_kernel.h"
#include <algorithm>
#include <functional>
#include <numeric>
#include "mindspore/ops/infer/adam.h"
#include "kernel/cpu/nnacl/errorcode.h"
#include "kernel/cpu/nnacl/fp32/adam_fp32.h"

#include "ops_utils/op_utils.h"

namespace mindspore {
namespace kernel {
namespace adam_cpu {
namespace {
constexpr size_t kAdamInputsNum = 10;
constexpr size_t kAdamOutputsNum = 3;
constexpr size_t kScalarIndex = 0;
constexpr size_t kIndexVar = 0;
constexpr size_t kIndexM = 1;
constexpr size_t kIndexV = 2;
constexpr size_t kIndexBeta1Power = 3;
constexpr size_t kIndexBeta2Power = 4;
constexpr size_t kIndexLr = 5;
constexpr size_t kIndexBeta1 = 6;
constexpr size_t kIndexBeta2 = 7;
constexpr size_t kIndexEpsilon = 8;
constexpr size_t kIndexGrad = 9;
constexpr float kAdamBlock = 1000;

struct AdamInputShapes {
  ShapeVector var;
  ShapeVector m;
  ShapeVector v;
  ShapeVector beta1_power;
  ShapeVector beta2_power;
  ShapeVector lr;
  ShapeVector beta1;
  ShapeVector beta2;
  ShapeVector epsilon;
  ShapeVector grad;
};

bool IsScalarShape(const ShapeVector &shape) { return shape.empty() || (shape.size() == 1 && shape[0] == 1); }

bool IsScalarOrSameBatchShape(const ShapeVector &shape, const ShapeVector &batch_shape) {
  return IsScalarShape(shape) || IsSameShape(shape, batch_shape);
}

AdamInputShapes GetAdamInputShapes(const std::vector<KernelTensor *> &inputs) {
  return {inputs[kIndexVar]->GetShapeVector(),        inputs[kIndexM]->GetShapeVector(),
          inputs[kIndexV]->GetShapeVector(),          inputs[kIndexBeta1Power]->GetShapeVector(),
          inputs[kIndexBeta2Power]->GetShapeVector(), inputs[kIndexLr]->GetShapeVector(),
          inputs[kIndexBeta1]->GetShapeVector(),      inputs[kIndexBeta2]->GetShapeVector(),
          inputs[kIndexEpsilon]->GetShapeVector(),    inputs[kIndexGrad]->GetShapeVector()};
}

bool IsValidShapeVector(const ShapeVector &shape) {
  return std::all_of(shape.begin(), shape.end(), [](int64_t dim) { return dim >= 0; });
}

bool AreValidAdamInputShapes(const AdamInputShapes &shapes) {
  const std::vector<ShapeVector> all_shapes = {shapes.var,         shapes.m,   shapes.v,     shapes.beta1_power,
                                               shapes.beta2_power, shapes.lr,  shapes.beta1, shapes.beta2,
                                               shapes.epsilon,     shapes.grad};
  return std::all_of(all_shapes.begin(), all_shapes.end(), IsValidShapeVector);
}

bool AreScalarShapes(const std::vector<ShapeVector> &shapes) {
  return std::all_of(shapes.begin(), shapes.end(), IsScalarShape);
}

void LogAdamInputShapes(const std::string &kernel_name, const AdamInputShapes &shapes) {
  MS_LOG(ERROR) << "For '" << kernel_name
                << "', input shapes must be valid non-negative shapes, but got 'var': " << shapes.var
                << ", 'm': " << shapes.m << ", 'v': " << shapes.v << ", 'beta1_power': " << shapes.beta1_power
                << ", 'beta2_power': " << shapes.beta2_power << ", 'lr': " << shapes.lr << ", 'beta1': " << shapes.beta1
                << ", 'beta2': " << shapes.beta2 << ", 'epsilon': " << shapes.epsilon << ", 'grad': " << shapes.grad;
}

int CheckAdamStateShapes(const std::string &kernel_name, const AdamInputShapes &shapes) {
  if (!AreValidAdamInputShapes(shapes)) {
    LogAdamInputShapes(kernel_name, shapes);
    return KRET_RESIZE_FAILED;
  }
  if (!IsSameShape(shapes.var, shapes.m) || !IsSameShape(shapes.var, shapes.v) ||
      !IsSameShape(shapes.var, shapes.grad)) {
    MS_LOG(ERROR) << "For '" << kernel_name << "', the shapes of 'var', 'm', 'v' and 'grad' must be the same, "
                  << "but got the shapes of 'var': " << shapes.var << ", 'm': " << shapes.m << ", 'v': " << shapes.v
                  << " and 'grad': " << shapes.grad;
    return KRET_RESIZE_FAILED;
  }
  return KRET_OK;
}

int CheckAdamScalarParameterShapes(const std::string &kernel_name, const AdamInputShapes &shapes) {
  if (AreScalarShapes(
        {shapes.beta1_power, shapes.beta2_power, shapes.lr, shapes.beta1, shapes.beta2, shapes.epsilon})) {
    return KRET_OK;
  }
  MS_LOG(ERROR) << "For '" << kernel_name
                << "', when 'batch_rank' is 0, 'beta1_power', 'beta2_power', 'lr', 'beta1', 'beta2' and "
                   "'epsilon' must be scalar or 1D tensors with one element, but got 'beta1_power': "
                << shapes.beta1_power << ", 'beta2_power': " << shapes.beta2_power << ", 'lr': " << shapes.lr
                << ", 'beta1': " << shapes.beta1 << ", 'beta2': " << shapes.beta2
                << " and 'epsilon': " << shapes.epsilon;
  return KRET_RESIZE_FAILED;
}

int CheckAdamBatchParameterShapes(const std::string &kernel_name, const AdamInputShapes &shapes, int64_t batch_rank) {
  if (batch_rank < 0) {
    MS_LOG(ERROR) << "For '" << kernel_name << "', 'batch_rank' must be greater than or equal to 0, but got "
                  << batch_rank;
    return KRET_RESIZE_FAILED;
  }
  if (batch_rank == 0) {
    return CheckAdamScalarParameterShapes(kernel_name, shapes);
  }
  if (shapes.lr.size() != static_cast<size_t>(batch_rank)) {
    MS_LOG(ERROR) << "For '" << kernel_name
                  << "', the shape size of 'lr' must be equal to 'batch_rank', but got the shape of 'lr': " << shapes.lr
                  << " and 'batch_rank': " << batch_rank;
    return KRET_RESIZE_FAILED;
  }
  if (!IsSameShape(shapes.beta1_power, shapes.lr) || !IsSameShape(shapes.beta2_power, shapes.lr)) {
    MS_LOG(ERROR) << "For '" << kernel_name
                  << "', when 'batch_rank' is greater than 0, the shapes of 'beta1_power' and 'beta2_power' must be "
                     "the same as the shape of 'lr', but got 'beta1_power': "
                  << shapes.beta1_power << ", 'beta2_power': " << shapes.beta2_power << " and 'lr': " << shapes.lr;
    return KRET_RESIZE_FAILED;
  }
  if (!IsScalarOrSameBatchShape(shapes.beta1, shapes.lr) || !IsScalarOrSameBatchShape(shapes.beta2, shapes.lr) ||
      !IsScalarOrSameBatchShape(shapes.epsilon, shapes.lr)) {
    MS_LOG(ERROR) << "For '" << kernel_name
                  << "', when 'batch_rank' is greater than 0, the shapes of 'beta1', 'beta2' and 'epsilon' must be "
                     "scalar, 1D tensors with one element, or the same as the shape of 'lr', but got 'beta1': "
                  << shapes.beta1 << ", 'beta2': " << shapes.beta2 << ", 'epsilon': " << shapes.epsilon
                  << " and 'lr': " << shapes.lr;
    return KRET_RESIZE_FAILED;
  }
  return KRET_OK;
}

int64_t GetAdamBatchSize(const ShapeVector &lr_shape) {
  if (lr_shape.empty()) {
    return 1;
  }
  return std::accumulate(lr_shape.begin(), lr_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
}

int CheckAdamBatchPrefixShape(const std::string &kernel_name, const ShapeVector &var_shape, const ShapeVector &lr_shape,
                              int64_t batch_rank) {
  if (batch_rank == 0) {
    return KRET_OK;
  }
  auto batch_rank_size = LongToSize(batch_rank);
  if (var_shape.size() < batch_rank_size) {
    MS_LOG(ERROR) << "For '" << kernel_name
                  << "', the shape size of 'var' must be greater than or equal to 'batch_rank', but got the shape of "
                     "'var': "
                  << var_shape << " and 'lr_shape': " << lr_shape;
    return KRET_RESIZE_FAILED;
  }
  std::vector<int64_t> var_batch_shape(var_shape.begin(), var_shape.begin() + batch_rank_size);
  if (!IsSameShape(lr_shape, var_batch_shape)) {
    MS_LOG(ERROR) << "For '" << kernel_name
                  << "', the batch shape of 'var' must be the same as the shape of 'lr', but got the batch shape of "
                  << "'var': " << var_batch_shape << " and the shape of 'lr': " << lr_shape;
    return KRET_RESIZE_FAILED;
  }
  return KRET_OK;
}
}  // namespace

template <typename T>
void AdamCpuKernelMod::LaunchAdam(const std::vector<kernel::KernelTensor *> &inputs,
                                  const std::vector<kernel::KernelTensor *> &) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kAdamInputsNum, kernel_name_);
  T *var = GetDeviceAddress<T>(inputs, kIndexVar);
  T *m = GetDeviceAddress<T>(inputs, kIndexM);
  T *v = GetDeviceAddress<T>(inputs, kIndexV);
  float *beta1_power = GetDeviceAddress<float>(inputs, kIndexBeta1Power);
  float *beta2_power = GetDeviceAddress<float>(inputs, kIndexBeta2Power);
  float *lr = GetDeviceAddress<float>(inputs, kIndexLr);
  float *beta1_addr = GetDeviceAddress<float>(inputs, kIndexBeta1);
  float *beta2_addr = GetDeviceAddress<float>(inputs, kIndexBeta2);
  float *epsilon_addr = GetDeviceAddress<float>(inputs, kIndexEpsilon);
  const bool beta1_is_scalar = IsScalarShape(inputs[kIndexBeta1]->GetShapeVector());
  const bool beta2_is_scalar = IsScalarShape(inputs[kIndexBeta2]->GetShapeVector());
  const bool epsilon_is_scalar = IsScalarShape(inputs[kIndexEpsilon]->GetShapeVector());
  T *gradient = GetDeviceAddress<T>(inputs, kIndexGrad);
  constexpr float ONE = 1.0;

  for (int64_t b = 0; b < batch_size_; b++) {
    size_t batch_index = LongToSize(b);
    T beta1 = static_cast<T>(beta1_addr[beta1_is_scalar ? kScalarIndex : batch_index]);
    T beta2 = static_cast<T>(beta2_addr[beta2_is_scalar ? kScalarIndex : batch_index]);
    T epsilon = static_cast<T>(epsilon_addr[epsilon_is_scalar ? kScalarIndex : batch_index]);
    T new_lr = static_cast<T>(lr[b] * std::sqrt(ONE - beta2_power[b]) / (ONE - beta1_power[b]));
    // multithreading
    auto task = [this, &var, &m, &v, &gradient, new_lr, beta1, beta2, epsilon](size_t start, size_t end) {
      T one = static_cast<T>(1.0);
      for (size_t i = start; i < end; i++) {
        m[i] += (gradient[i] - m[i]) * (one - beta1);
        v[i] += (gradient[i] * gradient[i] - v[i]) * (one - beta2);
        T sqrt_v = static_cast<T>(std::sqrt(static_cast<double>(v[i])));
        if (use_nesterov_) {
          var[i] -= new_lr * (m[i] * beta1 + (one - beta1) * gradient[i]) / (sqrt_v + epsilon);
        } else {
          var[i] -= new_lr * m[i] / (sqrt_v + epsilon);
        }
      }
    };
    ParallelLaunch(task, input_elements_, kAdamBlock, this);
    var = var + input_elements_;
    m = m + input_elements_;
    v = v + input_elements_;
    gradient = gradient + input_elements_;
  }
}

void AdamCpuKernelMod::LaunchAdamNnacl(const std::vector<kernel::KernelTensor *> &inputs,
                                       const std::vector<kernel::KernelTensor *> &) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kAdamInputsNum, kernel_name_);
  float *var = GetDeviceAddress<float>(inputs, kIndexVar);
  float *m = GetDeviceAddress<float>(inputs, kIndexM);
  float *v = GetDeviceAddress<float>(inputs, kIndexV);
  float *beta1_power = GetDeviceAddress<float>(inputs, kIndexBeta1Power);
  float *beta2_power = GetDeviceAddress<float>(inputs, kIndexBeta2Power);
  float *lr = GetDeviceAddress<float>(inputs, kIndexLr);
  float *beta1_addr = GetDeviceAddress<float>(inputs, kIndexBeta1);
  float *beta2_addr = GetDeviceAddress<float>(inputs, kIndexBeta2);
  float *epsilon_addr = GetDeviceAddress<float>(inputs, kIndexEpsilon);
  const bool beta1_is_scalar = IsScalarShape(inputs[kIndexBeta1]->GetShapeVector());
  const bool beta2_is_scalar = IsScalarShape(inputs[kIndexBeta2]->GetShapeVector());
  const bool epsilon_is_scalar = IsScalarShape(inputs[kIndexEpsilon]->GetShapeVector());
  float *gradient = GetDeviceAddress<float>(inputs, kIndexGrad);
  constexpr float ONE = 1.0;
  for (int64_t b = 0; b < batch_size_; b++) {
    size_t batch_index = LongToSize(b);
    float beta1 = beta1_addr[beta1_is_scalar ? kScalarIndex : batch_index];
    float beta2 = beta2_addr[beta2_is_scalar ? kScalarIndex : batch_index];
    float epsilon = epsilon_addr[epsilon_is_scalar ? kScalarIndex : batch_index];
    float new_lr = lr[b] * std::sqrt(ONE - beta2_power[b]) / (ONE - beta1_power[b]);
    // multithreading
    auto task = [this, &var, &m, &v, &gradient, new_lr, beta1, beta2, epsilon](size_t start, size_t end) {
      int ret = AdamFp32(var, m, v, new_lr, beta1, beta2, epsilon, gradient, start, end, use_nesterov_);
      if (ret != NNACL_OK) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', AdamFp32 failed. Error no: " << ret;
      }
    };
    ParallelLaunch(task, input_elements_, kAdamBlock, this);
    var = var + input_elements_;
    m = m + input_elements_;
    v = v + input_elements_;
    gradient = gradient + input_elements_;
  }
}

bool AdamCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (primitive_->HasAttr("use_locking")) {
    use_locking_ = GetValue<bool>(primitive_->GetAttr("use_locking"));
  }
  if (primitive_->HasAttr("use_nesterov")) {
    use_nesterov_ = GetValue<bool>(primitive_->GetAttr("use_nesterov"));
  }

  dtype_ = inputs.at(kIndex0)->dtype_id();
  batch_rank_ = ops::get_batch_rank(primitive_);
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kAdamInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kAdamOutputsNum, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int AdamCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    return ret;
  }
  input_elements_ = 0;

  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kAdamInputsNum, kernel_name_);
  const auto shapes = GetAdamInputShapes(inputs);

  ret = CheckAdamStateShapes(kernel_name_, shapes);
  if (ret != KRET_OK) {
    return ret;
  }
  ret = CheckAdamBatchParameterShapes(kernel_name_, shapes, batch_rank_);
  if (ret != KRET_OK) {
    return ret;
  }

  batch_size_ = GetAdamBatchSize(shapes.lr);
  if (batch_size_ <= 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', batch_size_ must be greater than 0, but got batch_size: " << batch_size_;
    return KRET_RESIZE_FAILED;
  }

  ret = CheckAdamBatchPrefixShape(kernel_name_, shapes.var, shapes.lr, batch_rank_);
  if (ret != KRET_OK) {
    return ret;
  }

  input_elements_ = IntToSize(
    std::accumulate(shapes.var.begin(), shapes.var.end(), static_cast<int64_t>(1), std::multiplies<int64_t>()));
  input_elements_ = input_elements_ / LongToSize(batch_size_);

  return KRET_OK;
}

template <typename T>
bool AdamCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                    const std::vector<kernel::KernelTensor *> &workspace,
                                    const std::vector<kernel::KernelTensor *> &outputs) {
  if (dtype_ == kNumberTypeFloat32) {
    LaunchAdamNnacl(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat16) {
    LaunchAdam<float16>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of 'var' must be Float16 or Float32, but got "
                      << TypeIdToType(dtype_)->ToString();
  }
  return true;
}

std::vector<std::pair<KernelAttr, AdamCpuKernelMod::AdamFunc>> AdamCpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &AdamCpuKernelMod::LaunchKernel<float>},
};

std::vector<KernelAttr> AdamCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, AdamFunc> &item) { return item.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Adam, AdamCpuKernelMod);
}  // namespace adam_cpu
}  // namespace kernel
}  // namespace mindspore
