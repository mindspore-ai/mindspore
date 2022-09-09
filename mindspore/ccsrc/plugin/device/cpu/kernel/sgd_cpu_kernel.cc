/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/sgd_cpu_kernel.h"
#include <algorithm>
#include <vector>
#include <utility>
#include <memory>
#include <map>
#include "ops/sgd.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSGDInputsNum = 6;
constexpr size_t kSGDOutputsNum = 1;
constexpr size_t kIndexParm = 0;
constexpr size_t kIndexGrad = 1;
constexpr size_t kIndexLr = 2;
constexpr size_t kIndexAccum = 3;
constexpr size_t kIndexMomentum = 4;
constexpr size_t kIndexStat = 5;

using KernelRunFunc = SGDCpuKernelMod::KernelRunFunc;
}  // namespace

bool SGDCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                           const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  if (inputs.size() != kSGDInputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input size must be " << kSGDInputsNum << ", but got "
                  << inputs.size();
    return false;
  }
  if (outputs.size() != kSGDOutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', output size must be " << kSGDOutputsNum << ", but got "
                  << outputs.size();
    return false;
  }
  auto sgd_op = std::make_shared<ops::SGD>(base_operator->GetPrim());

  dampening_ = sgd_op->get_dampening();
  weight_decay_ = sgd_op->get_weight_decay();
  nesterov_ = sgd_op->get_nesterov();

  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  return true;
}

int SGDCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                            const std::vector<KernelTensorPtr> &outputs,
                            const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  // "parameters", "gradient", "learning_rate", "accum", "momentum", "stat"
  std::vector<int64_t> parm_shape = inputs[kIndexParm]->GetShapeVector();
  std::vector<int64_t> grad_shape = inputs[kIndexGrad]->GetShapeVector();
  std::vector<int64_t> accum_shape = inputs[kIndexAccum]->GetShapeVector();
  std::vector<int64_t> stat_shape = inputs[kIndexStat]->GetShapeVector();

  std::vector<int64_t> momentum_shape = inputs[kIndexMomentum]->GetShapeVector();
  std::vector<int64_t> lr_shape = inputs[kIndexLr]->GetShapeVector();

  if (!IsSameShape(parm_shape, grad_shape)) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape of 'parameters' must be the gradient as the shape of 'gradient',"
                  << " but got the shape of 'parameters': " << Vector2Str(parm_shape)
                  << " and the shape of 'gradient': " << Vector2Str(grad_shape);
    return KRET_RESIZE_FAILED;
  }
  if (!IsSameShape(parm_shape, accum_shape)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the shape of 'parameters' must be the same as the shape of 'accum',"
                  << " but got the shape of 'parameters': " << Vector2Str(parm_shape)
                  << " and the shape of 'accum': " << Vector2Str(accum_shape);
    return KRET_RESIZE_FAILED;
  }
  if (!IsSameShape(parm_shape, stat_shape)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the shape of 'parameters' must be the same as the shape of 'stat',"
                  << " but got the shape of 'parameters': " << Vector2Str(parm_shape)
                  << " and the shape of 'stat': " << Vector2Str(stat_shape);
    return KRET_RESIZE_FAILED;
  }
  auto is_scalar_shape = [](const std::vector<int64_t> &shape) {
    return shape.empty() || (shape.size() == 1 && shape[0] == 1);
  };
  if (!is_scalar_shape(lr_shape)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'learning rate' should be a scalar. but got shape "
                  << Vector2Str(lr_shape);
    return KRET_RESIZE_FAILED;
  }
  if (!is_scalar_shape(momentum_shape)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'momentum' should be a scalar. but got shape "
                  << Vector2Str(momentum_shape);
    return KRET_RESIZE_FAILED;
  }
  return KRET_OK;
}

template <typename T>
bool SGDCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                   const std::vector<kernel::AddressPtr> &,
                                   const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSGDInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSGDOutputsNum, kernel_name_);
  auto param = reinterpret_cast<T *>(inputs[kIndexParm]->addr);
  auto grad = reinterpret_cast<T *>(inputs[kIndexGrad]->addr);
  auto lr = reinterpret_cast<T *>(inputs[kIndexLr]->addr);
  auto accum = reinterpret_cast<T *>(inputs[kIndexAccum]->addr);
  auto momentum = reinterpret_cast<T *>(inputs[kIndexMomentum]->addr);
  auto stat = reinterpret_cast<T *>(inputs[kIndexStat]->addr);
  auto output_param = reinterpret_cast<T *>(outputs[0]->addr);
  size_t elem_num = inputs[0]->size / sizeof(T);

  auto task = [this, &param, &grad, &lr, &accum, &momentum, &stat, &output_param](size_t start, size_t end) {
    T ZERO = static_cast<T>(0);
    T ONE = static_cast<T>(1);
    for (size_t i = start; i < end; i++) {
      T grad_new = grad[i];
      if (weight_decay_ > static_cast<float>(0.0)) {
        grad_new += param[i] * static_cast<T>(weight_decay_);
      }
      if (momentum[0] > ZERO) {
        if (stat[i] > ZERO) {
          accum[i] = grad_new;
          stat[i] = ZERO;
        } else {
          accum[i] = accum[i] * momentum[0] + (ONE - static_cast<T>(dampening_)) * grad_new;
        }
        if (nesterov_) {
          grad_new += accum[i] * momentum[0];
        } else {
          grad_new = accum[i];
        }
      }
      param[i] -= lr[0] * grad_new;
      output_param[i] = param[i];
    }
  };
  ParallelLaunchAutoSearch(task, elem_num, this, &parallel_search_info_);
  return true;
}

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &SGDCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &SGDCpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &SGDCpuKernelMod::LaunchKernel<float16>}};
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SGD, SGDCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
