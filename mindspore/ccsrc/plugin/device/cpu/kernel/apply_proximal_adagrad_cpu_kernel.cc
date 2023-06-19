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

#include "plugin/device/cpu/kernel/apply_proximal_adagrad_cpu_kernel.h"
#include <algorithm>
#include "kernel/common_utils.h"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"
#include "plugin/device/cpu/kernel/nnacl/fp32_grad/apply_proximal_adagrad_fp32.h"
#include "plugin/device/cpu/kernel/nnacl/intrinsics/ms_simd_instructions.h"

namespace mindspore {
namespace kernel {
constexpr size_t kApplyProximalAdagradInputsNum = 6;
constexpr size_t kVarIndex = 0;
constexpr size_t kAccIndex = 1;
constexpr size_t kLRIndex = 2;
constexpr size_t kL1Index = 3;
constexpr size_t kL2Index = 4;
constexpr size_t kGradIndex = 5;

bool ApplyProximalAdagradCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                            const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  batch_rank_ = base_operator->get_batch_rank();

  auto input_type_id = inputs[0]->GetDtype();
  if (input_type_id != kNumberTypeFloat32) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "',  does not support " << TypeIdToString(input_type_id);
    return false;
  }
  unit_size_ = sizeof(float);

  return true;
}

int ApplyProximalAdagradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                             const std::vector<KernelTensorPtr> &inputs,
                                             const std::vector<KernelTensorPtr> &outputs,
                                             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  if (input_size_list_.size() != kApplyProximalAdagradInputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' input size must be equal 6.";
    return KRET_RESIZE_FAILED;
  }
  std::vector<int64_t> var_shape = inputs[kVarIndex]->GetShapeVector();
  std::vector<int64_t> accum_shape = inputs[kAccIndex]->GetShapeVector();
  std::vector<int64_t> lr_shape = inputs[kLRIndex]->GetShapeVector();
  std::vector<int64_t> l1_shape = inputs[kL1Index]->GetShapeVector();
  std::vector<int64_t> l2_shape = inputs[kL2Index]->GetShapeVector();
  std::vector<int64_t> grad_shape = inputs[kGradIndex]->GetShapeVector();
  if (var_shape.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension of 'var' must be at least 1-D, but got scalar or None.";
    return KRET_RESIZE_FAILED;
  }
  if (!IsSameShape(var_shape, accum_shape)) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape of 'accum' must be the same as the shape of 'var', "
                     "but got the shape of 'accum': "
                  << accum_shape << " and the shape of 'var': " << var_shape;
    return KRET_RESIZE_FAILED;
  }
  if (!IsSameShape(var_shape, grad_shape)) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape of 'grad' must be the same as the shape of 'var', "
                     "but got the shape of 'grad': "
                  << grad_shape << " and the shape of 'var': " << var_shape;
    return KRET_RESIZE_FAILED;
  }

  if (!IsSameShape(lr_shape, l1_shape)) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape of 'lr' must be the same as the shape of 'l1', "
                     "but got the shape of 'lr': "
                  << lr_shape << " and the shape of 'l1': " << l1_shape;
    return KRET_RESIZE_FAILED;
  }

  if (!IsSameShape(lr_shape, l2_shape)) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape of 'lr' must be the same as the shape of 'l2', "
                     "but got the shape of 'lr': "
                  << lr_shape << " and the shape of 'l2': " << l2_shape;
    return KRET_RESIZE_FAILED;
  }
  if (batch_rank_ < 0 || lr_shape.size() != static_cast<size_t>(batch_rank_)) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape size of 'lr' must be equal to 'batch_rank', "
                     "but got the shape of 'lr': "
                  << lr_shape << " and 'batch_rank': " << batch_rank_;
    return KRET_RESIZE_FAILED;
  }

  batch_size_ = 1;
  if (!lr_shape.empty()) {
    batch_size_ = std::accumulate(lr_shape.begin(), lr_shape.end(), batch_size_, std::multiplies<int64_t>());
  }
  if (batch_size_ <= 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', batch_size_ must be greater than 0, but got batch_size: " << batch_size_;
    return KRET_RESIZE_FAILED;
  }

  input_elements_ = std::accumulate(var_shape.begin(), var_shape.end(), 1, std::multiplies<int64_t>());
  input_elements_ = input_elements_ / batch_size_;
  if (batch_rank_ > 1) {
    if (var_shape.size() < lr_shape.size()) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', the shape size of 'var' must be greater than 'lr_shape', but got the shape of 'var': "
                    << var_shape << " and 'lr_shape': " << lr_shape;
      return KRET_RESIZE_FAILED;
    }
    std::vector<int64_t> var_batch_shape(var_shape.begin(), var_shape.begin() + batch_rank_);
    if (!IsSameShape(lr_shape, var_batch_shape)) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', the batch shape of 'var' must be the same as the shape of 'lr', "
                       "but got the batch shape of 'var': "
                    << var_batch_shape << " and the shape of 'lr': " << lr_shape;
      return KRET_RESIZE_FAILED;
    }
  }

  return ret;
}

bool ApplyProximalAdagradCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                              const std::vector<kernel::AddressPtr> &workspace,
                                              const std::vector<kernel::AddressPtr> &) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kApplyProximalAdagradInputsNum, kernel_name_);
  auto var = reinterpret_cast<float *>(inputs[kVarIndex]->addr);
  auto accum = reinterpret_cast<float *>(inputs[kAccIndex]->addr);
  auto lr = reinterpret_cast<float *>(inputs[kLRIndex]->addr);
  auto l1 = reinterpret_cast<float *>(inputs[kL1Index]->addr);
  auto l2 = reinterpret_cast<float *>(inputs[kL2Index]->addr);
  auto grad = reinterpret_cast<float *>(inputs[kGradIndex]->addr);

  auto task = [this, &var, &accum, &lr, &l1, &l2, &grad](size_t start, size_t end) {
    auto cur_input_elements = end - start;
    for (int64_t b = 0; b < batch_size_; b++) {
      auto offset = b * input_elements_ + start;
      auto var_cur = var + offset;
      auto accum_cur = accum + offset;
      auto grad_cur = grad + offset;

      ApplyProximalAdagradOpt(var_cur, accum_cur, lr[b], l1[b], l2[b], grad_cur, cur_input_elements);
    }
  };

  ParallelLaunchAutoSearch(task, input_elements_, this, &parallel_search_info_, pool_);

  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ApplyProximalAdagrad, ApplyProximalAdagradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
