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

#include "plugin/device/cpu/kernel/sparse_apply_r_m_s_prop_cpu_kernel.h"
#include <algorithm>
#include <iostream>
#include "kernel/common_utils.h"
#include "mindspore/core/ops/sparse_apply_r_m_s_prop.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSparseApplyRMSPropOutputsNum = 3;
constexpr size_t kSparseApplyRMSPropInputsNum = 6;
constexpr size_t kIndicesDim = 1;
constexpr size_t kSparseApplyRMSPropWorkspaceSize = 4;
constexpr char kKernelName[] = "SparseApplyRMSProp";
using KernelRunFunc = SparseApplyRMSPropCpuKernelMod::KernelRunFunc;
#define ADD_INPUT_ATTR(var_type, indices_type) \
  .AddInputAttr(var_type)                      \
    .AddInputAttr(var_type)                    \
    .AddInputAttr(var_type)                    \
    .AddInputAttr(var_type)                    \
    .AddInputAttr(var_type)                    \
    .AddInputAttr(indices_type)

#define ADD_OI_REF_SAME_PLACE(ind1, ind2, ind3) .AddOutInRef(ind1, ind1).AddOutInRef(ind2, ind2).AddOutInRef(ind3, ind3)

#define CPU_FUNLIST_KERNEL_REGISTER(var_type, var_fun_type, indices_type, indices_fun_type) \
  {                                                                                         \
    KernelAttr() ADD_INPUT_ATTR(var_type, indices_type)                                     \
      .AddOutputAttr(var_type)                                                              \
      .AddOutputAttr(var_type)                                                              \
      .AddOutputAttr(var_type) ADD_OI_REF_SAME_PLACE(0, 1, 2),                              \
      &SparseApplyRMSPropCpuKernelMod::LaunchKernel<indices_fun_type, var_fun_type>         \
  }
}  // namespace

bool SparseApplyRMSPropCpuKernelMod::ResizedInputSize(const std::vector<KernelTensorPtr> &inputs) {
  var_shape_ = inputs.at(kIndex0)->GetShapeVector();
  if (var_shape_.empty()) {
    MS_EXCEPTION(ValueError) << "For '" << kKernelName
                             << "', the dimension of 'var' must be at least 1, but got scalar or None.";
    return false;
  }
  var_first_dim_size_ = var_shape_[kDim0];

  auto ms_shape = inputs.at(kIndex1)->GetShapeVector();
  if (!IsSameShape(var_shape_, ms_shape)) {
    MS_EXCEPTION(ValueError) << "For '" << kKernelName
                             << "', the shape of 'ms' must be the same as the shape of 'var', "
                                "but got the shape of 'ms': "
                             << Vector2Str(ms_shape) << " and the shape of 'var': " << Vector2Str(var_shape_);
    return false;
  }
  auto mom_shape = inputs.at(kIndex2)->GetShapeVector();
  if (!IsSameShape(var_shape_, mom_shape)) {
    MS_EXCEPTION(ValueError) << "For '" << kKernelName
                             << "', the shape of 'mom' must be the same as the shape of 'var', "
                                "but got the shape of 'mom': "
                             << Vector2Str(mom_shape) << " and the shape of 'var': " << Vector2Str(var_shape_);
    return false;
  }
  // scalar
  auto lr_shape = inputs.at(kIndex3)->GetShapeVector();
  if (!lr_shape.empty()) {
    MS_EXCEPTION(ValueError)
      << "For '" << kKernelName
      << "', 'lr' must be a scalar; thus, its dimension must be 0, but got the dimension of 'lr': "
      << Vector2Str(lr_shape);
    return false;
  }
  auto grad_shape = inputs.at(kIndex4)->GetShapeVector();
  for (size_t i = 1; i < var_shape_.size(); ++i) {
    if (var_shape_[i] != grad_shape[i]) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the shape of 'var' and 'grad' must be equal in dimension i=" << i
                    << ", but got 'var_shape[i]': " << var_shape_[i] << " and 'grad_shape[i]': " << grad_shape[i];
      return KRET_RESIZE_FAILED;
    }
    var_outer_dim_size_ *= var_shape_[i];
  }

  if (!IsSameShape(var_shape_, grad_shape)) {
    MS_EXCEPTION(ValueError) << "For '" << kKernelName
                             << "', the shape of 'grad' must be the same as the shape of 'var', "
                                "but got the shape of 'grad': "
                             << Vector2Str(mom_shape) << " and the shape of 'var': " << Vector2Str(var_shape_);
    return false;
  }
  auto indices_shape = inputs.at(kIndex5)->GetShapeVector();
  if (indices_shape.size() != kIndicesDim) {
    MS_LOG(EXCEPTION) << "For '" << kKernelName
                      << "', the 'indices' must be a 1-D Tensor, but got shape: " << Vector2Str(indices_shape);
    return false;
  }
  if (indices_shape[kDim0] != var_shape_[kDim0]) {
    MS_EXCEPTION(ValueError) << "For '" << kKernelName
                             << "', the indices.shape[0] must be equal to var.shape[0], but got 'var_shape[0]': "
                             << var_shape_[kDim0] << " and 'indices_shape[0]': " << indices_shape[kDim0];
    return false;
  }
  indices_size_ = indices_shape[kDim0];
  return true;
}

bool SparseApplyRMSPropCpuKernelMod::ResizedOutputSize(const std::vector<KernelTensorPtr> &outputs) {
  auto output_var_shape = outputs[kIndex0]->GetShapeVector();
  if (!IsSameShape(var_shape_, output_var_shape)) {
    MS_EXCEPTION(ValueError) << "For '" << kKernelName
                             << "', the shape of output 'var' must be the same as the shape of input 'var', but got "
                                "the shape of output 'var': "
                             << Vector2Str(output_var_shape)
                             << ", and the shape of input 'var': " << Vector2Str(var_shape_);
    return false;
  }
  auto output_ms_shape = outputs[kIndex1]->GetShapeVector();
  if (!IsSameShape(var_shape_, output_ms_shape)) {
    MS_EXCEPTION(ValueError) << "For '" << kKernelName
                             << "', the shape of output 'ms' must be the same as the shape of input 'ms', "
                                "but got the shape of output 'ms': "
                             << Vector2Str(output_ms_shape)
                             << " and the shape of input 'ms': " << Vector2Str(var_shape_);
    return false;
  }
  auto output_mom_shape = outputs[kIndex2]->GetShapeVector();
  if (!IsSameShape(var_shape_, output_mom_shape)) {
    MS_EXCEPTION(ValueError) << "For '" << kKernelName
                             << "', the shape of output 'mom' must be the same as the shape of output 'mom', "
                                "but got the shape of output 'mom': "
                             << Vector2Str(output_mom_shape)
                             << " and the shape of output 'mom': " << Vector2Str(var_shape_);
    return false;
  }
  return true;
}

void SparseApplyRMSPropCpuKernelMod::ResetResource() noexcept {
  input_size_list_.clear();
  output_size_list_.clear();
  indices_size_ = 0;
  var_first_dim_size_ = 0;
  var_outer_dim_size_ = 1;
}

int SparseApplyRMSPropCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                           const std::vector<KernelTensorPtr> &inputs,
                                           const std::vector<KernelTensorPtr> &outputs,
                                           const std::map<uint32_t, tensor::TensorPtr> &) {
  MS_EXCEPTION_IF_NULL(base_operator);
  ResetResource();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSparseApplyRMSPropInputsNum, kKernelName);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSparseApplyRMSPropOutputsNum, kKernelName);
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto kernel_ptr = std::dynamic_pointer_cast<ops::SparseApplyRMSProp>(base_operator);
  if (kernel_ptr == nullptr) {
    MS_LOG(ERROR) << "Cast op from BaseOperator to SparseApplyRMSProp failed.";
    return KRET_RESIZE_FAILED;
  }
  if (!ResizedInputSize(inputs)) {
    return KRET_RESIZE_FAILED;
  }
  if (!ResizedOutputSize(outputs)) {
    return KRET_RESIZE_FAILED;
  }
  return KRET_OK;
}

bool SparseApplyRMSPropCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::SparseApplyRMSProp>(base_operator);
  if (kernel_ptr == nullptr) {
    MS_LOG(ERROR) << "Cast op from BaseOperator to SparseApplyRMSProp failed.";
    return false;
  }
  rho_ = kernel_ptr->get_rho();
  if (rho_ > 1 || rho_ < 0) {
    MS_EXCEPTION(ValueError) << "For '" << kKernelName
                             << "', the argument rho should be between 0 and 1, but got the value of rho: " << rho_;
    return false;
  }
  momentum_ = kernel_ptr->get_momentum();
  if (momentum_ < 0) {
    MS_EXCEPTION(ValueError) << "For '" << kKernelName
                             << "', the argument momentum should be no less than 0, but got the value of momentum: "
                             << momentum_;
    return false;
  }
  epsilon_ = kernel_ptr->get_epsilon();
  if (epsilon_ <= 0) {
    MS_EXCEPTION(ValueError) << "For '" << kKernelName
                             << "', the argument momentum should be greater than 0, but got the value of epsilon: "
                             << epsilon_;
    return false;
  }
  return MatchKernelFunc(base_operator, inputs, outputs);
}

template <typename I, typename T>
bool SparseApplyRMSPropCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                  const std::vector<kernel::AddressPtr> &workspace,
                                                  const std::vector<kernel::AddressPtr> &outputs) {
  auto *var = reinterpret_cast<T *>(inputs.at(kIndex0)->addr);
  auto *ms = reinterpret_cast<T *>(inputs.at(kIndex1)->addr);
  auto *mom = reinterpret_cast<T *>(inputs.at(kIndex2)->addr);
  auto lr = reinterpret_cast<T *>(inputs.at(kIndex3)->addr)[kDim0];
  auto *grad = reinterpret_cast<T *>(inputs.at(kIndex4)->addr);
  auto *indices = reinterpret_cast<I *>(inputs.at(kIndex5)->addr);
  const auto rho = this->rho_;
  const auto momentum = this->momentum_;
  const auto epsilon = this->epsilon_;
  auto var_first_dim_size = static_cast<size_t>(this->var_first_dim_size_);
  auto var_outer_dim_size = this->var_outer_dim_size_;
  auto task = [var, ms, mom, grad, indices, &lr, &rho, &momentum, &epsilon, &var_first_dim_size, &var_outer_dim_size](
                size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      const int indices_pos = i / var_outer_dim_size;
      const int inner_pos = i % var_outer_dim_size;
      size_t index = static_cast<size_t>(indices[indices_pos]);
      if (LongToSize(index) >= var_first_dim_size) {
        MS_LOG(EXCEPTION) << "For '" << kKernelName << "', each element in 'indices' must be in range [0, "
                          << SizeToLong(var_first_dim_size) << "), but got " << index;
      }
      const size_t cur_pos = index * var_outer_dim_size + inner_pos;
      const float grad_t = static_cast<float>(grad[i]);
      float msf = static_cast<float>(ms[cur_pos]);
      if (grad_t != 0) {
        msf = msf * rho + grad_t * grad_t * (1.0f - rho);
        ms[cur_pos] = static_cast<T>(msf);
      }
      mom[cur_pos] = static_cast<T>(static_cast<float>(mom[cur_pos]) * momentum +
                                    1 / sqrt(msf + epsilon) * static_cast<float>(lr) * grad_t);
      var[cur_pos] -= mom[cur_pos];
    }
  };
  ParallelLaunchAutoSearch(task, var_first_dim_size * var_outer_dim_size, this, &parallel_search_info_);
  return true;
}

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &SparseApplyRMSPropCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    CPU_FUNLIST_KERNEL_REGISTER(kNumberTypeFloat32, float, kNumberTypeInt32, int),
    CPU_FUNLIST_KERNEL_REGISTER(kNumberTypeFloat32, float, kNumberTypeInt64, int64_t),
    CPU_FUNLIST_KERNEL_REGISTER(kNumberTypeFloat16, float16, kNumberTypeInt32, int),
    CPU_FUNLIST_KERNEL_REGISTER(kNumberTypeFloat16, float16, kNumberTypeInt64, int64_t),
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseApplyRMSProp, SparseApplyRMSPropCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
