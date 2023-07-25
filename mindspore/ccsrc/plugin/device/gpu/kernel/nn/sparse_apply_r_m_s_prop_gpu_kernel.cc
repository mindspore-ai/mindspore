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
#include "plugin/device/gpu/kernel/nn/sparse_apply_r_m_s_prop_gpu_kernel.h"
#include "mindspore/core/ops/nn_optimizer_ops.h"
#include "kernel/common_utils.h"
#include "mindspore/core/ops/sparse_apply_r_m_s_prop.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSparseApplyRMSPropInputsNum = 6;
constexpr size_t kSparseApplyRMSPropOutputsNum = 3;
constexpr size_t kIndicesDim = 1;
using KernelRunFunc = SparseApplyRMSPropGpuKernelMod::KernelRunFunc;
}  // namespace

bool SparseApplyRMSPropGpuKernelMod::ResizedInputSize(const std::vector<KernelTensorPtr> &inputs) {
  var_shape_ = inputs.at(kIndex0)->GetShapeVector();
  if (var_shape_.empty()) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', the dimension of 'var' must be at least 1, but got scalar or None.";
    return false;
  }
  var_first_dim_size_ = var_shape_[kDim0];

  auto ms_shape = inputs.at(kIndex1)->GetShapeVector();
  if (!IsSameShape(var_shape_, ms_shape)) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', the shape of 'ms' must be the same as the shape of 'var', "
                                "but got the shape of 'ms': "
                             << ms_shape << " and the shape of 'var': " << var_shape_;
    return false;
  }
  auto mom_shape = inputs.at(kIndex2)->GetShapeVector();
  if (!IsSameShape(var_shape_, mom_shape)) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', the shape of 'mom' must be the same as the shape of 'var', "
                                "but got the shape of 'mom': "
                             << mom_shape << " and the shape of 'var': " << var_shape_;
    return false;
  }
  // scalar
  auto lr_shape = inputs.at(kIndex3)->GetShapeVector();
  if (!lr_shape.empty()) {
    MS_EXCEPTION(ValueError)
      << "For '" << kernel_name_
      << "', 'lr' must be a scalar; thus, its dimension must be 0, but got the dimension of 'lr': " << lr_shape;
    return false;
  }
  auto grad_shape = inputs.at(kIndex4)->GetShapeVector();
  var_outer_dim_size_ = 1;
  for (size_t i = 1; i < var_shape_.size(); ++i) {
    if (var_shape_[i] != grad_shape[i]) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the shape of 'var' and 'grad' must be equal in dimension i=" << i
                    << ", but got 'var_shape[i]': " << var_shape_[i] << " and 'grad_shape[i]': " << grad_shape[i];
      return KRET_RESIZE_FAILED;
    }
    var_outer_dim_size_ *= var_shape_[i];
  }

  if (!IsSameShape(var_shape_, grad_shape)) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', the shape of 'grad' must be the same as the shape of 'var', "
                                "but got the shape of 'grad': "
                             << mom_shape << " and the shape of 'var': " << var_shape_;
    return false;
  }
  auto indices_shape = inputs.at(kIndex5)->GetShapeVector();
  if (indices_shape.size() != kIndicesDim) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the 'indices' must be a 1-D Tensor, but got shape: " << indices_shape;
    return false;
  }
  if (indices_shape[kDim0] != var_shape_[kDim0]) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', the indices.shape[0] must be equal to var.shape[0], but got 'var_shape[0]': "
                             << var_shape_[kDim0] << " and 'indices_shape[0]': " << indices_shape[kDim0];
    return false;
  }
  return true;
}
bool SparseApplyRMSPropGpuKernelMod::ResizedOutputSize(const std::vector<KernelTensorPtr> &outputs) {
  auto output_var_shape = outputs[kIndex0]->GetShapeVector();
  if (!IsSameShape(var_shape_, output_var_shape)) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', the shape of output 'var' must be the same as the shape of input 'var', but got "
                                "the shape of output 'var': "
                             << output_var_shape << ", and the shape of input 'var': " << var_shape_;
    return false;
  }
  auto output_ms_shape = outputs[kIndex1]->GetShapeVector();
  if (!IsSameShape(var_shape_, output_ms_shape)) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', the shape of output 'ms' must be the same as the shape of input 'ms', "
                                "but got the shape of output 'ms': "
                             << output_ms_shape << " and the shape of input 'ms': " << var_shape_;
    return false;
  }
  auto output_mom_shape = outputs[kIndex2]->GetShapeVector();
  if (!IsSameShape(var_shape_, output_mom_shape)) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', the shape of output 'mom' must be the same as the shape of output 'mom', "
                                "but got the shape of output 'mom': "
                             << output_mom_shape << " and the shape of output 'mom': " << var_shape_;
    return false;
  }
  return true;
}

bool SparseApplyRMSPropGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  if (kernel_name_ != prim::kPrimSparseApplyRMSProp->name()) {
    MS_LOG(ERROR) << "For 'SparseApplyRMSProp', the kernel name must be 'SparseApplyRMSProp', but got " << kernel_name_;
    return false;
  }

  auto kernel_ptr = std::dynamic_pointer_cast<ops::SparseApplyRMSProp>(base_operator);
  MS_EXCEPTION_IF_NULL(kernel_ptr);
  rho_ = kernel_ptr->get_rho();
  if (rho_ > 1 || rho_ < 0) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', the argument rho should be between 0 and 1, but got the value of rho: " << rho_;
    return false;
  }
  momentum_ = kernel_ptr->get_momentum();
  if (momentum_ < 0) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', the argument momentum should be no less than 0, but got the value of momentum: "
                             << momentum_;
    return false;
  }
  epsilon_ = kernel_ptr->get_epsilon();
  if (epsilon_ <= 0) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', the argument momentum should be greater than 0, but got the value of epsilon: "
                             << epsilon_;
    return false;
  }
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int SparseApplyRMSPropGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                           const std::vector<KernelTensorPtr> &inputs,
                                           const std::vector<KernelTensorPtr> &outputs,
                                           const std::map<uint32_t, tensor::TensorPtr> &) {
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSparseApplyRMSPropInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSparseApplyRMSPropOutputsNum, kernel_name_);
  int ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != 0) {
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

template <typename T, typename S>
bool SparseApplyRMSPropGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                  const std::vector<AddressPtr> &workspace,
                                                  const std::vector<AddressPtr> &outputs) {
  auto var = GetDeviceAddress<T>(inputs, kIndex0);
  auto ms = GetDeviceAddress<T>(inputs, kIndex1);
  auto mom = GetDeviceAddress<T>(inputs, kIndex2);
  auto lr = GetDeviceAddress<T>(inputs, kIndex3);
  auto grad = GetDeviceAddress<T>(inputs, kIndex4);
  auto indices = GetDeviceAddress<S>(inputs, kIndex5);
  const float rho = this->rho_;
  const float momentum = this->momentum_;
  const float epsilon = this->epsilon_;
  size_t var_first_dim_size = this->var_first_dim_size_;
  size_t var_outer_dim_size = this->var_outer_dim_size_;
  auto status =
    CalSparseApplyRMSProp(var_first_dim_size * var_outer_dim_size, var_outer_dim_size, rho, momentum, epsilon, lr, grad,
                          indices, var, ms, mom, reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &SparseApplyRMSPropGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutInRef(0, 0)
       .AddOutInRef(1, 1)
       .AddOutInRef(2, 2),
     &SparseApplyRMSPropGpuKernelMod::LaunchKernel<float, int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutInRef(0, 0)
       .AddOutInRef(1, 1)
       .AddOutInRef(2, 2),
     &SparseApplyRMSPropGpuKernelMod::LaunchKernel<float, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutInRef(0, 0)
       .AddOutInRef(1, 1)
       .AddOutInRef(2, 2),
     &SparseApplyRMSPropGpuKernelMod::LaunchKernel<half, int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutInRef(0, 0)
       .AddOutInRef(1, 1)
       .AddOutInRef(2, 2),
     &SparseApplyRMSPropGpuKernelMod::LaunchKernel<half, int64_t>}};
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SparseApplyRMSProp, SparseApplyRMSPropGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
