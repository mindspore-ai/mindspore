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

#include "plugin/device/cpu/kernel/sparse_slice_grad_cpu_kernel.h"
#include <algorithm>
#include <complex>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/eigen/eigen_common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
constexpr int64_t kSparseSliceGradInputsNum = 4;
constexpr int64_t kSparseSliceGradOutputsNum = 1;
constexpr int64_t kDim0Num = 1;
constexpr int64_t kDim1Num = 2;

#define ADD_KERNEL(dtype, type)                        \
  {                                                    \
    KernelAttr()                                       \
      .AddInputAttr(kNumberType##dtype)                \
      .AddInputAttr(kNumberTypeInt64)                  \
      .AddInputAttr(kNumberTypeInt64)                  \
      .AddInputAttr(kNumberTypeInt64)                  \
      .AddOutputAttr(kNumberType##dtype),              \
      &SparseSliceGradCpuKernelMod::LaunchKernel<type> \
  }
}  // namespace

bool SparseSliceGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs) {
  MS_ERROR_IF_NULL_W_RET_VAL(base_operator, false);
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  kernel_name_ = base_operator->name();
  return true;
}

int SparseSliceGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs,
                                        const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  const auto backprop_val_grad_shape = inputs[kIndex0]->GetShapeVector();
  const auto indices_shape = inputs[kIndex1]->GetShapeVector();
  const auto start_shape = inputs[kIndex2]->GetShapeVector();
  const auto new_indices_shape = inputs[kIndex3]->GetShapeVector();

  // Check shape
  if (backprop_val_grad_shape.size() != kDim0Num) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', it requires 'brackprop_val_gard' must be 1D Tensor "
                      << ", but got " << backprop_val_grad_shape.size() << "-D";
  }
  if (indices_shape.size() != kDim1Num) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', it requires 'indices_shape' must be 2D Tensor "
                      << ", but got " << indices_shape.size() << "-D";
  }
  if (start_shape.size() != kDim0Num) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', it requires 'start_shape' must be 1D Tensor "
                      << ", but got " << start_shape.size() << "-D";
  }
  if (new_indices_shape.size() != kDim1Num) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', it requires 'new_indices_shape' must be 2D Tensor "
                      << ", but got " << new_indices_shape.size() << "-D";
  }
  if (backprop_val_grad_shape[0] != new_indices_shape[0]) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', it requires the first dimension length of 'backprop_val_grad'  "
                         "must be equal to the first length of 'new_indices', but got 'backprop_val_grad' shape: "
                      << backprop_val_grad_shape[0] << " and 'new_indices' shape: " << new_indices_shape[0];
  }
  nnz_ = indices_shape[0];
  slice_nnz_ = backprop_val_grad_shape[0];
  rank_ = indices_shape[1];

  return KRET_OK;
}

template <typename T>
bool SparseSliceGradCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                               const std::vector<kernel::AddressPtr> &,
                                               const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSparseSliceGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSparseSliceGradOutputsNum, kernel_name_);
  auto backprop_val_grad = static_cast<T *>(inputs[kIndex0]->addr);
  auto indices = static_cast<int64_t *>(inputs[kIndex1]->addr);
  auto start = static_cast<int64_t *>(inputs[kIndex2]->addr);
  auto new_indices = static_cast<int64_t *>(inputs[kIndex3]->addr);
  auto y_grad = static_cast<T *>(outputs[kIndex0]->addr);

  SliceGradCompute<T>(backprop_val_grad, indices, start, new_indices, y_grad);

  return true;
}

template <typename T>
void SparseSliceGradCpuKernelMod::SliceGradCompute(T *backprop_val_grad, int64_t *indices, int64_t *start,
                                                   int64_t *new_indices, T *y_grad) const {
  (void)memset_s(y_grad, LongToSize(nnz_) * sizeof(T), 0, LongToSize(nnz_) * sizeof(T));

  ShapeVector indices_shape = ShapeVector({nnz_, rank_});
  ShapeVector new_indices_shape = ShapeVector({slice_nnz_, rank_});
  ShapeVector start_shape = ShapeVector({rank_});

  const auto indices_t = (EigenTensor(indices_shape, indices)).matrix<int64_t>();
  const auto new_indices_t = (EigenTensor(new_indices_shape, new_indices)).matrix<int64_t>();
  const auto start_flat = EigenTensor(start_shape, start).flat<int64_t>();

  int64_t j = 0;
  for (int64_t i = 0; i < nnz_ && j < slice_nnz_; ++i) {
    bool is_same = true;
    for (int64_t d = 0; d < rank_; ++d) {
      const int64_t indices_value = indices_t(i, d);
      const int64_t new_indices_value = new_indices_t(j, d);
      const int64_t offset = start_flat(d);
      if (indices_value != new_indices_value + offset) {
        is_same = false;
        break;
      }
    }
    if (is_same) {
      y_grad[i] = *(backprop_val_grad + j);
      ++j;
    }
  }
}

const std::vector<std::pair<KernelAttr, SparseSliceGradCpuKernelMod::KernelRunFunc>>
  &SparseSliceGradCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, SparseSliceGradCpuKernelMod::KernelRunFunc>> func_list = {
    ADD_KERNEL(Bool, bool),           ADD_KERNEL(UInt8, uint8_t),         ADD_KERNEL(UInt16, uint16_t),
    ADD_KERNEL(Int8, int8_t),         ADD_KERNEL(Int16, int16_t),         ADD_KERNEL(Int32, int),
    ADD_KERNEL(UInt32, uint32_t),     ADD_KERNEL(UInt64, uint64_t),       ADD_KERNEL(Int64, int64_t),
    ADD_KERNEL(Float16, float16),     ADD_KERNEL(Float32, float),         ADD_KERNEL(Float64, double),
    ADD_KERNEL(Complex64, complex64), ADD_KERNEL(Complex128, complex128),
  };
  return func_list;
}  // namespace kernel
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseSliceGrad, SparseSliceGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
