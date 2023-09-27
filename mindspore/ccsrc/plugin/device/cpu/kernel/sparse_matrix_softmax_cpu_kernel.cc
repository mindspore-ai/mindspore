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

#include "plugin/device/cpu/kernel/sparse_matrix_softmax_cpu_kernel.h"

#include <cmath>
#include <thread>
#include <vector>
#include <algorithm>
#include <map>
#include <string>
#include <functional>

#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputNum = 5;
constexpr size_t kOutputNum = 5;
constexpr size_t logits_dense_shape = 0;
constexpr size_t logits_batch_pointers = 1;
constexpr size_t logits_col_indices = 2;
constexpr size_t logits_row_pointers = 3;
constexpr size_t logits_values = 4;
constexpr char kKernelName[] = "sparse_matrix_softmax";
}  // namespace
bool SparseMatrixSoftmaxCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &outputs) {
  dtype_ = inputs[logits_values]->dtype_id();
  size_t input_num = inputs.size();
  if (input_num != kInputNum) {
    MS_LOG(ERROR) << "For " << kernel_name_
                  << ", input should be x_dense_shape, x_batch_pointers, x_row_pointers, x_col_indices, x_values "
                  << kInputNum << " tensors, but get " << input_num;
    return false;
  }
  return true;
}
int SparseMatrixSoftmaxCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    return ret;
  }
  return 0;
}

bool SparseMatrixSoftmaxCpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                                             const std::vector<kernel::KernelTensor *> &,
                                             const std::vector<kernel::KernelTensor *> &outputs) {
  if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat64) {
    LaunchKernel<double>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dtype of 'logits_values' must be Float32 or Float64, but got "
                      << TypeIdToType(dtype_)->ToString();
  }
  return true;
}

template <typename T>

void SparseMatrixSoftmaxCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                                   const std::vector<kernel::KernelTensor *> &) {
  auto *input_logits_values = reinterpret_cast<T *>(inputs[logits_values]->device_ptr());
  auto *input_logits_dense_shape = reinterpret_cast<int *>(inputs[logits_dense_shape]->device_ptr());
  auto *input_logits_col_indices = reinterpret_cast<int *>(inputs[logits_col_indices]->device_ptr());
  T total = 0;
  T MAX = input_logits_values[0];
  int row_index = input_logits_dense_shape[0];
  int start = 0;
  for (int i = 1; i <= row_index; i++) {
    int single_index = (input_logits_col_indices[i] - input_logits_col_indices[i - 1]);
    for (int k = 0; k < single_index; k++) {
      if (input_logits_values[k + start] > MAX) {
        MAX = input_logits_values[k + start];
      }
    }
    for (int k = 0; k < single_index; k++) {
      total = total + exp(input_logits_values[k + start] - MAX);
    }
    for (int k = 0; k < single_index; k++) {
      input_logits_values[k + start] = exp(input_logits_values[k + start] - MAX) / total;
    }
    start = start + single_index;
    MAX = input_logits_values[start];
    total = 0;
  }
}

std::vector<KernelAttr> SparseMatrixSoftmaxCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> kernel_attr_list = {KernelAttr()
                                                       .AddInputAttr(kNumberTypeInt32)
                                                       .AddInputAttr(kNumberTypeInt32)
                                                       .AddInputAttr(kNumberTypeInt32)
                                                       .AddInputAttr(kNumberTypeInt32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddOutputAttr(kNumberTypeInt32)
                                                       .AddOutputAttr(kNumberTypeInt32)
                                                       .AddOutputAttr(kNumberTypeInt32)
                                                       .AddOutputAttr(kNumberTypeInt32)
                                                       .AddOutputAttr(kNumberTypeFloat32)
                                                       .AddOutInRef(0, 0)
                                                       .AddOutInRef(1, 1)
                                                       .AddOutInRef(2, 2)
                                                       .AddOutInRef(3, 3)
                                                       .AddOutInRef(4, 4),
                                                     KernelAttr()
                                                       .AddInputAttr(kNumberTypeInt64)
                                                       .AddInputAttr(kNumberTypeInt64)
                                                       .AddInputAttr(kNumberTypeInt64)
                                                       .AddInputAttr(kNumberTypeInt64)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddOutputAttr(kNumberTypeInt64)
                                                       .AddOutputAttr(kNumberTypeInt64)
                                                       .AddOutputAttr(kNumberTypeInt64)
                                                       .AddOutputAttr(kNumberTypeInt64)
                                                       .AddOutputAttr(kNumberTypeFloat32)
                                                       .AddOutInRef(0, 0)
                                                       .AddOutInRef(1, 1)
                                                       .AddOutInRef(2, 2)
                                                       .AddOutInRef(3, 3)
                                                       .AddOutInRef(4, 4),
                                                     KernelAttr()
                                                       .AddInputAttr(kNumberTypeInt32)
                                                       .AddInputAttr(kNumberTypeInt32)
                                                       .AddInputAttr(kNumberTypeInt32)
                                                       .AddInputAttr(kNumberTypeInt32)
                                                       .AddInputAttr(kNumberTypeFloat64)
                                                       .AddOutputAttr(kNumberTypeInt32)
                                                       .AddOutputAttr(kNumberTypeInt32)
                                                       .AddOutputAttr(kNumberTypeInt32)
                                                       .AddOutputAttr(kNumberTypeInt32)
                                                       .AddOutputAttr(kNumberTypeFloat64)
                                                       .AddOutInRef(0, 0)
                                                       .AddOutInRef(1, 1)
                                                       .AddOutInRef(2, 2)
                                                       .AddOutInRef(3, 3)
                                                       .AddOutInRef(4, 4),
                                                     KernelAttr()
                                                       .AddInputAttr(kNumberTypeInt64)
                                                       .AddInputAttr(kNumberTypeInt64)
                                                       .AddInputAttr(kNumberTypeInt64)
                                                       .AddInputAttr(kNumberTypeInt64)
                                                       .AddInputAttr(kNumberTypeFloat64)
                                                       .AddOutputAttr(kNumberTypeInt64)
                                                       .AddOutputAttr(kNumberTypeInt64)
                                                       .AddOutputAttr(kNumberTypeInt64)
                                                       .AddOutputAttr(kNumberTypeInt64)
                                                       .AddOutputAttr(kNumberTypeFloat64)
                                                       .AddOutInRef(0, 0)
                                                       .AddOutInRef(1, 1)
                                                       .AddOutInRef(2, 2)
                                                       .AddOutInRef(3, 3)
                                                       .AddOutInRef(4, 4)};
  return kernel_attr_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseMatrixSoftmax, SparseMatrixSoftmaxCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
