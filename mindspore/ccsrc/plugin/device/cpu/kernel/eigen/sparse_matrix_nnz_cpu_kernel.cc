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

#include "plugin/device/cpu/kernel/eigen/sparse_matrix_nnz_cpu_kernel.h"
#include <algorithm>
#include "plugin/device/cpu/kernel/eigen/eigen_common_utils.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputsNum = 5;
constexpr size_t kOutputsNum = 1;
constexpr size_t kInputIndex0 = 0;
constexpr size_t kInputIndex1 = 1;
constexpr size_t kOutputIndex0 = 0;

#define ADD_KERNEL(t1, t2, t3, t4, t5, t6) \
  KernelAttr()                             \
    .AddInputAttr(kNumberType##t1)         \
    .AddInputAttr(kNumberType##t2)         \
    .AddInputAttr(kNumberType##t3)         \
    .AddInputAttr(kNumberType##t4)         \
    .AddInputAttr(kNumberType##t5)         \
    .AddOutputAttr(kNumberType##t6)
}  // namespace

void SparseMatrixNNZCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  CHECK_KERNEL_INPUTS_NUM(input_num, kInputsNum, kernel_name_);
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kOutputsNum, kernel_name_);
  auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kInputIndex0);
  const int rank_x = input_shape[0];
  const int kInputNoBatch = 2;
  const int kInputWithBatch = 3;
  if (rank_x != kInputNoBatch && rank_x != kInputWithBatch) {
    MS_LOG(EXCEPTION) << "For SparseMatrixNNZ, the shape of x_dense_shape must be (2,) or (3,), but got (" << rank_x
                      << ",).";
  }
  batch_size_ = static_cast<size_t>(common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kInputIndex1)[0] - 1);
  value_type_ = AnfAlgo::GetInputDeviceDataType(kernel_node, kInputIndex0);
}

bool SparseMatrixNNZCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                         const std::vector<kernel::AddressPtr> &,
                                         const std::vector<kernel::AddressPtr> &outputs) {
  switch (value_type_) {
    case kNumberTypeInt32:
      DoLaunch<int32_t>(inputs, outputs);
      break;
    case kNumberTypeInt64:
      DoLaunch<int64_t>(inputs, outputs);
      break;
    default:
      MS_LOG(EXCEPTION) << "For SparseMatrixNNZ, data type of (x_dense_shape | x_batch_pointers | x_row_pointers | "
                           "x_col_indices) is not int32 or int64";
  }

  return true;
}

template <typename T>
void SparseMatrixNNZCpuKernelMod::DoLaunch(const std::vector<kernel::AddressPtr> &inputs,
                                           const std::vector<kernel::AddressPtr> &outputs) {
  auto batch_pointers_x = static_cast<T *>(inputs[kInputIndex1]->addr);
  auto output_addr = static_cast<int32_t *>(outputs[kOutputIndex0]->addr);

  int64_t curr = 0;
  for (size_t i = 1; i < batch_size_ + 1; i++) {
    output_addr[i - 1] = batch_pointers_x[i] - curr;
    // update curr
    curr = batch_pointers_x[i];
  }
}

std::vector<KernelAttr> SparseMatrixNNZCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> kernel_attr_list = {ADD_KERNEL(Int32, Int32, Int32, Int32, Int8, Int32),
                                                     ADD_KERNEL(Int32, Int32, Int32, Int32, UInt8, Int32),
                                                     ADD_KERNEL(Int32, Int32, Int32, Int32, Int16, Int32),
                                                     ADD_KERNEL(Int32, Int32, Int32, Int32, UInt16, Int32),
                                                     ADD_KERNEL(Int32, Int32, Int32, Int32, Int32, Int32),
                                                     ADD_KERNEL(Int32, Int32, Int32, Int32, Int64, Int32),
                                                     ADD_KERNEL(Int32, Int32, Int32, Int32, Float16, Int32),
                                                     ADD_KERNEL(Int32, Int32, Int32, Int32, Float32, Int32),
                                                     ADD_KERNEL(Int32, Int32, Int32, Int32, Float64, Int32),
                                                     ADD_KERNEL(Int32, Int32, Int32, Int32, Bool, Int32),
                                                     ADD_KERNEL(Int32, Int32, Int32, Int32, Complex64, Int32),
                                                     ADD_KERNEL(Int32, Int32, Int32, Int32, Complex128, Int32),
                                                     ADD_KERNEL(Int64, Int64, Int64, Int64, Int8, Int32),
                                                     ADD_KERNEL(Int64, Int64, Int64, Int64, UInt8, Int32),
                                                     ADD_KERNEL(Int64, Int64, Int64, Int64, Int16, Int32),
                                                     ADD_KERNEL(Int64, Int64, Int64, Int64, UInt16, Int32),
                                                     ADD_KERNEL(Int64, Int64, Int64, Int64, Int32, Int32),
                                                     ADD_KERNEL(Int64, Int64, Int64, Int64, Int64, Int32),
                                                     ADD_KERNEL(Int64, Int64, Int64, Int64, Float16, Int32),
                                                     ADD_KERNEL(Int64, Int64, Int64, Int64, Float32, Int32),
                                                     ADD_KERNEL(Int64, Int64, Int64, Int64, Float64, Int32),
                                                     ADD_KERNEL(Int64, Int64, Int64, Int64, Bool, Int32),
                                                     ADD_KERNEL(Int64, Int64, Int64, Int64, Complex64, Int32),
                                                     ADD_KERNEL(Int64, Int64, Int64, Int64, Complex128, Int32)};

  return kernel_attr_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseMatrixNNZ, SparseMatrixNNZCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
