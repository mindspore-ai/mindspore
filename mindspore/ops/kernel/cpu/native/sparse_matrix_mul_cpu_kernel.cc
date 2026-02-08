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

#include "kernel/cpu/native/sparse_matrix_mul_cpu_kernel.h"
#include <algorithm>
#include <map>
#include <set>
#include <utility>
#include "include/utils/thread_pool.h"
#include "mindspore/ops/infer/sparse_matrix_mul.h"

namespace mindspore {
namespace kernel {
namespace sparse_matrix_mul_cpu {
namespace {
constexpr size_t kInputNum = 6;
constexpr size_t kOutputNum = 5;
constexpr size_t kAShapeIdx = 0;
constexpr size_t kABatchPointersIdx = 1;
constexpr size_t kAIndptrIdx = 2;
constexpr size_t kAIndicesIdx = 3;
constexpr size_t kAValuesIdx = 4;
constexpr size_t kBDenseIdx = 5;
constexpr size_t kOutShapeIdx = 0;
constexpr size_t kOutBatchPointersIdx = 1;
constexpr size_t kOutIndptrIdx = 2;
constexpr size_t kOutIndicesIdx = 3;
constexpr size_t kOutValuesIdx = 4;
constexpr size_t bShapeNum1 = 1;
constexpr size_t bShapeNum2 = 2;
using KernelRunFunc = SparseMatrixMulCpuKernelMod::KernelRunFunc;
}  // namespace
bool SparseMatrixMulCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }
  return true;
}

int SparseMatrixMulCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  int ret = KRET_OK;
  if ((ret = KernelMod::Resize(inputs, outputs)) != 0) {
    MS_LOG(ERROR) << kernel_name_ << " reinit failed.";
    return ret;
  }
  std::vector<int64_t> b_shape = inputs[kBDenseIdx]->GetShapeVector();
  size_t b_shape_num = b_shape.size();
  if (b_shape_num == bShapeNum1) {
    col_ = LongToSize(b_shape[0]);
  } else if (b_shape_num == bShapeNum2) {
    row_ = LongToSize(b_shape[0]);
    col_ = LongToSize(b_shape[1]);
  }
  return ret;
}

template <typename T, typename S>
const bool SparseMatrixMulCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                     const std::vector<KernelTensor *> &,
                                                     const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  const auto a_shape = reinterpret_cast<T *>(inputs[kAShapeIdx]->device_ptr());
  const auto a_batch_pointers = reinterpret_cast<T *>(inputs[kABatchPointersIdx]->device_ptr());
  const auto a_indptr = reinterpret_cast<T *>(inputs[kAIndptrIdx]->device_ptr());
  const auto a_indices = reinterpret_cast<T *>(inputs[kAIndicesIdx]->device_ptr());
  const auto a_values = reinterpret_cast<S *>(inputs[kAValuesIdx]->device_ptr());
  const auto b_dense = reinterpret_cast<S *>(inputs[kBDenseIdx]->device_ptr());

  auto c_shape = reinterpret_cast<T *>(outputs[kOutShapeIdx]->device_ptr());
  auto c_batch_pointers = reinterpret_cast<T *>(outputs[kOutBatchPointersIdx]->device_ptr());
  auto c_indptr = reinterpret_cast<T *>(outputs[kOutIndptrIdx]->device_ptr());
  auto c_indices = reinterpret_cast<T *>(outputs[kOutIndicesIdx]->device_ptr());
  auto c_values = reinterpret_cast<S *>(outputs[kOutValuesIdx]->device_ptr());
  const int64_t a_indices_num = SizeToLong(inputs[kAIndicesIdx]->size() / (sizeof(T)));
  const int64_t b_dense_num = SizeToLong(inputs[kBDenseIdx]->size() / (sizeof(S)));

  errno_t ret = memcpy_s(c_batch_pointers, inputs[kABatchPointersIdx]->size(), a_batch_pointers,
                         inputs[kABatchPointersIdx]->size());
  ret += memcpy_s(c_shape, inputs[kAShapeIdx]->size(), a_shape, inputs[kAShapeIdx]->size());
  ret += memcpy_s(c_indptr, inputs[kAIndptrIdx]->size(), a_indptr, inputs[kAIndptrIdx]->size());
  ret += memcpy_s(c_indices, inputs[kAIndicesIdx]->size(), a_indices, inputs[kAIndicesIdx]->size());
  if (ret != EOK) {
    MS_LOG(ERROR) << kernel_name_ << "memcpy_s failed.";
  }

  int64_t index = 0;
  for (int i = 0; i < a_indices_num; i++) {
    int64_t col = a_indices[i];
    int64_t row = 0;
    while (true) {
      if (i >= a_indptr[index] && i < a_indptr[index + 1]) {
        row = index;
        break;
      } else {
        index++;
      }
    }
    int64_t absIndex = row * SizeToLong(col_) + col;
    if (absIndex < b_dense_num) {
      c_values[i] = a_values[i] * b_dense[absIndex];
    } else {
      c_values[i] = 0;
    }
  }
  return true;
}

#define CPU_SPARSE_MATRIX_MUL_KERNEL_REGISTER(ms_index_type, ms_value_type, index_type, value_type) \
  {KernelAttr()                                                                                     \
     .AddInputAttr(ms_index_type)                                                                   \
     .AddInputAttr(ms_index_type)                                                                   \
     .AddInputAttr(ms_index_type)                                                                   \
     .AddInputAttr(ms_index_type)                                                                   \
     .AddInputAttr(ms_value_type)                                                                   \
     .AddInputAttr(ms_value_type)                                                                   \
     .AddOutputAttr(ms_index_type)                                                                  \
     .AddOutputAttr(ms_index_type)                                                                  \
     .AddOutputAttr(ms_index_type)                                                                  \
     .AddOutputAttr(ms_index_type)                                                                  \
     .AddOutputAttr(ms_value_type),                                                                 \
   &SparseMatrixMulCpuKernelMod::LaunchKernel<index_type, value_type>}

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &SparseMatrixMulCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    // float values
    CPU_SPARSE_MATRIX_MUL_KERNEL_REGISTER(kNumberTypeInt32, kNumberTypeFloat32, int, float),
    CPU_SPARSE_MATRIX_MUL_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeFloat32, int64_t, float),
    // double values
    CPU_SPARSE_MATRIX_MUL_KERNEL_REGISTER(kNumberTypeInt32, kNumberTypeFloat64, int, double),
    CPU_SPARSE_MATRIX_MUL_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeFloat64, int64_t, double),
    // int values
    CPU_SPARSE_MATRIX_MUL_KERNEL_REGISTER(kNumberTypeInt32, kNumberTypeInt32, int, int),
    CPU_SPARSE_MATRIX_MUL_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeInt32, int64_t, int),
    // int64 values
    CPU_SPARSE_MATRIX_MUL_KERNEL_REGISTER(kNumberTypeInt32, kNumberTypeInt64, int, int64_t),
    CPU_SPARSE_MATRIX_MUL_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeInt64, int64_t, int64_t),
    // int16 values
    CPU_SPARSE_MATRIX_MUL_KERNEL_REGISTER(kNumberTypeInt32, kNumberTypeInt16, int, int16_t),
    CPU_SPARSE_MATRIX_MUL_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeInt16, int64_t, int16_t),
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseMatrixMul, SparseMatrixMulCpuKernelMod);
}  // namespace sparse_matrix_mul_cpu
}  // namespace kernel
}  // namespace mindspore
