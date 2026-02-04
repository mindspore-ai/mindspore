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

#include "kernel/gpu/cuda/sparse/sparse_matrix_add_gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
#define GPU_SPARSE_MATRIX_ADD_KERNEL_REGISTER(ms_dtype, cuda_type) \
  MS_REG_GPU_KERNEL_ONE(SparseMatrixAdd,                           \
                        KernelAttr()                               \
                          .AddInputAttr(kNumberTypeInt32)          \
                          .AddInputAttr(kNumberTypeInt32)          \
                          .AddInputAttr(kNumberTypeInt32)          \
                          .AddInputAttr(kNumberTypeInt32)          \
                          .AddInputAttr(ms_dtype)                  \
                          .AddInputAttr(kNumberTypeInt32)          \
                          .AddInputAttr(kNumberTypeInt32)          \
                          .AddInputAttr(kNumberTypeInt32)          \
                          .AddInputAttr(kNumberTypeInt32)          \
                          .AddInputAttr(ms_dtype)                  \
                          .AddInputAttr(ms_dtype)                  \
                          .AddInputAttr(ms_dtype)                  \
                          .AddOutputAttr(kNumberTypeInt32)         \
                          .AddOutputAttr(kNumberTypeInt32)         \
                          .AddOutputAttr(kNumberTypeInt32)         \
                          .AddOutputAttr(kNumberTypeInt32)         \
                          .AddOutputAttr(ms_dtype),                \
                        SparseMatrixAddGpuKernel, cuda_type)

GPU_SPARSE_MATRIX_ADD_KERNEL_REGISTER(kNumberTypeFloat32, float)
GPU_SPARSE_MATRIX_ADD_KERNEL_REGISTER(kNumberTypeFloat64, double)
GPU_SPARSE_MATRIX_ADD_KERNEL_REGISTER(kNumberTypeComplex64, cuComplex)
GPU_SPARSE_MATRIX_ADD_KERNEL_REGISTER(kNumberTypeComplex128, cuDoubleComplex)
}  // namespace kernel
}  // namespace mindspore
