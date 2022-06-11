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

#include "plugin/device/gpu/kernel/sparse/dense_to_csr_sparse_matrix_gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_TWO(DenseToCSRSparseMatrix,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeBool)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeBool),
                      DenseToCSRSparseMatrixKernelMod, bool, int)
MS_REG_GPU_KERNEL_TWO(DenseToCSRSparseMatrix,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt8)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt8),
                      DenseToCSRSparseMatrixKernelMod, int8_t, int)
MS_REG_GPU_KERNEL_TWO(DenseToCSRSparseMatrix,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt16)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt16),
                      DenseToCSRSparseMatrixKernelMod, int16_t, int)
MS_REG_GPU_KERNEL_TWO(DenseToCSRSparseMatrix,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32),
                      DenseToCSRSparseMatrixKernelMod, int, int)
MS_REG_GPU_KERNEL_TWO(DenseToCSRSparseMatrix,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt64),
                      DenseToCSRSparseMatrixKernelMod, int64_t, int)
MS_REG_GPU_KERNEL_TWO(DenseToCSRSparseMatrix,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeUInt8),
                      DenseToCSRSparseMatrixKernelMod, uint8_t, int)
MS_REG_GPU_KERNEL_TWO(DenseToCSRSparseMatrix,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt16)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeUInt16),
                      DenseToCSRSparseMatrixKernelMod, uint16_t, int)
MS_REG_GPU_KERNEL_TWO(DenseToCSRSparseMatrix,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeUInt32),
                      DenseToCSRSparseMatrixKernelMod, uint, int)
MS_REG_GPU_KERNEL_TWO(DenseToCSRSparseMatrix,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeUInt64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeUInt64),
                      DenseToCSRSparseMatrixKernelMod, uint64_t, int)
MS_REG_GPU_KERNEL_TWO(DenseToCSRSparseMatrix,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeFloat16),
                      DenseToCSRSparseMatrixKernelMod, half, int)
MS_REG_GPU_KERNEL_TWO(DenseToCSRSparseMatrix,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeFloat32),
                      DenseToCSRSparseMatrixKernelMod, float, int)
MS_REG_GPU_KERNEL_TWO(DenseToCSRSparseMatrix,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeFloat64),
                      DenseToCSRSparseMatrixKernelMod, double, int)
MS_REG_GPU_KERNEL_TWO(DenseToCSRSparseMatrix,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeComplex64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeComplex64),
                      DenseToCSRSparseMatrixKernelMod, cuComplex, int)
MS_REG_GPU_KERNEL_TWO(DenseToCSRSparseMatrix,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeComplex128)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeComplex128),
                      DenseToCSRSparseMatrixKernelMod, cuDoubleComplex, int)
}  // namespace kernel
}  // namespace mindspore
