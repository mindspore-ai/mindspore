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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPARSE_MATRIX_SPARSE_MATMUL_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPARSE_MATRIX_SPARSE_MATMUL_GPU_KERNEL_H_
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <algorithm>
#include <functional>
#include <map>
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"
#include "mindspore/ops/infer/sparse_matrix_sparse_mat_mul.h"
namespace mindspore {
namespace kernel {
class SparseMatrixSparseMatMulGpuKernelMod : public NativeGpuKernelMod {
 public:
  SparseMatrixSparseMatMulGpuKernelMod() {
    handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCuSparseHandle();
    cusparseSetPointerMode(handle_, CUSPARSE_POINTER_MODE_HOST);
    ResetResource();
  }
  ~SparseMatrixSparseMatMulGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *cuda_stream) override {
    stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 protected:
  void ResetResource() noexcept {
    x1_num_rows = 0;
    x1_num_cols = 0;
    x1_nnz = 0;
    x2_num_rows = 0;
    x2_num_cols = 0;
    x2_nnz = 0;
    C_num_rows1 = 0;
    C_num_cols1 = 0;
    C_nnz1 = 0;
    transpose_a = false;
    transpose_b = false;
    output_size_list_.clear();
  }

  std::vector<KernelAttr> GetOpSupport() override;
  bool IsNeedUpdateOutputShapeAndSize() override { return true; }
  void UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                const std::vector<KernelTensor *> &outputs) override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);
  template <typename T>
  void MatrixTranspose(int m, int n, int nnz, void *csrVal, const int *csrRowPtr, const int *csrColInd, void *csrVal_t,
                       int *csrRowPtr_t, int *csrColInd_t);
  using SparseMatrixSparseMatMulFunc =
    std::function<bool(SparseMatrixSparseMatMulGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &)>;
  void Compute(cusparseHandle_t handle, cusparseMatDescr_t desc, csrgemm2Info_t info, const int m, const int n,
               const int k, float *A_val, const int *const A_colind, const int *const A_rowptr, const int A_nnz,
               float *B_val, const int *const B_colind, const int *const B_rowptr, const int B_nnz, float **C_val,
               int **const C_colind, int **const C_rowptr, int *const C_nnz);
  void Compute(cusparseHandle_t handle, cusparseMatDescr_t desc, csrgemm2Info_t info, const int m, const int n,
               const int k, double *A_val, const int *const A_colind, const int *const A_rowptr, const int A_nnz,
               double *B_val, const int *const B_colind, const int *const B_rowptr, const int B_nnz, double **C_val,
               int **const C_colind, int **const C_rowptr, int *const C_nnz);
  void Compute(cusparseHandle_t handle, cusparseMatDescr_t desc, csrgemm2Info_t info, const int m, const int n,
               const int k, mindspore::utils::Complex<float> *A_val, const int *const A_colind,
               const int *const A_rowptr, const int A_nnz, mindspore::utils::Complex<float> *B_val,
               const int *const B_colind, const int *const B_rowptr, const int B_nnz,
               mindspore::utils::Complex<float> **C_val, int **const C_colind, int **const C_rowptr, int *const C_nnz);
  void Compute(cusparseHandle_t handle, cusparseMatDescr_t desc, csrgemm2Info_t info, const int m, const int n,
               const int k, mindspore::utils::Complex<double> *A_val, const int *const A_colind,
               const int *const A_rowptr, const int A_nnz, mindspore::utils::Complex<double> *B_val,
               const int *const B_colind, const int *const B_rowptr, const int B_nnz,
               mindspore::utils::Complex<double> **C_val, int **const C_colind, int **const C_rowptr, int *const C_nnz);
  template <typename T>
  void Core(int x1_num_rows, int x1_num_cols, int x2_num_rows, int x2_num_cols, int batch_index,
            const std::vector<int> &h_x1_batch_num, const std::vector<int> &h_x2_batch_num, int *h_y_batch_num,
            int *x1_row_pointers, int *x1_col_indices, T *x1_values, int *x2_row_pointers, int *x2_col_indices,
            T *x2_values, int *y_row_pointers, int *y_col_indices, T *y_values, bool adjoint_a, bool adjoint_b,
            bool transpose_a, bool transpose_b);
  template <typename T>
  void MemFree(bool adjoint_a, bool adjoint_b, bool transpose_a, bool transpose_b, T *x1_Val_c, void *x1_Val_t,
               int *x1_RowPtr_t, int *x1_ColInd_t, T *x2_Val_c, void *x2_Val_t, int *x2_RowPtr_t, int *x2_ColInd_t);

 private:
  cusparseHandle_t handle_{nullptr};
  cusparseMatDescr_t desc;
  csrgemm2Info_t info = NULL;
  SparseMatrixSparseMatMulFunc kernel_func_{};
  cusparseIndexBase_t idxBase = CUSPARSE_INDEX_BASE_ZERO;

  cudaDataType_t computeType;
  cusparseAction_t action = CUSPARSE_ACTION_NUMERIC;
  cusparseCsr2CscAlg_t alg = CUSPARSE_CSR2CSC_ALG1;
  int x1_num_rows;
  int x1_num_cols;
  int x1_nnz;
  int x2_num_rows;
  int x2_num_cols;
  int x2_nnz;
  int64_t C_num_rows1 = 0;
  int64_t C_num_cols1 = 0;
  int64_t C_nnz1 = 0;
  bool transpose_a;
  bool transpose_b;
  bool adjoint_a{false};
  bool adjoint_b{false};
  device::gpu::GPUMemoryAllocator &allocator = device::gpu::GPUMemoryAllocator::GetInstance();

  cudaStream_t stream;
  int rank = 0;
  static std::vector<std::pair<KernelAttr, SparseMatrixSparseMatMulFunc>> func_list_;
  std::vector<std::vector<size_t>> ele_size_vec;
  enum InputList {
    X1_DENSE_SHAPE,
    X1_BATCH_POINTERS,
    X1_ROW_POINTERS,
    X1_COL_INDICES,
    X1_VALUES,
    X2_DENSE_SHAPE,
    X2_BATCH_POINTERS,
    X2_ROW_POINTERS,
    X2_COL_INDICES,
    X2_VALUES
  };
  int batch_ele = 1;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPARSE_MATRIX_SPARSE_MATMUL_GPU_KERNEL_H_
