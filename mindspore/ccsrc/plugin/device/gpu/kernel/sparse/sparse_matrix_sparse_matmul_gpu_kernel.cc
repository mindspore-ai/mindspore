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
#include "plugin/device/gpu/kernel/sparse/sparse_matrix_sparse_matmul_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/unary_op_impl.cuh"
namespace mindspore {
namespace kernel {
template <typename T>
void SparseMatrixSparseMatMulGpuKernelMod::MatrixTranspose(int m, int n, int nnz, void *csrVal, const int *csrRowPtr,
                                                           const int *csrColInd, void *csrVal_t, int *csrRowPtr_t,
                                                           int *csrColInd_t) {
  size_t csc_buffer_size;
  void *csc_buffer;

  CHECK_CUSPARSE_RET_WITH_EXCEPT(
    cusparseCsr2cscEx2_bufferSize(handle_, m, n, nnz, csrVal, csrRowPtr, csrColInd, csrVal_t, csrRowPtr_t, csrColInd_t,
                                  computeType, action, idxBase, alg, &csc_buffer_size),
    "cusparseCsr2cscEx2_bufferSize failed.");
  csc_buffer = allocator.AllocTensorMem(csc_buffer_size);
  CHECK_CUSPARSE_RET_WITH_EXCEPT(
    cusparseCsr2cscEx2(handle_, m, n, nnz, csrVal, csrRowPtr, csrColInd, csrVal_t, csrRowPtr_t, csrColInd_t,
                       computeType, action, idxBase, alg, csc_buffer),
    "cusparseCsr2cscEx2 failed.");
  allocator.FreeTensorMem(csc_buffer);
}

bool SparseMatrixSparseMatMulGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                                const std::vector<KernelTensorPtr> &inputs,
                                                const std::vector<KernelTensorPtr> &outputs) {
  cusparseCreateMatDescr(&desc);
  cusparseSetMatType(desc, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(desc, CUSPARSE_INDEX_BASE_ZERO);

  cusparseCreateCsrgemm2Info(&info);
  kernel_name_ = base_operator->GetPrim()->name();
  auto kernel_ptr = std::dynamic_pointer_cast<ops::SparseMatrixSparseMatMul>(base_operator);
  transpose_a = kernel_ptr->get_transpose_a();
  transpose_b = kernel_ptr->get_transpose_b();
  adjoint_a = kernel_ptr->get_adjoint_a();
  adjoint_b = kernel_ptr->get_adjoint_b();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the kernel type should be in [float32, float64, complex64, complex128], but got: "
                  << kernel_attr << ".";
    return false;
  }
  kernel_func_ = func_list_[index].second;
  if (inputs.at(kIndex4)->GetDtype() == TypeId::kNumberTypeFloat32) {
    computeType = CUDA_R_32F;
  }
  if (inputs.at(kIndex4)->GetDtype() == TypeId::kNumberTypeFloat64) {
    computeType = CUDA_R_64F;
  }
  if (inputs.at(kIndex4)->GetDtype() == TypeId::kNumberTypeComplex64) {
    computeType = CUDA_C_32F;
  }
  if (inputs.at(kIndex4)->GetDtype() == TypeId::kNumberTypeComplex128) {
    computeType = CUDA_C_64F;
  }
  is_need_retrieve_output_shape_ = true;
  for (size_t i = 0; i < inputs.size(); i++) {
    std::vector<int64_t> input_shape = std::vector<int64_t>(inputs.at(i)->GetDeviceShapeAdaptively().begin(),
                                                            inputs.at(i)->GetDeviceShapeAdaptively().end());

    size_t input_elements_ = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int64_t>());
    size_t unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(i).first);

    std::vector<size_t> temp{input_elements_, unit_size_};
    ele_size_vec.push_back(temp);
    input_size_list_.push_back(input_elements_ * unit_size_);
  }

  rank = ele_size_vec[InputList::X1_DENSE_SHAPE][0];
  x1_nnz = ele_size_vec[InputList::X1_VALUES][0];
  x2_nnz = ele_size_vec[InputList::X2_VALUES][0];
  batch_ele = ele_size_vec[InputList::X1_BATCH_POINTERS][0];
  for (size_t i = 0; i < outputs.size(); i++) {
    std::vector<int64_t> output_shape = std::vector<int64_t>(outputs.at(i)->GetDeviceShapeAdaptively().begin(),
                                                             outputs.at(i)->GetDeviceShapeAdaptively().end());
    size_t output_elements_ = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int64_t>());
    size_t unit_size_ = abstract::TypeIdSize(kernel_attr.GetOutputAttr(i).first);
    output_size_list_.push_back(output_elements_ * unit_size_);
  }
  return true;
}

int SparseMatrixSparseMatMulGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                 const std::vector<KernelTensorPtr> &inputs,
                                                 const std::vector<KernelTensorPtr> &outputs,
                                                 const std::map<uint32_t, tensor::TensorPtr> &) {
  outputs_ = outputs;
  return 0;
}

void SparseMatrixSparseMatMulGpuKernelMod::Compute(
  cusparseHandle_t handle, cusparseMatDescr_t desc, csrgemm2Info_t info, const int m, const int n, const int k,
  mindspore::utils::Complex<double> *A_val, const int *const A_colind, const int *const A_rowptr, const int A_nnz,
  mindspore::utils::Complex<double> *B_val, const int *const B_colind, const int *const B_rowptr, const int B_nnz,
  mindspore::utils::Complex<double> **C_val, int **const C_colind, int **const C_rowptr, int *const C_nnz) {
  cuDoubleComplex alpha{1.0, 0.0};
  size_t buffer_size;
  cusparseZcsrgemm2_bufferSizeExt(handle, m, n, k, &alpha, desc, A_nnz, A_rowptr, A_colind, desc, B_nnz, B_rowptr,
                                  B_colind, NULL, desc, B_nnz, B_rowptr, B_colind, info, &buffer_size);
  void *buffer = NULL;
  buffer = allocator.AllocTensorMem(buffer_size);

  cusparseXcsrgemm2Nnz(handle, m, n, k, desc, A_nnz, A_rowptr, A_colind, desc, B_nnz, B_rowptr, B_colind, desc, B_nnz,
                       B_rowptr, B_colind, desc, *C_rowptr, C_nnz, info, buffer);
  cusparseZcsrgemm2(handle, m, n, k, &alpha, desc, A_nnz, reinterpret_cast<cuDoubleComplex *>(A_val), A_rowptr,
                    A_colind, desc, B_nnz, reinterpret_cast<cuDoubleComplex *>(B_val), B_rowptr, B_colind, NULL, desc,
                    B_nnz, reinterpret_cast<cuDoubleComplex *>(B_val), B_rowptr, B_colind, desc,
                    *reinterpret_cast<cuDoubleComplex **>(C_val), *C_rowptr, *C_colind, info, buffer);
  allocator.FreeTensorMem(buffer);
}

void SparseMatrixSparseMatMulGpuKernelMod::Compute(
  cusparseHandle_t handle, cusparseMatDescr_t desc, csrgemm2Info_t info, const int m, const int n, const int k,
  mindspore::utils::Complex<float> *A_val, const int *const A_colind, const int *const A_rowptr, const int A_nnz,
  mindspore::utils::Complex<float> *B_val, const int *const B_colind, const int *const B_rowptr, const int B_nnz,
  mindspore::utils::Complex<float> **C_val, int **const C_colind, int **const C_rowptr, int *const C_nnz) {
  cuComplex alpha{1.0, 0.0};
  size_t buffer_size;
  cusparseCcsrgemm2_bufferSizeExt(handle, m, n, k, &alpha, desc, A_nnz, A_rowptr, A_colind, desc, B_nnz, B_rowptr,
                                  B_colind, NULL, desc, B_nnz, B_rowptr, B_colind, info, &buffer_size);
  void *buffer = NULL;
  buffer = allocator.AllocTensorMem(buffer_size);

  cusparseXcsrgemm2Nnz(handle, m, n, k, desc, A_nnz, A_rowptr, A_colind, desc, B_nnz, B_rowptr, B_colind, desc, B_nnz,
                       B_rowptr, B_colind, desc, *C_rowptr, C_nnz, info, buffer);
  cusparseCcsrgemm2(handle, m, n, k, &alpha, desc, A_nnz, reinterpret_cast<cuComplex *>(A_val), A_rowptr, A_colind,
                    desc, B_nnz, reinterpret_cast<cuComplex *>(B_val), B_rowptr, B_colind, NULL, desc, B_nnz,
                    reinterpret_cast<cuComplex *>(B_val), B_rowptr, B_colind, desc,
                    *reinterpret_cast<cuComplex **>(C_val), *C_rowptr, *C_colind, info, buffer);
  allocator.FreeTensorMem(buffer);
}

void SparseMatrixSparseMatMulGpuKernelMod::Compute(cusparseHandle_t handle, cusparseMatDescr_t desc,
                                                   csrgemm2Info_t info, const int m, const int n, const int k,
                                                   double *A_val, const int *const A_colind, const int *const A_rowptr,
                                                   const int A_nnz, double *B_val, const int *const B_colind,
                                                   const int *const B_rowptr, const int B_nnz, double **C_val,
                                                   int **const C_colind, int **const C_rowptr, int *const C_nnz) {
  double alpha = 1.0;
  size_t buffer_size;
  cusparseDcsrgemm2_bufferSizeExt(handle, m, n, k, &alpha, desc, A_nnz, A_rowptr, A_colind, desc, B_nnz, B_rowptr,
                                  B_colind, NULL, desc, B_nnz, B_rowptr, B_colind, info, &buffer_size);
  void *buffer = NULL;
  buffer = allocator.AllocTensorMem(buffer_size);
  cusparseXcsrgemm2Nnz(handle, m, n, k, desc, A_nnz, A_rowptr, A_colind, desc, B_nnz, B_rowptr, B_colind, desc, B_nnz,
                       B_rowptr, B_colind, desc, *C_rowptr, C_nnz, info, buffer);
  cusparseDcsrgemm2(handle, m, n, k, &alpha, desc, A_nnz, A_val, A_rowptr, A_colind, desc, B_nnz, B_val, B_rowptr,
                    B_colind, NULL, desc, B_nnz, B_val, B_rowptr, B_colind, desc, *C_val, *C_rowptr, *C_colind, info,
                    buffer);
  allocator.FreeTensorMem(buffer);
}

bool CheckShape(bool transpose_a, bool transpose_b, bool adjoint_a, bool adjoint_b, int A_row, int A_col, int B_row,
                int B_col) {
  bool trans_A = (transpose_a || adjoint_a);
  bool trans_B = (transpose_b || adjoint_b);
  if (trans_A & trans_B) {
    return (A_row == B_col);
  }
  if (trans_A) {
    return (A_row == B_row);
  }
  if (trans_B) {
    return (A_col == B_col);
  }
  return (A_col == B_row);
}

void SparseMatrixSparseMatMulGpuKernelMod::Compute(cusparseHandle_t handle, cusparseMatDescr_t desc,
                                                   csrgemm2Info_t info, const int m, const int n, const int k,
                                                   float *A_val, const int *const A_colind, const int *const A_rowptr,
                                                   const int A_nnz, float *B_val, const int *const B_colind,
                                                   const int *const B_rowptr, const int B_nnz, float **C_val,
                                                   int **const C_colind, int **const C_rowptr, int *const C_nnz) {
  float alpha = 1.0;
  size_t buffer_size;
  cusparseScsrgemm2_bufferSizeExt(handle, m, n, k, &alpha, desc, A_nnz, A_rowptr, A_colind, desc, B_nnz, B_rowptr,
                                  B_colind, NULL, desc, B_nnz, B_rowptr, B_colind, info, &buffer_size);
  void *buffer = NULL;
  buffer = allocator.AllocTensorMem(buffer_size);
  cusparseXcsrgemm2Nnz(handle, m, n, k, desc, A_nnz, A_rowptr, A_colind, desc, B_nnz, B_rowptr, B_colind, desc, B_nnz,
                       B_rowptr, B_colind, desc, *C_rowptr, C_nnz, info, buffer);
  cusparseScsrgemm2(handle, m, n, k, &alpha, desc, A_nnz, A_val, A_rowptr, A_colind, desc, B_nnz, B_val, B_rowptr,
                    B_colind, NULL, desc, B_nnz, B_val, B_rowptr, B_colind, desc, *C_val, *C_rowptr, *C_colind, info,
                    buffer);
  allocator.FreeTensorMem(buffer);
}

template <typename T>
void SparseMatrixSparseMatMulGpuKernelMod::MemFree(bool adjoint_a, bool adjoint_b, bool transpose_a, bool transpose_b,
                                                   T *x1_Val_c, void *x1_Val_t, int *x1_RowPtr_t, int *x1_ColInd_t,
                                                   T *x2_Val_c, void *x2_Val_t, int *x2_RowPtr_t, int *x2_ColInd_t) {
  if (transpose_a) {
    allocator.FreeTensorMem(x1_Val_t);
    allocator.FreeTensorMem(x1_RowPtr_t);
    allocator.FreeTensorMem(x1_ColInd_t);
  }
  if (transpose_b) {
    allocator.FreeTensorMem(x2_Val_t);
    allocator.FreeTensorMem(x2_RowPtr_t);
    allocator.FreeTensorMem(x2_ColInd_t);
  }
  if (adjoint_a) {
    allocator.FreeTensorMem(x1_Val_c);
    allocator.FreeTensorMem(x1_Val_t);
    allocator.FreeTensorMem(x1_RowPtr_t);
    allocator.FreeTensorMem(x1_ColInd_t);
  }
  if (adjoint_b) {
    allocator.FreeTensorMem(x2_Val_c);
    allocator.FreeTensorMem(x2_Val_t);
    allocator.FreeTensorMem(x2_RowPtr_t);
    allocator.FreeTensorMem(x2_ColInd_t);
  }
}

template <typename T>
void SparseMatrixSparseMatMulGpuKernelMod::Core(int x1_num_rows, int x1_num_cols, int x2_num_rows, int x2_num_cols,
                                                int batch_index, const std::vector<int> &h_x1_batch_num,
                                                const std::vector<int> &h_x2_batch_num, int *h_y_batch_num,
                                                int *x1_row_pointers, int *x1_col_indices, T *x1_values,
                                                int *x2_row_pointers, int *x2_col_indices, T *x2_values,
                                                int *y_row_pointers, int *y_col_indices, T *y_values, bool adjoint_a,
                                                bool adjoint_b, bool transpose_a, bool transpose_b) {
  int d_C_nnz, m, n, k;
  m = x1_num_rows;
  n = x1_num_cols;
  k = x2_num_cols;
  T *x1_Val_c = nullptr;
  void *x1_Val_t = nullptr;
  int *x1_RowPtr_t = nullptr;
  int *x1_ColInd_t = nullptr;
  T *x2_Val_c = nullptr;
  void *x2_Val_t = nullptr;
  int *x2_RowPtr_t = nullptr;
  int *x2_ColInd_t = nullptr;

  int current_x1_nnz = h_x1_batch_num[batch_index + 1] - h_x1_batch_num[batch_index];
  int current_x2_nnz = h_x2_batch_num[batch_index + 1] - h_x2_batch_num[batch_index];
  auto x1_compute_row = x1_row_pointers + batch_index * (x1_num_rows + 1);
  auto x1_compute_col = x1_col_indices + h_x1_batch_num[batch_index];
  auto x1_compute_val = x1_values + h_x1_batch_num[batch_index];
  auto x2_compute_row = x2_row_pointers + batch_index * (x2_num_rows + 1);
  auto x2_compute_col = x2_col_indices + h_x2_batch_num[batch_index];
  auto x2_compute_val = x2_values + h_x2_batch_num[batch_index];

  if (transpose_a) {
    x1_Val_t = allocator.AllocTensorMem(current_x1_nnz * sizeof(T));
    x1_RowPtr_t = reinterpret_cast<int *>(allocator.AllocTensorMem((x1_num_cols + 1) * sizeof(int)));
    x1_ColInd_t = reinterpret_cast<int *>(allocator.AllocTensorMem(current_x1_nnz * sizeof(int)));
    MatrixTranspose<T>(x1_num_rows, x1_num_cols, current_x1_nnz, x1_compute_val, x1_compute_row, x1_compute_col,
                       x1_Val_t, x1_RowPtr_t, x1_ColInd_t);
    x1_compute_row = x1_RowPtr_t;
    x1_compute_col = x1_ColInd_t;
    x1_compute_val = reinterpret_cast<T *>(x1_Val_t);
    m = x1_num_cols;
    n = x1_num_rows;
  }
  if (transpose_b) {
    x2_Val_t = allocator.AllocTensorMem(current_x2_nnz * sizeof(T));
    x2_RowPtr_t = reinterpret_cast<int *>(allocator.AllocTensorMem((x2_num_cols + 1) * sizeof(int)));
    x2_ColInd_t = reinterpret_cast<int *>(allocator.AllocTensorMem(current_x2_nnz * sizeof(int)));
    MatrixTranspose<T>(x2_num_rows, x2_num_cols, current_x2_nnz, x2_compute_val, x2_compute_row, x2_compute_col,
                       x2_Val_t, x2_RowPtr_t, x2_ColInd_t);
    x2_compute_row = x2_RowPtr_t;
    x2_compute_col = x2_ColInd_t;
    x2_compute_val = reinterpret_cast<T *>(x2_Val_t);
    n = x2_num_cols;
    k = x2_num_rows;
  }

  if (adjoint_a) {
    x1_Val_c = reinterpret_cast<T *>(allocator.AllocTensorMem(current_x1_nnz * sizeof(T)));
    x1_Val_t = allocator.AllocTensorMem(current_x1_nnz * sizeof(T));
    x1_RowPtr_t = reinterpret_cast<int *>(allocator.AllocTensorMem((x1_num_cols + 1) * sizeof(int)));
    x1_ColInd_t = reinterpret_cast<int *>(allocator.AllocTensorMem(current_x1_nnz * sizeof(int)));
    Conj(x1_compute_val, x1_Val_c, current_x1_nnz, stream);
    MatrixTranspose<T>(x1_num_rows, x1_num_cols, current_x1_nnz, x1_Val_c, x1_compute_row, x1_compute_col, x1_Val_t,
                       x1_RowPtr_t, x1_ColInd_t);
    x1_compute_row = x1_RowPtr_t;
    x1_compute_col = x1_ColInd_t;
    x1_compute_val = reinterpret_cast<T *>(x1_Val_t);
    m = x1_num_cols;
    n = x1_num_rows;
  }

  if (adjoint_b) {
    x2_Val_c = reinterpret_cast<T *>(allocator.AllocTensorMem(current_x2_nnz * sizeof(T)));
    x2_Val_t = allocator.AllocTensorMem(current_x2_nnz * sizeof(T));
    x2_RowPtr_t = reinterpret_cast<int *>(allocator.AllocTensorMem((x2_num_cols + 1) * sizeof(int)));
    x2_ColInd_t = reinterpret_cast<int *>(allocator.AllocTensorMem(current_x2_nnz * sizeof(int)));
    Conj(x2_compute_val, x2_Val_c, current_x2_nnz, stream);
    MatrixTranspose<T>(x2_num_rows, x2_num_cols, current_x2_nnz, x2_Val_c, x2_compute_row, x2_compute_col, x2_Val_t,
                       x2_RowPtr_t, x2_ColInd_t);
    x2_compute_row = x2_RowPtr_t;
    x2_compute_col = x2_ColInd_t;
    x2_compute_val = reinterpret_cast<T *>(x2_Val_t);
    n = x2_num_cols;
    k = x2_num_rows;
  }
  auto batch_y_val = y_values + h_y_batch_num[batch_index];
  auto batch_y_col = y_col_indices + h_y_batch_num[batch_index];
  auto batch_y_row = y_row_pointers + (m + 1) * batch_index;
  Compute(handle_, desc, info, m, n, k, x1_compute_val, x1_compute_col, x1_compute_row, current_x1_nnz, x2_compute_val,
          x2_compute_col, x2_compute_row, current_x2_nnz, &batch_y_val, &batch_y_col, &batch_y_row, &d_C_nnz);
  h_y_batch_num[batch_index + 1] = h_y_batch_num[batch_index] + d_C_nnz;
  MemFree<T>(adjoint_a, adjoint_b, transpose_a, transpose_b, x1_Val_c, x1_Val_t, x1_RowPtr_t, x1_ColInd_t, x2_Val_c,
             x2_Val_t, x2_RowPtr_t, x2_ColInd_t);
}

int ShapeCalRow(bool transpose_a, bool adjoint_a, int A_row, int A_col) {
  bool trans_A = (transpose_a || adjoint_a);
  if (trans_A) {
    return A_col;
  }
  return A_row;
}

int ShapeCalCol(bool transpose_b, bool adjoint_b, int B_row, int B_col) {
  bool trans_B = (transpose_b || adjoint_b);
  if (trans_B) {
    return B_row;
  }
  return B_col;
}

template <typename T>
bool SparseMatrixSparseMatMulGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                        const std::vector<AddressPtr> &workspace,
                                                        const std::vector<AddressPtr> &outputs) {
  int *x1_dense_shape = GetDeviceAddress<int>(inputs, 0);
  int *x1_batch_pointers = GetDeviceAddress<int>(inputs, 1);
  int *x1_row_pointers = GetDeviceAddress<int>(inputs, 2);
  int *x1_col_indices = GetDeviceAddress<int>(inputs, 3);
  T *x1_values = GetDeviceAddress<T>(inputs, 4);

  int *x2_dense_shape = GetDeviceAddress<int>(inputs, 5);
  int *x2_batch_pointers = GetDeviceAddress<int>(inputs, 6);
  int *x2_row_pointers = GetDeviceAddress<int>(inputs, 7);
  int *x2_col_indices = GetDeviceAddress<int>(inputs, 8);
  T *x2_values = GetDeviceAddress<T>(inputs, 9);

  int *y_dense_shape = GetDeviceAddress<int>(outputs, 0);
  int *y_batch_pointers = GetDeviceAddress<int>(outputs, 1);
  int *y_row_pointers = GetDeviceAddress<int>(outputs, 2);
  int *y_col_indices = GetDeviceAddress<int>(outputs, 3);
  T *y_values = GetDeviceAddress<T>(outputs, 4);

  std::vector<int> h_x1_dense_shape(rank);
  std::vector<int> h_x2_dense_shape(rank);
  std::vector<int> h_x1_batch_num(batch_ele);
  std::vector<int> h_x2_batch_num(batch_ele);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(h_x1_dense_shape.data(), x1_dense_shape, sizeof(int) * rank, cudaMemcpyDeviceToHost, stream),
    "cudaMemcpy failed.");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(h_x2_dense_shape.data(), x2_dense_shape, sizeof(int) * rank, cudaMemcpyDeviceToHost, stream),
    "cudaMemcpy failed.");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(h_x1_batch_num.data(), x1_batch_pointers, sizeof(int) * batch_ele, cudaMemcpyDeviceToHost, stream),
    "cudaMemcpy failed.");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(h_x2_batch_num.data(), x2_batch_pointers, sizeof(int) * batch_ele, cudaMemcpyDeviceToHost, stream),
    "cudaMemcpy failed.");
  int idx = 0;
  int batch = 1;
  int c_rank = 3;
  if (rank == c_rank) {
    idx = 1;
    batch = h_x1_dense_shape[0];
  }

  x1_num_rows = h_x1_dense_shape[idx];
  x1_num_cols = h_x1_dense_shape[idx + 1];
  x2_num_rows = h_x2_dense_shape[idx];
  x2_num_cols = h_x2_dense_shape[idx + 1];
  bool is_valid_shape =
    CheckShape(transpose_a, transpose_b, adjoint_a, adjoint_b, x1_num_rows, x1_num_cols, x2_num_rows, x2_num_cols);
  if (!is_valid_shape) {
    MS_LOG(ERROR) << "SparseMatrixSparseMatMul :: Please check matrix shape and transpose or adjoint option";
    return false;
  }
  int shape_y_row = ShapeCalRow(transpose_a, adjoint_a, x1_num_rows, x1_num_cols);
  int shape_y_col = ShapeCalCol(transpose_b, adjoint_b, x2_num_rows, x2_num_cols);
  std::vector<int> h_y_batch_num(batch + 1);
  h_y_batch_num[0] = 0;
  for (int batch_index = 0; batch_index < batch; batch_index++) {
    Core<T>(x1_num_rows, x1_num_cols, x2_num_rows, x2_num_cols, batch_index, h_x1_batch_num, h_x2_batch_num,
            h_y_batch_num.data(), x1_row_pointers, x1_col_indices, x1_values, x2_row_pointers, x2_col_indices,
            x2_values, y_row_pointers, y_col_indices, y_values, adjoint_a, adjoint_b, transpose_a, transpose_b);
  }

  std::vector<int> res_dense_shape;
  if (rank == c_rank) {
    res_dense_shape = {batch, shape_y_row, shape_y_col};
  } else {
    res_dense_shape = {shape_y_row, shape_y_col};
  }

  C_nnz1 = h_y_batch_num[h_y_batch_num.size() - 1];
  C_num_rows1 = (shape_y_row + 1) * batch;
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(y_batch_pointers, h_y_batch_num.data(), sizeof(int) * h_y_batch_num.size(), cudaMemcpyHostToDevice,
                    stream),
    "cudaMemcpy failed.");

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(y_dense_shape, res_dense_shape.data(), sizeof(int) * res_dense_shape.size(), cudaMemcpyHostToDevice,
                    stream),
    "cudaMemcpy failed.");
  return true;
}

void SparseMatrixSparseMatMulGpuKernelMod::SyncData() {
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(stream),
                                     "SparseMatrixSparseMatMul cudaStreamSynchronized failed");
  std::vector<int64_t> output2_shape = {
    SizeToLong(C_num_rows1),
  };
  std::vector<int64_t> output3_shape = {
    SizeToLong(C_nnz1),
  };
  std::vector<int64_t> output4_shape = {
    SizeToLong(C_nnz1),
  };
  outputs_[kIndex2]->SetShapeVector(output2_shape);
  outputs_[kIndex3]->SetShapeVector(output3_shape);
  outputs_[kIndex4]->SetShapeVector(output4_shape);
}

template <typename T>
using Complex = mindspore::utils::Complex<T>;
std::vector<std::pair<KernelAttr, SparseMatrixSparseMatMulGpuKernelMod::SparseMatrixSparseMatMulFunc>>
  SparseMatrixSparseMatMulGpuKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32),
     &SparseMatrixSparseMatMulGpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat64),
     &SparseMatrixSparseMatMulGpuKernelMod::LaunchKernel<double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeComplex64),
     &SparseMatrixSparseMatMulGpuKernelMod::LaunchKernel<Complex<float>>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeComplex128),
     &SparseMatrixSparseMatMulGpuKernelMod::LaunchKernel<Complex<double>>}};

std::vector<KernelAttr> SparseMatrixSparseMatMulGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SparseMatrixSparseMatMulFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SparseMatrixSparseMatMul, SparseMatrixSparseMatMulGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
