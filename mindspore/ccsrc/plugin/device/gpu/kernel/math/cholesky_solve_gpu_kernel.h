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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MATH_CHOLESKY_SOLVE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MATH_CHOLESKY_SOLVE_GPU_KERNEL_H_
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <map>
#include <utility>
#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/triangle_matrix_copy_impl.cuh"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"
#include "include/common/utils/convert_utils.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/matrix_transpose_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr size_t kCholeskyInputsNum = 1;
constexpr size_t kInputIndex = 0;
constexpr size_t kCholeskyOutputsNum = 1;
constexpr size_t kOutputIndex = 0;
constexpr size_t kRowIndex = 2;
constexpr size_t kColIndex = 1;
inline cublasStatus_t cublasXtrsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
                                  cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float *alpha,
                                  const float *A, int lda, float *B, int ldb) {
  return cublasStrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}
inline cublasStatus_t cublasXtrsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
                                  cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double *alpha,
                                  const double *A, int lda, double *B, int ldb) {
  return cublasDtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}
inline cublasStatus_t cublasXtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
                                         cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
                                         const float *alpha, const float *const A[], int lda, float *const B[], int ldb,
                                         int batchCount) {
  return cublasStrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount);
}
inline cublasStatus_t cublasXtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
                                         cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
                                         const double *alpha, const double *const A[], int lda, double *const B[],
                                         int ldb, int batchCount) {
  return cublasDtrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount);
}

class CholeskySolveGpuKernelMod : public NativeGpuKernelMod {
 public:
  CholeskySolveGpuKernelMod() = default;
  ~CholeskySolveGpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override {
    kernel_name_ = base_operator->name();
    upper_ = GetValue<bool>(base_operator->GetAttr("upper"));
    handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCublasHandle();

    auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
    auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
    if (!is_match) {
      MS_LOG(ERROR) << "For 'CholeskySolve', it does not support this kernel type: " << kernel_attr;
      return false;
    }
    kernel_func_ = func_list_[index].second;
    return true;
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    cuda_stream_ = reinterpret_cast<cudaStream_t>(stream_ptr);
    return kernel_func_(this, inputs, workspace, outputs);
  }

  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs) {
    using pointer = T *;
    CHECK_CUBLAS_RET_WITH_ERROR(cublasSetStream(handle_, reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                "cholesky solve cublasSetStream failed");
    auto input_a_addr = GetDeviceAddress<T>(inputs, kDim0);
    auto input_b_addr = GetDeviceAddress<T>(inputs, kDim1);
    auto output_addr = GetDeviceAddress<T>(outputs, kDim0);
    auto d_a_array_addr = GetDeviceAddress<pointer>(workspace, kDim0);
    auto d_b_array_addr = GetDeviceAddress<pointer>(workspace, kDim1);
    auto d_c_array_addr = GetDeviceAddress<pointer>(workspace, kDim2);
    std::vector<pointer> h_a_array(batch_num_);
    std::vector<pointer> h_b_array(batch_num_);
    std::vector<pointer> h_c_array(batch_num_);
    for (size_t i = 0; i < batch_num_; i++) {
      h_a_array[i] = input_a_addr + i * lda_ * nrhs_;
      h_b_array[i] = input_b_addr + i * ldb_ * m_;
      h_c_array[i] = output_addr + i * lda_ * nrhs_;
    }
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(d_a_array_addr, h_a_array.data(), sizeof(pointer) * batch_num_, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(cuda_stream_)),
      "cuda memcopy Fail");
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(d_b_array_addr, h_b_array.data(), sizeof(pointer) * batch_num_, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(cuda_stream_)),
      "cuda memcopy Fail");
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(d_c_array_addr, h_c_array.data(), sizeof(pointer) * batch_num_, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(cuda_stream_)),
      "cuda memcopy Fail");
    MatrixTranspose(input_a_addr, SizeToInt(batch_num_ * lda_ * nrhs_), SizeToInt(lda_), SizeToInt(nrhs_), output_addr,
                    device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
    cublasFillMode_t uplo_ = CUBLAS_FILL_MODE_UPPER;
    if (upper_) {
      uplo_ = CUBLAS_FILL_MODE_LOWER;
      transa_ = CUBLAS_OP_N;
      transa_t_ = CUBLAS_OP_T;
    }
    T alpha = 1;
    if (batch_num_ == 1) {
      CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(cublasXtrsm(handle_, CUBLAS_SIDE_LEFT, uplo_, transa_, CUBLAS_DIAG_NON_UNIT,
                                                       lda_, nrhs_, &alpha, input_b_addr, ldb_, output_addr, lda_),
                                           "cholesky solve cublasXtrsm failed!");
      CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(
        cublasXtrsm(handle_, CUBLAS_SIDE_LEFT, uplo_, transa_t_, CUBLAS_DIAG_NON_UNIT, lda_, nrhs_, &alpha,
                    input_b_addr, ldb_, output_addr, lda_),
        "cholesky solve cublasXtrsm failed!");
    } else {
      CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(
        cublasXtrsmBatched(handle_, CUBLAS_SIDE_LEFT, uplo_, transa_, CUBLAS_DIAG_NON_UNIT, lda_, nrhs_, &alpha,
                           d_b_array_addr, ldb_, d_c_array_addr, lda_, batch_num_),
        "cholesky solve cublasXgetrsBatched failed!");
      CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(
        cublasXtrsmBatched(handle_, CUBLAS_SIDE_LEFT, uplo_, transa_t_, CUBLAS_DIAG_NON_UNIT, lda_, nrhs_, &alpha,
                           d_b_array_addr, ldb_, d_c_array_addr, lda_, batch_num_),
        "cholesky solve cublasXgetrsBatched failed!");
    }
    MatrixTranspose(output_addr, SizeToInt(batch_num_ * lda_ * nrhs_), SizeToInt(nrhs_), SizeToInt(lda_), input_a_addr,
                    device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
    auto output_elements = batch_num_ * lda_ * nrhs_;
    MatrixCopy(input_a_addr, output_addr, output_elements, reinterpret_cast<cudaStream_t>(cuda_stream_));
    return true;
  }

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override {
    if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
      return ret;
    }

    const auto b_shape = inputs.at(kIndex0)->GetShapeVector();
    const auto cho_shape = inputs.at(kIndex1)->GetShapeVector();

    is_null_input_ = CHECK_SHAPE_NULL(LongVecToSizeVec(b_shape), kernel_name_, "input_a") ||
                     CHECK_SHAPE_NULL(LongVecToSizeVec(cho_shape), kernel_name_, "input_b");
    batch_num_ = std::accumulate(b_shape.begin(), b_shape.end() - kIndex2, int64_t(1), std::multiplies{});
    m_ = cho_shape.back();
    ldb_ = m_;
    lda_ = m_;
    nrhs_ = b_shape.back();

    workspace_size_list_.clear();
    workspace_size_list_ = {batch_num_ * sizeof(float *), batch_num_ * sizeof(float *), batch_num_ * sizeof(float *),
                            batch_num_ * sizeof(int)};

    return KRET_OK;
  }

  std::vector<KernelAttr> GetOpSupport() override {
    std::vector<KernelAttr> support_list;
    (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                         [](const std::pair<KernelAttr, CholeskySolveFunc> &pair) { return pair.first; });
    return support_list;
  }

 private:
  using CholeskySolveFunc =
    std::function<bool(CholeskySolveGpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;
  size_t nrhs_{0};
  size_t batch_num_{0};
  size_t m_{0};
  size_t lda_{0};
  size_t ldb_{0};
  cublasHandle_t handle_{nullptr};
  cublasOperation_t transa_{CUBLAS_OP_T};
  cublasOperation_t transa_t_{CUBLAS_OP_N};
  bool upper_{false};
  bool is_null_input_;
  CholeskySolveFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, CholeskySolveFunc>> func_list_;
  void *cuda_stream_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MATH_CHOLESKY_SOLVE_GPU_KERNEL_H_
