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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_LU_SCIPY_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_LU_SCIPY_GPU_KERNEL_H_
#include <algorithm>
#include <map>
#include <string>
#include <type_traits>
#include <vector>
#include "utils/convert_utils.h"
#include "kernel/gpu/cuda_impl/cuda_ops/transpose_impl.cuh"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/kernel_constants.h"

namespace mindspore {
namespace kernel {
template <typename T>
class LUGpuKernelMod : public NativeGpuKernelMod {
 public:
  LUGpuKernelMod() : is_null_input_(false) {}
  ~LUGpuKernelMod() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    CHECK_CUSOLVER_RET_WITH_ERROR(cusolverDnSetStream(handle_, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                  "cusolverDnSetStream failed");
    T *batch_input_addr = GetDeviceAddress<T>(inputs, kDim0);
    T *batch_output_addr = GetDeviceAddress<T>(outputs, kDim0);
    int *batch_piv_output_addr = nullptr;
    if (pivot_on_) {
      batch_piv_output_addr = GetDeviceAddress<int>(outputs, kDim1);
    }
    // workspace
    int *batch_permutation_addr = GetDeviceAddress<int>(outputs, kDim2);
    int *info_output_addr = GetDeviceAddress<int>(workspace, kDim0);
    T *dev_transpose_work = GetDeviceAddress<T>(workspace, kDim1);

    TransposeInfo info;
    TransposeInfo work_info;
    info.input_shape = std::vector<int64_t>{m_, n_};
    info.perm = std::vector<int32_t>{1, 0};
    work_info.input_shape = std::vector<int64_t>{n_, m_};
    work_info.perm = std::vector<int32_t>{1, 0};

    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(batch_output_addr, batch_input_addr, batch_size_ * m_ * n_ * unit_size_, cudaMemcpyDeviceToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemcpyAsync failed in LUGpuKernelMod::Launch.");

    // 4. query working space of getrf
    if constexpr (std::is_same_v<T, float>) {
      CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
        cusolverDnSgetrf_bufferSize(handle_, m_, n_, batch_output_addr, lda_, &lwork_),
        "cusolver query lu work size fail");
    } else if constexpr (std::is_same_v<T, double>) {
      CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
        cusolverDnDgetrf_bufferSize(handle_, m_, n_, batch_output_addr, lda_, &lwork_),
        "cusolver query lu work size fail");
    } else {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the data type only should be float or double, right now.";
    }
    // 5. malloc device working space of getrf
    d_work_ = reinterpret_cast<T *>(device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(unit_size_ * lwork_));
    for (size_t batch = 0; batch < batch_size_; ++batch) {
      T *output_addr = batch_output_addr + batch * m_ * n_;
      int *permutation_addr = batch_permutation_addr + batch * k_ * k_;
      int *piv_output_addr = batch_piv_output_addr + batch * k_;
      auto s1 = CalTranspose<T, false>(m_ * n_, output_addr, info, dev_transpose_work,
                                       reinterpret_cast<cudaStream_t>(stream_ptr));
      CHECK_CUDA_STATUS(s1, "Transpose called by " + kernel_name_);

      // 6.lu factorization according to cuSolver api, outputs have been written to input's matrix.
      if constexpr (std::is_same_v<T, float>) {
        CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(

          cusolverDnSgetrf(handle_, m_, n_, dev_transpose_work, lda_, d_work_, piv_output_addr, info_output_addr),
          "cusolver lu fail");
      } else if constexpr (std::is_same_v<T, double>) {
        // 6.lu factorization according to cuSolver api, outputs have been written to input's matrix.
        CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
          cusolverDnDgetrf(handle_, m_, n_, dev_transpose_work, lda_, d_work_, piv_output_addr, info_output_addr),
          "cusolver lu fail");
      } else {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the data type only should be float or double, right now.";
      }
      auto s2 = CalTranspose<T, false>(m_ * n_, dev_transpose_work, work_info, output_addr,
                                       reinterpret_cast<cudaStream_t>(stream_ptr));
      CHECK_CUDA_STATUS(s2, "Transpose called by " + kernel_name_);
      std::vector<int> host_permuted(k_, 0);
      std::vector<int> host_pivots(k_, 0);
      std::vector<int> host_permutation(k_ * k_, 0);
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
        cudaMemcpyAsync(host_pivots.data(), piv_output_addr, sizeof(int) * k_, cudaMemcpyDeviceToHost,
                        reinterpret_cast<cudaStream_t>(stream_ptr)),
        "For 'LuScipy', cudaMemcpyAsync failed in LUGpuKernelMod::Launch copy pivots to host.");
      if (cudaStreamQuery(reinterpret_cast<cudaStream_t>(stream_ptr)) != cudaSuccess) {
        CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_ptr)),
                                           "For 'LuScipy', cuda Stream Sync Failed.");
      }

      // cal pivots && permutation major by row.
      for (size_t i = 0; i < k_; ++i) {
        host_pivots[i] -= 1;
        host_permuted[i] = i;
      }
      for (size_t i = 0; i < k_; ++i) {
        int tmp_value = host_permuted[i];
        host_permuted[i] = host_permuted[host_pivots[i]];
        host_permuted[host_pivots[i]] = tmp_value;
      }
      // gpu default is P.A = LU, so here is col swap.
      for (size_t i = 0; i < k_; ++i) {
        host_permutation[host_permuted[i] * k_ + i] = 1;
      }
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
        cudaMemcpyAsync(permutation_addr, host_permutation.data(), sizeof(int) * k_ * k_, cudaMemcpyHostToDevice,
                        reinterpret_cast<cudaStream_t>(stream_ptr)),
        "cudaMemcpyAsync failed in LUGpuKernelMod::Launch copy permutation matrix.");
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
        cudaMemcpyAsync(piv_output_addr, host_pivots.data(), sizeof(int) * k_, cudaMemcpyHostToDevice,
                        reinterpret_cast<cudaStream_t>(stream_ptr)),
        "cudaMemcpyAsync failed in LUGpuKernelMod::Launch copy pivots array.");
    }
    device::gpu::GPUMemoryAllocator::GetInstance().FreeTensorMem(d_work_);
    return true;
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCusolverDnHandle();
    return true;
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
      return ret;
    }
    batch_size_ = 1;
    auto shape_signed = inputs[kIndex0]->GetShapeVector();
    auto in_shape = Convert2SizeT(shape_signed);
    // 2. check input shape not null
    is_null_input_ = CHECK_SHAPE_NULL(in_shape, kernel_name_, "input");
    if (is_null_input_) {
      InitSizeLists();
      return KRET_OK;
    }
    // 3. calculate input size
    if (!InitInputSize(in_shape)) {
      MS_LOG(ERROR) << "For 'PureCholeskyGpuKernel', input shape init failed.";
      return KRET_RESIZE_FAILED;
    }
    return KRET_OK;
  }

  std::vector<KernelAttr> GetOpSupport() override {
    static std::vector<KernelAttr> support_list = {
      KernelAttr()
        .AddInputAttr(kNumberTypeFloat32)
        .AddOutputAttr(kNumberTypeFloat32)
        .AddOutputAttr(kNumberTypeInt32)
        .AddOutputAttr(kNumberTypeInt32),
      KernelAttr()
        .AddInputAttr(kNumberTypeFloat64)
        .AddOutputAttr(kNumberTypeFloat64)
        .AddOutputAttr(kNumberTypeInt32)
        .AddOutputAttr(kNumberTypeInt32),
    };
    return support_list;
  }

 private:
  bool InitInputSize(const std::vector<size_t> &in_shape) {
    constexpr size_t lu_min_dim = 1;
    if (in_shape.size() <= lu_min_dim) {
      MS_LOG_EXCEPTION << kernel_name_ << " input shape is " << in_shape.size() << " which is invalid.";
    }
    constexpr size_t lu_reverse_row_dim = 2;
    lu_row_ = in_shape.at(in_shape.size() - lu_reverse_row_dim);
    lu_col_ = in_shape.at(in_shape.size() - 1);
    batch_size_ = lu_min_dim;
    for (int batch = 0; batch < static_cast<int>(in_shape.size() - lu_reverse_row_dim); ++batch) {
      batch_size_ *= in_shape.at(batch);
    }
    // set matrix row or col to be lead dimension
    m_ = SizeToInt(lu_row_);
    n_ = SizeToInt(lu_col_);
    k_ = std::min(lu_row_, lu_col_);
    lda_ = m_;
    ldb_ = n_;
    InitSizeLists();
    return true;
  }

  void InitSizeLists() {
    size_t output_size = batch_size_ * lu_row_ * lu_col_ * unit_size_;

    size_t output_piv_size = 0;
    if (pivot_on_) {
      output_piv_size = batch_size_ * k_ * sizeof(int);
    }
    size_t output_permutation_size = batch_size_ * k_ * k_ * sizeof(int);
    output_size_list_.resize(kDim3);
    output_size_list_[kDim0] = output_size;
    output_size_list_[kDim1] = output_piv_size;
    output_size_list_[kDim2] = output_permutation_size;

    // a device addr to place lu factor return code
    workspace_size_list_.push_back(sizeof(int));

    // transpose 2d matrix scalar args workspace
    constexpr size_t shape_2d = 2;
    workspace_size_list_.push_back(shape_2d * sizeof(size_t));
    workspace_size_list_.push_back(shape_2d * sizeof(size_t));

    // transpose workspace
    workspace_size_list_.push_back(m_ * n_ * unit_size_);
  }

  size_t unit_size_{sizeof(T)};
  size_t batch_size_{1};
  size_t lu_row_{0};
  size_t lu_col_{0};
  size_t k_{0};
  int64_t m_{0};
  int64_t n_{0};
  size_t lda_{0};
  size_t ldb_{0};
  int lwork_{0};
  bool pivot_on_{true};
  T *d_work_{nullptr};
  cusolverDnHandle_t handle_{nullptr};
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_LU_SCIPY_GPU_KERNEL_H_
