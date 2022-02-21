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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_BAND_PART_GPU_KERNEL_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_BAND_PART_GPU_KERNEL_H

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/matrix_band_part_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"

namespace mindspore {
namespace kernel {
template <typename T>
using Complex = mindspore::utils::Complex<T>;

template <typename T>
class MatrixBandPartGpuKernelMod : public NativeGpuKernelMod {
 public:
  MatrixBandPartGpuKernelMod() : is_null_input_(false) {}
  ~MatrixBandPartGpuKernelMod() = default;

  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
    shapes_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    dim_size_ = shapes_.size();
    if (shapes_.size() < kDim2) {
      MS_LOG(EXCEPTION) << "Wrong array shape, matrix shape should not less than 2.";
    }
    m_ = shapes_[dim_size_ - kDim2];
    n_ = shapes_[dim_size_ - kDim1];
    for (size_t i = 0; i < shapes_.size() - kDim2; i++) {
      out_range_size_ *= shapes_[i];
    }
    matrix_size_ = out_range_size_ * m_ * n_;
    InitSizeLists();
    return true;
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    auto input_matrix_addr = GetDeviceAddress<T>(inputs, kDim0);
    auto lower_addr = GetDeviceAddress<int64_t>(inputs, kDim1);
    auto upper_addr = GetDeviceAddress<int64_t>(inputs, kDim2);
    auto output_matrix_addr = GetDeviceAddress<T>(outputs, kDim0);
    cudaMemsetAsync(output_matrix_addr, 0, matrix_size_ * sizeof(T), reinterpret_cast<cudaStream_t>(stream_ptr));
    int64_t lower = 0;
    int64_t upper = 0;
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(&lower, lower_addr, sizeof(int64_t), cudaMemcpyDeviceToHost,
                                               reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "Copy input lower to host failed");
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(&upper, upper_addr, sizeof(int64_t), cudaMemcpyDeviceToHost,
                                               reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "Copy input upper to host failed");
    const size_t l = (lower < 0 || lower > static_cast<int64_t>(m_)) ? m_ : lower;
    const size_t u = (upper < 0 || upper > static_cast<int64_t>(n_)) ? n_ : upper;
    // Return all
    if (l >= m_ && u >= n_) {
      CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                                 cudaMemcpyAsync(output_matrix_addr, input_matrix_addr, matrix_size_ * sizeof(T),
                                                 cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "Copy return all input matrix failed");
      return true;
    }
    size_t diag_len = std::min(m_, l + n_);
    MatrixBandPart(out_range_size_ * diag_len, input_matrix_addr, m_, n_, l, u, output_matrix_addr,
                   reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(matrix_size_ * sizeof(T));   // Input
    input_size_list_.push_back(sizeof(int64_t));            // Lower
    input_size_list_.push_back(sizeof(int64_t));            // Upper
    output_size_list_.push_back(matrix_size_ * sizeof(T));  // Output
  }

 private:
  TypeId dtype_{kNumberTypeFloat32};
  bool is_null_input_;
  std::vector<size_t> shapes_{};
  size_t dim_size_{1};
  size_t matrix_size_{0};
  size_t out_range_size_{1};
  size_t m_{1};
  size_t n_{1};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_BAND_PART_GPU_KERNEL_H
