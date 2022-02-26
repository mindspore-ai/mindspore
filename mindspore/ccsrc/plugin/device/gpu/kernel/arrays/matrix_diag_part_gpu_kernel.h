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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_DIAG_PART_GPU_KERNEL_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_DIAG_PART_GPU_KERNEL_H

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/matrix_diag_part_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
#include "kernel/common_utils.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"

namespace mindspore {
namespace kernel {
template <typename T>
using Complex = mindspore::utils::Complex<T>;

template <typename T>
class MatrixDiagPartGpuKernelMod : public NativeGpuKernelMod {
 public:
  MatrixDiagPartGpuKernelMod() : is_null_input_(false) { ResetResource(); }

  ~MatrixDiagPartGpuKernelMod() = default;

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
    alignment_ = GetAlignments(common::AnfAlgo::GetNodeAttr<std::string>(kernel_node, kAlignment));
    kernel_node_ = kernel_node;
    return true;
  }

  void UpdateOp() override {
    auto output_shape = AnfAlgo::GetOutputDeviceShapeAdaptively(kernel_node_.lock(), 0);
    output_shape[shapes_.size() - kDim1] = max_diag_len_;
    // If the out shape m' * n', the m' dimension is 1, then remove this dimension
    output_shape[shapes_.size() - kDim2] = num_diags_;
    if (num_diags_ == 1) {
      output_shape.erase(output_shape.begin() + shapes_.size() - kDim2);
    }
    auto data_type = AnfAlgo::GetInputDeviceDataType(kernel_node_.lock(), 0);
    common::AnfAlgo::SetOutputInferTypeAndShape({data_type}, {output_shape}, kernel_node_.lock().get());
  }

  void ResetResource() noexcept override {
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    auto input_matrix_addr = GetDeviceAddress<T>(inputs, kDim0);
    auto d_k_range = GetDeviceAddress<int64_t>(inputs, kDim1);
    auto padding_value = GetDeviceAddress<T>(inputs, kDim2);
    auto output_matrix_addr = GetDeviceAddress<T>(outputs, kDim0);

    int64_t k_range[kDim2]{0, 0};
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(&k_range, d_k_range, kDim2 * sizeof(int64_t), cudaMemcpyDeviceToHost,
                                               reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "Copy input lower to host failed");
    int64_t l = k_range[0];
    int64_t u = k_range[1];
    // New diagonal matrix m*n matrix, m dimension ;
    if (l > u) {
      MS_LOG(EXCEPTION) << "The k[1] must not less than k[0].";
    }
    u = std::min(u, static_cast<int64_t>(n_) - 1);
    l = std::max(-(static_cast<int64_t>(m_) - 1), l);
    num_diags_ = u - l + 1;
    // New diagonal matrix m * n matrix, n dimension
    max_diag_len_ = std::min(m_ + std::min(u, static_cast<int64_t>(0)), n_ + std::min(-l, static_cast<int64_t>(0)));
    MatrixDiagPart(out_range_size_ * num_diags_ * max_diag_len_, input_matrix_addr, m_, n_, l, u, num_diags_,
                   max_diag_len_, alignment_.first, alignment_.second, padding_value, output_matrix_addr,
                   reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(matrix_size_ * sizeof(T));   // Input
    input_size_list_.push_back(kDim2 * sizeof(int64_t));    // k_range
    input_size_list_.push_back(sizeof(T));                  // padding_value
    output_size_list_.push_back(matrix_size_ * sizeof(T));  // Output
  }

 private:
  TypeId dtype_{kNumberTypeFloat32};
  bool is_null_input_;
  std::vector<size_t> shapes_{};
  size_t dim_size_{1};
  size_t matrix_size_{0};
  size_t out_range_size_{1};
  int64_t num_diags_{1};
  int64_t max_diag_len_{1};
  size_t m_{1};
  size_t n_{1};
  std::pair<MatrixDiag::Alignment, MatrixDiag::Alignment> alignment_{MatrixDiag::RIGHT, MatrixDiag::LEFT};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_DIAG_PART_GPU_KERNEL_H
