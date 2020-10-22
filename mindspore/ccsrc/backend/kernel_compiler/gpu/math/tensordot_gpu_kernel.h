/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_TENSORDOT_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_TENSORDOT_H_

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <vector>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/kernel_constants.h"
#include "backend/kernel_compiler/gpu/cuda_impl/transpose_impl.cuh"
#include "utils/convert_utils.h"

namespace mindspore {
namespace kernel {
template <typename T>
class TensorDotGpuKernel : public GpuKernel {
 public:
  TensorDotGpuKernel()
      : batch_(0),
        m_(0),
        n_(0),
        k_(0),
        is_null_input_(false),
        handle_(nullptr),
        dtype_a_(CUDA_R_32F),
        dtype_b_(CUDA_R_32F),
        dtype_c_(CUDA_R_32F),
        algo_(CUBLAS_GEMM_DEFAULT) {}
  ~TensorDotGpuKernel() = default;
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *x1_input = GetDeviceAddress<T>(inputs, 0);
    T *x2_input = GetDeviceAddress<T>(inputs, 1);
    size_t *x1_input_shape = GetDeviceAddress<size_t>(workspace, 0);
    size_t *x2_input_shape = GetDeviceAddress<size_t>(workspace, 1);
    size_t *x1_input_trans_axes = GetDeviceAddress<size_t>(workspace, 2);
    size_t *x2_input_trans_axes = GetDeviceAddress<size_t>(workspace, 3);
    // transposed interim values moved to workspace, then multiplied
    T *x1_reshape = GetDeviceAddress<T>(workspace, 4);
    T *x2_reshape = GetDeviceAddress<T>(workspace, 5);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);

    // Transpose X1
    CHECK_CUDA_RET_WITH_EXCEPT(
      cudaMemcpyAsync(x1_input_shape, &x1_input_shape_[0], x1_input_shape_.size() * sizeof(size_t),
                      cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemcpyAsync x1_input_shape failed");
    CHECK_CUDA_RET_WITH_EXCEPT(
      cudaMemcpyAsync(x1_input_trans_axes, &x1_transpose_fwd_[0], x1_input_shape_.size() * sizeof(size_t),
                      cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemcpyAsync input_axis_x1 failed");
    int size_x1 = SizeToInt(input_size_x1_ / sizeof(T));
    CalTranspose(size_x1, x1_input, x1_input_shape, x1_input_trans_axes, SizeToInt(x1_input_shape_.size()), x1_reshape,
                 reinterpret_cast<cudaStream_t>(stream_ptr));

    // Transpose X2
    CHECK_CUDA_RET_WITH_EXCEPT(
      cudaMemcpyAsync(x2_input_shape, &x2_input_shape_[0], (x2_input_shape_.size() * sizeof(size_t)),
                      cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemcpyAsync x2_input_shape failed");
    CHECK_CUDA_RET_WITH_EXCEPT(
      cudaMemcpyAsync(x2_input_trans_axes, &x2_transpose_fwd_[0], (x2_input_shape_.size() * sizeof(size_t)),
                      cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemcpyAsync input_axis_x2 failed");
    int size_x2 = SizeToInt(input_size_x2_ / sizeof(T));
    CalTranspose(size_x2, x2_input, x2_input_shape, x2_input_trans_axes, SizeToInt(x2_input_shape_.size()), x2_reshape,
                 reinterpret_cast<cudaStream_t>(stream_ptr));

    // Matrix Mulitply interim transposed values with GEMM
    const float alpha = 1;  // constants for cublas API
    const float beta = 0;
    const int lda = SizeToInt(k_);
    const int ldb = SizeToInt(n_);
    const int ldc = n_;
    auto stride_a = SizeToInt(m_ * k_);
    auto stride_b = SizeToInt(k_ * n_);
    auto stride_c = SizeToInt(m_ * n_);
    try {
      CHECK_CUBLAS_RET_WITH_EXCEPT(
        cublasGemmStridedBatchedEx(handle_, CUBLAS_OP_N, CUBLAS_OP_N, SizeToInt(n_), SizeToInt(m_), SizeToInt(k_),
                                   &alpha, x2_reshape, dtype_b_, ldb, stride_b, x1_reshape, dtype_a_, lda, stride_a,
                                   &beta, output_addr, dtype_c_, ldc, stride_c, batch_, CUDA_R_32F, algo_),
        "cublasSgemm Call Fail");
    } catch (const std::exception &e) {
      MS_LOG(EXCEPTION) << "Encountered an exception: " << e.what() << " when invoke cublas cublasGemmStridedBatchedEx";
    }
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCublasHandle();
    // checking for FP16 op, switch to Tensor Core if available
    dtype_a_ = GetCudaDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
    dtype_b_ = GetCudaDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 1)));
    dtype_c_ = GetCudaDataType(TypeIdLabel(AnfAlgo::GetOutputDeviceDataType(kernel_node, 0)));
    if (dtype_a_ == CUDA_R_16F && dtype_b_ == CUDA_R_16F && dtype_c_ == CUDA_R_16F) {
      MS_LOG(INFO) << "Input and output type is float16, allow to use Tensor Core operations if possible";
      algo_ = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    }

    auto tmp_x1_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto tmp_x2_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    input_size_x1_ = sizeof(T);
    for (size_t i = 0; i < tmp_x1_shape.size(); i++) {
      x1_input_shape_.push_back(tmp_x1_shape[i]);
      input_size_x1_ *= tmp_x1_shape[i];
    }
    input_size_x2_ = sizeof(T);
    for (size_t i = 0; i < tmp_x2_shape.size(); i++) {
      x2_input_shape_.push_back(tmp_x2_shape[i]);
      input_size_x2_ *= tmp_x2_shape[i];
    }

    // holding in temp values to convert to size_t vectors
    auto x1_transpose_fwd_temp = GetAttr<std::vector<int>>(kernel_node, "x1_transpose_fwd");
    auto x2_transpose_fwd_temp = GetAttr<std::vector<int>>(kernel_node, "x2_transpose_fwd");

    for (size_t i = 0; i < x1_transpose_fwd_temp.size(); i++) {
      x1_transpose_fwd_.push_back(x1_transpose_fwd_temp[i]);
    }

    for (size_t i = 0; i < x2_transpose_fwd_temp.size(); i++) {
      x2_transpose_fwd_.push_back(x2_transpose_fwd_temp[i]);
    }

    // values to decide multiplication call specifics
    x1_reshape_fwd_ = GetAttr<std::vector<int>>(kernel_node, "x1_reshape_fwd");
    x2_reshape_fwd_ = GetAttr<std::vector<int>>(kernel_node, "x2_reshape_fwd");
    auto output_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    output_size_ = sizeof(T);
    for (size_t i = 0; i < output_shape.size(); i++) {
      output_size_ *= output_shape[i];
    }
    is_null_input_ = CHECK_NULL_INPUT(output_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "input is null";
      InitSizeLists();
      return true;
    }
    m_ = x1_reshape_fwd_[0];
    k_ = x1_reshape_fwd_[1];
    n_ = x2_reshape_fwd_[1];
    batch_ = 1;  // kept as a single multiplication operation
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    size_t size_t_size = sizeof(size_t);
    input_size_list_.push_back(input_size_x1_);
    input_size_list_.push_back(input_size_x2_);
    workspace_size_list_.push_back(x1_input_shape_.size() * size_t_size);
    workspace_size_list_.push_back(x2_input_shape_.size() * size_t_size);
    workspace_size_list_.push_back(x1_transpose_fwd_.size() * size_t_size);
    workspace_size_list_.push_back(x2_transpose_fwd_.size() * size_t_size);
    workspace_size_list_.push_back(input_size_x1_);
    workspace_size_list_.push_back(input_size_x2_);
    output_size_list_.push_back(output_size_);
  }

 private:
  size_t batch_;
  size_t m_;
  size_t n_;
  size_t k_;
  bool is_null_input_;
  std::vector<size_t> x1_input_shape_;
  std::vector<size_t> x2_input_shape_;
  size_t input_size_x1_;
  size_t input_size_x2_;
  size_t output_size_;
  std::vector<size_t> x1_transpose_fwd_;  // For transpose
  std::vector<size_t> x2_transpose_fwd_;
  std::vector<int> x1_reshape_fwd_;  // For mulitplication shape
  std::vector<int> x2_reshape_fwd_;
  cublasHandle_t handle_;
  cudaDataType_t dtype_a_;
  cudaDataType_t dtype_b_;
  cudaDataType_t dtype_c_;
  cublasGemmAlgo_t algo_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif
