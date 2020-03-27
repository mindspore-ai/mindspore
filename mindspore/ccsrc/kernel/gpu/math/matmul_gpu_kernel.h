/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MATMUL_GPU_KERNEL_H
#define MINDSPORE_MATMUL_GPU_KERNEL_H

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <vector>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/kernel_constants.h"
#include "utils/convert_utils.h"

namespace mindspore {
namespace kernel {
template <typename T>
class MatMulGpuKernel : public GpuKernel {
 public:
  MatMulGpuKernel()
      : batch_(0),
        m_(0),
        n_(0),
        k_(0),
        transpose_x1_(CUBLAS_OP_N),
        transpose_x2_(CUBLAS_OP_N),
        handle_(nullptr),
        cudaDataType_(CUDA_R_32F) {}
  ~MatMulGpuKernel() = default;
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, uintptr_t stream_ptr) override {
    VARIABLE_NOT_USED(workspace);
    VARIABLE_NOT_USED(stream_ptr);
    auto input1_addr = GetDeviceAddress<T>(inputs, 0);
    auto input2_addr = GetDeviceAddress<T>(inputs, 1);
    auto output_addr = GetDeviceAddress<T>(outputs, 0);

    const float alpha = 1;
    const float beta = 0;
    const int lda = (transpose_x2_ == CUBLAS_OP_T) ? SizeToInt(k_) : SizeToInt(n_);
    const int ldb = (transpose_x1_ == CUBLAS_OP_T) ? SizeToInt(m_) : SizeToInt(k_);

    for (size_t i = 0; i < batch_; i++) {
      auto input1_slice = input1_addr + i * m_ * k_;
      auto input2_slice = input2_addr + i * k_ * n_;
      auto output_slice = output_addr + i * m_ * n_;

      CHECK_CUBLAS_RET_WITH_EXCEPT(cublasSgemmEx(handle_, transpose_x2_, transpose_x1_, SizeToInt(n_), SizeToInt(m_),
                                                 SizeToInt(k_), &alpha, input2_slice, cudaDataType_, lda, input1_slice,
                                                 cudaDataType_, ldb, &beta, output_slice, cudaDataType_, SizeToInt(n_)),
                                   "cublasSgemm Call Fail");
    }
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCublasHandle();
    cudaDataType_ = kCudaDtypeMap[TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0))];
    auto output_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    auto dims = output_shape.size();
    if (dims < 2) {
      MS_LOG(EXCEPTION) << "Output dims " << dims << " not support.";
    }

    m_ = output_shape[dims - 2];
    n_ = output_shape[dims - 1];
    batch_ = 1;
    for (size_t i = 0; i < dims - 2; i++) {
      batch_ *= output_shape[i];
    }

    bool transpose = GetAttr<bool>(kernel_node, "transpose_x1");
    transpose_x1_ = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
    auto input1_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    k_ = transpose ? input1_shape[dims - 2] : input1_shape[dims - 1];

    transpose = GetAttr<bool>(kernel_node, "transpose_x2");
    transpose_x2_ = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;

    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    size_t unit_size = sizeof(T);

    size_t input_size = batch_ * m_ * k_ * unit_size;
    input_size_list_.push_back(input_size);

    input_size = batch_ * n_ * k_ * unit_size;
    input_size_list_.push_back(input_size);

    size_t output_size = batch_ * m_ * n_ * unit_size;
    output_size_list_.push_back(output_size);
  }

 private:
  size_t batch_;
  size_t m_;
  size_t n_;
  size_t k_;

  cublasOperation_t transpose_x1_;
  cublasOperation_t transpose_x2_;

  cublasHandle_t handle_;
  cudaDataType_t cudaDataType_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif
