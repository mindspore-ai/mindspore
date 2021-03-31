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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_MATRIX_INVERSE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_MATRIX_INVERSE_GPU_KERNEL_H_
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <type_traits>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/kernel_constants.h"

namespace mindspore {
namespace kernel {
template <typename T>
class MatrixInverseGpuKernel : public GpuKernel {
 public:
  MatrixInverseGpuKernel() : input_size_(0), adjoint_(false), batch_size_(1), size_(1) {}
  ~MatrixInverseGpuKernel() override = default;
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);
    auto lu_batch_addr = GetDeviceAddress<T *>(workspace, 0);
    auto inv_batch_addr = GetDeviceAddress<T *>(workspace, 1);
    auto pivo_addr = GetDeviceAddress<int>(workspace, 2);
    auto info_addr = GetDeviceAddress<int>(workspace, 3);

    int len = SizeToInt(size_);
    int batchsize = SizeToInt(batch_size_);
    for (size_t i = 0; i < batch_size_; i++) {
      lu_addr_[i] = input_addr + i * len * len;
      inv_addr_[i] = output_addr + i * len * len;
    }
    CHECK_CUDA_RET_WITH_ERROR(kernel_node_,
                              cudaMemcpyAsync(lu_batch_addr, lu_addr_.data(), sizeof(T *) * batch_size_,
                                              cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                              "cuda memcopy Fail");
    CHECK_CUDA_RET_WITH_ERROR(kernel_node_,
                              cudaMemcpyAsync(inv_batch_addr, inv_addr_.data(), sizeof(T *) * batch_size_,
                                              cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                              "cuda memcopy Fail");
    if (std::is_same<T, float>::value) {
      CHECK_CUBLAS_RET_WITH_EXCEPT(kernel_node_,
                                   cublasSgetrfBatched(handle_, len, reinterpret_cast<float **>(lu_batch_addr), len,
                                                       pivo_addr, info_addr, batchsize),
                                   "cublas trsm batched Fail");
      CHECK_CUBLAS_RET_WITH_EXCEPT(
        kernel_node_,
        cublasSgetriBatched(handle_, len, reinterpret_cast<float **>(lu_batch_addr), len, pivo_addr,
                            reinterpret_cast<float **>(inv_batch_addr), len, info_addr, batchsize),
        "cublas trsm batched Fail");
    } else if (std::is_same<T, double>::value) {
      CHECK_CUBLAS_RET_WITH_EXCEPT(kernel_node_,
                                   cublasDgetrfBatched(handle_, len, reinterpret_cast<double **>(lu_batch_addr), len,
                                                       pivo_addr, info_addr, batchsize),
                                   "cublas trsm batched Fail");
      CHECK_CUBLAS_RET_WITH_EXCEPT(
        kernel_node_,
        cublasDgetriBatched(handle_, len, reinterpret_cast<double **>(lu_batch_addr), len, pivo_addr,
                            reinterpret_cast<double **>(inv_batch_addr), len, info_addr, batchsize),
        "cublas trsm batched Fail");
    } else {
      MS_LOG(EXCEPTION) << "The data type entered must be float or double.";
    }

    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCublasHandle();
    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    if (input_shape.empty() || input_shape.size() < 2) {
      MS_LOG(EXCEPTION) << "The dim entered needs to be greater than 2, but " << input_shape.size() << " was taken";
    }
    size_t last_index = input_shape.size() - 1;
    if (input_shape[last_index] != input_shape[last_index - 1]) {
      MS_LOG(EXCEPTION) << "The last two dimensions of the input matrix should be equal!";
    }
    size_ = input_shape[last_index];
    for (size_t i = 0; i < last_index - 1; i++) {
      batch_size_ *= input_shape[i];
    }

    input_size_ = sizeof(T);
    for (auto dim : input_shape) {
      input_size_ *= dim;
    }
    adjoint_ = GetAttr<bool>(kernel_node, "adjoint");
    lu_addr_.resize(batch_size_);
    inv_addr_.resize(batch_size_);
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(input_size_);
    size_t lu_size = batch_size_ * sizeof(T *);
    workspace_size_list_.push_back(lu_size);
    size_t inv_size = batch_size_ * sizeof(T *);
    workspace_size_list_.push_back(inv_size);
    size_t pivo_size = batch_size_ * size_ * sizeof(int);
    workspace_size_list_.push_back(pivo_size);
    size_t info_size = batch_size_ * sizeof(int);
    workspace_size_list_.push_back(info_size);
  }

 private:
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  size_t input_size_;
  bool adjoint_;
  cublasHandle_t handle_;
  size_t batch_size_;
  size_t size_;
  std::vector<T *> lu_addr_;
  std::vector<T *> inv_addr_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_MATRIX_INVERSE_GPU_KERNEL_H_
