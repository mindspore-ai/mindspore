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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CTCLOSS_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CTCLOSS_GPU_KERNEL_H_

#include <cuda_runtime_api.h>
#include <vector>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "runtime/device/gpu/gpu_memory_allocator.h"

namespace mindspore {
namespace kernel {
template <typename T>
class CtcLossGpuKernel : public GpuKernel {
 public:
  CtcLossGpuKernel()
      : cudnn_handle_(nullptr),
        probs_desc_(nullptr),
        ctcloss_desc_(nullptr),
        label_size_(0),
        input_lengths_size_(0),
        label_lengths_size_(0) {}
  ~CtcLossGpuKernel() override { DestroyResource(); }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    float *probs = GetDeviceAddress<float>(inputs, 0);
    int *labels = GetDeviceAddress<int>(inputs, 1);
    int *input_lengths = GetDeviceAddress<int>(inputs, 2);
    int *label_lengths = GetDeviceAddress<int>(inputs, 3);
    float *costs = GetDeviceAddress<float>(outputs, 0);
    float *grads = GetDeviceAddress<float>(outputs, 1);

    // Copy labels/input_lengths/label_length to host as cudnn7.x.x requires
    int *labels_host = nullptr;
    int *no_blank_labels_host = nullptr;
    void *input_lengths_host = nullptr;
    void *label_lengths_host = nullptr;
    CHECK_CUDA_RET_WITH_EXCEPT(cudaMallocHost(&labels_host, inputs[1]->size), "cudaMallocHost failed.");
    CHECK_CUDA_RET_WITH_EXCEPT(cudaMallocHost(&no_blank_labels_host, inputs[1]->size), "cudaMallocHost failed.");
    CHECK_CUDA_RET_WITH_EXCEPT(cudaMallocHost(&input_lengths_host, inputs[2]->size), "cudaMallocHost failed.");
    CHECK_CUDA_RET_WITH_EXCEPT(cudaMallocHost(&label_lengths_host, inputs[3]->size), "cudaMallocHost failed.");
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    CHECK_CUDA_RET_WITH_EXCEPT(cudaMemcpyAsync(labels_host, labels, inputs[1]->size, cudaMemcpyDeviceToHost, stream),
                               "cudaMemcpyAsync failed.");
    CHECK_CUDA_RET_WITH_EXCEPT(
      cudaMemcpyAsync(input_lengths_host, input_lengths, inputs[2]->size, cudaMemcpyDeviceToHost, stream),
      "cudaMemcpyAsync failed.");
    CHECK_CUDA_RET_WITH_EXCEPT(
      cudaMemcpyAsync(label_lengths_host, label_lengths, inputs[3]->size, cudaMemcpyDeviceToHost, stream),
      "cudaMemcpyAsync failed.");

    CHECK_CUDA_RET_WITH_EXCEPT(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed.");

    size_t j = 0;
    for (size_t i = 0; i < inputs[1]->size / sizeof(int); i++) {
      if (labels_host[i] != 0) {
        no_blank_labels_host[j] = labels_host[i];
        j++;
      }
    }

    size_t workspace_size = 0;
    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnGetCTCLossWorkspaceSize(
        cudnn_handle_, probs_desc_, probs_desc_, reinterpret_cast<int *>(no_blank_labels_host),
        reinterpret_cast<int *>(label_lengths_host), reinterpret_cast<int *>(input_lengths_host),
        CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, ctcloss_desc_, &workspace_size),
      "cudnnGetCTCLossWorkspaceSize failed.");
    void *workspace = device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(workspace_size);
    if (workspace == nullptr) {
      MS_LOG(EXCEPTION) << "Failed to alloc workspace, size: " << workspace_size;
    }

    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnCTCLoss(cudnn_handle_, probs_desc_, probs, reinterpret_cast<int *>(no_blank_labels_host),
                   reinterpret_cast<int *>(label_lengths_host), reinterpret_cast<int *>(input_lengths_host), costs,
                   probs_desc_, grads, CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, ctcloss_desc_, workspace, workspace_size),
      "cudnnCtcLoss failed.");
    CHECK_CUDA_RET_WITH_EXCEPT(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed.");

    device::gpu::GPUMemoryAllocator::GetInstance().FreeTensorMem(workspace);
    CHECK_CUDA_RET_WITH_EXCEPT(cudaFreeHost(label_lengths_host), "cudaFreeHost failed.");
    CHECK_CUDA_RET_WITH_EXCEPT(cudaFreeHost(input_lengths_host), "cudaFreeHost failed.");
    CHECK_CUDA_RET_WITH_EXCEPT(cudaFreeHost(labels_host), "cudaFreeHost failed.");
    CHECK_CUDA_RET_WITH_EXCEPT(cudaFreeHost(no_blank_labels_host), "cudaFreeHost failed.");
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    InitResource();
    auto probs_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    if (probs_shape.size() != 3) {
      MS_LOG(EXCEPTION) << "probs dims: " << probs_shape.size() << " not support.";
    }
    probs_dims_[0] = probs_shape[0];
    probs_dims_[1] = probs_shape[1];
    probs_dims_[2] = probs_shape[2];

    auto labels_dims = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    if (labels_dims.size() != 1 && labels_dims.size() != 2) {
      MS_LOG(EXCEPTION) << "labels dims: " << labels_dims.size() << " not support.";
    }
    label_size_ = sizeof(int);
    for (auto i : labels_dims) {
      label_size_ *= i;
    }

    auto input_length_dims = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
    input_lengths_size_ = input_length_dims[0] * sizeof(int);
    auto label_length_dims = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 3);
    label_lengths_size_ = label_length_dims[0] * sizeof(int);
    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetTensorNdDescriptorEx(probs_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 3, probs_dims_),
      "cudnnSetTensorNdDescriptorEx failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSetCTCLossDescriptorEx(ctcloss_desc_, CUDNN_DATA_FLOAT,
                                                            CUDNN_LOSS_NORMALIZATION_SOFTMAX, CUDNN_PROPAGATE_NAN),
                                "cudnnSetCTCLossDescriptorEx failed.");
    InitSizeLists();
    return true;
  }

 protected:
  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&probs_desc_), "cudnnCreateTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateCTCLossDescriptor(&ctcloss_desc_), "cudnnCreateCTCLossDescriptor failed.");
  }

  void InitSizeLists() override {
    input_size_list_.push_back(probs_dims_[0] * probs_dims_[1] * probs_dims_[2] * sizeof(float));
    input_size_list_.push_back(label_size_);
    input_size_list_.push_back(input_lengths_size_);
    input_size_list_.push_back(label_lengths_size_);

    output_size_list_.push_back(probs_dims_[1] * sizeof(float));
    output_size_list_.push_back(probs_dims_[0] * probs_dims_[1] * probs_dims_[2] * sizeof(float));
  }

 private:
  void DestroyResource() noexcept {
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyCTCLossDescriptor(ctcloss_desc_), "cudnnDestroyCTCLossDescriptor failed.");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(probs_desc_), "cudnnDestroyTensorDescriptor failed.");
  }
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;

  cudnnHandle_t cudnn_handle_;
  cudnnTensorDescriptor_t probs_desc_;
  cudnnCTCLossDescriptor_t ctcloss_desc_;
  int probs_dims_[3] = {0};
  int label_size_;
  int input_lengths_size_;
  int label_lengths_size_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CTCLOSS_GPU_KERNEL_H_
