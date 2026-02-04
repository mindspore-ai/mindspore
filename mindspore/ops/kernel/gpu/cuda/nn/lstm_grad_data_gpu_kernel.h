/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_LSTM_GRAD_DATA_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_LSTM_GRAD_DATA_GPU_KERNEL_H_

#include <cuda_runtime_api.h>
#include <vector>
#include <memory>
#include <map>
#include <utility>
#include "kernel/gpu/gpu_kernel.h"

namespace mindspore {
namespace kernel {
constexpr size_t k3DSize = 3;
constexpr size_t kInputDimLowerLimit = 2;
constexpr size_t kWeightDimLowerLimit = 3;
class LstmGradDataGpuKernelMod : public NativeGpuKernelMod {
 public:
  LstmGradDataGpuKernelMod()
      : batch_size_(0),
        seq_len_(0),
        input_size_(0),
        hidden_size_(0),
        num_layers_(0),
        has_bias_(false),
        bidirectional_(false),
        states_init_(false),
        is_null_input_(false),
        dropout_(0),
        weight_size_(0),
        reserved_size_(0),
        rnn_desc_(nullptr),
        y_desc_(nullptr),
        dy_desc_(nullptr),
        dhy_desc_(nullptr),
        dcy_desc_(nullptr),
        w_desc_(nullptr),
        hx_desc_(nullptr),
        cx_desc_(nullptr),
        dropout_desc_(nullptr),
        dx_desc_(nullptr),
        dhx_desc_(nullptr),
        dcx_desc_(nullptr),
        handle_(nullptr),
        cudnn_data_type_(CUDNN_DATA_FLOAT) {}
  ~LstmGradDataGpuKernelMod() override { DestroyResource(); }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyRNNDescriptor(rnn_desc_), "destroy rnn_desc failed");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyDropoutDescriptor(dropout_desc_), "destroy dropout_desc failed");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(dcx_desc_), "destroy dcx_desc_ failed");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(dhx_desc_), "destroy dhx_desc_ failed");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyFilterDescriptor(w_desc_), "destroy w_desc_ failed");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(cx_desc_), "destroy cx_desc_ failed");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(hx_desc_), "destroy hx_desc_ failed");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(dcy_desc_), "destroy dcy_desc_ failed");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(dhy_desc_), "destroy dhy_desc_ failed");
    DestroyTensorDescGrp();
  }

  void InitResource() override {
    handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&dhy_desc_), "create dhy_desc_ failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&dcy_desc_), "create dcy_desc_ failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&hx_desc_), "create hx_desc_ failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&cx_desc_), "create cx_desc_ failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateFilterDescriptor(&w_desc_), "create w_desc_ failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&dhx_desc_), "create dhx_desc_ failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&dcx_desc_), "create dcx_desc_ failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateDropoutDescriptor(&dropout_desc_), "create dropout_desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateRNNDescriptor(&rnn_desc_), "create rnn_desc failed");
  }

  void InitSizeLists() {
    size_t h_size = 0;
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetTensorSizeInBytes(hx_desc_, &h_size), "get h size failed");
    size_t state_size = 0;
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnDropoutGetStatesSize(handle_, &state_size),
                                        "get dropout states size failed");

    output_size_list_.clear();
    size_t x_size = IntToSize(seq_len_ * batch_size_ * input_size_) * type_size_;
    output_size_list_.push_back(x_size);
    output_size_list_.push_back(h_size);
    output_size_list_.push_back(h_size);

    size_t workspace_size = 0;
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnGetRNNWorkspaceSize(handle_, rnn_desc_, seq_len_, dx_desc_.get(), &workspace_size),
      "get workspace size failed");
    workspace_size_list_.push_back(workspace_size);
  }

  void CreateTensorDescGrp() {
    int x_dims[3]{batch_size_, input_size_, 1};
    int y_dims[3]{batch_size_, hidden_size_ * (bidirectional_ ? 2 : 1), 1};

    dx_desc_ = std::make_unique<cudnnTensorDescriptor_t[]>(seq_len_);
    y_desc_ = std::make_unique<cudnnTensorDescriptor_t[]>(seq_len_);
    dy_desc_ = std::make_unique<cudnnTensorDescriptor_t[]>(seq_len_);

    for (size_t i = 0; i < IntToSize(seq_len_); ++i) {
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&dx_desc_[i]), "create x_desc failed");
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnSetTensorNdDescriptorEx(dx_desc_[i], CUDNN_TENSOR_NCHW, cudnn_data_type_, k3DSize, x_dims),
        "set dx_desc failed");

      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&y_desc_[i]), "create y_desc failed");
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnSetTensorNdDescriptorEx(y_desc_[i], CUDNN_TENSOR_NCHW, cudnn_data_type_, k3DSize, y_dims),
        "set y_desc failed");

      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&dy_desc_[i]), "create dy_desc_ failed");
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnSetTensorNdDescriptorEx(dy_desc_[i], CUDNN_TENSOR_NCHW, cudnn_data_type_, k3DSize, y_dims),
        "set dy_desc_ failed");
    }
  }

  void DestroyTensorDescGrp() {
    if (dy_desc_ != nullptr) {
      for (size_t i = 0; i < IntToSize(seq_len_); ++i) {
        CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnDestroyTensorDescriptor(dy_desc_[i]), "destroy dy_desc_ failed");
      }
    }
    if (y_desc_ != nullptr) {
      for (size_t i = 0; i < IntToSize(seq_len_); ++i) {
        CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnDestroyTensorDescriptor(y_desc_[i]), "destroy y_desc failed");
      }
    }
    if (dx_desc_ != nullptr) {
      for (size_t i = 0; i < IntToSize(seq_len_); ++i) {
        CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnDestroyTensorDescriptor(dx_desc_[i]), "destroy dx_desc_ failed");
      }
    }
  }

  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);
  using LstmGradDataGpuLaunchFunc =
    std::function<bool(LstmGradDataGpuKernelMod *, const std::vector<KernelTensor *> &,
                       const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &, void *)>;
  static std::vector<std::pair<KernelAttr, LstmGradDataGpuLaunchFunc>> func_list_;
  LstmGradDataGpuLaunchFunc kernel_func_;

  int batch_size_;
  int seq_len_;
  int input_size_;
  int hidden_size_;
  int num_layers_;

  bool has_bias_;
  bool bidirectional_;
  bool states_init_;
  bool is_null_input_;
  float dropout_;

  size_t weight_size_;
  size_t reserved_size_;
  size_t type_size_ = 0;

  cudnnRNNDescriptor_t rnn_desc_;

  // input desc
  std::unique_ptr<cudnnTensorDescriptor_t[]> y_desc_;
  std::unique_ptr<cudnnTensorDescriptor_t[]> dy_desc_;
  cudnnTensorDescriptor_t dhy_desc_;
  cudnnTensorDescriptor_t dcy_desc_;
  cudnnFilterDescriptor_t w_desc_;
  cudnnTensorDescriptor_t hx_desc_;
  cudnnTensorDescriptor_t cx_desc_;

  cudnnDropoutDescriptor_t dropout_desc_;

  // output desc
  std::unique_ptr<cudnnTensorDescriptor_t[]> dx_desc_;
  cudnnTensorDescriptor_t dhx_desc_;
  cudnnTensorDescriptor_t dcx_desc_;

  cudnnHandle_t handle_;
  cudnnDataType_t cudnn_data_type_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_LSTM_GRAD_DATA_GPU_KERNEL_H_
