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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_LSTM_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_LSTM_GPU_KERNEL_H_

#include <cuda_runtime_api.h>
#include <vector>
#include <memory>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/kernel_constants.h"

namespace mindspore {
namespace kernel {
template <typename T>
class LstmGpuKernel : public GpuKernel {
 public:
  LstmGpuKernel()
      : batch_size_(0),
        seq_len_(0),
        input_size_(0),
        hidden_size_(0),
        num_layers_(0),
        has_bias_(false),
        bidirectional_(false),
        states_init_(false),
        dropout_(0),
        weight_size_(0),
        reserved_size_(0),
        x_desc_(nullptr),
        hx_desc_(nullptr),
        cx_desc_(nullptr),
        w_desc_(nullptr),
        dropout_desc_(nullptr),
        y_desc_(nullptr),
        hy_desc_(nullptr),
        cy_desc_(nullptr),
        rnn_desc_(nullptr),
        handle_(nullptr),
        cudnn_data_type_(CUDNN_DATA_FLOAT) {}
  ~LstmGpuKernel() override { DestroyResource(); }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    VARIABLE_NOT_USED(stream_ptr);
    auto x_addr = GetDeviceAddress<T>(inputs, 0);
    auto hx_addr = GetDeviceAddress<T>(inputs, 1);
    auto cx_addr = GetDeviceAddress<T>(inputs, 2);
    auto w_addr = GetDeviceAddress<T>(inputs, 3);
    auto y_addr = GetDeviceAddress<T>(outputs, 0);
    auto hy_addr = GetDeviceAddress<T>(outputs, 1);
    auto cy_addr = GetDeviceAddress<T>(outputs, 2);
    auto reserved_addr = GetDeviceAddress<T>(outputs, 3);
    auto states_addr = GetDeviceAddress<T>(outputs, 4);
    void *workspace_addr = GetDeviceAddress<T>(workspace, 0);

    if (!states_init_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_, cudnnSetDropoutDescriptor(dropout_desc_, handle_, dropout_, states_addr, output_size_list_[4], 0),
        "set dropout descriptor failed. Possible reasons: the GPU is out of memory.");
      states_init_ = true;
    }

    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_,
      cudnnRNNForwardTraining(handle_, rnn_desc_, seq_len_, x_desc_.get(), x_addr, hx_desc_, hx_addr, cx_desc_, cx_addr,
                              w_desc_, w_addr, y_desc_.get(), y_addr, hy_desc_, hy_addr, cy_desc_, cy_addr,
                              workspace_addr, workspace_size_list_[0], reserved_addr, reserved_size_),
      "launch lstm kernel failed");

    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    InitResource();
    cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    seq_len_ = SizeToInt(input_shape[0]);
    batch_size_ = SizeToInt(input_shape[1]);
    input_size_ = SizeToInt(input_shape[2]);

    input_size_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "input_size"));
    hidden_size_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "hidden_size"));
    num_layers_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "num_layers"));
    has_bias_ = GetAttr<bool>(kernel_node, "has_bias");
    bidirectional_ = GetAttr<bool>(kernel_node, "bidirectional");
    dropout_ = GetAttr<float>(kernel_node, "dropout");

    cudnnRNNInputMode_t input_mode = CUDNN_LINEAR_INPUT;
    cudnnDirectionMode_t direction = bidirectional_ ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL;
    cudnnRNNMode_t rnn_mode = CUDNN_LSTM;
    cudnnRNNAlgo_t algo = CUDNN_RNN_ALGO_STANDARD;
    CreateTensorDescGrp();
    int hx_dims[3]{num_layers_ * (bidirectional_ ? 2 : 1), batch_size_, hidden_size_};
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnSetTensorNdDescriptorEx(hx_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, 3, hx_dims),
                                "set hx_desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnSetTensorNdDescriptorEx(cx_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, 3, hx_dims),
                                "set cx_desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnSetTensorNdDescriptorEx(hy_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, 3, hx_dims),
                                "set hy_desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnSetTensorNdDescriptorEx(cy_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, 3, hx_dims),
                                "set cy_desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnSetDropoutDescriptor(dropout_desc_, handle_, dropout_, nullptr, 0, 0),
                                "set dropout_desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnSetRNNDescriptor(handle_, rnn_desc_, hidden_size_, num_layers_, dropout_desc_,
                                                      input_mode, direction, rnn_mode, algo, cudnn_data_type_),
                                "set rnn_desc failed");
    cudnnRNNBiasMode_t bias_mode = has_bias_ ? CUDNN_RNN_DOUBLE_BIAS : CUDNN_RNN_NO_BIAS;
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnSetRNNBiasMode(rnn_desc_, bias_mode), "set bias_mode failed");
    auto weight_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 3);
    size_t weight_size = weight_shape[0] * weight_shape[1] * weight_shape[2] * sizeof(T);
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnGetRNNParamsSize(handle_, rnn_desc_, x_desc_[0], &weight_size_, cudnn_data_type_),
                                "get weight_size_ failed");
    if (weight_size != weight_size_) {
      MS_LOG(EXCEPTION) << "weight size: " << weight_size << " error, expect: " << weight_size_ << " .";
    }
    int w_dims[3] = {SizeToInt(weight_size_ / 4), 1, 1};
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnSetFilterNdDescriptor(w_desc_, cudnn_data_type_, CUDNN_TENSOR_NCHW, 3, w_dims),
                                "set w_desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_, cudnnGetRNNTrainingReserveSize(handle_, rnn_desc_, seq_len_, x_desc_.get(), &reserved_size_),
      "get reserve size failed");
    InitSizeLists();
    return true;
  }
  void CreateTensorDescGrp() {
    int x_dims[3]{batch_size_, input_size_, 1};
    int y_dims[3]{batch_size_, hidden_size_ * (bidirectional_ ? 2 : 1), 1};

    x_desc_ = std::make_unique<cudnnTensorDescriptor_t[]>(seq_len_);
    y_desc_ = std::make_unique<cudnnTensorDescriptor_t[]>(seq_len_);

    for (size_t i = 0; i < IntToSize(seq_len_); ++i) {
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&x_desc_[i]), "create x_desc failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_, cudnnSetTensorNdDescriptorEx(x_desc_[i], CUDNN_TENSOR_NCHW, cudnn_data_type_, 3, x_dims),
        "set x_desc failed");

      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&y_desc_[i]), "create y_desc failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_, cudnnSetTensorNdDescriptorEx(y_desc_[i], CUDNN_TENSOR_NCHW, cudnn_data_type_, 3, y_dims),
        "set y_desc failed");
    }
  }

  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyRNNDescriptor(rnn_desc_), "destroy rnn_desc failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyDropoutDescriptor(dropout_desc_),
                               "destroy dropout_desc failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(cy_desc_), "destroy cy_desc failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(hy_desc_), "destroy hy_desc failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyFilterDescriptor(w_desc_), "destroy w_desc failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(hx_desc_), "destroy hx_desc failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(cx_desc_), "destroy cx_desc failed");

    for (size_t i = 0; i < IntToSize(seq_len_); ++i) {
      CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(y_desc_[i]), "destroy y_desc failed");
      CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(x_desc_[i]), "destroy x_desc failed");
    }
  }

 protected:
  void InitResource() override {
    handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&hx_desc_), "create hx_desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&cx_desc_), "create cx_desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateFilterDescriptor(&w_desc_), "create w_desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&hy_desc_), "create hy_desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&cy_desc_), "create cy_desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateDropoutDescriptor(&dropout_desc_),
                                "create dropout_desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateRNNDescriptor(&rnn_desc_), "create rnn_desc failed");
  }
  void InitSizeLists() override {
    size_t x_size = IntToSize(seq_len_ * batch_size_ * input_size_) * sizeof(T);

    size_t h_size = 0;
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnGetTensorSizeInBytes(hx_desc_, &h_size), "get h size failed");

    input_size_list_.push_back(x_size);
    input_size_list_.push_back(h_size);
    input_size_list_.push_back(h_size);
    input_size_list_.push_back(weight_size_);

    size_t y_size = IntToSize(seq_len_ * batch_size_ * hidden_size_ * (bidirectional_ ? 2 : 1)) * sizeof(T);
    output_size_list_.push_back(y_size);
    output_size_list_.push_back(h_size);
    output_size_list_.push_back(h_size);
    output_size_list_.push_back(reserved_size_);
    size_t state_size = 0;
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnDropoutGetStatesSize(handle_, &state_size),
                                "get dropout states size failed");
    output_size_list_.push_back(state_size);

    size_t workspace_size = 0;
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnGetRNNWorkspaceSize(handle_, rnn_desc_, seq_len_, x_desc_.get(), &workspace_size),
                                "get workspace size failed");
    workspace_size_list_.push_back(workspace_size);
  }

 private:
  int batch_size_;
  int seq_len_;
  int input_size_;
  int hidden_size_;
  int num_layers_;

  bool has_bias_;
  bool bidirectional_;
  bool states_init_;
  float dropout_;

  size_t weight_size_;
  size_t reserved_size_;

  // input desc
  std::unique_ptr<cudnnTensorDescriptor_t[]> x_desc_;
  cudnnTensorDescriptor_t hx_desc_;
  cudnnTensorDescriptor_t cx_desc_;
  cudnnFilterDescriptor_t w_desc_;
  cudnnDropoutDescriptor_t dropout_desc_;
  std::unique_ptr<cudnnTensorDescriptor_t[]> y_desc_;
  cudnnTensorDescriptor_t hy_desc_;
  cudnnTensorDescriptor_t cy_desc_;
  cudnnRNNDescriptor_t rnn_desc_;

  cudnnHandle_t handle_;
  cudnnDataType_t cudnn_data_type_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_LSTM_GPU_KERNEL_H_
