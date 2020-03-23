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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_NN_LSTM_GRAD_DATA_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_NN_LSTM_GRAD_DATA_GPU_KERNEL_H_

#include <cuda_runtime_api.h>
#include <vector>
#include <memory>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/kernel_constants.h"
#include "dataset/util/make_unique.h"

namespace mindspore {
namespace kernel {
template <typename T>
class LstmGradDataGpuKernel : public GpuKernel {
 public:
  LstmGradDataGpuKernel()
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
  ~LstmGradDataGpuKernel() override { DestroyResource(); }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, uintptr_t stream_ptr) override {
    VARIABLE_NOT_USED(stream_ptr);
    auto y_addr = GetDeviceAddress<T>(inputs, 0);
    auto dy_addr = GetDeviceAddress<T>(inputs, 1);
    auto dhy_addr = GetDeviceAddress<T>(inputs, 2);
    auto dcy_addr = GetDeviceAddress<T>(inputs, 3);
    auto w_addr = GetDeviceAddress<T>(inputs, 4);
    auto hx_addr = GetDeviceAddress<T>(inputs, 5);
    auto cx_addr = GetDeviceAddress<T>(inputs, 6);
    auto reserved_addr = GetDeviceAddress<T>(inputs, 7);
    auto states_addr = GetDeviceAddress<T>(inputs, 8);
    auto dx_addr = GetDeviceAddress<T>(outputs, 0);
    auto dhx_addr = GetDeviceAddress<T>(outputs, 1);
    auto dcx_addr = GetDeviceAddress<T>(outputs, 2);
    void *workspace_addr = GetDeviceAddress<T>(workspace, 0);

    if (!states_init_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnRestoreDropoutDescriptor(dropout_desc_, handle_, dropout_, states_addr, input_size_list_[8], 0),
        "restore dropout state failed");
      states_init_ = true;
    }

    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnRNNBackwardData(handle_, rnn_desc_, seq_len_, y_desc_.get(), y_addr, dy_desc_.get(), dy_addr, dhy_desc_,
                           dhy_addr, dcy_desc_, dcy_addr, w_desc_, w_addr, hx_desc_, hx_addr, cx_desc_, cx_addr,
                           dx_desc_.get(), dx_addr, dhx_desc_, dhx_addr, dcx_desc_, dcx_addr, workspace_addr,
                           workspace_size_list_[0], reserved_addr, reserved_size_),
      "launch lstm back data kernel failed");

    CHECK_CUDA_RET_WITH_EXCEPT(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "stream synchronize failed.");
    return true;
  }
  void GetAttrs(const CNodePtr &kernel_node) {
    input_size_ = GetAttr<int>(kernel_node, "input_size");
    hidden_size_ = GetAttr<int>(kernel_node, "hidden_size");
    num_layers_ = GetAttr<int>(kernel_node, "num_layers");
    has_bias_ = GetAttr<bool>(kernel_node, "has_bias");
    bidirectional_ = GetAttr<bool>(kernel_node, "bidirectional");
    dropout_ = GetAttr<float>(kernel_node, "dropout");
  }
  bool Init(const CNodePtr &kernel_node) override {
    InitResource();
    cudnn_data_type_ = kCudnnDtypeMap[TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0))];
    auto input_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    seq_len_ = SizeToInt(input_shape[0]);
    batch_size_ = SizeToInt(input_shape[1]);
    GetAttrs(kernel_node);
    cudnnRNNInputMode_t input_mode = CUDNN_LINEAR_INPUT;
    cudnnDirectionMode_t direction = bidirectional_ ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL;
    cudnnRNNMode_t rnn_mode = CUDNN_LSTM;
    cudnnRNNAlgo_t algo = CUDNN_RNN_ALGO_STANDARD;
    CreateTensorDescGrp();
    int hx_dims[3]{num_layers_ * (bidirectional_ ? 2 : 1), batch_size_, hidden_size_};
    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetTensorNdDescriptorEx(dhy_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, 3, hx_dims), "set dhy_desc_ failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetTensorNdDescriptorEx(dcy_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, 3, hx_dims), "set dcy_desc_ failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSetTensorNdDescriptorEx(hx_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, 3, hx_dims),
                                "set hx_desc_ failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSetTensorNdDescriptorEx(cx_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, 3, hx_dims),
                                "set cx_desc_ failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetTensorNdDescriptorEx(dhx_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, 3, hx_dims), "set dhx_desc_ failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetTensorNdDescriptorEx(dcx_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, 3, hx_dims), "set dcx_desc_ failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSetDropoutDescriptor(dropout_desc_, handle_, dropout_, nullptr, 0, 0),
                                "set dropout_desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSetRNNDescriptor(handle_, rnn_desc_, hidden_size_, num_layers_, dropout_desc_,
                                                      input_mode, direction, rnn_mode, algo, cudnn_data_type_),
                                "set rnn_desc failed");
    cudnnRNNBiasMode_t bias_mode = has_bias_ ? CUDNN_RNN_DOUBLE_BIAS : CUDNN_RNN_NO_BIAS;
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSetRNNBiasMode(rnn_desc_, bias_mode), "set bias_mode failed");
    auto weight_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 4);
    size_t weight_size = weight_shape[0] * weight_shape[1] * weight_shape[2] * sizeof(T);
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetRNNParamsSize(handle_, rnn_desc_, dx_desc_[0], &weight_size_, cudnn_data_type_),
                                "get weight_size_ failed");
    if (weight_size != weight_size_) {
      MS_LOG(EXCEPTION) << "weight size: " << weight_size << " error, expect: " << weight_size_ << " .";
    }
    int w_dims[3] = {SizeToInt(weight_size_ / 4), 1, 1};
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSetFilterNdDescriptor(w_desc_, cudnn_data_type_, CUDNN_TENSOR_NCHW, 3, w_dims),
                                "set w_desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnGetRNNTrainingReserveSize(handle_, rnn_desc_, seq_len_, dx_desc_.get(), &reserved_size_), "get size failed");
    InitSizeLists();
    return true;
  }

 protected:
  void InitResource() override {
    handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&dhy_desc_), "create dhy_desc_ failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&dcy_desc_), "create dcy_desc_ failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&hx_desc_), "create hx_desc_ failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&cx_desc_), "create cx_desc_ failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateFilterDescriptor(&w_desc_), "create w_desc_ failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&dhx_desc_), "create dhx_desc_ failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&dcx_desc_), "create dcx_desc_ failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateDropoutDescriptor(&dropout_desc_), "create dropout_desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateRNNDescriptor(&rnn_desc_), "create rnn_desc failed");
  }

  void InitSizeLists() override {
    size_t y_size = IntToSize(seq_len_ * batch_size_ * hidden_size_ * (bidirectional_ ? 2 : 1)) * sizeof(T);

    size_t h_size = 0;
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetTensorSizeInBytes(hx_desc_, &h_size), "get h size failed");

    input_size_list_.push_back(y_size);
    input_size_list_.push_back(y_size);
    input_size_list_.push_back(h_size);
    input_size_list_.push_back(h_size);
    input_size_list_.push_back(weight_size_);
    input_size_list_.push_back(h_size);
    input_size_list_.push_back(h_size);
    input_size_list_.push_back(reserved_size_);
    size_t state_size = 0;
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnDropoutGetStatesSize(handle_, &state_size), "get dropout states size failed");
    input_size_list_.push_back(state_size);

    size_t x_size = IntToSize(seq_len_ * batch_size_ * input_size_) * sizeof(T);
    output_size_list_.push_back(x_size);
    output_size_list_.push_back(h_size);
    output_size_list_.push_back(h_size);

    size_t workspace_size = 0;
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetRNNWorkspaceSize(handle_, rnn_desc_, seq_len_, dx_desc_.get(), &workspace_size),
                                "get workspace size failed");
    workspace_size_list_.push_back(workspace_size);
  }

 private:
  void DestroyResource() noexcept {
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyRNNDescriptor(rnn_desc_), "destroy rnn_desc failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyDropoutDescriptor(dropout_desc_), "destroy dropout_desc failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(dcx_desc_), "destroy dcx_desc_ failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(dhx_desc_), "destroy dhx_desc_ failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyFilterDescriptor(w_desc_), "destroy w_desc_ failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(cx_desc_), "destroy cx_desc_ failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(hx_desc_), "destroy hx_desc_ failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(dcy_desc_), "destroy dcy_desc_ failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(dhy_desc_), "destroy dhy_desc_ failed");
    DestroyTensorDescGrp();
  }
  void CreateTensorDescGrp() {
    int x_dims[3]{batch_size_, input_size_, 1};
    int y_dims[3]{batch_size_, hidden_size_ * (bidirectional_ ? 2 : 1), 1};

    dx_desc_ = mindspore::make_unique<cudnnTensorDescriptor_t[]>(seq_len_);
    y_desc_ = mindspore::make_unique<cudnnTensorDescriptor_t[]>(seq_len_);
    dy_desc_ = mindspore::make_unique<cudnnTensorDescriptor_t[]>(seq_len_);

    for (size_t i = 0; i < IntToSize(seq_len_); ++i) {
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&dx_desc_[i]), "create x_desc failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnSetTensorNdDescriptorEx(dx_desc_[i], CUDNN_TENSOR_NCHW, cudnn_data_type_, 3, x_dims),
        "set dx_desc failed");

      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&y_desc_[i]), "create y_desc failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnSetTensorNdDescriptorEx(y_desc_[i], CUDNN_TENSOR_NCHW, cudnn_data_type_, 3, y_dims), "set y_desc failed");

      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&dy_desc_[i]), "create dy_desc_ failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnSetTensorNdDescriptorEx(dy_desc_[i], CUDNN_TENSOR_NCHW, cudnn_data_type_, 3, y_dims),
        "set dy_desc_ failed");
    }
  }

  void DestroyTensorDescGrp() {
    for (size_t i = 0; i < IntToSize(seq_len_); ++i) {
      CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(dy_desc_[i]), "destroy dy_desc failed");
      CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(y_desc_[i]), "destroy y_desc failed");
      CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(dx_desc_[i]), "destroy x_desc failed");
    }
  }

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
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_NN_LSTM_GRAD_DATA_GPU_KERNEL_H_
