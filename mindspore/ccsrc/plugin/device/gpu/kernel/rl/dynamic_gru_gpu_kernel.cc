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

#include "plugin/device/gpu/kernel/rl/dynamic_gru_gpu_kernel.h"
#include <functional>
#include <utility>
#include <string>
#include <algorithm>
#include "mindspore/core/abstract/utils.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr int kDynamicGruGpuInputsNum = 4;
constexpr int kDynamicGruGpuOutputsNum = 4;
constexpr size_t DimOfTensor = 3;
}  // namespace

bool DynamicGruGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs) {
#if CUDNN_VERSION >= 8000
  MS_ERROR_IF_NULL_W_RET_VAL(base_operator, false);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kDynamicGruGpuInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kDynamicGruGpuOutputsNum, kernel_name_);
  InitResource();
  if (!GetCudnnDataType(TypeIdLabel(inputs[0]->GetDtype()), &cudnn_data_type_)) {
    MS_LOG(ERROR) << kernel_name_ << " get cudnn data type failed.";
    return false;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }

  auto input_shape = inputs[kIndex0]->GetShapeVector();
  input_type_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).first);
  max_seq_len_ = static_cast<int>(input_shape[0]);
  auto input_size = static_cast<int>(input_shape[kIndexTwo]);
  input_size_ = static_cast<int>(GetValue<int64_t>(base_operator->GetAttr("input_size")));
  if (input_size != input_size_) {
    MS_LOG(EXCEPTION) << "The input size from inputs :" << input_size
                      << " is not equal to input size from attrs: " << input_size_;
  }
  hidden_size_ = static_cast<int>(GetValue<int64_t>(base_operator->GetAttr("hidden_size")));
  num_layers_ = static_cast<int>(GetValue<int64_t>(base_operator->GetAttr("num_layers")));
  has_bias_ = GetValue<bool>(base_operator->GetAttr("has_bias"));
  bidirectional_ = GetValue<bool>(base_operator->GetAttr("bidirectional"));
  dropout_ = GetValue<float>(base_operator->GetAttr("dropout"));
  is_train_ = GetValue<bool>(base_operator->GetAttr("is_train"));
  kernel_func_ = func_list_[index].second;
#endif
  return true;
}

void DynamicGruGpuKernelMod::ResetResource() noexcept {
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
  reserved_size_ = 0;
}

int DynamicGruGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs,
                                   const std::map<uint32_t, tensor::TensorPtr> &) {
#if CUDNN_VERSION >= 8000
  ResetResource();
  auto input_shape = inputs[kIndex0]->GetShapeVector();
  batch_size_ = static_cast<int>(input_shape[1]);
  if (batch_size_ == -1) {
    return KRET_UNKNOWN_SHAPE;
  }
  CreateTensorNdDesc(inputs);
  size_t workspace_size = 0;
  cudnnForwardMode_t rnn_fwd_mode = is_train_ ? CUDNN_FWD_MODE_TRAINING : CUDNN_FWD_MODE_INFERENCE;
  // create a temp x_desc to get the workspace and reservespace size.
  x_desc_max_ = std::make_unique<cudnnRNNDataDescriptor_t>();
  float padding_fill = 0.0f;
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateRNNDataDescriptor(x_desc_max_.get()), "create x_desc_max failed");
  for (size_t i = 0; i < size_t(batch_size_); ++i) {
    seq_lens_.push_back(max_seq_len_);
  }
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetRNNDataDescriptor(*(x_desc_max_.get()), cudnn_data_type_, CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,
                              max_seq_len_, batch_size_, input_size_, seq_lens_.data(),
                              reinterpret_cast<void *>(&padding_fill)),
    "set x_desc_max failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnGetRNNTempSpaceSizes(handle_, rnn_desc_, rnn_fwd_mode, *(x_desc_max_.get()), &workspace_size, &reserved_size_),
    "get workspace_size and reserved_size size failed");
  workspace_size_list_.push_back(workspace_size);
  size_t x_size = IntToSize(max_seq_len_ * batch_size_ * input_size_) * input_type_size_;
  size_t seq_len_size = IntToSize(batch_size_) * sizeof(int32_t);
  size_t h_size = 0;
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetTensorSizeInBytes(hx_desc_, &h_size), "get h size failed");

  input_size_list_.push_back(x_size);
  input_size_list_.push_back(h_size);
  input_size_list_.push_back(weight_size_);
  input_size_list_.push_back(seq_len_size);

  size_t y_size = IntToSize(max_seq_len_ * batch_size_ * hidden_size_ * (bidirectional_ ? 2 : 1)) * input_type_size_;
  output_size_list_.push_back(y_size);
  output_size_list_.push_back(h_size);
  output_size_list_.push_back(reserved_size_);
  size_t state_size = 0;
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnDropoutGetStatesSize(handle_, &state_size),
                                      "get dropout states size failed");
  output_size_list_.push_back(state_size);
#endif
  return KRET_OK;
}

template <typename T>
bool DynamicGruGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                          const std::vector<AddressPtr> &workspace,
                                          const std::vector<AddressPtr> &outputs, void *stream_ptr) {
#if CUDNN_VERSION >= 8000
  cuda_stream_ = reinterpret_cast<cudaStream_t>(stream_ptr);

  VARIABLE_NOT_USED(stream_ptr);
  auto x_addr = GetDeviceAddress<T>(inputs, 0);
  auto hx_addr = GetDeviceAddress<T>(inputs, 1);
  auto w_addr = GetDeviceAddress<T>(inputs, 2);
  auto seq_addr = GetDeviceAddress<int>(inputs, 3);
  auto cx_addr = nullptr;
  auto y_addr = GetDeviceAddress<T>(outputs, 0);
  auto hy_addr = GetDeviceAddress<T>(outputs, 1);
  auto cy_addr = nullptr;
  auto reserved_addr = GetPossiblyNullDeviceAddress<T>(outputs, 2);
  auto states_addr = GetPossiblyNullDeviceAddress<T>(outputs, 3);
  void *workspace_addr = GetPossiblyNullDeviceAddress<T>(workspace, 0);
  cudnnForwardMode_t rnn_fwd_mode = is_train_ ? CUDNN_FWD_MODE_TRAINING : CUDNN_FWD_MODE_INFERENCE;

  // copy seq_lens_ from seq_addr
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpy(seq_lens_.data(), seq_addr, batch_size_ * sizeof(int32_t), cudaMemcpyDeviceToHost),
    "cudaMemcpy seq_lengths from device to host failed.");
  CreateRNNDataDescGrp();
  if (!states_init_ && is_train_ && dropout_ > 0) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetDropoutDescriptor(dropout_desc_, handle_, dropout_, states_addr, output_size_list_[kIndexThree], 0),
      "set dropout descriptor failed. Possible reasons: the GPU is out of memory.");
    states_init_ = true;
  }

  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnRNNForward(/* handle= */ handle_, /* rnnDesc= */ rnn_desc_,
                    /* fwdMode= */ rnn_fwd_mode,
                    /* devSeqLengths= */ reinterpret_cast<const int *>(seq_addr),
                    /* xDesc= */ *(x_desc_.get()), /* x= */ x_addr,
                    /* yDesc= */ *(y_desc_.get()), /* y= */ y_addr,
                    /* hDesc= */ hx_desc_, /* hx= */ hx_addr,
                    /* hy= */ hy_addr,
                    /* cxDesc= */ cx_desc_, /* cx= */ cx_addr,
                    /* cy= */ cy_addr,
                    /* weightSpaceSize= */ weight_size_,
                    /* weightSpace= */ w_addr,
                    /* workSpaceSize= */ workspace_size_list_[0], /* workspace= */ workspace_addr,
                    /* reserveSpaceSize= */ reserved_size_,
                    /* reserveSpace= */ reserved_addr),
    "launch gru kernel failed");
#endif
  return true;
}

std::vector<std::pair<KernelAttr, DynamicGruGpuKernelMod::DynamicGruGpuFunc>> DynamicGruGpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &DynamicGruGpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &DynamicGruGpuKernelMod::LaunchKernel<half>}};

void DynamicGruGpuKernelMod::CreateTensorNdDesc(const std::vector<KernelTensorPtr> &inputs) {
#if CUDNN_VERSION >= 8000
  auto weight_shape = inputs[kIndex2]->GetShapeVector();
  cudnnRNNInputMode_t input_mode = CUDNN_LINEAR_INPUT;
  cudnnDirectionMode_t direction = bidirectional_ ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL;
  cudnnRNNMode_t rnn_mode = CUDNN_GRU;
  cudnnRNNAlgo_t algo = CUDNN_RNN_ALGO_STANDARD;
  int hx_dims[] = {num_layers_ * (bidirectional_ ? 2 : 1), batch_size_, hidden_size_};
  int strides[] = {hx_dims[1] * hx_dims[2], hx_dims[2], 1};
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensorNdDescriptor(hx_desc_, cudnn_data_type_, DimOfTensor, hx_dims, strides), "set hx_desc failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensorNdDescriptor(cx_desc_, cudnn_data_type_, DimOfTensor, hx_dims, strides), "set cx_desc failed");
  if (is_train_ && dropout_ > 0) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnSetDropoutDescriptor(dropout_desc_, handle_, dropout_, nullptr, 0, 0),
                                        "set dropout_desc failed");
  }
  cudnnRNNBiasMode_t bias_mode = has_bias_ ? CUDNN_RNN_DOUBLE_BIAS : CUDNN_RNN_NO_BIAS;

  cudnnMathType_t math_type = (cudnn_data_type_ == CUDNN_DATA_HALF) ? CUDNN_TENSOR_OP_MATH : CUDNN_FMA_MATH;
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetRNNDescriptor_v8(rnn_desc_, algo, rnn_mode, bias_mode, direction, input_mode, cudnn_data_type_,
                             cudnn_data_type_, math_type, input_size_, hidden_size_, hidden_size_, num_layers_,
                             dropout_desc_, CUDNN_RNN_PADDED_IO_ENABLED),
    "set rnn_desc failed");

  size_t weight_size = weight_shape[0] * weight_shape[1] * weight_shape[kIndexTwo] * input_type_size_;
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetRNNWeightSpaceSize(handle_, rnn_desc_, &weight_size_),
                                      "get weight_size_ failed");
  if (weight_size != weight_size_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the size of weight should be equal to " << weight_size_
                      << " but got " << weight_size;
  }
#endif
}

void DynamicGruGpuKernelMod::CreateRNNDataDescGrp() {
#if CUDNN_VERSION >= 8000
  x_desc_ = std::make_unique<cudnnRNNDataDescriptor_t>();
  y_desc_ = std::make_unique<cudnnRNNDataDescriptor_t>();
  float padding_fill = 0.0f;
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateRNNDataDescriptor(x_desc_.get()), "create x_desc failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetRNNDataDescriptor(*(x_desc_.get()), cudnn_data_type_, CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,
                              max_seq_len_, batch_size_, input_size_, seq_lens_.data(),
                              reinterpret_cast<void *>(&padding_fill)),
    "set x_desc failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateRNNDataDescriptor(y_desc_.get()), "create y_desc failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetRNNDataDescriptor(*(y_desc_.get()), cudnn_data_type_, CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,
                              max_seq_len_, batch_size_, hidden_size_ * (bidirectional_ ? 2 : 1), seq_lens_.data(),
                              reinterpret_cast<void *>(&padding_fill)),
    "set x_desc failed");
#endif
}

std::vector<KernelAttr> DynamicGruGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, DynamicGruGpuKernelMod::DynamicGruGpuFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, GRUV2, DynamicGruGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
