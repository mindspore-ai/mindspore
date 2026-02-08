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

#include "kernel/gpu/cudnn/lstm_grad_weight_gpu_kernel.h"
#include <algorithm>
#include "mindspore/ops/infer/grad/lstm_grad_weight.h"

namespace mindspore {
namespace kernel {
bool LstmGradWeightGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;

  InitResource();
  cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(inputs[kIndex0]->dtype_id()));
  type_size_ = GetTypeByte(TypeIdToType(inputs[kIndex0]->dtype_id()));

  bidirectional_ = GetValue<bool>(primitive_->GetAttr("bidirectional"));
  input_size_ = static_cast<int>(GetValue<int64_t>(primitive_->GetAttr("input_size")));
  hidden_size_ = static_cast<int>(GetValue<int64_t>(primitive_->GetAttr("hidden_size")));
  num_layers_ = static_cast<int>(GetValue<int64_t>(primitive_->GetAttr("num_layers")));
  has_bias_ = GetValue<bool>(primitive_->GetAttr("has_bias"));
  dropout_ = GetValue<float>(primitive_->GetAttr("dropout"));
  auto proj_size = GetValue<int64_t>(primitive_->GetAttr("proj_size"));
  if (proj_size != 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', 'proj_size' could only be 0 in GPU, but got proj_size=" << proj_size;
  }
  return true;
}

int LstmGradWeightGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  DestroyTensorDescGrp();

  auto input_shape = inputs[kIndex0]->GetShapeVector();
  is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name_, "input");
  if (is_null_input_) {
    InitSizeLists();
    return KRET_OK;
  }
  if (input_shape.size() < kLstmInputShapeSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be less than 3, but got "
                      << input_shape.size();
  }
  seq_len_ = LongToInt(input_shape[0]);
  batch_size_ = LongToInt(input_shape[1]);

  cudnnRNNInputMode_t input_mode = CUDNN_LINEAR_INPUT;
  cudnnDirectionMode_t direction = bidirectional_ ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL;
  cudnnRNNMode_t rnn_mode = CUDNN_LSTM;
  cudnnRNNAlgo_t algo = CUDNN_RNN_ALGO_STANDARD;

  CreateTensorDescGrp();
  int hx_dims[3]{num_layers_ * (bidirectional_ ? 2 : 1), batch_size_, hidden_size_};

  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensorNdDescriptorEx(hx_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, kLstmInputShapeSize, hx_dims),
    "set hx_desc_ failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnSetDropoutDescriptor(dropout_desc_, handle_, dropout_, nullptr, 0, 0),
                                      "set dropout_desc failed");
  cudnnRNNBiasMode_t bias_mode = has_bias_ ? CUDNN_RNN_DOUBLE_BIAS : CUDNN_RNN_NO_BIAS;
#if CUDNN_VERSION < 8000
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetRNNDescriptor_v6(handle_, rnn_desc_, hidden_size_, num_layers_, dropout_desc_, input_mode, direction,
                             rnn_mode, algo, cudnn_data_type_),
    "set rnn_desc failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnSetRNNBiasMode(rnn_desc_, bias_mode), "set bias_mode failed");
#else
  cudnnMathType_t math_type = (cudnn_data_type_ == CUDNN_DATA_HALF) ? CUDNN_TENSOR_OP_MATH : CUDNN_FMA_MATH;
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetRNNDescriptor_v8(rnn_desc_, algo, rnn_mode, bias_mode, direction, input_mode, cudnn_data_type_,
                             cudnn_data_type_, math_type, input_size_, hidden_size_, hidden_size_, num_layers_,
                             dropout_desc_, 0),
    "set rnn_desc failed");
#endif
  auto shape_signed = outputs[kIndex0]->GetShapeVector();
  auto weight_shape = Convert2SizeTClipNeg(shape_signed);
  is_null_input_ = CHECK_SHAPE_NULL(weight_shape, kernel_name_, "weight");
  if (is_null_input_) {
    InitSizeLists();
    return KRET_OK;
  }
  if (weight_shape.size() < kLstmInputShapeSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of weight cannot be less than 3, but got "
                      << weight_shape.size();
  }
  size_t weight_size = weight_shape[0] * weight_shape[1] * weight_shape[2] * type_size_;

  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnGetRNNParamsSize(handle_, rnn_desc_, x_desc_[0], &weight_size_, cudnn_data_type_), "get weight_size_ failed");
  if (weight_size != weight_size_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the size of weight must be equal to " << weight_size_
                      << " but got " << weight_size;
  }
  int w_dims[3] = {SizeToInt(weight_size_ / type_size_), 1, 1};
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetFilterNdDescriptor(dw_desc_, cudnn_data_type_, CUDNN_TENSOR_NCHW, kLstmInputShapeSize, w_dims),
    "set dw_desc failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnGetRNNTrainingReserveSize(handle_, rnn_desc_, seq_len_, x_desc_.get(), &reserved_size_),
    "get reserve size failed");
  InitSizeLists();
  return KRET_OK;
}

template <typename T>
bool LstmGradWeightGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &workspace,
                                              const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  auto x_addr = GetDeviceAddress<T>(inputs, 0);
  auto hx_addr = GetDeviceAddress<T>(inputs, 1);
  auto y_addr = GetDeviceAddress<T>(inputs, 2);
  auto reserved_addr = GetDeviceAddress<T>(inputs, 3);
  auto states_addr = GetDeviceAddress<T>(inputs, 4);
  auto dw_addr = GetDeviceAddress<T>(outputs, 0);
  void *workspace_addr = GetDeviceAddress<T>(workspace, 0);

  if (!states_init_) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnRestoreDropoutDescriptor(dropout_desc_, handle_, dropout_, states_addr, inputs[kIndex4]->size(), 0),
      "restore dropout state failed");
    states_init_ = true;
  }

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemsetAsync(dw_addr, 0, outputs[0]->size(), reinterpret_cast<cudaStream_t>(stream_ptr)), "cudaMemSet Failed");

  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnRNNBackwardWeights(handle_, rnn_desc_, seq_len_, x_desc_.get(), x_addr, hx_desc_, hx_addr, y_desc_.get(),
                            y_addr, workspace_addr, workspace_size_list_[0], dw_desc_, dw_addr, reserved_addr,
                            reserved_size_),
    "launch lstm back weight kernel failed");

  return true;
}

std::vector<std::pair<KernelAttr, LstmGradWeightGpuKernelMod::LstmGradWeightGpuLaunchFunc>>
  LstmGradWeightGpuKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &LstmGradWeightGpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &LstmGradWeightGpuKernelMod::LaunchKernel<half>},
};

std::vector<KernelAttr> LstmGradWeightGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, LstmGradWeightGpuKernelMod::LstmGradWeightGpuLaunchFunc> &pair) {
                         return pair.first;
                       });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, LSTMGradWeight, LstmGradWeightGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
