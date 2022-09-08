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

#include "plugin/device/gpu/kernel/nn/lstm_grad_data_gpu_kernel.h"
#include <algorithm>
#include "mindspore/core/ops/grad/lstm_grad_data.h"

namespace mindspore {
namespace kernel {
bool LstmGradDataGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs) {
  MS_ERROR_IF_NULL_W_RET_VAL(base_operator, false);
  kernel_name_ = base_operator->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;

  InitResource();
  cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(inputs[kIndex0]->GetDtype()));
  type_size_ = GetTypeByte(TypeIdToType(inputs[kIndex0]->GetDtype()));

  auto kernel_ptr = std::dynamic_pointer_cast<ops::LSTMGradData>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "Cast LSTMGradData ops failed!";
    return false;
  }
  bidirectional_ = kernel_ptr->get_bidirectional();
  input_size_ = kernel_ptr->get_input_size();
  hidden_size_ = kernel_ptr->get_hidden_size();
  num_layers_ = kernel_ptr->get_num_layers();
  has_bias_ = kernel_ptr->get_has_bias();
  dropout_ = kernel_ptr->get_dropout();
  return true;
}

int LstmGradDataGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs,
                                     const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }

  auto input_shape = inputs[kIndex0]->GetShapeVector();
  is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name_, "input");
  if (is_null_input_) {
    InitSizeLists();
    return KRET_OK;
  }
  if (input_shape.size() < kInputDimLowerLimit) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be less than 2, but got "
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
    cudnnSetTensorNdDescriptorEx(dhy_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, k3DSize, hx_dims),
    "set dhy_desc_ failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensorNdDescriptorEx(dcy_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, k3DSize, hx_dims),
    "set dcy_desc_ failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensorNdDescriptorEx(hx_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, k3DSize, hx_dims),
    "set hx_desc_ failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensorNdDescriptorEx(cx_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, k3DSize, hx_dims),
    "set cx_desc_ failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensorNdDescriptorEx(dhx_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, k3DSize, hx_dims),
    "set dhx_desc_ failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensorNdDescriptorEx(dcx_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, k3DSize, hx_dims),
    "set dcx_desc_ failed");
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
  const size_t kPrevOutput4th = 4;
  auto weight_shape = inputs[kPrevOutput4th]->GetShapeVector();
  is_null_input_ = CHECK_SHAPE_NULL(weight_shape, kernel_name_, "weight");
  if (is_null_input_ || IsDynamic(weight_shape)) {
    InitSizeLists();
    return KRET_OK;
  }
  if (weight_shape.size() < kWeightDimLowerLimit) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of weight cannot be less than 3, but got "
                      << weight_shape.size();
  }
  size_t weight_size = LongToSizeClipNeg(weight_shape[0] * weight_shape[1] * weight_shape[2]) * type_size_;
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnGetRNNParamsSize(handle_, rnn_desc_, dx_desc_[0], &weight_size_, cudnn_data_type_), "get weight_size_ failed");
  if (weight_size != weight_size_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the size of weight must be equal to " << weight_size_
                      << " but got " << weight_size;
  }
  int w_dims[3] = {SizeToInt(weight_size_ / type_size_), 1, 1};
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetFilterNdDescriptor(w_desc_, cudnn_data_type_, CUDNN_TENSOR_NCHW, k3DSize, w_dims), "set w_desc failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnGetRNNTrainingReserveSize(handle_, rnn_desc_, seq_len_, dx_desc_.get(), &reserved_size_), "get size failed");
  InitSizeLists();
  return KRET_OK;
}

template <typename T>
bool LstmGradDataGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                            const std::vector<AddressPtr> &workspace,
                                            const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
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
  void *workspace_addr = GetPossiblyNullDeviceAddress<T>(workspace, 0);

  if (!states_init_) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnRestoreDropoutDescriptor(dropout_desc_, handle_, dropout_, states_addr, input_size_list_[kIndex8], 0),
      "restore dropout state failed");
    states_init_ = true;
  }

  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnRNNBackwardData(handle_, rnn_desc_, seq_len_, y_desc_.get(), y_addr, dy_desc_.get(), dy_addr, dhy_desc_,
                         dhy_addr, dcy_desc_, dcy_addr, w_desc_, w_addr, hx_desc_, hx_addr, cx_desc_, cx_addr,
                         dx_desc_.get(), dx_addr, dhx_desc_, dhx_addr, dcx_desc_, dcx_addr, workspace_addr,
                         workspace_size_list_[0], reserved_addr, reserved_size_),
    "launch lstm back data kernel failed");

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_ptr)),
                                     "stream synchronize failed.");
  return true;
}

std::vector<std::pair<KernelAttr, LstmGradDataGpuKernelMod::LstmGradDataGpuLaunchFunc>>
  LstmGradDataGpuKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &LstmGradDataGpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &LstmGradDataGpuKernelMod::LaunchKernel<half>},
};

std::vector<KernelAttr> LstmGradDataGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, LstmGradDataGpuKernelMod::LstmGradDataGpuLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, LSTMGradData, LstmGradDataGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
