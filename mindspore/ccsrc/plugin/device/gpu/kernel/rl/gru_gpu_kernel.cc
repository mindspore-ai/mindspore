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

#include "plugin/device/gpu/kernel/rl/gru_gpu_kernel.h"

#include <algorithm>

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t DimOfTensor = 3;
constexpr size_t LeastWeightShape = 3;
constexpr size_t LeastInputShapeSize = 3;
constexpr size_t kInputsXIndex = 0;
constexpr size_t kInputsHxIndex = 1;
constexpr size_t kInputsWIndex = 2;
constexpr size_t kOutputsYIndex = 0;
constexpr size_t kOutputsHyIndex = 1;
constexpr size_t kOutputsReservedAddrIndex = 2;
constexpr size_t kOutputsStatedAddrIndex = 3;
constexpr size_t kGruInputsNum = 3;
constexpr size_t kGruOutputsNum = 4;
constexpr size_t kCudnnGRUInputDim = 3;
constexpr size_t kCudnnGRUHDim = 3;
constexpr size_t kCudnnGRUWDim = 3;
}  // namespace
bool GruGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                           const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kGruInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kGruOutputsNum, kernel_name_);
  InitResource();

  if (!GetCudnnDataType(TypeIdLabel(inputs[kInputsXIndex]->GetDtype()), &cudnn_data_type_)) {
    MS_LOG(ERROR) << kernel_name_ << ": Get cudnn data type failed.";
    return false;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }

  input_type_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kInputsXIndex).dtype);
  input_size_ = static_cast<int>(GetValue<int64_t>(base_operator->GetAttr("input_size")));
  hidden_size_ = static_cast<int>(GetValue<int64_t>(base_operator->GetAttr("hidden_size")));
  num_layers_ = static_cast<int>(GetValue<int64_t>(base_operator->GetAttr("num_layers")));
  has_bias_ = GetValue<bool>(base_operator->GetAttr("has_bias"));
  bidirectional_ = GetValue<bool>(base_operator->GetAttr("bidirectional"));
  dropout_ = GetValue<float>(base_operator->GetAttr("dropout"));
  kernel_func_ = func_list_[index].second;
  return true;
}

void GruGpuKernelMod::ResetResource() noexcept {
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
  reserved_size_ = 0;
}

int GruGpuKernelMod::CheckInputsShape(const std::vector<KernelTensorPtr> &inputs) {
  auto input_shape = inputs[kInputsXIndex]->GetShapeVector();  // (seq_len, batch_size, input_size)
  auto hx_shape = inputs[kInputsHxIndex]->GetShapeVector();    // (num_directions * num_layers, batch_size, hidden_size)
  auto w_shape = inputs[kInputsWIndex]->GetShapeVector();
  if (IsDynamic(input_shape) || IsDynamic(hx_shape) || IsDynamic(w_shape)) {
    return KRET_UNKNOWN_SHAPE;
  }
  (void)CheckAndConvertUtils::CheckInteger("input_dims", input_shape.size(), kEqual, SizeToLong(kCudnnGRUInputDim),
                                           kernel_name_);
  (void)CheckAndConvertUtils::CheckInteger("hx_dims", hx_shape.size(), kEqual, SizeToLong(kCudnnGRUHDim), kernel_name_);
  (void)CheckAndConvertUtils::CheckInteger("w_dims", w_shape.size(), kEqual, SizeToLong(kCudnnGRUWDim), kernel_name_);
  if (input_shape[kIndex1] != hx_shape[kIndex1]) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', input_shape[1] must be equal to hx_shape[1], but got "
                      << input_shape[kIndex1] << " and " << hx_shape[kIndex1] << ".";
  }
  (void)CheckAndConvertUtils::CheckInteger("input_shape[2]", input_shape[kIndex2], kEqual, IntToLong(input_size_),
                                           kernel_name_);
  int64_t real_num_layers = bidirectional_ ? IntToLong(num_layers_ * 2) : IntToLong(num_layers_);
  (void)CheckAndConvertUtils::CheckInteger("hx_shape[0]", hx_shape[kIndex0], kEqual, real_num_layers, kernel_name_);
  (void)CheckAndConvertUtils::CheckInteger("hx_shape[2]", hx_shape[kIndex2], kEqual, hidden_size_, kernel_name_);
  return KRET_OK;
}

int GruGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                            const std::vector<KernelTensorPtr> &outputs,
                            const std::map<uint32_t, tensor::TensorPtr> &) {
  ResetResource();
  auto ret = CheckInputsShape(inputs);
  if (ret != KRET_OK) {
    return ret;
  }
  auto input_shape = inputs[kInputsXIndex]->GetShapeVector();
  seq_len_ = LongToInt(input_shape[0]);
  batch_size_ = LongToInt(input_shape[1]);

  cudnnRNNInputMode_t input_mode = CUDNN_LINEAR_INPUT;
  cudnnDirectionMode_t direction = bidirectional_ ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL;
  cudnnRNNMode_t rnn_mode = CUDNN_GRU;
  cudnnRNNAlgo_t algo = CUDNN_RNN_ALGO_STANDARD;
  CreateTensorDescGrp();
  int hx_dims[3]{num_layers_ * (bidirectional_ ? 2 : 1), batch_size_, hidden_size_};
  CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(
    cudnnSetTensorNdDescriptorEx(hx_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, DimOfTensor, hx_dims),
    "set hx_desc failed");
  CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(
    cudnnSetTensorNdDescriptorEx(cx_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, DimOfTensor, hx_dims),
    "set cx_desc failed");
  CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(
    cudnnSetTensorNdDescriptorEx(hy_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, DimOfTensor, hx_dims),
    "set hy_desc failed");
  CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(
    cudnnSetTensorNdDescriptorEx(cy_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, DimOfTensor, hx_dims),
    "set cy_desc failed");
  CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnSetDropoutDescriptor(dropout_desc_, handle_, dropout_, nullptr, 0, 0),
                                     "set dropout_desc failed");
  cudnnRNNBiasMode_t bias_mode = has_bias_ ? CUDNN_RNN_DOUBLE_BIAS : CUDNN_RNN_NO_BIAS;
#if CUDNN_VERSION < 8000
  cudnnMathType_t math_type = (cudnn_data_type_ == CUDNN_DATA_HALF) ? CUDNN_TENSOR_OP_MATH : CUDNN_DEFAULT_MATH;
  CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(
    cudnnSetRNNDescriptor_v6(handle_, rnn_desc_, hidden_size_, num_layers_, dropout_desc_, input_mode, direction,
                             rnn_mode, algo, cudnn_data_type_),
    "set rnn_desc failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnSetRNNMatrixMathType(rnn_desc_, math_type), "Set math type failed.");
  CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnSetRNNBiasMode(rnn_desc_, bias_mode), "set bias_mode failed");
#else
  cudnnMathType_t math_type = (cudnn_data_type_ == CUDNN_DATA_HALF) ? CUDNN_TENSOR_OP_MATH : CUDNN_FMA_MATH;
  CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(
    cudnnSetRNNDescriptor_v8(rnn_desc_, algo, rnn_mode, bias_mode, direction, input_mode, cudnn_data_type_,
                             cudnn_data_type_, math_type, input_size_, hidden_size_, hidden_size_, num_layers_,
                             dropout_desc_, 0),
    "set rnn_desc failed");
#endif
  auto weight_shape = inputs[kInputsWIndex]->GetShapeVector();
  size_t weight_size = LongToSizeClipNeg(weight_shape[0] * weight_shape[1] * weight_shape[kIndex2]) * input_type_size_;
  CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(
    cudnnGetRNNParamsSize(handle_, rnn_desc_, x_desc_[0], &weight_size_, cudnn_data_type_), "get weight_size_ failed");
  if (weight_size != weight_size_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the size of weight should be equal to " << weight_size_
                      << " but got " << weight_size;
  }
  int w_dims[3] = {SizeToInt(weight_size_ / input_type_size_), 1, 1};
  CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(
    cudnnSetFilterNdDescriptor(w_desc_, cudnn_data_type_, CUDNN_TENSOR_NCHW, DimOfTensor, w_dims), "set w_desc failed");
  CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(
    cudnnGetRNNTrainingReserveSize(handle_, rnn_desc_, seq_len_, x_desc_.get(), &reserved_size_),
    "get reserve size failed");

  InitSizeLists();
  return KRET_OK;
}

template <typename T>
bool GruGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                   const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  cuda_stream_ = reinterpret_cast<cudaStream_t>(stream_ptr);
  VARIABLE_NOT_USED(stream_ptr);

  auto x_addr = GetDeviceAddress<T>(inputs, kInputsXIndex);
  auto hx_addr = GetDeviceAddress<T>(inputs, kInputsHxIndex);
  auto cx_addr = nullptr;
  auto w_addr = GetDeviceAddress<T>(inputs, kInputsWIndex);
  auto y_addr = GetDeviceAddress<T>(outputs, kOutputsYIndex);
  auto hy_addr = GetDeviceAddress<T>(outputs, kOutputsHyIndex);
  auto cy_addr = nullptr;
  auto reserved_addr = GetDeviceAddress<T>(outputs, kOutputsReservedAddrIndex);
  auto states_addr = GetDeviceAddress<T>(outputs, kOutputsStatedAddrIndex);
  void *workspace_addr = GetPossiblyNullDeviceAddress<T>(workspace, 0);

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_ptr)),
                                     "stream synchronize failed.");

  if (!states_init_) {
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnSetDropoutDescriptor(dropout_desc_, handle_, dropout_, states_addr,
                                                                 output_size_list_[kOutputsStatedAddrIndex], 0),
                                       "set dropout descriptor failed. Possible reasons: the GPU is out of memory.");
    states_init_ = true;
  }

  CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(
    cudnnRNNForwardTraining(handle_, rnn_desc_, seq_len_, x_desc_.get(), x_addr, hx_desc_, hx_addr, cx_desc_, cx_addr,
                            w_desc_, w_addr, y_desc_.get(), y_addr, hy_desc_, hy_addr, cy_desc_, cy_addr,
                            workspace_addr, workspace_size_list_[0], reserved_addr, reserved_size_),
    "launch gru kernel failed");

  return true;
}

void GruGpuKernelMod::InitResource() {
  handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
  CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnCreateTensorDescriptor(&hx_desc_), "create hx_desc failed");
  CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnCreateTensorDescriptor(&cx_desc_), "create cx_desc failed");
  CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnCreateFilterDescriptor(&w_desc_), "create w_desc failed");
  CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnCreateTensorDescriptor(&hy_desc_), "create hy_desc failed");
  CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnCreateTensorDescriptor(&cy_desc_), "create cy_desc failed");
  CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnCreateDropoutDescriptor(&dropout_desc_), "create dropout_desc failed");
  CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnCreateRNNDescriptor(&rnn_desc_), "create rnn_desc failed");
}

void GruGpuKernelMod::DestroyResource() noexcept {
  CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyRNNDescriptor(rnn_desc_), "destroy rnn_desc failed");
  CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyDropoutDescriptor(dropout_desc_), "destroy dropout_desc failed");
  CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(cy_desc_), "destroy cy_desc failed");
  CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(hy_desc_), "destroy hy_desc failed");
  CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyFilterDescriptor(w_desc_), "destroy w_desc failed");
  CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(hx_desc_), "destroy hx_desc failed");
  CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(cx_desc_), "destroy cx_desc failed");

  for (size_t i = 0; i < IntToSize(seq_len_); ++i) {
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(y_desc_[i]), "destroy y_desc failed");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(x_desc_[i]), "destroy x_desc failed");
  }
}

std::vector<KernelAttr> GruGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, GruGpuKernelFunc> &pair) { return pair.first; });
  return support_list;
}

void GruGpuKernelMod::InitSizeLists() {
  size_t x_size = IntToSize(seq_len_ * batch_size_ * input_size_) * input_type_size_;

  size_t h_size = 0;
  CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnGetTensorSizeInBytes(hx_desc_, &h_size), "get h size failed");

  input_size_list_.push_back(x_size);
  input_size_list_.push_back(h_size);
  input_size_list_.push_back(weight_size_);

  size_t y_size = IntToSize(seq_len_ * batch_size_ * hidden_size_ * (bidirectional_ ? 2 : 1)) * input_type_size_;
  output_size_list_.push_back(y_size);
  output_size_list_.push_back(h_size);
  output_size_list_.push_back(reserved_size_);
  size_t state_size = 0;
  CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDropoutGetStatesSize(handle_, &state_size), "get dropout states size failed");
  output_size_list_.push_back(state_size);

  size_t workspace_size = 0;
  CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(
    cudnnGetRNNWorkspaceSize(handle_, rnn_desc_, seq_len_, x_desc_.get(), &workspace_size),
    "get workspace size failed");
  workspace_size_list_.push_back(workspace_size);
}

void GruGpuKernelMod::CreateTensorDescGrp() {
  int x_dims[3]{batch_size_, input_size_, 1};
  int y_dims[3]{batch_size_, hidden_size_ * (bidirectional_ ? 2 : 1), 1};

  x_desc_ = std::make_unique<cudnnTensorDescriptor_t[]>(seq_len_);
  y_desc_ = std::make_unique<cudnnTensorDescriptor_t[]>(seq_len_);
  for (size_t i = 0; i < IntToSize(seq_len_); ++i) {
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnCreateTensorDescriptor(&x_desc_[i]), "create x_desc failed");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(
      cudnnSetTensorNdDescriptorEx(x_desc_[i], CUDNN_TENSOR_NCHW, cudnn_data_type_, DimOfTensor, x_dims),
      "set x_desc failed");

    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnCreateTensorDescriptor(&y_desc_[i]), "create y_desc failed");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(
      cudnnSetTensorNdDescriptorEx(y_desc_[i], CUDNN_TENSOR_NCHW, cudnn_data_type_, DimOfTensor, y_dims),
      "set y_desc failed");
  }
}

std::vector<std::pair<KernelAttr, GruGpuKernelFunc>> GruGpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &GruGpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &GruGpuKernelMod::LaunchKernel<half>}};

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, CudnnGRU, GruGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
