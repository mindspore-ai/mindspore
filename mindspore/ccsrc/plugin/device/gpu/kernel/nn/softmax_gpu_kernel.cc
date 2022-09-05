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

#include "plugin/device/gpu/kernel/nn/softmax_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/transpose_impl.cuh"
#include "mindspore/core/ops/softmax.h"
#include "mindspore/core/ops/log_softmax.h"

namespace mindspore {
namespace kernel {
bool SoftmaxGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->GetPrim()->name();
  cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&input_descriptor_),
                                      kernel_name_ + " create input_descriptor failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&output_descriptor_),
                                      kernel_name_ + " create output_descriptor failed");
  constexpr size_t input_num = 1;
  constexpr size_t output_num = 1;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  auto input_data_type = inputs.at(kIndex0)->GetDtype();
  type_id_size_ = abstract::TypeIdSize(input_data_type);
  cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(input_data_type));
  return true;
}

int SoftmaxGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs,
                                const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  // input, workspace and output will be assign in InitSizeLists.
  ResetResource();
  auto input_shape = LongVecToSizeVec(inputs[kIndex0]->GetShapeVector());
  shape_size_ = input_shape.size();
  if (kernel_name_ == "LogSoftmax") {
    algo_ = CUDNN_SOFTMAX_LOG;
    auto log_soft_max_ptr = std::dynamic_pointer_cast<ops::LogSoftmax>(base_operator);
    auto axis = LongToInt(log_soft_max_ptr->get_axis());
    InitSizeByAxis(input_shape, axis);
  } else {
    algo_ = CUDNN_SOFTMAX_ACCURATE;
    std::vector<int> axis;
    auto soft_max_ptr = std::dynamic_pointer_cast<ops::Softmax>(base_operator);
    auto axis_me = soft_max_ptr->get_axis();
    (void)std::transform(axis_me.begin(), axis_me.end(), std::back_inserter(axis),
                         [](const int64_t &value) { return LongToInt(value); });
    if (axis.size() < 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'axis' cannot be equal to 0, but got "
                        << axis.size();
    }
    if (axis.size() > 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'axis' cannot be greater than 1, but got "
                        << axis.size();
    }
    InitSizeByAxis(input_shape, axis[0]);
  }
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensor4dDescriptor(input_descriptor_, CUDNN_TENSOR_NCHW, cudnn_data_type_, SizeToInt(batch_size_),
                               SizeToInt(channel_size_), SizeToInt(height_), SizeToInt(width_)),
    "set input_descriptor failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensor4dDescriptor(output_descriptor_, CUDNN_TENSOR_NCHW, cudnn_data_type_, SizeToInt(batch_size_),
                               SizeToInt(channel_size_), SizeToInt(height_), SizeToInt(width_)),
    "set output_descriptor failed");
  InitSizeLists();
  return KRET_OK;
}

template <typename T>
bool SoftmaxGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                       const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  T *input_addr = GetDeviceAddress<T>(inputs, 0);
  MS_ERROR_IF_NULL_W_RET_VAL(input_addr, false);
  T *output_addr = GetDeviceAddress<T>(outputs, 0);
  MS_ERROR_IF_NULL_W_RET_VAL(output_addr, false);
  const float alpha = 1;
  const float beta = 0;
  if (need_transpose_ == false) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnSoftmaxForward(cudnn_handle_, algo_, mode_, &alpha, input_descriptor_,
                                                            input_addr, &beta, output_descriptor_, output_addr),
                                        kernel_name_ + " cudnnSoftmaxForward failed");
  } else {
    T *transpose_input_addr = GetDeviceAddress<T>(workspace, kIndex0);
    MS_ERROR_IF_NULL_W_RET_VAL(transpose_input_addr, false);
    T *transpose_output_addr = GetDeviceAddress<T>(workspace, kIndex1);
    MS_ERROR_IF_NULL_W_RET_VAL(transpose_output_addr, false);
    size_t *input_shape = GetDeviceAddress<size_t>(workspace, kIndex2);
    size_t *transpose_shape = GetDeviceAddress<size_t>(workspace, kIndex3);
    size_t *transpose_axis = GetDeviceAddress<size_t>(workspace, kIndex4);
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(input_shape, &input_shape_[0], workspace_size_, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      kernel_name_ + " cudaMemcpyAsync input_shape failed");
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(transpose_shape, &transpose_shape_[0], workspace_size_, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      kernel_name_ + " cudaMemcpyAsync input_shape failed");
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(transpose_axis, &transpose_axis_[0], workspace_size_, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      kernel_name_ + " cudaMemcpyAsync input_axis failed");
    size_t size = input_size_ / type_id_size_;
    CalTranspose(size, input_addr, input_shape, transpose_axis, shape_size_, transpose_input_addr,
                 reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSoftmaxForward(cudnn_handle_, algo_, mode_, &alpha, input_descriptor_, transpose_input_addr, &beta,
                          output_descriptor_, transpose_output_addr),
      kernel_name_ + " cudnnSoftmaxForward failed");
    CalTranspose(size, transpose_output_addr, transpose_shape, transpose_axis, shape_size_, output_addr,
                 reinterpret_cast<cudaStream_t>(stream_ptr));
  }
  return true;
}

std::vector<std::pair<KernelAttr, SoftmaxGpuKernelMod::SoftmaxGpuLaunchFunc>> SoftmaxGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &SoftmaxGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &SoftmaxGpuKernelMod::LaunchKernel<half>},
};

std::vector<KernelAttr> SoftmaxGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, SoftmaxGpuKernelMod::SoftmaxGpuLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Softmax, SoftmaxGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, LogSoftmax, SoftmaxGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
