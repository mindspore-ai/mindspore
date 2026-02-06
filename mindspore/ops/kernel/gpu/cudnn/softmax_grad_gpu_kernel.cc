/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/cudnn/softmax_grad_gpu_kernel.h"
#include "mindspore/ops/infer/grad/softmax_grad.h"
#include "kernel/gpu/cuda_impl/cuda_ops/transpose_impl.cuh"

namespace mindspore {
namespace kernel {

bool SoftmaxGradGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&y_desc_),
                                      kernel_name_ + "create input_descriptor failed");
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  auto input_data_type = inputs[kIndex0]->dtype_id();
  type_id_size_ = abstract::TypeIdSize(input_data_type);
  cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(input_data_type));
  return true;
}

int SoftmaxGradGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  ResetResource();
  auto input_shape = LongVecToSizeVec(inputs[kIndex0]->GetShapeVector());
  shape_size_ = input_shape.size();
  algo_ = CUDNN_SOFTMAX_ACCURATE;
  std::vector<int> axis;
  auto axis_me = GetValue<std::vector<int64_t>>(primitive_->GetAttr("axis"));
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
  use_workspace_ = (axis_ != static_cast<int>(input_shape_.size()) - 1);
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensor4dDescriptor(y_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, SizeToInt(batch_size_),
                               SizeToInt(channel_size_), SizeToInt(height_), SizeToInt(width_)),
    kernel_name_ + "set input_descriptor failed");
  InitSizeLists();
  return KRET_OK;
}

template <typename T>
bool SoftmaxGradGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &workspace,
                                           const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  T *y_addr = GetDeviceAddress<T>(inputs, kIndex0);
  T *dy_addr = GetDeviceAddress<T>(inputs, kIndex1);
  T *dx_addr = GetDeviceAddress<T>(outputs, kIndex0);

  const float alpha = 1;
  const float beta = 0;
  if (!use_workspace_) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnSoftmaxBackward(cudnn_handle_, algo_, mode_, &alpha, y_desc_, y_addr,
                                                             y_desc_, dy_addr, &beta, y_desc_, dx_addr),
                                        kernel_name_ + "cudnnSoftmaxBackward failed");
  } else {
    T *transpose_y_addr = GetDeviceAddress<T>(workspace, kIndex0);
    T *transpose_dy_addr = GetDeviceAddress<T>(workspace, kIndex1);
    T *transpose_dx_addr = GetDeviceAddress<T>(workspace, kIndex2);

    TransposeInfo x_info;
    TransposeInfo y_info;
    for (size_t i = 0; i < shape_size_; ++i) {
      x_info.input_shape.push_back(static_cast<int64_t>(input_shape_[i]));
      x_info.perm.push_back(static_cast<int32_t>(transpose_axis_[i]));
      y_info.input_shape.push_back(static_cast<int64_t>(transpose_shape_[i]));
      y_info.perm.push_back(static_cast<int32_t>(transpose_axis_[i]));
    }

    size_t size = input_size_ / sizeof(T);
    auto s1 = CalTranspose<T, true>(size, y_addr, x_info, transpose_y_addr, reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(s1, "Transpose called by " + kernel_name_);
    auto s2 =
      CalTranspose<T, true>(size, dy_addr, x_info, transpose_dy_addr, reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(s2, "Transpose called by " + kernel_name_);
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSoftmaxBackward(cudnn_handle_, algo_, mode_, &alpha, y_desc_, transpose_y_addr, y_desc_, transpose_dy_addr,
                           &beta, y_desc_, transpose_dx_addr),
      kernel_name_ + "cudnnSoftmaxBackward failed");
    auto s3 =
      CalTranspose<T, true>(size, transpose_dx_addr, y_info, dx_addr, reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(s3, "Transpose called by " + kernel_name_);
  }
  return true;
}

std::vector<std::pair<KernelAttr, SoftmaxGradGpuKernelMod::SoftmaxGradGpuLaunchFunc>>
  SoftmaxGradGpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &SoftmaxGradGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &SoftmaxGradGpuKernelMod::LaunchKernel<half>},
};

std::vector<KernelAttr> SoftmaxGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, SoftmaxGradGpuKernelMod::SoftmaxGradGpuLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SoftmaxGrad, SoftmaxGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
