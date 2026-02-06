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

#include "kernel/gpu/cudnn/bias_add_gpu_kernel.h"

#include <map>

namespace mindspore {
namespace kernel {
bool BiasAddGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(inputs[kIndex1]->dtype_id()));
  return true;
}

int BiasAddGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto x_shape = LongVecToSizeVec(inputs[kIndex0]->GetShapeVector());
  auto num_dims = x_shape.size();
  is_null_input_ = CHECK_SHAPE_NULL(x_shape, kernel_name_, "input_x");

  format_ = inputs[kIndex2]->GetValueWithCheck<int64_t>();
  if (format_ == Format::NHWC) {
    input_shape_ = Convert2SizeTClipNeg(inputs[kIndex0]->GetShapeVector());
    bias_shape_ = Convert2SizeTClipNeg(inputs[kIndex1]->GetShapeVector());
    return KRET_OK;
  }

  // Expand to 4 dims for cudnnSetTensorNdDescriptorEx.
  constexpr size_t four_4D = 4;
  size_t pos = format_ == Format::NHWC ? num_dims - 1 : 1;
  size_t cudnn_dims = std::max(num_dims, four_4D);
  std::unique_ptr<int[]> x_dims = std::make_unique<int[]>(cudnn_dims);
  std::unique_ptr<int[]> b_dims = std::make_unique<int[]>(cudnn_dims);
  for (size_t i = 0; i < cudnn_dims; i++) {
    x_dims[i] = (i < num_dims) ? LongToInt(x_shape[i]) : 1;
    b_dims[i] = (i == pos) ? LongToInt(x_shape[i]) : 1;
  }

  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensorNdDescriptorEx(x_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, SizeToInt(cudnn_dims), x_dims.get()),
    "cudnnSetTensorNdDescriptor failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensorNdDescriptorEx(b_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, SizeToInt(cudnn_dims), b_dims.get()),
    "cudnnSetTensorNdDescriptor failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetOpTensorDescriptor(op_desc_, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN),
    "cudnnSetOpTensorDescriptor failed");
  return KRET_OK;
}

template <typename T>
bool BiasAddGpuKernelMod::ComputeNHWC(const T *src_addr, const T *bias_addr, T *output_addr, const size_t num_value,
                                      const size_t num_bias) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream_);
  auto status = CalBiasAddNHWC<T>(num_value, num_bias, src_addr, bias_addr, output_addr, device_id_, stream);
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

template <typename T>
bool BiasAddGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &workspace,
                                       const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  VARIABLE_NOT_USED(workspace);
  cuda_stream_ = stream_ptr;
  if (is_null_input_) {
    return true;
  }
  if (format_ == Format::NHWC) {
    T *src_addr = GetDeviceAddress<T>(inputs, 0);
    T *bias_addr = GetDeviceAddress<T>(inputs, 1);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);
    size_t num_value = 1;
    size_t num_bias = bias_shape_[0];
    for (size_t i = 0; i < input_shape_.size(); ++i) {
      num_value *= input_shape_[i];
    }
    ComputeNHWC<T>(src_addr, bias_addr, output_addr, num_value, num_bias);
    return true;
  }
  T *x_addr = GetDeviceAddress<T>(inputs, 0);
  T *b_addr = GetDeviceAddress<T>(inputs, 1);
  T *output_addr = GetDeviceAddress<T>(outputs, 0);
  const float alpha = 1;
  const float beta = 0;
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnOpTensor(cudnn_handle_, op_desc_, &alpha, x_desc_, x_addr, &alpha, b_desc_,
                                                    b_addr, &beta, x_desc_, output_addr),
                                      "cudnnOpTensor failed");
  return true;
}

std::vector<std::pair<KernelAttr, BiasAddGpuKernelMod::BiasAddLaunchFunc>> BiasAddGpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat16),
   &BiasAddGpuKernelMod::LaunchKernel<half>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &BiasAddGpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt8),
   &BiasAddGpuKernelMod::LaunchKernel<int8_t>}};

std::vector<KernelAttr> BiasAddGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, BiasAddGpuKernelMod::BiasAddLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, BiasAdd, BiasAddGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
