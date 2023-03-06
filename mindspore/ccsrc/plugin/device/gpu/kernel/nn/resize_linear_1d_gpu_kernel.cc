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

#include "plugin/device/gpu/kernel/nn/resize_linear_1d_gpu_kernel.h"
#include "mindspore/core/abstract/utils.h"

namespace {
constexpr const size_t kResizeLinear1DInputsNum = 2;
constexpr const size_t kResizeLinear1DOutputsNum = 1;
constexpr const size_t kResizeInputDims = 3;
}  // namespace

namespace mindspore {
namespace kernel {
bool ResizeLinear1DGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs) {
  MS_ERROR_IF_NULL_W_RET_VAL(base_operator, false);

  auto kernel_ptr = std::dynamic_pointer_cast<ops::ResizeLinear1D>(base_operator);
  MS_ERROR_IF_NULL_W_RET_VAL(kernel_ptr, false);

  kernel_name_ = kernel_ptr->name();
  if (inputs.size() != kResizeLinear1DInputsNum || outputs.size() != kResizeLinear1DOutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input and output size must be " << kResizeLinear1DInputsNum
                  << " and " << kResizeLinear1DOutputsNum << ", but got " << inputs.size() << " and " << outputs.size();
    return false;
  }

  std::string coordinate_transformation_mode = kernel_ptr->get_coordinate_transformation_mode();
  if (coordinate_transformation_mode == "align_corners") {
    mode_ = ResizeLinearCoordinateTransformationMode::ALIGN_CORNERS;
  } else if (coordinate_transformation_mode == "half_pixel") {
    mode_ = ResizeLinearCoordinateTransformationMode::HALF_PIXEL;
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', coordinate_transformation_mode: " << coordinate_transformation_mode
                  << " not support now.";
    return false;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto pair = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!pair.first) {
    MS_LOG(ERROR) << "'" << kernel_name_ << "' does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[pair.second].second;
  return true;
}

int ResizeLinear1DGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs,
                                       const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost)) != KRET_OK) {
    return ret;
  }

  input_shape_ = std::vector<int64_t>(inputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                      inputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
  batch_ = LongToSize(input_shape_[kIndex0]);
  channel_ = LongToSize(input_shape_[kIndex1]);
  in_width_ = input_shape_[kIndex2];
  output_shape_ = std::vector<int64_t>(outputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                       outputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
  out_width_ = output_shape_[kIndex2];

  if (input_shape_.size() != kResizeInputDims) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension of 'input_x' should be greater than or equal to 1, but got "
                  << input_shape_.size() << ".";
    return KRET_RESIZE_FAILED;
  }
  return KRET_OK;
}

template <typename T>
bool ResizeLinear1DGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                              const std::vector<AddressPtr> &workspace,
                                              const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kResizeLinear1DInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kResizeLinear1DOutputsNum, kernel_name_);
  T *input = GetDeviceAddress<T>(inputs, kIndex0);
  MS_ERROR_IF_NULL_W_RET_VAL(input, false);

  T *output = GetDeviceAddress<T>(outputs, kIndex0);
  MS_ERROR_IF_NULL_W_RET_VAL(output, false);

  int64_t output_size = batch_ * channel_ * out_width_;

  ResizeLinear1D(mode_, output_size, in_width_, out_width_, input, output, device_id_,
                 reinterpret_cast<cudaStream_t>(stream_ptr));

  return true;
}

#define RESIZE_LINEAR_1D_GPU_REG(MS_T, MS_S, T) \
  KernelAttr().AddInputAttr(MS_T).AddInputAttr(MS_S).AddOutputAttr(MS_T), &ResizeLinear1DGpuKernelMod::LaunchKernel<T>

std::vector<std::pair<KernelAttr, ResizeLinear1DGpuKernelMod::ResizeLinear1DFunc>>
  ResizeLinear1DGpuKernelMod::func_list_ = {
    {RESIZE_LINEAR_1D_GPU_REG(kNumberTypeFloat16, kNumberTypeInt32, half)},
    {RESIZE_LINEAR_1D_GPU_REG(kNumberTypeFloat32, kNumberTypeInt32, float)},
    {RESIZE_LINEAR_1D_GPU_REG(kNumberTypeFloat64, kNumberTypeInt32, double)},
    {RESIZE_LINEAR_1D_GPU_REG(kNumberTypeFloat16, kNumberTypeInt64, half)},
    {RESIZE_LINEAR_1D_GPU_REG(kNumberTypeFloat32, kNumberTypeInt64, float)},
    {RESIZE_LINEAR_1D_GPU_REG(kNumberTypeFloat64, kNumberTypeInt64, double)},
};

std::vector<KernelAttr> ResizeLinear1DGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ResizeLinear1DFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ResizeLinear1D, ResizeLinear1DGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
